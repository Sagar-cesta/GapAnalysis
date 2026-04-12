"""
MCP Chat Backend
================
FastAPI server that:
  1. Exposes /api/tools  — lists all MCP tools from LiteLLM
  2. Exposes /api/chat   — runs the full agentic loop and streams events back
  3. Serves  index.html  at /

Run:  python app.py
"""

import json
import os
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

LITELLM_BASE_URL = "http://localhost:4000"
LITELLM_API_KEY  = os.getenv("LITELLM_MASTER_KEY", "sk-local-test-key")
MODEL            = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful assistant with access to real-world tools via MCP (Model Context Protocol).

IMPORTANT RULES:
- You MUST use the available tools whenever a question involves real-world data, live information, files, repositories, calendars, or the web.
- NEVER say you "don't have access" or "can't retrieve" something if a relevant tool exists — use the tool instead.
- NEVER invent or guess API keys. If a service requires an API key you don't have, use a free alternative instead.
- For GitHub questions: use github-* tools (e.g. github-get_me to get the current user, github-search_repositories to find repos).
- For web content / scraping: use fetch-fetch.
- For files/directories: use filesystem-* tools.
- For calendar/scheduling: use zapier-google_calendar_* tools.
- Always prefer tool results over your training knowledge for any live or user-specific data.

BLOCKED SITES — NEVER fetch these (robots.txt blocks automated access):
- google.com, google.co.*, news.google.com, youtube.com
- reddit.com, twitter.com, instagram.com, facebook.com, linkedin.com
- medium.com, wikipedia.org (use the API instead)
- Any URL that starts with https://www.google

FREE SERVICES TO USE (no API key needed):
- Weather: You MUST use wttr.in. NEVER use openweathermap, weatherapi.com, weatherstack, or any other weather service.
  Exact URL format: https://wttr.in/CITY_NAME?format=j1
  Examples:
    Raleigh NC  → https://wttr.in/Raleigh,NC?format=j1
    New York    → https://wttr.in/New+York?format=j1
    London      → https://wttr.in/London?format=j1
  The response is JSON with current_condition[0] containing temp_F, temp_C, weatherDesc, humidity, windspeedMiles.
- IP info: https://ipinfo.io/json
- Public holidays: https://date.nager.at/api/v3/PublicHolidays/{year}/{countryCode}
- Current time / date: https://worldtimeapi.org/api/timezone/America/New_York
- News (AI/Tech): https://hnrss.org/frontpage  (Hacker News — RSS, always open)
- News (BBC Tech): https://feeds.bbci.co.uk/news/technology/rss.xml
- News (TechCrunch): https://techcrunch.com/feed/
- News (any topic search): https://hnrss.org/search?q=TOPIC&count=5
  Example AI news: https://hnrss.org/search?q=artificial+intelligence&count=5
- Wikipedia summary: https://en.wikipedia.org/api/rest_v1/page/summary/TOPIC
  Example: https://en.wikipedia.org/api/rest_v1/page/summary/Docker_(software)
"""

app = FastAPI(title="MCP Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = AsyncOpenAI(
    base_url=f"{LITELLM_BASE_URL}/v1",
    api_key=LITELLM_API_KEY,
)


# ── URL guard: intercept fetch-fetch calls with blocked domains ────────────────

BLOCKED_DOMAINS = [
    "google.com", "google.co.", "reuters.com", "bloomberg.com",
    "nytimes.com", "wsj.com", "ft.com", "twitter.com", "x.com",
    "reddit.com", "instagram.com", "facebook.com", "linkedin.com",
    "medium.com", "youtube.com", "bing.com", "yahoo.com",
]

NEWS_KEYWORDS = [
    "news", "conflict", "war", "iran", "politics", "election",
    "economy", "market", "stock", "ai", "artificial intelligence",
    "llm", "model", "tech", "technology", "science",
]

def rewrite_fetch_url(url: str, original_args: dict) -> tuple[str, str | None]:
    """
    If the URL is on a blocked domain, replace it with a working alternative.
    Returns (new_url, explanation_or_None).
    """
    url_lower = url.lower()

    if not any(b in url_lower for b in BLOCKED_DOMAINS):
        return url, None  # URL is fine

    # Extract a search keyword from the original URL
    import re, urllib.parse
    query = ""
    for param in ["q", "query", "search", "blob", "keyword"]:
        match = re.search(rf"[?&]{param}=([^&]+)", url_lower)
        if match:
            query = urllib.parse.unquote_plus(match.group(1)).replace("+", " ")
            break

    if not query:
        # Try to pull a topic from the path
        path = urllib.parse.urlparse(url).path
        query = path.strip("/").replace("-", " ").replace("_", " ").split("/")[-1]

    if query:
        encoded = urllib.parse.quote_plus(query)
        new_url = f"https://hnrss.org/search?q={encoded}&count=10"
        return new_url, f"Redirected blocked domain to Hacker News RSS: {new_url}"
    else:
        return "https://hnrss.org/frontpage?count=10", \
               "Redirected blocked domain to Hacker News front page"


# ── Helpers ────────────────────────────────────────────────────────────────────

async def fetch_mcp_tools() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{LITELLM_BASE_URL}/v1/mcp/tools",
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
            )
            resp.raise_for_status()
        raw = resp.json().get("tools", [])
    except Exception:
        return []   # LiteLLM not reachable — return empty list gracefully
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t.get("description") or "",
                "parameters":  t.get("inputSchema") or {"type": "object", "properties": {}},
            },
        }
        for t in raw
    ]


async def execute_tool(name: str, arguments: dict) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id":      "1",
        "method":  "tools/call",
        "params":  {"name": name, "arguments": arguments},
    }
    result_data: dict = {}
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream(
            "POST",
            f"{LITELLM_BASE_URL}/mcp/call_tool",
            headers={
                "Authorization": f"Bearer {LITELLM_API_KEY}",
                "Content-Type":  "application/json",
                "Accept":        "application/json, text/event-stream",
            },
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    try:
                        result_data = json.loads(line[5:].strip())
                        break
                    except json.JSONDecodeError:
                        pass

    result  = result_data.get("result", result_data)
    content = result.get("content", [])
    parts   = [c.get("text", "") for c in content if c.get("type") == "text"]
    return "\n".join(parts) if parts else json.dumps(result)


def sse(event: str, data: dict) -> str:
    """Format a server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("index.html")


@app.get("/api/tools")
async def get_tools():
    tools = await fetch_mcp_tools()
    servers: dict[str, list] = {}
    for t in tools:
        prefix = t["function"]["name"].split("-")[0]
        servers.setdefault(prefix, []).append(t["function"]["name"])
    litellm_up = len(tools) > 0
    return {
        "total":      len(tools),
        "servers":    servers,
        "tools":      tools,
        "litellm_up": litellm_up,
    }


class ChatRequest(BaseModel):
    message: str
    selected_servers: list[str] = []   # empty = use all


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Runs the full agentic loop and streams back Server-Sent Events:
      - event: tool_call   — LLM is calling a tool
      - event: tool_result — tool returned a result
      - event: answer      — final text answer
      - event: error       — something went wrong
    """
    async def stream() -> AsyncGenerator[str, None]:
        try:
            all_tools = await fetch_mcp_tools()

            # Filter to selected servers if specified
            if req.selected_servers:
                tools = [
                    t for t in all_tools
                    if t["function"]["name"].split("-")[0] in req.selected_servers
                ]
            else:
                tools = all_tools

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": req.message},
            ]
            iteration = 0

            # Emit how many tools were loaded so the UI can show it
            yield sse("status", {"tools_loaded": len(tools)})

            while iteration < 10:   # safety cap on loops
                iteration += 1
                response = await llm.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto",
                )
                msg = response.choices[0].message

                if not msg.tool_calls:
                    yield sse("answer", {"text": msg.content})
                    return

                # Add assistant message to history
                messages.append(msg.model_dump(exclude_unset=True))

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    # Intercept fetch-fetch calls with blocked domains
                    redirect_note = None
                    if tool_name == "fetch-fetch" and "url" in tool_args:
                        new_url, redirect_note = rewrite_fetch_url(tool_args["url"], tool_args)
                        tool_args["url"] = new_url

                    yield sse("tool_call", {
                        "tool":      tool_name,
                        "arguments": tool_args,
                        **({"redirect": redirect_note} if redirect_note else {}),
                    })

                    result = await execute_tool(tool_name, tool_args)

                    yield sse("tool_result", {
                        "tool":   tool_name,
                        "result": result[:1000],   # trim for display
                    })

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })

        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)
