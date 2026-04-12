"""
STEP 2 - GitHub MCP + LiteLLM (Direct, No Proxy)
--------------------------------------------------
Clean direct integration:
  - GitHub MCP remote endpoint: https://api.githubcopilot.com/mcp/
  - Auth: static Bearer token (GitHub PAT)
  - No proxy layer, no token refresh, no middleware

Flow:
  This Script --> LiteLLM --> GitHub MCP (remote, via Bearer token)

Run: python step2_github_mcp.py
"""

import os
import asyncio
import json
import litellm
from dotenv import load_dotenv
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"


async def main():

    # ── STEP A: Connect directly to GitHub MCP remote endpoint ──
    print("=" * 60)
    print("STEP A: Connecting to GitHub MCP (direct, no proxy)...")
    print(f"  URL   : {GITHUB_MCP_URL}")
    print(f"  Auth  : Bearer token (static PAT)")
    print("=" * 60)

    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

    async with streamablehttp_client(GITHUB_MCP_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected!\n")

            # ── STEP B: List all tools ───────────────────────────
            tools_result = await session.list_tools()
            tools = tools_result.tools

            print(f"Tools available: {len(tools)}\n")
            for t in tools:
                print(f"  [{t.name}] {(t.description or '')[:70]}")
            print()

            # ── STEP C: Call a tool directly ─────────────────────
            print("=" * 60)
            print("STEP B: Calling search_repositories directly...")
            print("=" * 60)

            result = await session.call_tool(
                "search_repositories",
                arguments={"query": "model context protocol", "perPage": 3}
            )
            raw = result.content[0].text if result.content else "{}"
            data = json.loads(raw)
            repos = data.get("items", [])[:3]

            print("\nTop 3 repos:\n")
            for i, r in enumerate(repos, 1):
                print(f"  {i}. {r.get('full_name')} ({r.get('stargazers_count', '?')} stars)")
                print(f"     {r.get('description','N/A')[:80]}\n")

            # ── STEP D: LiteLLM with MCP tools ───────────────────
            print("=" * 60)
            print("STEP C: LiteLLM using GitHub MCP tools...")
            print("=" * 60)

            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema if hasattr(t, "inputSchema") else {}
                    }
                }
                for t in tools
            ]

            messages = [{
                "role": "user",
                "content": "Search GitHub for the top 3 repos about 'model context protocol' and summarise each one."
            }]

            response = litellm.completion(
                model="gpt-4o-mini",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto"
            )

            choice = response.choices[0]
            print(f"LLM finish reason: {choice.finish_reason}\n")

            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    print(f"LLM called: {tc.function.name}({args})")
                    res = await session.call_tool(tc.function.name, arguments=args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": res.content[0].text if res.content else "{}"
                    })

                final = litellm.completion(model="gpt-4o-mini", messages=messages)
                answer = final.choices[0].message.content
            else:
                answer = choice.message.content

            print("\nFinal Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            print("\nSUCCESS: Direct GitHub MCP + LiteLLM flow complete!")


asyncio.run(main())
