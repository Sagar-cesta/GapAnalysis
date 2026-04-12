"""
STEP 3 - Fetch MCP + LiteLLM
------------------------------
MCP Server : mcp-server-fetch (Python, run via uvx — no proxy layer)
What it does: Fetches any URL and returns clean readable content
Auth        : None required
Transport   : stdio (spawned as subprocess)

Pattern demonstrated:
  LiteLLM --> Fetch MCP (stdio) --> Fetches real webpage --> LLM summarises

Run: python step3_fetch_mcp.py
"""

import asyncio
import os
import sys
import json
import litellm
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# uvx runs mcp-server-fetch in an isolated environment — no version conflicts
UVX_PATH = os.path.join(os.path.dirname(sys.executable), "uvx.exe")


async def main():

    # ── STEP A: Connect to Fetch MCP server via stdio ───────────
    print("=" * 60)
    print("STEP A: Connecting to Fetch MCP server (via uvx)...")
    print("  Transport : stdio (subprocess)")
    print("  Auth      : none required")
    print("=" * 60)

    server_params = StdioServerParameters(
        command=UVX_PATH,
        args=["mcp-server-fetch"],
        env=os.environ.copy()
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected!\n")

            # ── STEP B: List all tools ───────────────────────────
            tools_result = await session.list_tools()
            tools = tools_result.tools

            print(f"Tools available: {len(tools)}\n")
            for t in tools:
                print(f"  [{t.name}] {(t.description or '')[:80]}")
            print()

            # ── STEP C: Call fetch tool directly ─────────────────
            print("=" * 60)
            print("STEP B: Calling 'fetch' tool directly...")
            print("  Fetching: https://modelcontextprotocol.io/introduction")
            print("=" * 60)

            result = await session.call_tool(
                "fetch",
                arguments={
                    "url": "https://modelcontextprotocol.io/introduction",
                    "max_length": 3000
                }
            )
            raw = result.content[0].text if result.content else ""
            print(f"\nFetched content preview (first 500 chars):\n{raw[:500]}...\n")

            # ── STEP D: LiteLLM with Fetch tools ─────────────────
            print("=" * 60)
            print("STEP C: LiteLLM using Fetch MCP to answer a question...")
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
                "content": "Fetch the MCP introduction page at https://modelcontextprotocol.io/introduction and give me a 3-sentence summary of what MCP is."
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
                    print(f"LLM called: {tc.function.name}({list(args.keys())})")
                    res = await session.call_tool(tc.function.name, arguments=args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": res.content[0].text if res.content else ""
                    })

                final = litellm.completion(model="gpt-4o-mini", messages=messages)
                answer = final.choices[0].message.content
            else:
                answer = choice.message.content

            print("\nFinal Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            print("\nSUCCESS: Fetch MCP + LiteLLM flow complete!")


asyncio.run(main())
