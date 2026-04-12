"""
STEP 4 - Filesystem MCP + LiteLLM
------------------------------------
MCP Server : @modelcontextprotocol/server-filesystem (npm, stdio)
What it does: Read/write/list files on your local machine
Auth        : None — scoped to allowed directories only
Transport   : stdio (spawned as subprocess)

Pattern demonstrated:
  LiteLLM --> Filesystem MCP (stdio) --> Reads project files --> LLM summarises

Run: python step4_filesystem_mcp.py
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

PROJECT_DIR = r"C:\Users\SAGAR\OneDrive\Desktop\Sagar\Cesta Inc\GapAnalysis"

# npm global bin path
NPX_PATH = r"C:\Program Files\nodejs\npx.cmd"
FILESYSTEM_SERVER = "@modelcontextprotocol/server-filesystem"


async def main():

    # ── STEP A: Connect to Filesystem MCP server ────────────────
    print("=" * 60)
    print("STEP A: Connecting to Filesystem MCP server...")
    print(f"  Transport  : stdio (subprocess)")
    print(f"  Auth       : none")
    print(f"  Allowed dir: {PROJECT_DIR}")
    print("=" * 60)

    server_params = StdioServerParameters(
        command=NPX_PATH,
        args=["-y", FILESYSTEM_SERVER, PROJECT_DIR],
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

            # ── STEP C: Call list_directory directly ─────────────
            print("=" * 60)
            print("STEP B: Calling 'list_directory' tool directly...")
            print("=" * 60)

            result = await session.call_tool(
                "list_directory",
                arguments={"path": PROJECT_DIR}
            )
            raw = result.content[0].text if result.content else ""
            print(f"\nProject files:\n{raw}\n")

            # ── STEP D: LiteLLM with Filesystem tools ────────────
            print("=" * 60)
            print("STEP C: LiteLLM using Filesystem MCP to analyse project...")
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
                "content": f"List the files in {PROJECT_DIR} and read the litellm_config.yaml file. Then explain in plain English what this project is set up to do."
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

                # Handle multiple tool calls in a loop
                while choice.finish_reason == "tool_calls":
                    for tc in choice.message.tool_calls:
                        args = json.loads(tc.function.arguments)
                        print(f"LLM called: {tc.function.name}({list(args.keys())})")
                        res = await session.call_tool(tc.function.name, arguments=args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": res.content[0].text if res.content else ""
                        })

                    next_resp = litellm.completion(
                        model="gpt-4o-mini",
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto"
                    )
                    choice = next_resp.choices[0]
                    messages.append(choice.message)

                answer = choice.message.content
            else:
                answer = choice.message.content

            print("\nFinal Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            print("\nSUCCESS: Filesystem MCP + LiteLLM flow complete!")


asyncio.run(main())
