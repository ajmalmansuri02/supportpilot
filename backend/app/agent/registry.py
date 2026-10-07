"""Which tools the agent gets, and where they come from (week 8).

By default every tool is a local Python function. With TICKETS_VIA_MCP=true, the backend
starts the MCP server as a subprocess, *discovers* its tools at runtime with list_tools, and
replaces the local ticket tools with them. The agent code doesn't change at all: that's the
point of MCP. Point it at someone else's MCP server and the agent gains new abilities.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import Client, StdioServerParameters

from app.agent.tools import AgentContext, Tool, ToolRegistry, default_registry

log = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parents[2]

_registry: ToolRegistry = default_registry()


def get_registry() -> ToolRegistry:
    return _registry


def _risk(tool) -> str:
    ann = tool.annotations
    if ann and ann.read_only_hint:
        return "read"
    if ann and ann.destructive_hint:
        return "approval"  # never let the model run destructive remote tools unattended
    return "write"


def _result_to_data(result) -> dict:
    if result.structured_content is not None:
        data = result.structured_content
    else:
        text = "".join(getattr(block, "text", "") for block in result.content)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"text": text}
    if result.is_error:
        return {"error": data if isinstance(data, str) else json.dumps(data)}
    return data if isinstance(data, dict) else {"result": data}


def simplify_schema(schema):
    """Make MCP's generated JSON schema friendlier to model APIs.

    Optional parameters arrive as {"anyOf": [{"type": "string"}, {"type": "null"}]}, which
    some providers (Gemini's OpenAI endpoint among them) handle poorly. We collapse them to
    the non-null type and drop the auto-generated "title" fields.
    """
    if isinstance(schema, list):
        return [simplify_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    options = schema.get("anyOf")
    if options and len(options) == 2 and {"type": "null"} in options:
        other = next(o for o in options if o != {"type": "null"})
        schema = {**{k: v for k, v in schema.items() if k != "anyOf"}, **other}
    return {k: simplify_schema(v) for k, v in schema.items() if k != "title"}


def mcp_tools(client: Client, listed) -> list[Tool]:
    tools = []
    for remote in listed.tools:
        async def handler(args: dict, ctx: AgentContext, _name: str = remote.name) -> dict:
            result = await client.call_tool(_name, {k: v for k, v in args.items() if v is not None})
            return _result_to_data(result)

        tools.append(Tool(name=remote.name, description=remote.description or "",
                          parameters=simplify_schema(remote.input_schema), handler=handler, risk=_risk(remote)))
    return tools


async def connect_mcp(stack: AsyncExitStack) -> list[str]:
    """Start the ticket MCP server over stdio and register its tools. Returns their names."""
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "app.mcp_server"],
        env=dict(os.environ), cwd=str(BACKEND_DIR),
    )
    client = await stack.enter_async_context(Client(params))
    listed = await client.list_tools()
    tools = mcp_tools(client, listed)
    _registry.replace(tools)
    names = [t.name for t in tools]
    log.info("Loaded %d tools from the MCP server: %s", len(names), names)
    return names
