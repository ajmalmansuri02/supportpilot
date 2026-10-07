"""The MCP server, tested two ways: in-process, and as a real stdio subprocess."""

import json
import os
from contextlib import AsyncExitStack

from mcp import Client

from app import db
from app.agent.registry import _result_to_data, connect_mcp, get_registry
from app.agent.tools import AgentContext, default_registry
from app.llm import LLMClient, ToolCall
from app.mcp_server import server


async def test_server_lists_tools_resources_and_prompts():
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert {"list_tickets", "get_ticket", "create_ticket", "update_ticket"} <= set(tools)
        assert tools["list_tickets"].annotations.read_only_hint is True

        created = await client.call_tool("create_ticket", {
            "subject": "Sync broken", "description": "Android won't sync", "priority": "high"})
        ticket_id = _result_to_data(created)["ticket_id"]
        updated = await client.call_tool("update_ticket", {"ticket_id": ticket_id, "note": "Asked for logs"})
        assert _result_to_data(updated)["notes"] == ["Asked for logs"]

        templates = await client.list_resource_templates()
        assert any("tickets://" in t.uri_template for t in templates.resource_templates)
        prompts = await client.list_prompts()
        assert prompts.prompts[0].name == "triage_ticket"


async def test_agent_can_use_tickets_through_a_stdio_mcp_subprocess():
    os.environ.setdefault("LLM_PROVIDER", "fake")
    await db.open_pool()
    try:
        async with AsyncExitStack() as stack:
            names = await connect_mcp(stack)
            assert "create_ticket" in names
            registry = get_registry()
            ctx = AgentContext(llm=LLMClient())
            out = json.loads(await registry.execute(ToolCall("1", "create_ticket", {
                "subject": "Via MCP", "description": "Created over stdio", "category": "technical"}), ctx))
            assert out["ticket_id"] > 0
            listed = json.loads(await registry.execute(ToolCall("2", "list_tickets", {}), ctx))
            assert listed["tickets"][0]["subject"] == "Via MCP"
    finally:
        get_registry().replace(list(default_registry().tools.values()))
        await db.close_pool()
