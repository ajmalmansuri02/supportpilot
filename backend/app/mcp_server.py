"""MCP server exposing the CloudNotes ticket system (week 8).

MCP (Model Context Protocol) is a standard way to give any AI app access to your tools and
data. Write the server once, and SupportPilot's agent, Claude Desktop, VS Code, Cursor or the
MCP Inspector can all use it.

The three MCP building blocks, all shown here:
- Tools:     actions the model can call (list/create/update tickets)
- Resources: read-only data an app can load as context (tickets://open, tickets://{id})
- Prompts:   reusable prompt templates the user can pick (triage_ticket)

Run it:
    uv run python -m app.mcp_server                 # stdio (how apps usually launch it)
    uv run python -m app.mcp_server --http          # http://localhost:8765/mcp
    npx @modelcontextprotocol/inspector uv run python -m app.mcp_server   # click around
"""

from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from app import db
from app.services import tickets

Status = Literal["open", "pending", "resolved"]
Priority = Literal["low", "medium", "high"]
Category = Literal["billing", "technical", "account", "feature_request", "other"]


@asynccontextmanager
async def lifespan(server: MCPServer):
    await db.open_pool()
    await db.init_schema()
    try:
        yield {}
    finally:
        await db.close_pool()


server = MCPServer(
    "cloudnotes-tickets",
    instructions="Support tickets for CloudNotes. List, read, create and update tickets.",
    lifespan=lifespan,
)


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_tickets(customer_email: str | None = None, status: Status | None = None,
                       limit: int = 20) -> dict:
    """List support tickets, newest first, optionally filtered by customer email and status."""
    return {"tickets": await tickets.list_tickets(customer_email, status, limit)}


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
async def get_ticket(ticket_id: int) -> dict:
    """Get one ticket with its notes."""
    return await tickets.get_ticket(ticket_id) or {"error": f"Ticket {ticket_id} not found"}


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
async def create_ticket(subject: str, description: str, category: Category = "other",
                        priority: Priority = "medium", customer_email: str | None = None,
                        conversation_id: str | None = None) -> dict:
    """Open a support ticket so the support team follows up with the customer by email."""
    ticket = await tickets.create_ticket(subject, description, category=category, priority=priority,
                                         customer_email=customer_email, conversation_id=conversation_id)
    return {"ticket_id": ticket["id"], "status": ticket["status"], "priority": ticket["priority"]}


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
async def update_ticket(ticket_id: int, status: Status | None = None, priority: Priority | None = None,
                        note: str | None = None) -> dict:
    """Change a ticket's status or priority, or add an internal note."""
    return (await tickets.update_ticket(ticket_id, status=status, priority=priority, note=note)
            or {"error": f"Ticket {ticket_id} not found"})


@server.resource("tickets://open", mime_type="application/json")
async def open_tickets() -> str:
    """All open tickets."""
    return json.dumps(await tickets.list_tickets(status="open", limit=100))


@server.resource("tickets://{ticket_id}", mime_type="application/json")
async def ticket_resource(ticket_id: str) -> str:
    """One ticket by id."""
    return json.dumps(await tickets.get_ticket(int(ticket_id)))


@server.prompt()
def triage_ticket(ticket_id: str) -> str:
    """Ask the model to triage a ticket."""
    return (f"Read ticket {ticket_id} with get_ticket. Decide its category and priority using the "
            "CloudNotes support rules (money lost, no account access or data loss = high), "
            "update the ticket, and add a note explaining why.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="serve over Streamable HTTP")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.http:
        server.run("streamable-http", host="127.0.0.1", port=args.port)
    else:
        server.run()
