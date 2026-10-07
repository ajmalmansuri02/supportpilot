"""Tools the agent can call (week 6).

A tool is a name, a description, a JSON schema for its arguments and a Python function.
The model never runs anything itself: it *asks* for a tool call, we validate and run it,
and we send the result back as a "tool" message.

Every tool has a risk level, which is the core of safe agents (weeks 7 and 10):
- "read":     looks things up, runs freely
- "write":    low-risk changes (opening a ticket), runs freely and is logged
- "approval": money or account changes; never runs directly, a human must approve it
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from app import db
from app.llm import LLMClient, ToolCall
from app.services import accounts, tickets

log = logging.getLogger(__name__)

Risk = Literal["read", "write", "approval"]
Emit = Callable[[dict], None]

# Arguments the server fills in itself; the model never sees or sets them.
HIDDEN_ARGS = {"conversation_id"}


@dataclass
class AgentContext:
    llm: LLMClient
    conversation_id: str | None = None
    emit: Emit = lambda event: None
    tool_calls: list[dict] = field(default_factory=list)  # audit trail for this turn
    # Emails the customer typed in this conversation; None = no restriction (scripts, tests).
    allowed_emails: set[str] | None = None


Handler = Callable[[dict[str, Any], AgentContext], Awaitable[Any]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    risk: Risk = "read"

    def spec(self) -> dict[str, Any]:
        """The OpenAI-style tool definition sent to the model."""
        props = {k: v for k, v in self.parameters.get("properties", {}).items() if k not in HIDDEN_ARGS}
        required = [r for r in self.parameters.get("required", []) if r not in HIDDEN_ARGS]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }


def _obj(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


_STR = {"type": "string"}


# --- read tools -----------------------------------------------------------------------

async def _search_docs(args, ctx: AgentContext):
    from app.chat import retrieve, sources_payload  # local import avoids a cycle

    hits = await retrieve(ctx.llm, args["query"])
    if not hits:
        return {"results": [], "note": "Nothing relevant in the documentation."}
    return {"results": [
        {"n": s["n"], "source": s["source"], "heading": s["heading"], "content": s["content"]}
        for s in sources_payload(hits)
    ]}


async def _get_account(args, ctx):
    account = await accounts.get_account(args["email"])
    return account or {"error": f"No CloudNotes account found for {args['email']}"}


async def _list_tickets(args, ctx):
    return {"tickets": await tickets.list_tickets(args.get("customer_email"), args.get("status"))}


# --- write tools ------------------------------------------------------------------------

async def _create_ticket(args, ctx):
    ticket = await tickets.create_ticket(
        args["subject"], args["description"], category=args.get("category", "other"),
        priority=args.get("priority", "medium"), customer_email=args.get("customer_email"),
        conversation_id=args.get("conversation_id"),
    )
    return {"ticket_id": ticket["id"], "status": ticket["status"], "priority": ticket["priority"]}


async def _escalate(args, ctx: AgentContext):
    ticket = await tickets.create_ticket(
        "Escalated: " + args["reason"][:150], args["reason"], priority="high",
        customer_email=args.get("customer_email"), conversation_id=ctx.conversation_id,
        assignee="human",
    )
    if ctx.conversation_id:
        async with db.pool().connection() as conn:
            await conn.execute("UPDATE conversations SET escalated = true WHERE id = %s",
                               (ctx.conversation_id,))
    ctx.emit({"type": "escalated", "ticket_id": ticket["id"]})
    return {"escalated": True, "ticket_id": ticket["id"],
            "message": "A human agent will reply by email."}


# --- approval tools -----------------------------------------------------------------------

async def _request_refund(args, ctx: AgentContext):
    from app.agent.actions import create_pending_action

    account = await accounts.get_account(args["customer_email"])
    if not account:
        return {"error": "No account for that email, so no refund can be requested."}
    invoice = next((i for i in account["invoices"] if i["id"] == args["invoice_id"].upper()), None)
    if invoice is None:
        return {"error": f"Invoice {args['invoice_id']} does not belong to {args['customer_email']}."}
    if invoice["status"] != "paid":
        return {"error": f"Invoice {invoice['id']} is {invoice['status']} and cannot be refunded."}
    description = (f"Refund ${invoice['amount_usd']:.2f} for {invoice['id']} "
                   f"({args['customer_email']}): {args['reason']}")
    return await create_pending_action(ctx, "issue_refund", {
        "invoice_id": invoice["id"], "reason": args["reason"]}, description)


async def _request_plan_change(args, ctx: AgentContext):
    from app.agent.actions import create_pending_action

    if not await accounts.get_account(args["customer_email"]):
        return {"error": "No account for that email."}
    description = f"Change {args['customer_email']} to the {args['new_plan']} plan"
    return await create_pending_action(ctx, "change_plan", {
        "email": args["customer_email"], "new_plan": args["new_plan"]}, description)


LOCAL_TOOLS = [
    Tool("search_docs",
         "Search the CloudNotes help centre. Use it for any question about plans, prices, "
         "policies, features or how-to steps before answering.",
         _obj({"query": {**_STR, "description": "What to search for, as a full question"}}, ["query"]),
         _search_docs, "read"),
    Tool("get_account",
         "Look up a customer's account by email: plan, billing cycle and recent invoices.",
         _obj({"email": {**_STR, "description": "The customer's account email"}}, ["email"]),
         _get_account, "read"),
    Tool("list_tickets",
         "List support tickets, optionally for one customer email and/or status.",
         _obj({"customer_email": _STR, "status": {"type": "string", "enum": ["open", "pending", "resolved"]}}, []),
         _list_tickets, "read"),
    Tool("create_ticket",
         "Open a support ticket so the support team follows up by email.",
         _obj({
             "subject": _STR,
             "description": {**_STR, "description": "Everything the team needs, including invoice numbers"},
             "category": {"type": "string", "enum": ["billing", "technical", "account", "feature_request", "other"]},
             "priority": {"type": "string", "enum": ["low", "medium", "high"]},
             "customer_email": _STR,
             "conversation_id": _STR,
         }, ["subject", "description"]),
         _create_ticket, "write"),
    Tool("escalate_to_human",
         "Hand the conversation to a human agent. Use when the customer asks for a person, "
         "for account recovery or data loss, or when you cannot resolve the issue.",
         _obj({"reason": {**_STR, "description": "Short summary for the human agent"},
               "customer_email": _STR}, ["reason"]),
         _escalate, "write"),
    Tool("issue_refund",
         "Request a refund of one invoice. Only for duplicate charges or refunds allowed by the "
         "refund policy, after checking the account. A human agent must approve it.",
         _obj({"invoice_id": {**_STR, "description": "Invoice number, e.g. INV-204519"},
               "customer_email": _STR,
               "reason": {**_STR, "description": "Why the refund is allowed, citing the policy"}},
              ["invoice_id", "customer_email", "reason"]),
         _request_refund, "approval"),
    Tool("change_plan",
         "Request a plan change for a customer. A human agent must approve it.",
         _obj({"customer_email": _STR, "new_plan": {"type": "string", "enum": ["free", "pro", "team"]}},
              ["customer_email", "new_plan"]),
         _request_plan_change, "approval"),
]


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools.values()]

    def replace(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.tools[tool.name] = tool

    async def execute(self, call: ToolCall, ctx: AgentContext) -> str:
        """Run one tool call and return its result as a JSON string for the model.

        Errors are returned to the model as data instead of raised, so it can recover
        (ask for the missing email, try another invoice...) rather than crash.
        """
        from app.guardrails import email_allowed, log_event

        tool = self.tools.get(call.name)
        args = dict(call.arguments)
        email = args.get("email") or args.get("customer_email")
        if tool is not None and not email_allowed(email, ctx.allowed_emails):
            result: Any = {"error": "For privacy, only the account whose email the customer gave "
                                    "in this chat can be used. Ask the customer for their email."}
            if ctx.conversation_id:
                await log_event(ctx.conversation_id, "tool_denied", f"{call.name} for {email}")
        elif tool is None:
            result = {"error": f"Unknown tool {call.name}. Available: {sorted(self.tools)}"}
        elif missing := [r for r in tool.spec()["function"]["parameters"]["required"] if not args.get(r)]:
            result = {"error": f"Missing required arguments: {missing}"}
        else:
            if "conversation_id" in tool.parameters.get("properties", {}):
                args["conversation_id"] = ctx.conversation_id
            try:
                result = await tool.handler(args, ctx)
            except Exception as exc:
                log.exception("tool %s failed", call.name)
                result = {"error": f"{type(exc).__name__}: {exc}"}
        ctx.tool_calls.append({"tool": call.name, "args": call.arguments,
                               "risk": tool.risk if tool else None,
                               "error": isinstance(result, dict) and "error" in result})
        return json.dumps(result, default=str)


def default_registry() -> ToolRegistry:
    return ToolRegistry(list(LOCAL_TOOLS))
