"""Human-in-the-loop approval for risky actions (week 7).

When the agent wants to refund money or change a plan, we do NOT run it. We store a
pending action, show an Approve / Reject card in the UI, and only run it after a human
(a support agent, in a real deployment) decides. This is the standard defence against
"excessive agency": the model can propose, but cannot act alone on anything costly.
"""

from __future__ import annotations

import json

from app import db, memory
from app.agent.tools import AgentContext
from app.services import accounts

EXECUTORS = {
    "issue_refund": lambda a: accounts.refund_invoice(a["invoice_id"], a["reason"]),
    "change_plan": lambda a: accounts.change_plan(a["email"], a["new_plan"]),
}


async def create_pending_action(ctx: AgentContext, tool: str, args: dict, description: str) -> dict:
    async with db.pool().connection() as conn:
        row = await (await conn.execute(
            "INSERT INTO pending_actions (conversation_id, tool, args, description)"
            " VALUES (%s, %s, %s, %s) RETURNING id",
            (ctx.conversation_id, tool, json.dumps(args), description),
        )).fetchone()
    action_id = str(row["id"])
    ctx.emit({"type": "approval_required", "action_id": action_id, "tool": tool,
              "description": description})
    return {"status": "awaiting_approval", "action_id": action_id,
            "message": "Submitted for approval by a support agent. Tell the customer it is "
                       "being reviewed; do not say it is done."}


async def get_action(action_id: str) -> dict | None:
    async with db.pool().connection() as conn:
        return await (await conn.execute(
            "SELECT * FROM pending_actions WHERE id = %s", (action_id,)
        )).fetchone()


async def decide(action_id: str, approve: bool) -> dict:
    action = await get_action(action_id)
    if action is None:
        raise LookupError("Action not found")
    if action["status"] != "pending":
        return {"status": action["status"], "result": action["result"], "already_decided": True}

    if approve:
        result = await EXECUTORS[action["tool"]](action["args"])
        status = "approved" if result.get("ok") else "failed"
    else:
        result = {"ok": False, "rejected": True}
        status = "rejected"

    async with db.pool().connection() as conn:
        await conn.execute(
            "UPDATE pending_actions SET status = %s, result = %s, decided_at = now() WHERE id = %s",
            (status, json.dumps(result), action_id),
        )
    message = {
        "approved": f"Update: a support agent approved this: {action['description']}.",
        "failed": f"Update: this could not be completed: {result.get('error', 'unknown error')}.",
        "rejected": f"Update: a support agent reviewed and declined this request: "
                    f"{action['description']}. They will follow up by email.",
    }[status]
    if action["conversation_id"]:
        await memory.add_message(str(action["conversation_id"]), "assistant", message,
                                 {"mode": "agent", "action_id": action_id, "status": status})
    return {"status": status, "result": result, "message": message}
