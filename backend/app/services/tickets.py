"""Support tickets. Used directly by the agent, and exposed to any AI app by the MCP server."""

from __future__ import annotations

import json

from app import db

_COLUMNS = ("id, conversation_id, customer_email, subject, description, category, priority,"
            " status, assignee, notes, created_at, updated_at")


def _clean(row: dict) -> dict:
    out = dict(row)
    for key in ("created_at", "updated_at"):
        out[key] = out[key].isoformat()
    if out.get("conversation_id"):
        out["conversation_id"] = str(out["conversation_id"])
    return out


async def create_ticket(subject: str, description: str, *, category: str = "other",
                        priority: str = "medium", customer_email: str | None = None,
                        conversation_id: str | None = None, assignee: str = "ai") -> dict:
    if priority not in ("low", "medium", "high"):
        priority = "medium"
    async with db.pool().connection() as conn:
        row = await (await conn.execute(
            "INSERT INTO tickets (conversation_id, customer_email, subject, description, category,"
            f" priority, assignee) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING {_COLUMNS}",
            (conversation_id, customer_email, subject[:200], description, category, priority, assignee),
        )).fetchone()
    return _clean(row)


async def list_tickets(customer_email: str | None = None, status: str | None = None,
                       limit: int = 20) -> list[dict]:
    where, params = [], []
    if customer_email:
        where.append("lower(customer_email) = lower(%s)")
        params.append(customer_email)
    if status:
        where.append("status = %s")
        params.append(status)
    sql = f"SELECT {_COLUMNS} FROM tickets"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT %s"
    async with db.pool().connection() as conn:
        rows = await (await conn.execute(sql, (*params, limit))).fetchall()
    return [_clean(r) for r in rows]


async def get_ticket(ticket_id: int) -> dict | None:
    async with db.pool().connection() as conn:
        row = await (await conn.execute(
            f"SELECT {_COLUMNS} FROM tickets WHERE id = %s", (ticket_id,)
        )).fetchone()
    return _clean(row) if row else None


async def update_ticket(ticket_id: int, *, status: str | None = None, priority: str | None = None,
                        assignee: str | None = None, note: str | None = None) -> dict | None:
    sets, params = ["updated_at = now()"], []
    for column, value in (("status", status), ("priority", priority), ("assignee", assignee)):
        if value:
            sets.append(f"{column} = %s")
            params.append(value)
    if note:
        sets.append("notes = notes || %s::jsonb")
        params.append(json.dumps([note]))
    async with db.pool().connection() as conn:
        row = await (await conn.execute(
            f"UPDATE tickets SET {', '.join(sets)} WHERE id = %s RETURNING {_COLUMNS}",
            (*params, ticket_id),
        )).fetchone()
    return _clean(row) if row else None
