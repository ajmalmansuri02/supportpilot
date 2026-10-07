"""Customer accounts and billing actions (mock data from seed.py)."""

from __future__ import annotations

from app import db


async def get_account(email: str) -> dict | None:
    async with db.pool().connection() as conn:
        customer = await (await conn.execute(
            "SELECT id, email, name, plan, billing_cycle, seats, status FROM customers"
            " WHERE lower(email) = lower(%s)",
            (email.strip(),),
        )).fetchone()
        if not customer:
            return None
        invoices = await (await conn.execute(
            "SELECT id, amount_usd, description, status, issued_at FROM invoices"
            " WHERE customer_id = %s ORDER BY issued_at DESC LIMIT 10",
            (customer["id"],),
        )).fetchall()
    customer = {k: v for k, v in customer.items() if k != "id"}
    customer["invoices"] = [
        {**inv, "amount_usd": float(inv["amount_usd"]), "issued_at": inv["issued_at"].isoformat()}
        for inv in invoices
    ]
    return customer


async def refund_invoice(invoice_id: str, reason: str) -> dict:
    async with db.pool().connection() as conn:
        inv = await (await conn.execute(
            "UPDATE invoices SET status = 'refunded' WHERE id = %s AND status = 'paid'"
            " RETURNING id, amount_usd",
            (invoice_id.strip().upper(),),
        )).fetchone()
    if not inv:
        return {"ok": False, "error": f"Invoice {invoice_id} not found or not refundable"}
    return {"ok": True, "invoice_id": inv["id"], "refunded_usd": float(inv["amount_usd"]),
            "reason": reason, "eta": "5 to 10 business days"}


async def change_plan(email: str, new_plan: str) -> dict:
    async with db.pool().connection() as conn:
        row = await (await conn.execute(
            "UPDATE customers SET plan = %s WHERE lower(email) = lower(%s) RETURNING email, plan",
            (new_plan, email.strip()),
        )).fetchone()
    if not row:
        return {"ok": False, "error": f"No account for {email}"}
    return {"ok": True, **row}
