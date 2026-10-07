"""Mock customers and invoices so the agent has something real to look up.

Seeded automatically on startup when the customers table is empty, or run:
    uv run python -m app.services.seed --reset
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

from app import db

NOW = datetime.now(UTC).replace(hour=9, minute=0, second=0, microsecond=0)


def _ago(days: int) -> datetime:
    return NOW - timedelta(days=days)


# (email, name, plan, cycle, seats, [(invoice id, amount, description, days ago, status)])
CUSTOMERS = [
    ("priya@example.com", "Priya Sharma", "pro", "monthly", 1, [
        ("INV-204410", 8.00, "Pro monthly", 33, "paid"),
        # A duplicate charge: the same period billed twice a few minutes apart.
        ("INV-204518", 8.00, "Pro monthly", 3, "paid"),
        ("INV-204519", 8.00, "Pro monthly", 3, "paid"),
    ]),
    ("rahul@example.com", "Rahul Verma", "team", "annual", 5, [
        ("INV-198001", 750.00, "Team annual, 5 seats", 200, "paid"),
    ]),
    ("maria@example.com", "Maria Garcia", "pro", "annual", 1, [
        ("INV-203377", 80.00, "Pro annual", 10, "paid"),
    ]),
    ("alex@example.com", "Alex Kim", "free", None, 1, []),
    ("sam@example.com", "Sam Lee", "pro", "monthly", 1, [
        ("INV-204002", 8.00, "Pro monthly", 40, "paid"),
        ("INV-204390", 8.00, "Pro monthly", 10, "failed"),
    ]),
]


async def seed(reset: bool = False) -> int:
    async with db.pool().connection() as conn, conn.transaction():
        if reset:
            await conn.execute("TRUNCATE customers, invoices, tickets, pending_actions RESTART IDENTITY CASCADE")
        row = await (await conn.execute("SELECT count(*) AS n FROM customers")).fetchone()
        if row["n"]:
            return 0
        for email, name, plan, cycle, seats, invoices in CUSTOMERS:
            customer = await (await conn.execute(
                "INSERT INTO customers (email, name, plan, billing_cycle, seats, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (email, name, plan, cycle, seats, _ago(400)),
            )).fetchone()
            for inv_id, amount, desc, days, status in invoices:
                issued = _ago(days) + (timedelta(minutes=4) if inv_id == "INV-204519" else timedelta())
                await conn.execute(
                    "INSERT INTO invoices (id, customer_id, amount_usd, description, status, issued_at)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    (inv_id, customer["id"], amount, desc, status, issued),
                )
    return len(CUSTOMERS)


async def _main(reset: bool) -> None:
    await db.open_pool()
    try:
        await db.init_schema()
        print(f"Seeded {await seed(reset)} customers")
    finally:
        await db.close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="wipe customers, tickets and actions first")
    asyncio.run(_main(parser.parse_args().reset))
