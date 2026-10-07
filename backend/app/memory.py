"""Conversation memory (week 2).

The model has no memory between calls, so we store every message in Postgres and send
the history back each time. Long conversations would overflow the context window and
cost more, so once a conversation passes MEMORY_MAX_MESSAGES the older messages are
folded into a short summary and only the recent ones are sent word for word.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import get_settings
from app.db import pool
from app.llm import LLMClient
from app.prompts import load_prompt


async def create_conversation() -> str:
    async with pool().connection() as conn:
        row = await (await conn.execute("INSERT INTO conversations DEFAULT VALUES RETURNING id")).fetchone()
    return str(row["id"])


async def conversation_exists(conversation_id: str) -> bool:
    async with pool().connection() as conn:
        row = await (
            await conn.execute("SELECT 1 FROM conversations WHERE id = %s", (conversation_id,))
        ).fetchone()
    return row is not None


async def add_message(conversation_id: str, role: str, content: str,
                      meta: dict[str, Any] | None = None) -> None:
    async with pool().connection() as conn:
        await conn.execute(
            "INSERT INTO messages (conversation_id, role, content, meta) VALUES (%s, %s, %s, %s)",
            (conversation_id, role, content, json.dumps(meta or {})),
        )


async def replace_last_assistant(conversation_id: str, content: str) -> None:
    async with pool().connection() as conn:
        await conn.execute(
            "UPDATE messages SET content = %s WHERE id = (SELECT max(id) FROM messages"
            " WHERE conversation_id = %s AND role = 'assistant')",
            (content, conversation_id),
        )


async def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    async with pool().connection() as conn:
        conv = await (
            await conn.execute(
                "SELECT id, created_at, summary, escalated FROM conversations WHERE id = %s",
                (conversation_id,),
            )
        ).fetchone()
        if not conv:
            return None
        rows = await (
            await conn.execute(
                "SELECT role, content, meta, created_at FROM messages "
                "WHERE conversation_id = %s ORDER BY id",
                (conversation_id,),
            )
        ).fetchall()
    return {**conv, "id": str(conv["id"]), "messages": rows}


async def load_history(conversation_id: str) -> tuple[str | None, list[dict[str, str]]]:
    """Return (summary of older messages, recent messages not yet summarised)."""
    async with pool().connection() as conn:
        conv = await (
            await conn.execute("SELECT summary FROM conversations WHERE id = %s", (conversation_id,))
        ).fetchone()
        rows = await (
            await conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = %s "
                "AND NOT summarized ORDER BY id",
                (conversation_id,),
            )
        ).fetchall()
    return (conv["summary"] if conv else None), [dict(r) for r in rows]


async def compact_if_needed(llm: LLMClient, conversation_id: str) -> bool:
    """Summarise older messages once the conversation gets long. Returns True if it did."""
    settings = get_settings()
    summary, recent = await load_history(conversation_id)
    if len(recent) <= settings.memory_max_messages:
        return False
    keep = settings.memory_max_messages // 2
    to_fold = recent[:-keep]
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in to_fold)
    if summary:
        transcript = f"Earlier summary:\n{summary}\n\nNewer messages:\n{transcript}"
    result = await llm.chat(
        [
            {"role": "system", "content": load_prompt("summarize").text},
            {"role": "user", "content": transcript},
        ],
        model=settings.fast_model,
        purpose="summarize",
        conversation_id=conversation_id,
    )
    async with pool().connection() as conn, conn.transaction():
        await conn.execute(
            "UPDATE conversations SET summary = %s WHERE id = %s",
            (result.content, conversation_id),
        )
        # Mark everything except the most recent `keep` messages as summarised.
        await conn.execute(
            "UPDATE messages SET summarized = true WHERE conversation_id = %s AND id IN ("
            " SELECT id FROM messages WHERE conversation_id = %s AND NOT summarized"
            " ORDER BY id LIMIT %s)",
            (conversation_id, conversation_id, len(to_fold)),
        )
    return True


def build_messages(system_prompt: str, summary: str | None,
                   history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Assemble what the model sees: rules, then the summary, then recent turns."""
    system = system_prompt
    if summary:
        system += f"\n\nSummary of the earlier conversation:\n{summary}"
    return [{"role": "system", "content": system}, *history]
