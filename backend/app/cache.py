"""Semantic answer cache (week 9).

Many support questions repeat with different wording ("how much is Pro?" / "what does the
Pro plan cost?"). We embed each question anyway for retrieval, so we can look up earlier
answers to *similar* questions for free and skip the most expensive step: generation.

Only used for the first message of a conversation (later answers depend on chat history),
and keyed by prompt version and retrieval settings, so changing either starts fresh.
"""

from __future__ import annotations

import json

from app import db
from app.config import get_settings
from app.prompts import load_prompt


def settings_key() -> str:
    s = get_settings()
    return (f"system:{load_prompt('system').version}|rag:{load_prompt('rag').version}|"
            f"{s.chat_model}|{s.retrieval_mode}|{s.top_k}|{s.reranker}")


async def lookup(query_vector: list[float]) -> dict | None:
    s = get_settings()
    if not s.cache_enabled:
        return None
    async with db.pool().connection() as conn:
        row = await (await conn.execute(
            """SELECT id, question, answer, sources, 1 - (embedding <=> %s::vector) AS similarity
               FROM answer_cache
               WHERE settings_key = %s AND created_at > now() - make_interval(hours => %s)
               ORDER BY embedding <=> %s::vector LIMIT 1""",
            (db.to_vector(query_vector), settings_key(), s.cache_ttl_hours, db.to_vector(query_vector)),
        )).fetchone()
        if not row or row["similarity"] < s.cache_similarity:
            return None
        await conn.execute("UPDATE answer_cache SET hits = hits + 1 WHERE id = %s", (row["id"],))
    return row


async def store(question: str, query_vector: list[float], answer: str, sources: list[dict]) -> None:
    if not get_settings().cache_enabled:
        return
    async with db.pool().connection() as conn:
        await conn.execute(
            "INSERT INTO answer_cache (question, embedding, answer, sources, settings_key)"
            " VALUES (%s, %s::vector, %s, %s, %s)",
            (question, db.to_vector(query_vector), answer, json.dumps(sources), settings_key()),
        )


async def clear() -> None:
    async with db.pool().connection() as conn:
        await conn.execute("TRUNCATE answer_cache")
