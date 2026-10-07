"""Cost, latency and tracing (week 9).

Every LLM call already produces a CallRecord (see llm.py). Here we:
1. Store each record in Postgres (llm_calls), so /api/metrics can show cost and latency
   per purpose, p95 latency, and cost per conversation.
2. Optionally send it to Langfuse, an open-source LLM tracing UI you can self-host for free,
   with emails, phone numbers and card numbers redacted first.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app import db
from app.config import get_settings
from app.guardrails import redact
from app.llm import CallRecord, LLMClient

log = logging.getLogger(__name__)
_background: set[asyncio.Task] = set()


async def _store(record: CallRecord) -> None:
    try:
        async with db.pool().connection() as conn:
            await conn.execute(
                "INSERT INTO llm_calls (conversation_id, purpose, provider, model, input_tokens,"
                " output_tokens, latency_ms, cost_usd, error) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (record.conversation_id, record.purpose, record.provider, record.model,
                 record.input_tokens, record.output_tokens, record.latency_ms, record.cost_usd,
                 record.error),
            )
    except Exception:
        log.exception("could not store LLM call")


def db_listener(record: CallRecord) -> None:
    """Called synchronously after each LLM call; writes in the background."""
    try:
        task = asyncio.get_running_loop().create_task(_store(record))
    except RuntimeError:
        return  # no event loop (e.g. a script); skip
    _background.add(task)
    task.add_done_callback(_background.discard)


async def flush() -> None:
    if _background:
        await asyncio.gather(*list(_background), return_exceptions=True)


def _redact_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if messages is None:
        return None
    out = []
    for m in messages:
        content = m.get("content")
        out.append({**m, "content": redact(content, for_logs=True)[0] if isinstance(content, str) else content})
    return out


def langfuse_listener():
    """Return a listener that sends each call to Langfuse, or None if not configured."""
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning("LANGFUSE keys are set but the SDK is missing: uv sync --extra tracing")
        return None
    client = Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key,
                      host=s.langfuse_host)

    def listener(record: CallRecord) -> None:
        try:
            trace_context = None
            if record.conversation_id:
                # All calls of one conversation land in the same trace.
                trace_context = {"trace_id": client.create_trace_id(seed=record.conversation_id)}
            generation = client.start_observation(
                trace_context=trace_context,
                name=record.purpose,
                as_type="generation",
                model=record.model,
                input=_redact_messages(record.input),
                output=redact(record.output or "", for_logs=True)[0],
                usage_details={"input": record.input_tokens, "output": record.output_tokens},
                cost_details={"total": record.cost_usd},
                metadata={"latency_ms": record.latency_ms, "provider": record.provider},
                level="ERROR" if record.error else "DEFAULT",
                status_message=record.error,
            )
            generation.end()
        except Exception:
            log.exception("Langfuse export failed")

    log.info("Langfuse tracing enabled (%s)", s.langfuse_host)
    return listener


def install(llm: LLMClient) -> None:
    llm.add_listener(db_listener)
    if (listener := langfuse_listener()) is not None:
        llm.add_listener(listener)


async def metrics(hours: int = 24) -> dict[str, Any]:
    """Cost and latency summary for the last `hours`."""
    async with db.pool().connection() as conn:
        by_purpose = await (await conn.execute(
            """SELECT purpose, model, count(*) AS calls,
                      round(avg(latency_ms)) AS avg_latency_ms,
                      percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
                      sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
                      sum(cost_usd)::float AS cost_usd,
                      count(*) FILTER (WHERE error IS NOT NULL) AS errors
               FROM llm_calls WHERE created_at > now() - make_interval(hours => %s)
               GROUP BY purpose, model ORDER BY calls DESC""",
            (hours,),
        )).fetchall()
        per_conv = await (await conn.execute(
            """SELECT count(*) AS conversations,
                      avg(cost)::float AS avg_cost_usd, avg(calls)::float AS avg_calls,
                      avg(tokens)::float AS avg_tokens
               FROM (SELECT conversation_id, sum(cost_usd) AS cost, count(*) AS calls,
                            sum(input_tokens + output_tokens) AS tokens
                     FROM llm_calls
                     WHERE conversation_id IS NOT NULL
                       AND created_at > now() - make_interval(hours => %s)
                     GROUP BY conversation_id) c""",
            (hours,),
        )).fetchone()
        cache = await (await conn.execute(
            "SELECT count(*) AS entries, coalesce(sum(hits), 0) AS hits FROM answer_cache"
        )).fetchone()
        guards = await (await conn.execute(
            """SELECT kind, count(*) AS n FROM guardrail_events
               WHERE created_at > now() - make_interval(hours => %s) GROUP BY kind""",
            (hours,),
        )).fetchall()
    return {"hours": hours, "by_purpose": by_purpose, "per_conversation": per_conv,
            "cache": cache, "guardrails": {g["kind"]: g["n"] for g in guards}}
