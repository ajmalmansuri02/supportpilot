"""The chat pipeline. Each mode is an async generator of events that the API streams
to the browser as Server-Sent Events:

    {"type": "sources", "sources": [...]}   documents used for the answer (RAG mode)
    {"type": "classification", ...}        how the agent classified the message
    {"type": "tool_call" / "tool_result"}  what the agent is doing
    {"type": "approval_required", ...}      a risky action is waiting for a human
    {"type": "escalated", "ticket_id": 1}   handed to a human agent
    {"type": "token", "text": "..."}       a piece of the answer
    {"type": "done", "prompt_version": 1}  the answer is complete
    {"type": "error", "message": "..."}    something went wrong
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from app import cache, guardrails, memory
from app.config import get_settings
from app.llm import LLMClient
from app.prompts import load_prompt
from app.rag.search import Hit, format_context, is_relevant, search

Event = dict[str, Any]


async def simple_chat(llm: LLMClient, conversation_id: str, message: str) -> AsyncIterator[Event]:
    """Week 1-2: a plain conversation with memory, no documents or tools."""
    await memory.add_message(conversation_id, "user", message)
    await memory.compact_if_needed(llm, conversation_id)
    summary, history = await memory.load_history(conversation_id)
    system = load_prompt("system")
    messages = memory.build_messages(system.text, summary, history)

    parts: list[str] = []
    async for piece in llm.stream(messages, purpose="chat", conversation_id=conversation_id):
        parts.append(piece)
        yield {"type": "token", "text": piece}

    answer = "".join(parts)
    await memory.add_message(conversation_id, "assistant", answer,
                             {"mode": "chat", "prompt_version": system.version})
    yield {"type": "done", "prompt_version": system.version}


# --- Weeks 3-4: answering from the documentation (RAG) --------------------------------

IDK = "I don't know based on the CloudNotes documentation."


async def rewrite_query(llm: LLMClient, history: list[dict[str, str]], message: str,
                        conversation_id: str | None = None) -> str:
    """Turn a follow-up like "how much is it?" into "How much does the Pro plan cost?".

    Without this, retrieval would search for "how much is it" and find nothing useful.
    The first message of a conversation is already standalone, so it is used as is.
    """
    previous = history[:-1][-6:]  # the last item is the message we are rewriting
    if not previous:
        return message
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in previous)
    result = await llm.chat(
        [
            {"role": "system", "content": load_prompt("rewrite").text},
            {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {message}"},
        ],
        model=get_settings().fast_model,
        temperature=0,
        purpose="rewrite_query",
        conversation_id=conversation_id,
    )
    return result.content.strip().strip('"') or message


def rag_messages(system_text: str, hits: list[Hit], summary: str | None,
                 history: list[dict[str, str]]) -> list[dict[str, str]]:
    rag = load_prompt("rag").format(context=format_context(hits))
    return memory.build_messages(f"{system_text}\n\n{rag}", summary, history)


async def retrieve(llm: LLMClient, query: str, **search_kwargs) -> list[Hit]:
    hits = await search(llm, query, **search_kwargs)
    return hits if is_relevant(hits, get_settings().rag_min_similarity) else []


def sources_payload(hits: list[Hit]) -> list[dict]:
    return [
        {"n": i, "source": h.source, "heading": h.heading, "content": h.content,
         "score": round(h.score, 4)}
        for i, h in enumerate(hits, start=1)
    ]


async def rag_chat(llm: LLMClient, conversation_id: str, message: str) -> AsyncIterator[Event]:
    """Retrieve relevant doc chunks, then stream an answer that cites them."""
    await memory.add_message(conversation_id, "user", message)
    await memory.compact_if_needed(llm, conversation_id)
    summary, history = await memory.load_history(conversation_id)
    system = load_prompt("system")
    first_message = len(history) == 1 and not summary

    query = await rewrite_query(llm, history, message, conversation_id)
    [query_vector] = await llm.embed([query])

    # Week 9: a first question similar enough to an earlier one reuses its answer.
    if first_message and (hit := await cache.lookup(query_vector)):
        yield {"type": "sources", "query": query, "sources": hit["sources"],
               "cache_hit": True, "cached_question": hit["question"]}
        yield {"type": "token", "text": hit["answer"]}
        await memory.add_message(conversation_id, "assistant", hit["answer"], {
            "mode": "rag", "cache_hit": True, "sources": [s["source"] for s in hit["sources"]]})
        yield {"type": "done", "prompt_version": system.version}
        return

    hits = await retrieve(llm, query, query_vector=query_vector)
    sources = sources_payload(hits)
    yield {"type": "sources", "query": query, "sources": sources}

    if not hits:
        # Nothing relevant: don't let the model improvise an answer.
        answer = f"{IDK} Would you like me to open a support ticket for you?"
        yield {"type": "token", "text": answer}
    else:
        parts: list[str] = []
        messages = rag_messages(system.text, hits, summary, history)
        async for piece in llm.stream(messages, purpose="rag_answer", conversation_id=conversation_id):
            parts.append(piece)
            yield {"type": "token", "text": piece}
        answer = "".join(parts)
        if first_message and guardrails.check_output(answer)[1] is False:
            await cache.store(query, query_vector, answer, sources)

    await memory.add_message(conversation_id, "assistant", answer, {
        "mode": "rag", "prompt_version": system.version, "query": query,
        "sources": [h.source for h in hits],
    })
    yield {"type": "done", "prompt_version": system.version}


async def answer_question(llm: LLMClient, question: str, **search_kwargs) -> dict:
    """One-shot RAG answer without memory. Used by /api/answer and the eval suite."""
    hits = await retrieve(llm, question, **search_kwargs)
    if not hits:
        return {"answer": IDK, "sources": [], "contexts": []}
    system = load_prompt("system")
    result = await llm.chat(rag_messages(system.text, hits, None, [{"role": "user", "content": question}]),
                            purpose="rag_answer")
    return {
        "answer": result.content,
        "sources": sources_payload(hits),
        "contexts": [h.content for h in hits],
    }


# --- Weeks 6-8: the support agent with tools ------------------------------------------


async def agent_chat(llm: LLMClient, conversation_id: str, message: str) -> AsyncIterator[Event]:
    """Run the tool-using agent and stream what it does: tool calls, results, approvals."""
    from app.agent.graph import run_agent_graph
    from app.agent.loop import run_agent_loop
    from app.agent.registry import get_registry
    from app.agent.tools import AgentContext

    await memory.add_message(conversation_id, "user", message)
    await memory.compact_if_needed(llm, conversation_id)
    summary, history = await memory.load_history(conversation_id)
    system, agent_prompt = load_prompt("system"), load_prompt("agent")
    messages = memory.build_messages(f"{system.text}\n\n{agent_prompt.text}", summary, history)

    queue: asyncio.Queue[Event] = asyncio.Queue()
    ctx = AgentContext(
        llm=llm, conversation_id=conversation_id, emit=queue.put_nowait,
        # Week 10: the agent may only touch accounts the customer named in this chat.
        allowed_emails=guardrails.emails_in(
            [m["content"] for m in history if m["role"] == "user"] + [summary or ""]),
    )
    engine = run_agent_graph if get_settings().agent_engine == "graph" else run_agent_loop
    task = asyncio.create_task(engine(messages, get_registry(), ctx))

    # Forward events while the agent works, then the final answer.
    while not task.done() or not queue.empty():
        getter = asyncio.ensure_future(queue.get())
        done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
        if getter in done:
            yield getter.result()
        else:
            getter.cancel()
    answer = task.result()  # re-raises if the agent crashed

    for i, word in enumerate(answer.split(" ")):
        yield {"type": "token", "text": word if i == 0 else " " + word}
    await memory.add_message(conversation_id, "assistant", answer, {
        "mode": "agent", "engine": get_settings().agent_engine,
        "prompt_version": agent_prompt.version, "tool_calls": ctx.tool_calls,
    })
    yield {"type": "done", "prompt_version": agent_prompt.version}



# --- Week 10: guardrails around every mode ------------------------------------------------


def guarded(handler):
    """Wrap a chat mode with input and output guardrails.

    Input: secrets are redacted before the model or database sees them, and obvious
    injection attempts are answered with a fixed reply without calling the model.
    Output: a reply that leaks our prompts is replaced (the UI gets a "replace" event).
    """

    async def run(llm: LLMClient, conversation_id: str, message: str) -> AsyncIterator[Event]:
        check = await guardrails.check_input(message, llm)
        for flag in check.flags:
            await guardrails.log_event(conversation_id, "pii_redacted", flag)
            yield {"type": "guardrail", "kind": "pii_redacted", "detail": flag}
        if check.blocked:
            await guardrails.log_event(conversation_id, "injection", check.reason or "")
            await memory.add_message(conversation_id, "user", check.text, {"blocked": True})
            await memory.add_message(conversation_id, "assistant", guardrails.BLOCKED_REPLY,
                                     {"guardrail": check.reason})
            yield {"type": "guardrail", "kind": "injection", "detail": check.reason}
            yield {"type": "token", "text": guardrails.BLOCKED_REPLY}
            yield {"type": "done"}
            return

        parts: list[str] = []
        async for event in handler(llm, conversation_id, check.text):
            if event["type"] == "token":
                parts.append(event["text"])
            if event["type"] == "done":
                safe, leaked = guardrails.check_output("".join(parts))
                if leaked:
                    await guardrails.log_event(conversation_id, "prompt_leak", "reply replaced")
                    await memory.replace_last_assistant(conversation_id, safe)
                    yield {"type": "replace", "text": safe}
            yield event

    return run
