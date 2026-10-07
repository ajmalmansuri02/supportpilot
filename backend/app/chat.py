"""The chat pipeline. Each mode is an async generator of events that the API streams
to the browser as Server-Sent Events:

    {"type": "token", "text": "..."}       a piece of the answer
    {"type": "done", "prompt_version": 1}  the answer is complete
    {"type": "error", "message": "..."}    something went wrong
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app import memory
from app.llm import LLMClient
from app.prompts import load_prompt

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
