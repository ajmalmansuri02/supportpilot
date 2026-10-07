"""The agent loop in plain Python (week 6).

This is all an "agent" is: call the model with the tools; if it asks for tool calls, run
them and send back the results; repeat until it answers in plain text or we hit the step
limit. Frameworks like LangGraph (see graph.py) add structure on top of this same loop.
"""

from __future__ import annotations

import json
from typing import Any

from app.agent.tools import AgentContext, ToolRegistry
from app.llm import ChatResult

STEP_LIMIT_REPLY = ("I'm sorry, I couldn't finish that on my own. I've noted the details; "
                    "would you like me to pass this to a human agent?")


def assistant_message(result: ChatResult) -> dict[str, Any]:
    """Echo the model's tool calls back in the format the API expects."""
    return {
        "role": "assistant",
        "content": result.content or None,
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in result.tool_calls
        ],
    }


async def run_tool_calls(result: ChatResult, registry: ToolRegistry, ctx: AgentContext) -> list[dict]:
    messages = []
    for call in result.tool_calls:
        ctx.emit({"type": "tool_call", "id": call.id, "name": call.name, "args": call.arguments})
        output = await registry.execute(call, ctx)
        ctx.emit({"type": "tool_result", "id": call.id, "name": call.name, "result": json.loads(output)})
        messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
    return messages


async def run_agent_loop(messages: list[dict[str, Any]], registry: ToolRegistry, ctx: AgentContext,
                         *, max_steps: int = 6) -> str:
    messages = list(messages)
    for _ in range(max_steps):
        result = await ctx.llm.chat(messages, tools=registry.specs(), purpose="agent",
                                    conversation_id=ctx.conversation_id)
        if not result.tool_calls:
            return result.content
        messages.append(assistant_message(result))
        messages.extend(await run_tool_calls(result, registry, ctx))
    return STEP_LIMIT_REPLY
