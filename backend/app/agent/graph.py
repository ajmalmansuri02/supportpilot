"""The same agent as loop.py, rebuilt as a LangGraph state machine (week 7).

    START -> classify --(asks for a person)--> escalate -> END
                      \\-> agent <-> tools   (until the model answers or hits the step limit)
                                \\-> give_up -> END

What LangGraph adds over the plain loop:
- The flow is explicit and drawable (print(GRAPH.get_graph().draw_mermaid())).
- Each node does one job and can be tested on its own.
- Routing decisions (escalate? keep looping?) are plain functions, not buried in a loop.
- It supports checkpoints and interrupts for long-running or paused agents. We persist
  approvals in Postgres instead (see actions.py), which survives server restarts and is
  easy to inspect, but `langgraph.types.interrupt` is the built-in alternative.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agent.loop import STEP_LIMIT_REPLY, assistant_message, run_tool_calls
from app.agent.tools import AgentContext, ToolRegistry
from app.classify import classify_ticket
from app.llm import ChatResult, ToolCall

MAX_STEPS = 6
WANTS_HUMAN = re.compile(
    r"\b(talk|speak|chat)\s+(to|with)\s+(a|an|some)?\s*(human|person|real person|agent|someone|"
    r"representative|manager)\b|\breal person\b|\bhuman agent\b",
    re.IGNORECASE,
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], operator.add]
    classification: dict[str, str]
    steps: int
    final: str


def _deps(config: RunnableConfig) -> tuple[ToolRegistry, AgentContext]:
    c = config["configurable"]
    return c["registry"], c["ctx"]


def _last_user(state: AgentState) -> str:
    return next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")


async def classify(state: AgentState, config: RunnableConfig) -> AgentState:
    _, ctx = _deps(config)
    result = await classify_ticket(ctx.llm, _last_user(state), conversation_id=ctx.conversation_id)
    ctx.emit({"type": "classification", **result.model_dump()})
    return {"classification": result.model_dump()}


def route_after_classify(state: AgentState) -> str:
    return "escalate" if WANTS_HUMAN.search(_last_user(state)) else "agent"


async def agent(state: AgentState, config: RunnableConfig) -> AgentState:
    registry, ctx = _deps(config)
    result = await ctx.llm.chat(state["messages"], tools=registry.specs(), purpose="agent",
                                conversation_id=ctx.conversation_id)
    if not result.tool_calls:
        return {"final": result.content}
    return {"messages": [assistant_message(result)], "steps": state.get("steps", 0) + 1}


def route_after_agent(state: AgentState) -> str:
    if "final" in state:
        return END
    return "give_up" if state.get("steps", 0) >= MAX_STEPS else "tools"


async def tools(state: AgentState, config: RunnableConfig) -> AgentState:
    registry, ctx = _deps(config)
    last = state["messages"][-1]
    calls = [ToolCall(id=c["id"], name=c["function"]["name"],
                      arguments=_loads(c["function"]["arguments"])) for c in last["tool_calls"]]
    return {"messages": await run_tool_calls(ChatResult(content="", tool_calls=calls), registry, ctx)}


async def escalate(state: AgentState, config: RunnableConfig) -> AgentState:
    registry, ctx = _deps(config)
    call = ToolCall(id="escalate", name="escalate_to_human",
                    arguments={"reason": _last_user(state)[:500]})
    await run_tool_calls(ChatResult(content="", tool_calls=[call]), registry, ctx)
    return {"final": "Of course. I've passed your conversation to a human support agent, "
                     "who will reply by email. Is there anything I can note for them?"}


async def give_up(state: AgentState, config: RunnableConfig) -> AgentState:
    return {"final": STEP_LIMIT_REPLY}


def _loads(raw: str) -> dict:
    import json

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", classify)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("escalate", escalate)
    g.add_node("give_up", give_up)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_after_classify, ["agent", "escalate"])
    g.add_conditional_edges("agent", route_after_agent, ["tools", "give_up", END])
    g.add_edge("tools", "agent")
    g.add_edge("escalate", END)
    g.add_edge("give_up", END)
    return g.compile()


GRAPH = build_graph()


async def run_agent_graph(messages: list[dict[str, Any]], registry: ToolRegistry,
                          ctx: AgentContext) -> str:
    state = await GRAPH.ainvoke(
        {"messages": messages, "steps": 0},
        config={"configurable": {"registry": registry, "ctx": ctx}, "recursion_limit": 40},
    )
    return state["final"]
