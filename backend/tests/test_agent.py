import json

import pytest
from conftest import read_sse

from app.agent.registry import simplify_schema
from app.agent.tools import AgentContext, default_registry
from app.llm import LLMClient, ToolCall
from app.services.seed import seed


@pytest.fixture(autouse=True)
def fresh_data(client):
    client.portal.call(seed, True)  # reset customers, invoices, tickets and actions


def agent(client, message, conversation_id=None):
    events = read_sse(client.post("/api/chat", json={
        "message": message, "mode": "agent", "conversation_id": conversation_id}))
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    return events, answer


def test_docs_question_uses_search_docs(client):
    events, answer = agent(client, "How do I export all my notes?")
    calls = [e["name"] for e in events if e["type"] == "tool_call"]
    assert calls == ["search_docs"]
    assert "[1]" in answer
    assert any(e["type"] == "classification" for e in events)


def test_duplicate_charge_goes_through_human_approval(client):
    events, answer = agent(client, "I was charged twice this month! My email is priya@example.com")
    calls = [e["name"] for e in events if e["type"] == "tool_call"]
    assert calls == ["get_account", "issue_refund"]
    approval = next(e for e in events if e["type"] == "approval_required")
    assert "INV-204519" in approval["description"]
    assert "approve" in answer

    # Nothing is refunded until a human approves.
    account = client.get("/api/actions/" + approval["action_id"]).json()
    assert account["status"] == "pending"
    decision = client.post(f"/api/actions/{approval['action_id']}", json={"approve": True}).json()
    assert decision["status"] == "approved" and decision["result"]["refunded_usd"] == 8.0

    # Deciding twice does nothing.
    again = client.post(f"/api/actions/{approval['action_id']}", json={"approve": False}).json()
    assert again["already_decided"]

    conv = client.get(f"/api/conversations/{events[0]['conversation_id']}").json()
    assert conv["messages"][-1]["content"].startswith("Update: a support agent approved")


def test_rejected_actions_are_not_executed(client):
    events, _ = agent(client, "I was charged twice. priya@example.com")
    approval = next(e for e in events if e["type"] == "approval_required")
    result = client.post(f"/api/actions/{approval['action_id']}", json={"approve": False}).json()
    assert result["status"] == "rejected"


def test_asking_for_a_human_escalates(client):
    events, _ = agent(client, "I want to talk to a human please")
    assert any(e["type"] == "escalated" for e in events)
    conv = client.get(f"/api/conversations/{events[0]['conversation_id']}").json()
    assert conv["escalated"] is True
    tickets = client.get("/api/tickets", params={"status": "open"}).json()
    assert any(t["assignee"] == "human" for t in tickets)


def test_billing_question_without_duplicate_opens_a_ticket(client):
    events, answer = agent(client, "Question about my invoice, email maria@example.com")
    calls = [e["name"] for e in events if e["type"] == "tool_call"]
    assert calls == ["get_account", "create_ticket"]
    assert "ticket #" in answer


def test_plain_loop_engine_gives_the_same_result(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "agent_engine", "loop")
    events, _ = agent(client, "How do I reset my password?")
    assert [e["name"] for e in events if e["type"] == "tool_call"] == ["search_docs"]
    assert not any(e["type"] == "classification" for e in events)  # only the graph classifies


async def test_tool_errors_are_returned_to_the_model_not_raised():
    registry = default_registry()
    ctx = AgentContext(llm=LLMClient())
    unknown = json.loads(await registry.execute(ToolCall("1", "delete_everything", {}), ctx))
    missing = json.loads(await registry.execute(ToolCall("2", "get_account", {}), ctx))
    assert "Unknown tool" in unknown["error"] and "Missing" in missing["error"]


def test_hidden_args_are_not_shown_to_the_model():
    spec = next(t for t in default_registry().specs() if t["function"]["name"] == "create_ticket")
    assert "conversation_id" not in spec["function"]["parameters"]["properties"]


def test_refund_for_someone_elses_invoice_is_refused(client):
    registry = default_registry()
    ctx = AgentContext(llm=LLMClient())
    call = ToolCall("1", "issue_refund", {"invoice_id": "INV-198001",  # Rahul's invoice
                                          "customer_email": "priya@example.com", "reason": "test"})
    result = json.loads(client.portal.call(registry.execute, call, ctx))
    assert "does not belong" in result["error"]


def test_tools_endpoint_lists_risk_levels(client):
    risks = {t["name"]: t["risk"] for t in client.get("/api/tools").json()}
    assert risks["issue_refund"] == "approval" and risks["get_account"] == "read"


def test_simplify_schema_collapses_optional_types():
    schema = {"type": "object", "title": "x", "properties": {
        "email": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "Email"}}}
    assert simplify_schema(schema) == {"type": "object", "properties": {
        "email": {"type": "string", "default": None}}}
