import json

from app.classify import TicketClassification, classify_ticket
from app.llm import ChatResult, LLMClient, Usage


async def test_fake_chat_and_stream():
    llm = LLMClient()
    result = await llm.chat([{"role": "user", "content": "hello"}])
    assert "hello" in result.content
    assert result.usage.input_tokens > 0
    pieces = [p async for p in llm.stream([{"role": "user", "content": "hello there"}])]
    assert "".join(pieces) == result.content.replace("hello", "hello there")


async def test_listener_records_every_call():
    llm = LLMClient()
    records = []
    llm.add_listener(records.append)
    await llm.chat([{"role": "user", "content": "hi"}], purpose="test")
    assert records and records[0].purpose == "test" and records[0].latency_ms >= 0


async def test_embeddings_are_normalised_and_similar_for_similar_text():
    llm = LLMClient()
    a, b, c = await llm.embed(["reset my password", "how to reset password", "export to pdf"])
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))
    assert abs(dot(a, a) - 1) < 1e-6
    assert dot(a, b) > dot(a, c)


async def test_classify_returns_valid_json():
    result = await classify_ticket(LLMClient(), "I was charged twice this month!!")
    assert result == TicketClassification(category="billing", priority="high", sentiment="negative")


async def test_classify_retries_then_falls_back():
    class BrokenLLM(LLMClient):
        calls = 0

        async def chat(self, messages, **kwargs):
            BrokenLLM.calls += 1
            return ChatResult(content="not json", usage=Usage(1, 1))

    result = await classify_ticket(BrokenLLM(), "anything")
    assert BrokenLLM.calls == 2
    assert result.category == "other"


async def test_classify_accepts_fenced_json():
    class FencedLLM(LLMClient):
        async def chat(self, messages, **kwargs):
            body = json.dumps({"category": "technical", "priority": "low", "sentiment": "neutral"})
            return ChatResult(content=f"```json\n{body}\n```")

    result = await classify_ticket(FencedLLM(), "sync is slow")
    assert result.category == "technical"
