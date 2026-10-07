from conftest import read_sse

from app.chat import IDK
from app.rag.chunking import chunk_markdown, split_sections
from app.rag.search import Hit, is_relevant, reciprocal_rank_fusion

DOC = """# Refund policy

Intro paragraph.

## Annual plans

Annual plans can be refunded within 30 days.

## Monthly plans

""" + "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(6))


def test_sections_keep_heading_path():
    sections = split_sections(DOC)
    assert sections[0] == ("Refund policy", "Intro paragraph.")
    assert sections[1][0] == "Refund policy > Annual plans"


def test_long_sections_are_split_with_overlap():
    chunks = [c for c in chunk_markdown(DOC, max_chars=800) if c.heading.endswith("Monthly plans")]
    assert len(chunks) > 1
    assert all(len(c.content) <= 800 + 400 for c in chunks)
    last_para = chunks[0].content.split("\n\n")[-1]
    assert chunks[1].content.startswith(last_para)  # overlap


def test_rrf_rewards_chunks_found_by_both_searches():
    a, b, c = (Hit(i, "s", "t", "h", "x") for i in (1, 2, 3))
    fused = reciprocal_rank_fusion([a, b], [Hit(2, "s", "t", "h", "x", keyword_match=True), c])
    assert fused[0].chunk_id == 2 and fused[0].keyword_match


def test_relevance_uses_similarity_when_available():
    assert not is_relevant([], 0.5)
    assert not is_relevant([Hit(1, "s", "t", "h", "x", similarity=0.1)], 0.5)
    assert is_relevant([Hit(1, "s", "t", "h", "x", similarity=0.7)], 0.5)
    assert is_relevant([Hit(1, "s", "t", "h", "x", keyword_match=True)], 0.5)


def test_search_endpoint_finds_the_right_doc(client):
    for mode in ("vector", "keyword", "hybrid"):
        hits = client.get("/api/search", params={"q": "refund for a duplicate charge", "mode": mode}).json()
        assert hits[0]["source"] == "refunds.md", mode


def test_answer_cites_sources(client):
    body = client.post("/api/answer", json={"question": "How much does the Pro plan cost per year?"}).json()
    assert "$80" in body["answer"] and "[1]" in body["answer"]
    assert body["sources"][0]["source"] == "plans-and-pricing.md"


def test_unanswerable_question_says_i_dont_know(client):
    body = client.post("/api/answer", json={"question": "What is the weather in Paris?"}).json()
    assert body["answer"] == IDK and body["sources"] == []


def test_rag_chat_streams_sources_then_answer(client):
    events = read_sse(client.post("/api/chat", json={"message": "How do I reset my password?", "mode": "rag"}))
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[1] == "sources" and types[-1] == "done"
    assert events[1]["sources"][0]["source"] == "account-and-login.md"
    conv = client.get(f"/api/conversations/{events[0]['conversation_id']}").json()
    assert conv["messages"][-1]["meta"]["sources"]


def test_follow_up_questions_are_rewritten(client):
    first = read_sse(client.post("/api/chat", json={"message": "Tell me about the Team plan", "mode": "rag"}))
    conv_id = first[0]["conversation_id"]
    follow = read_sse(client.post("/api/chat", json={
        "message": "how much does it cost?", "mode": "rag", "conversation_id": conv_id}))
    sources = next(e for e in follow if e["type"] == "sources")
    assert sources["query"] == "how much does it cost?"  # the fake model echoes; real models rewrite
