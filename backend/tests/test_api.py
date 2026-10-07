from conftest import read_sse


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["provider"] == "fake"


def test_chat_streams_and_remembers(client):
    r = client.post("/api/chat", json={"message": "hello"})
    events = read_sse(r)
    assert events[0]["type"] == "meta"
    conv_id = events[0]["conversation_id"]
    assert any(e["type"] == "token" for e in events)
    assert events[-1]["type"] == "done"

    r2 = client.post("/api/chat", json={"message": "again", "conversation_id": conv_id})
    assert read_sse(r2)[0]["conversation_id"] == conv_id
    conv = client.get(f"/api/conversations/{conv_id}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]


def test_bad_conversation_id_starts_a_new_one(client):
    events = read_sse(client.post("/api/chat", json={"message": "hi", "conversation_id": "nope"}))
    assert events[0]["conversation_id"] != "nope"
    assert client.get("/api/conversations/nope").status_code == 404


def test_long_conversations_get_summarised(client):
    conv_id = None
    for i in range(8):  # 16 messages > MEMORY_MAX_MESSAGES (12)
        payload = {"message": f"message number {i}", "conversation_id": conv_id}
        conv_id = read_sse(client.post("/api/chat", json=payload))[0]["conversation_id"]
    conv = client.get(f"/api/conversations/{conv_id}").json()
    assert conv["summary"] and conv["summary"].startswith("Summary:")


def test_classify_endpoint(client):
    body = client.post("/api/classify", json={"message": "Please add a dark mode feature"}).json()
    assert body["category"] == "feature_request"
