import json
from argparse import Namespace

from app.classify import TicketClassification, classify_ticket
from app.config import Settings
from app.evals.classify_eval import load, summarize
from app.finetune.generate import label_grid, run
from app.llm import LLMClient


async def test_generate_writes_valid_chat_examples(tmp_path):
    args = Namespace(count=90, source="templates", verify=False, test_share=0.1, seed=1,
                     out=str(tmp_path))
    counts = await run(args)
    assert counts == {"train": 81, "test": 9}
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    roles = [m["role"] for m in rows[0]["messages"]]
    assert roles == ["system", "user", "assistant"]
    for row in rows:
        label = TicketClassification.model_validate_json(row["messages"][2]["content"])
        assert label.model_dump() == row["label"]
    # Every label combination is represented.
    assert len({json.dumps(r["label"], sort_keys=True) for r in rows}) == len(label_grid())


async def test_generate_with_fake_provider_falls_back_to_templates(tmp_path):
    args = Namespace(count=45, source="llm", verify=False, test_share=0.2, seed=1,
                     out=str(tmp_path))
    counts = await run(args)
    assert counts["train"] + counts["test"] == 45


def test_classify_eval_reads_both_dataset_formats(tmp_path):
    label = {"category": "billing", "priority": "high", "sentiment": "negative"}
    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps({"id": "c1", "message": "charged twice", "label": label}))
    chat = tmp_path / "test.jsonl"
    chat.write_text(json.dumps({"messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "charged twice"},
        {"role": "assistant", "content": json.dumps(label)}], "label": label}))
    assert load(golden)[0]["message"] == load(chat)[0]["message"] == "charged twice"

    records = [{"correct": {"category": True, "priority": True, "sentiment": False}},
               {"correct": {"category": True, "priority": True, "sentiment": True}}]
    summary = summarize(records, [])
    assert summary["category_acc"] == 1.0 and summary["sentiment_acc"] == 0.5
    assert summary["exact_match"] == 0.5


async def test_classify_uses_router_settings(monkeypatch):
    llm = LLMClient()
    seen = []
    llm.add_listener(seen.append)
    monkeypatch.setattr("app.classify.get_settings", lambda: Settings(
        llm_provider="fake", embed_provider="fake",
        classify_model="supportpilot-router", classify_prompt="classify_ft"))
    result = await classify_ticket(llm, "I was charged twice!!")
    assert result.category == "billing"
    assert seen[-1].model == "supportpilot-router"
    assert "Classify this CloudNotes support message" in seen[-1].input[0]["content"]


async def test_vertex_provider_uses_project_url_and_fresh_token():
    llm = LLMClient(Settings(llm_provider="vertex", embed_provider="fake",
                             vertex_project="demo-project", vertex_location="europe-west4"))
    assert str(llm._chat_client.base_url) == (
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/demo-project/"
        "locations/europe-west4/endpoints/openapi/")

    class StubToken:
        async def get(self):
            return "ya29.test-token"

    llm._google = StubToken()
    await llm._authorize()
    assert llm._chat_client.api_key == "ya29.test-token"


def test_vertex_provider_needs_a_project():
    import pytest

    with pytest.raises(RuntimeError, match="VERTEX_PROJECT"):
        LLMClient(Settings(llm_provider="vertex", embed_provider="fake"))
