"""Classification eval: prompted model vs fine-tuned model (weeks 2 and 11).

    uv run python -m app.evals.classify_eval --model qwen2.5:3b --tag prompted
    uv run python -m app.evals.classify_eval --model supportpilot-router --prompt classify_ft --tag finetuned
    uv run python -m app.evals.classify_eval --compare evals/results/*-classify-*.json

The default dataset is 30 hand-written, hand-labelled tickets (evals/datasets/classify_golden.jsonl).
`--dataset finetune/data/test.jsonl` also works, but synthetic test data shares the generator's
habits, so it flatters the fine-tuned model.

For each run we report accuracy per field, exact match (all three right), latency and prompt
tokens. The fine-tuned model should match the prompted one with a far shorter prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from app.classify import classify_ticket
from app.config import REPO_ROOT, get_settings
from app.llm import CallRecord, get_llm

DATASET = REPO_ROOT / "evals" / "datasets" / "classify_golden.jsonl"
RESULTS = REPO_ROOT / "evals" / "results"
FIELDS = ("category", "priority", "sentiment")


def load(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:  # fine-tuning format from app.finetune.generate
            message = next(m["content"] for m in row["messages"] if m["role"] == "user")
            row = {"id": f"t{i + 1:03d}", "message": message, "label": row["label"]}
        rows.append(row)
    return rows


def summarize(records: list[dict], calls: list[CallRecord]) -> dict:
    n = len(records) or 1
    summary = {f"{f}_acc": sum(r["correct"][f] for r in records) / n for f in FIELDS}
    summary["exact_match"] = sum(all(r["correct"].values()) for r in records) / n
    if calls:
        summary["latency_ms_p50"] = statistics.median(c.latency_ms for c in calls)
        summary["prompt_tokens_avg"] = sum(c.input_tokens for c in calls) / len(calls)
    return {k: round(v, 3) for k, v in summary.items()}


async def run(args) -> dict:
    settings = get_settings()
    llm = get_llm()
    calls: list[CallRecord] = []
    llm.add_listener(lambda c: calls.append(c) if c.purpose == "classify" else None)
    model = args.model or settings.classify_model or settings.fast_model
    rows = load(Path(args.dataset))
    records = []
    for row in rows:
        got = await classify_ticket(llm, row["message"], model=model, prompt=args.prompt)
        pred = got.model_dump()
        correct = {f: pred[f] == row["label"][f] for f in FIELDS}
        records.append({"id": row["id"], "message": row["message"], "label": row["label"],
                        "pred": pred, "correct": correct})
        if not all(correct.values()):
            print(f"  miss {row['id']}: want {row['label']} got {pred} | {row['message'][:60]}")
    summary = summarize(records, calls)
    result = {"tag": args.tag, "model": model, "prompt": args.prompt,
              "dataset": str(args.dataset), "summary": summary, "records": records}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}-classify-{args.tag}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {out.relative_to(REPO_ROOT)}")
    return result


def compare(paths: list[str]) -> None:
    runs = [json.loads(Path(p).read_text()) for p in paths]
    keys = list(runs[0]["summary"])
    print("| run | model | " + " | ".join(keys) + " |")
    print("| --- " * (len(keys) + 2) + "|")
    for r in runs:
        print(f"| {r['tag']} | {r['model']} | "
              + " | ".join(str(r["summary"].get(k, "")) for k in keys) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="defaults to CLASSIFY_MODEL, then FAST_MODEL")
    parser.add_argument("--prompt", default="classify", help="classify or classify_ft")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--tag", default="run")
    parser.add_argument("--compare", nargs="+", metavar="RESULT_JSON")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare)
        sys.exit(0)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
