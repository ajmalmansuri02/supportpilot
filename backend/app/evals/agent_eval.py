"""Agent evaluation: does the agent take the right actions? (week 7)

    uv run python -m app.evals.agent_eval
    uv run python -m app.evals.agent_eval --engine loop --tag loop

Each scenario in evals/datasets/agent_scenarios.jsonl says which tools must be called (in
order), which must NOT be called (e.g. no refund without an account check, no tools at all
for a prompt injection), which events must happen (approval_required, escalated) and words
the reply should contain. A scenario passes only if every check passes.

Mock data is reset before every scenario, so refunds can be tested repeatedly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from app import db, memory
from app.chat import agent_chat
from app.config import REPO_ROOT, get_settings
from app.llm import get_llm
from app.services.seed import seed

DATASET = REPO_ROOT / "evals" / "datasets" / "agent_scenarios.jsonl"
RESULTS = REPO_ROOT / "evals" / "results"


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    it = iter(actual)
    return all(name in it for name in expected)


async def run_scenario(row: dict) -> dict:
    await seed(reset=True)
    conversation_id = await memory.create_conversation()
    events = [e async for e in agent_chat(get_llm(), conversation_id, row["message"])]
    tools = [e["name"] for e in events if e["type"] == "tool_call"]
    kinds = {e["type"] for e in events}
    answer = "".join(e["text"] for e in events if e["type"] == "token")

    checks = {
        "expected_tools": is_subsequence(row["expect_tools"], tools),
        "no_forbidden_tools": not set(tools) & set(row["forbid_tools"]),
        "expected_events": set(row["expect_events"]) <= kinds,
        "no_unexpected_approval": "approval_required" in row["expect_events"]
        or "approval_required" not in kinds,
        "answer_mentions": not row["answer_contains_any"]
        or any(w.lower() in answer.lower() for w in row["answer_contains_any"]),
    }
    return {"id": row["id"], "message": row["message"], "tools": tools, "answer": answer,
            "checks": checks, "passed": all(checks.values())}


def summarize(records: list[dict]) -> dict:
    n = len(records)
    out = {"task_success": round(sum(r["passed"] for r in records) / n, 3),
           "avg_tool_calls": round(sum(len(r["tools"]) for r in records) / n, 2)}
    for check in records[0]["checks"]:
        out[check] = round(sum(r["checks"][check] for r in records) / n, 3)
    return out


async def run(args) -> int:
    if args.engine:
        get_settings().agent_engine = args.engine
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    s = get_settings()
    print(f"Running {len(rows)} agent scenarios | provider={s.llm_provider} model={s.chat_model} "
          f"engine={s.agent_engine}")
    await db.open_pool()
    try:
        await db.init_schema()
        started = time.perf_counter()
        records = [await run_scenario(r) for r in rows]  # sequential: each resets the data
        await seed(reset=True)
    finally:
        await db.close_pool()

    summary = summarize(records)
    tag = args.tag or s.agent_engine
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}-agent-{tag}.json"
    out.write_text(json.dumps({"tag": tag, "provider": s.llm_provider, "model": s.chat_model,
                               "seconds": round(time.perf_counter() - started, 1),
                               "summary": summary, "records": records}, indent=2))
    for key, value in summary.items():
        print(f"  {key:24} {value}")
    for r in records:
        if not r["passed"]:
            failed = [k for k, ok in r["checks"].items() if not ok]
            print(f"  FAIL {r['id']} {failed} tools={r['tools']} | {r['message']}")
    print(f"Saved {out.relative_to(REPO_ROOT)}")

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())["summary"]["task_success"]
        if summary["task_success"] < base - args.max_drop:
            print(f"REGRESSION: task_success {base:.3f} -> {summary['task_success']:.3f}")
            return 1
        print("No regression against baseline.")
    if args.save_baseline:
        Path(args.save_baseline).write_text(json.dumps({"tag": tag, "summary": summary}, indent=2) + "\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["graph", "loop"])
    parser.add_argument("--tag")
    parser.add_argument("--baseline")
    parser.add_argument("--max-drop", type=float, default=0.05)
    parser.add_argument("--save-baseline")
    sys.exit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
