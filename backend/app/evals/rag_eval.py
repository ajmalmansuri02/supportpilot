"""RAG evaluation suite (week 5).

    uv run python -m app.evals.rag_eval                         # default settings
    uv run python -m app.evals.rag_eval --retrieval vector --tag vector-only
    uv run python -m app.evals.rag_eval --compare results/a.json results/b.json
    uv run python -m app.evals.rag_eval --baseline ../evals/baseline.json   # CI gate

Metrics
- hit_rate      share of answerable questions where an expected source was retrieved
- mrr           mean reciprocal rank of the first expected source (1.0 = always ranked first)
- correctness   LLM judge: does the answer match the reference answer? (0-1)
- faithfulness  LLM judge: is every claim supported by the retrieved excerpts? (0-1)
- citation_rate share of answered questions that cite an excerpt like [1]
- abstain_ok    share of UNANSWERABLE questions where the bot correctly said "I don't know"
- false_abstain share of ANSWERABLE questions where the bot wrongly said "I don't know"

Use a strong judge if you can (e.g. --judge-model with LLM_PROVIDER=gemini); a small
local model is a noisy grader. Results are saved to evals/results/ for comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean

from app import db
from app.chat import answer_question
from app.config import REPO_ROOT, get_settings
from app.llm import LLMClient, get_llm
from app.prompts import load_prompt

DATASET = REPO_ROOT / "evals" / "datasets" / "rag_golden.jsonl"
RESULTS = REPO_ROOT / "evals" / "results"
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
}
ABSTAIN = re.compile(r"(don't|do not|cannot|can't) (know|find)|not (in|covered by) the (documentation|docs)", re.IGNORECASE)
CITATION = re.compile(r"\[\d+\]")
METRICS = ["hit_rate", "mrr", "correctness", "faithfulness", "citation_rate", "abstain_ok", "false_abstain"]
LOWER_IS_BETTER = {"false_abstain"}


async def judge(llm: LLMClient, prompt: str, content: str, model: str | None) -> float:
    result = await llm.chat(
        [{"role": "system", "content": load_prompt(prompt).text}, {"role": "user", "content": content}],
        json_schema=JUDGE_SCHEMA, temperature=0, model=model, purpose="eval_judge",
    )
    try:
        return max(0.0, min(1.0, float(json.loads(result.content)["score"])))
    except (ValueError, KeyError, TypeError):
        return 0.0


async def evaluate_one(llm: LLMClient, row: dict, args) -> dict:
    out = await answer_question(llm, row["question"], mode=args.retrieval, reranker=args.reranker)
    answer = out["answer"]
    retrieved = [s["source"] for s in out["sources"]]
    abstained = bool(ABSTAIN.search(answer))
    record = {"id": row["id"], "question": row["question"], "answer": answer,
              "retrieved": retrieved, "answerable": row["answerable"], "abstained": abstained}

    if not row["answerable"]:
        return record

    ranks = [retrieved.index(s) + 1 for s in row["sources"] if s in retrieved]
    record["hit"] = 1.0 if ranks else 0.0
    record["rr"] = 1.0 / min(ranks) if ranks else 0.0
    record["cited"] = 1.0 if CITATION.search(answer) else 0.0
    if args.judge:
        record["correctness"] = await judge(
            llm, "judge_correctness",
            f"QUESTION: {row['question']}\nREFERENCE: {row['expected']}\nANSWER: {answer}",
            args.judge_model,
        )
        context = "\n\n".join(out["contexts"]) or "(no excerpts)"
        record["faithfulness"] = await judge(
            llm, "judge_faithfulness", f"EXCERPTS:\n{context}\n\nANSWER: {answer}", args.judge_model
        )
    return record


def summarize(records: list[dict]) -> dict[str, float]:
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    answered = [r for r in answerable if not r["abstained"]]

    def avg(rows, key):
        values = [r[key] for r in rows if key in r]
        return round(mean(values), 3) if values else None

    return {
        "hit_rate": avg(answerable, "hit"),
        "mrr": avg(answerable, "rr"),
        "correctness": avg(answerable, "correctness"),
        "faithfulness": avg(answerable, "faithfulness"),
        "citation_rate": avg(answered, "cited"),
        "abstain_ok": round(mean(r["abstained"] for r in unanswerable), 3) if unanswerable else None,
        "false_abstain": round(mean(r["abstained"] for r in answerable), 3) if answerable else None,
    }


def table(columns: dict[str, dict]) -> str:
    names = list(columns)
    lines = ["| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for m in METRICS:
        cells = [("–" if columns[n].get(m) is None else f"{columns[n][m]:.3f}") for n in names]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check_regression(summary: dict, baseline: dict, max_drop: float) -> list[str]:
    problems = []
    for metric, base in baseline.items():
        now = summary.get(metric)
        if base is None or now is None:
            continue
        drop = (now - base) if metric in LOWER_IS_BETTER else (base - now)
        if drop > max_drop:
            problems.append(f"{metric}: {base:.3f} -> {now:.3f}")
    return problems


async def run(args) -> int:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    s = get_settings()
    print(f"Evaluating {len(rows)} questions | provider={s.llm_provider} model={s.chat_model} "
          f"retrieval={args.retrieval or s.retrieval_mode} reranker={args.reranker or s.reranker}")

    await db.open_pool()
    try:
        llm = get_llm()
        started = time.perf_counter()
        sem = asyncio.Semaphore(args.concurrency)

        async def guarded(row):
            async with sem:
                return await evaluate_one(llm, row, args)

        records = await asyncio.gather(*(guarded(r) for r in rows))
    finally:
        await db.close_pool()

    summary = summarize(records)
    tag = args.tag or f"{args.retrieval or s.retrieval_mode}-{args.reranker or s.reranker}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}.json"
    out.write_text(json.dumps({
        "tag": tag, "provider": s.llm_provider, "model": s.chat_model,
        "seconds": round(time.perf_counter() - started, 1), "summary": summary, "records": records,
    }, indent=2))
    print(table({tag: summary}))
    print(f"\nSaved {out.relative_to(REPO_ROOT)}")
    for r in records:
        if r["answerable"] and (r.get("hit") == 0 or r["abstained"]):
            print(f"  check {r['id']}: hit={r.get('hit')} abstained={r['abstained']} | {r['question']}")

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())["summary"]
        problems = check_regression(summary, baseline, args.max_drop)
        if problems:
            print("\nREGRESSION against baseline:\n  " + "\n  ".join(problems))
            return 1
        print("\nNo regression against baseline.")
    if args.save_baseline:
        Path(args.save_baseline).write_text(json.dumps({"tag": tag, "summary": summary}, indent=2) + "\n")
        print(f"Baseline written to {args.save_baseline}")
    return 0


def compare(paths: list[str]) -> None:
    columns = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        columns[data.get("tag", Path(p).stem)] = data["summary"]
    print(table(columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval", choices=["vector", "keyword", "hybrid"])
    parser.add_argument("--reranker", choices=["none", "cross_encoder"])
    parser.add_argument("--no-judge", dest="judge", action="store_false", help="skip LLM-judged metrics")
    parser.add_argument("--judge-model", help="model for grading (defaults to CHAT_MODEL)")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--tag", help="name for this run in result files and tables")
    parser.add_argument("--baseline", help="fail (exit 1) if a metric drops vs this file")
    parser.add_argument("--max-drop", type=float, default=0.05)
    parser.add_argument("--save-baseline", help="write this run's summary as a new baseline")
    parser.add_argument("--compare", nargs="+", metavar="RESULT_JSON", help="print runs side by side")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare)
        return
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
