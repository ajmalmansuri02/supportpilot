"""Generate a labelled ticket dataset for fine-tuning the router (week 11).

    uv run python -m app.finetune.generate --count 500            # uses CHAT_MODEL as the "teacher"
    uv run python -m app.finetune.generate --count 500 --verify   # also drop tickets the teacher mislabels
    uv run python -m app.finetune.generate --source templates     # offline, no model needed

This is distillation: a big model (Gemini or qwen2.5:7b) writes and labels examples, and a
small model (Qwen2.5 0.5B-1.5B) learns to copy it, so the small one can do the job faster and
with a much shorter prompt. Output goes to finetune/data/ in the chat format Unsloth expects:

    {"messages": [system, user, assistant], "label": {...}}

Synthetic data has the generator's blind spots, so the real test is the hand-written set in
evals/datasets/classify_golden.jsonl (see app/evals/classify_eval.py).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import re
from pathlib import Path
from typing import get_args

from app.classify import TicketClassification, classify_ticket
from app.config import REPO_ROOT, get_settings
from app.llm import LLMClient, get_llm
from app.prompts import load_prompt

OUT_DIR = REPO_ROOT / "finetune" / "data"

CATEGORIES = get_args(TicketClassification.model_fields["category"].annotation)
PRIORITIES = get_args(TicketClassification.model_fields["priority"].annotation)
SENTIMENTS = get_args(TicketClassification.model_fields["sentiment"].annotation)

TOPICS = {
    "billing": ["charged twice", "refund request", "invoice details", "card declined",
                "price of the Team plan", "annual vs monthly billing", "unexpected charge"],
    "technical": ["notes not syncing", "app crashes on start", "attachments won't upload",
                  "offline mode lost edits", "search returns nothing", "export to PDF fails",
                  "slow on Android"],
    "account": ["can't log in", "two-factor codes not arriving", "reset password email missing",
                "change account email", "delete my account", "account locked"],
    "feature_request": ["dark mode", "nested tags", "calendar integration", "public API",
                        "markdown tables", "handwriting on tablets"],
    "other": ["partnership proposal", "praise for the product", "press inquiry",
              "question about the company", "student discount", "job application"],
}
STYLES = ["short and casual", "detailed and formal", "frustrated, with a typo or two",
          "written by a non-native English speaker", "one line only", "polite but worried"]

GENERATE_PROMPT = """You write realistic customer support messages for CloudNotes, a note-taking
app with Free, Pro and Team plans. Write {k} different messages that a support team would
label exactly as:

- category: {category}
- priority: {priority}  (high = money lost, can't access account, data loss, service down;
  medium = something broken with a workaround; low = questions and ideas)
- sentiment: {sentiment}

Topic hint: {topic}. Style: {style}. Vary length, names and details. Do not mention the labels.
Return JSON: {{"tickets": ["...", "..."]}}"""

TICKETS_SCHEMA = {
    "type": "object",
    "properties": {"tickets": {"type": "array", "items": {"type": "string"}}},
    "required": ["tickets"],
}


def to_example(text: str, label: dict, system: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(label)},
        ],
        "label": label,
    }


def label_grid() -> list[dict]:
    return [{"category": c, "priority": p, "sentiment": s}
            for c, p, s in itertools.product(CATEGORIES, PRIORITIES, SENTIMENTS)]


async def generate_with_llm(llm: LLMClient, count: int, rng: random.Random,
                            per_call: int = 5, concurrency: int = 4) -> list[tuple[str, dict]]:
    grid = label_grid()
    calls = -(-count // per_call)
    jobs = [grid[i % len(grid)] for i in range(calls)]
    sem = asyncio.Semaphore(concurrency)

    async def one(label: dict) -> list[tuple[str, dict]]:
        prompt = GENERATE_PROMPT.format(k=per_call, topic=rng.choice(TOPICS[label["category"]]),
                                        style=rng.choice(STYLES), **label)
        async with sem:
            result = await llm.chat([{"role": "user", "content": prompt}],
                                    json_schema=TICKETS_SCHEMA, temperature=1.0,
                                    purpose="finetune_generate")
        try:
            tickets = json.loads(result.content).get("tickets", [])
        except json.JSONDecodeError:
            return []
        return [(t, label) for t in tickets if isinstance(t, str)]

    batches = await asyncio.gather(*(one(label) for label in jobs))
    return [pair for batch in batches for pair in batch]


# Offline generator: label-consistent sentences stitched from pieces. Good enough to test
# the pipeline and the notebook end to end, far too uniform to train a useful model.
_OPEN = {"positive": ["Hi team, love CloudNotes!", "Thanks for the great app.", "Big fan here."],
         "neutral": ["Hello,", "Hi,", "Quick question."],
         "negative": ["This is really frustrating.", "I'm very unhappy.", "Honestly this is ridiculous."]}
_URGENCY = {"high": ["I need this fixed today.", "This is urgent, I can't work.", "Please help ASAP."],
            "medium": ["There's a workaround but it's annoying.", "Not urgent but please look.", ""],
            "low": ["No rush.", "Just curious.", "Whenever you get a chance."]}
_BODY = {
    "billing": "I have a problem with {topic} on my invoice.",
    "technical": "I'm having an issue: {topic}.",
    "account": "I need help with my account: {topic}.",
    "feature_request": "Could you please add {topic}?",
    "other": "I'm writing about a {topic}.",
}


def generate_from_templates(count: int, rng: random.Random) -> list[tuple[str, dict]]:
    grid = label_grid()
    pairs = []
    for i in range(count):
        label = grid[i % len(grid)]
        text = " ".join(filter(None, [
            rng.choice(_OPEN[label["sentiment"]]),
            _BODY[label["category"]].format(topic=rng.choice(TOPICS[label["category"]])),
            rng.choice(_URGENCY[label["priority"]]),
            f"(ref {rng.randint(1000, 9999)})",
        ]))
        pairs.append((text, label))
    return pairs


async def verify(llm: LLMClient, pairs: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Keep only tickets where the teacher, using the full prompt, agrees on category and priority."""
    model = get_settings().chat_model
    sem = asyncio.Semaphore(4)

    async def check(text: str, label: dict) -> bool:
        async with sem:
            got = await classify_ticket(llm, text, model=model, prompt="classify")
        return got.category == label["category"] and got.priority == label["priority"]

    keep = await asyncio.gather(*(check(t, lbl) for t, lbl in pairs))
    return [p for p, ok in zip(pairs, keep, strict=True) if ok]


def clean(pairs: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    seen, out = set(), []
    for text, label in pairs:
        text = text.strip()
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if 10 <= len(text) <= 800 and key not in seen:
            seen.add(key)
            out.append((text, label))
    return out


def write(pairs: list[tuple[str, dict]], out_dir: Path, test_share: float,
          rng: random.Random) -> dict[str, int]:
    system = load_prompt("classify_ft").text
    rng.shuffle(pairs)
    n_test = max(1, int(len(pairs) * test_share))
    splits = {"test": pairs[:n_test], "train": pairs[n_test:]}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        lines = [json.dumps(to_example(text, label, system)) + "\n" for text, label in rows]
        (out_dir / f"{name}.jsonl").write_text("".join(lines))
    return {name: len(rows) for name, rows in splits.items()}


async def run(args) -> dict[str, int]:
    rng = random.Random(args.seed)
    llm = get_llm()
    source = args.source
    if source == "llm" and llm.provider == "fake":
        print("LLM_PROVIDER=fake can't write tickets; using --source templates instead.")
        source = "templates"
    if source == "templates":
        pairs = generate_from_templates(args.count, rng)
    else:
        pairs = await generate_with_llm(llm, args.count, rng)
    pairs = clean(pairs)
    print(f"Generated {len(pairs)} unique tickets")
    if args.verify:
        before = len(pairs)
        pairs = await verify(llm, pairs)
        print(f"Teacher agreed on {len(pairs)}/{before} ({len(pairs) / max(before, 1):.0%})")
    counts = write(pairs, Path(args.out), args.test_share, rng)
    print(f"Wrote {counts['train']} train / {counts['test']} test examples to {args.out}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--source", choices=["llm", "templates"], default="llm")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--test-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT_DIR))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
