# Evals

Evals are automated tests for AI quality. A normal unit test checks `2 + 2 == 4`; an eval
checks "does the bot still answer refund questions correctly after I changed the prompt?"

## Datasets

| File | What it holds |
| --- | --- |
| `datasets/rag_golden.jsonl` | 50 answerable questions with reference answers and the doc they come from, plus 5 the docs can't answer |

## Run the RAG eval

```bash
cd backend
uv run python -m app.evals.rag_eval                          # current .env settings
uv run python -m app.evals.rag_eval --retrieval vector --tag vector
uv run python -m app.evals.rag_eval --retrieval hybrid --tag hybrid
uv run python -m app.evals.rag_eval --compare ../evals/results/*-vector.json ../evals/results/*-hybrid.json
```

Each run prints a metrics table, saves the full results (every question, answer and score)
to `evals/results/`, and lists the questions that failed so you can look at them.

Tips:
- `--limit 10` for a quick check, `--no-judge` to skip the slower LLM-graded metrics.
- A small local model is a noisy judge. For more trustworthy scores, grade with Gemini:
  set `LLM_PROVIDER=gemini` for the eval run, or pass `--judge-model`.
- Record your own baseline with your real model:
  `uv run python -m app.evals.rag_eval --save-baseline ../evals/baseline.json`
  and later `--baseline ../evals/baseline.json` fails (exit code 1) if any metric drops by
  more than `--max-drop` (default 0.05).

## CI gate

`.github/workflows/ci.yml` runs the eval with the offline fake model against
`baseline-fake.json` on every pull request. The fake model can't judge answer quality, but
the gate still catches broken retrieval, missing citations and broken "I don't know"
handling. Update the baseline on purpose with `--save-baseline` when a change is expected
to move the numbers.

## promptfoo

`promptfooconfig.yaml` runs readable assertion-style tests against the running API:

```bash
npx promptfoo@latest eval -c evals/promptfooconfig.yaml
npx promptfoo@latest view
```
