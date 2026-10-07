# Fine-tuning the ticket router (week 11)

SupportPilot labels every message with a category, priority and sentiment before the agent
runs (the `classify` node in `backend/app/agent/graph.py`). Here you replace the prompted
model doing that job with a small model fine-tuned for it, then measure whether it was worth it.

Everything is free: data comes from your local or Gemini model, training runs on Colab's free GPU,
and the result runs in Ollama.

## 1. Generate training data (on your machine)

```bash
cd backend
# The "teacher" is CHAT_MODEL. Gemini (free tier) writes more varied tickets than a local 7B.
uv run python -m app.finetune.generate --count 500 --verify
```

This writes `finetune/data/train.jsonl` and `test.jsonl`. `--verify` asks the teacher to
label each ticket again with the full prompt and drops the ones where it disagrees, which
removes mislabelled examples. Open a few lines and read them: data quality decides
everything here.

No model yet? `--source templates` builds an offline dataset to test the pipeline. It is too
repetitive to train anything useful.

## 2. Train on Colab (free T4 GPU)

Open [`SupportPilot_finetune.ipynb`](SupportPilot_finetune.ipynb) in Google Colab
(File → Upload notebook), select a T4 GPU, upload the two `.jsonl` files and run all cells.
It measures the base model, trains LoRA adapters for 3 epochs, measures again and downloads a
`q4_k_m` GGUF file (about 1 GB).

## 3. Use the model

```bash
# put the downloaded file here as finetune/supportpilot-router.gguf
cd finetune
ollama create supportpilot-router -f Modelfile
ollama run supportpilot-router "I was charged twice!"
```

## 4. Compare it with the prompted model

```bash
cd backend
uv run python -m app.evals.classify_eval --model qwen2.5:3b --tag prompted-3b
uv run python -m app.evals.classify_eval --model qwen2.5:7b --tag prompted-7b
uv run python -m app.evals.classify_eval --model supportpilot-router --prompt classify_ft --tag finetuned
uv run python -m app.evals.classify_eval --compare ../evals/results/*-classify-*.json
```

The eval uses 30 hand-written tickets (`evals/datasets/classify_golden.jsonl`) that the
generator never saw. Put the comparison table in your README: accuracy, latency and prompt
tokens for each. A good outcome is the fine-tuned 1.5B model matching the prompted 7B with
a prompt about four times shorter.

If it wins, make it the router in `.env`:

```bash
CLASSIFY_MODEL=supportpilot-router
CLASSIFY_PROMPT=classify_ft
```

## When not to fine-tune

Fine-tuning teaches a format or a skill, not facts. Prices and policies change, so they stay
in RAG. Try a better prompt and a bigger model first. Fine-tune when a narrow, repeated task
needs to be cheaper, faster or more consistent, and only when you have an eval to prove it.
