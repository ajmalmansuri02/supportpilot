# SupportPilot

An AI customer-support agent for **CloudNotes**, a made-up note-taking app. It answers questions
from the product docs, looks up customer accounts, opens tickets and hands hard cases to a human.

It is a learning and portfolio project covering the 10 most in-demand "AI engineer" skills:
LLM APIs, prompt and context engineering, RAG, evals, agents and tool calling, LangGraph, MCP,
production concerns (cost, latency, tracing, guardrails), fine-tuning and a cloud AI platform.
Everything runs **locally and for free**.

| Phase | Weeks | What it adds | Status |
| --- | --- | --- | --- |
| 1. LLM APIs and prompting | 1–2 | Streaming chat, provider switch, structured output, memory | ✅ |
| 2. RAG and evals | 3–5 | Doc search with pgvector, hybrid search, reranking, eval suite | ✅ |
| 3. Agents and MCP | 6–8 | Tool-calling agent, LangGraph, MCP ticket server | ⏳ |
| 4. Production | 9–10 | Tracing, cost tracking, caching, guardrails, red-team evals | ⏳ |
| 5. Fine-tuning and cloud | 11–12 | Fine-tuned router, deploy to a managed AI platform | ⏳ |

See [LEARNING.md](LEARNING.md) for what each part teaches and where to find it in the code.

## Architecture

```
Next.js chat UI  ──SSE──▶  FastAPI backend  ──▶  LLM (Ollama / Gemini / any OpenAI-compatible)
 (frontend/)               (backend/app/)    ──▶  Postgres + pgvector (memory, docs, tickets)
                                             ──▶  data/docs/*.md  (the CloudNotes help centre)
```

The knowledge base is 15 Markdown help-centre pages in `data/docs/`. On first start the
backend loads them automatically. After editing them, run
`uv run python -m app.rag.ingest` to re-index (only changed files are re-embedded).

## Prerequisites

| Tool | Why | Get it |
| --- | --- | --- |
| Docker Desktop | Runs Postgres with pgvector | docker.com |
| Python 3.11+ and [uv](https://docs.astral.sh/uv/) | Backend | `pip install uv` or the uv installer |
| Node 20+ | Frontend | nodejs.org |
| Ollama | Free local models | ollama.com |
| Gemini API key (optional) | Stronger free model | aistudio.google.com/apikey |

Hardware: 16 GB RAM runs the 7B model comfortably. On 8 GB, use `qwen2.5:3b` for both
`CHAT_MODEL` and `FAST_MODEL`.

## Run it locally

```bash
# 1. Models (one time)
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 2. Settings
cp .env.example .env          # Windows PowerShell: copy .env.example .env

# 3. Database
docker compose up -d db

# 4. Backend  ->  http://localhost:8000/docs
cd backend
uv sync --extra dev
uv run uvicorn app.main:app --reload --port 8000

# 5. Frontend (new terminal)  ->  http://localhost:3000
cd frontend
npm install
npm run dev
```

No Ollama yet? Set `LLM_PROVIDER=fake` and `EMBED_PROVIDER=fake` in `.env` to click around with
the offline test model. Its answers are canned, but every screen works.

Want a stronger model for free? Set `LLM_PROVIDER=gemini`, add `GEMINI_API_KEY`, and set
`CHAT_MODEL` / `FAST_MODEL` to current Gemini model names from AI Studio.

Prefer everything in Docker? `docker compose --profile app up --build` (Ollama still runs on
your machine).

## Tests

```bash
docker compose up -d db
cd backend && uv run pytest -q     # uses the offline fake model, no Ollama needed
```

CI runs the same tests, the RAG eval regression gate and a frontend build on every pull
request. See [evals/README.md](evals/README.md) for running evals with your real model.

## API

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/api/health` | Shows which provider and models are active |
| POST | `/api/chat` | `{message, conversation_id?, mode: "rag" \| "chat"}` → Server-Sent Events stream |
| POST | `/api/answer` | `{question}` → `{answer, sources}` without memory (used by evals) |
| GET | `/api/search?q=...&mode=hybrid` | Shows exactly which chunks a question retrieves |
| POST | `/api/classify` | `{message}` → `{category, priority, sentiment}` as validated JSON |
| GET | `/api/conversations/{id}` | Full conversation, including the rolling summary |
