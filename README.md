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
| 3. Agents and MCP | 6–8 | Tool-calling agent, LangGraph, MCP ticket server | ✅ |
| 4. Production | 9–10 | Tracing, cost tracking, caching, guardrails, red-team evals | ✅ |
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

CI runs the same tests, three eval regression gates (RAG quality, agent task success and the
red-team defence rate) and a frontend build on every pull request. See [evals/README.md](evals/README.md) for running evals with your real model.

## API

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/api/health` | Shows which provider and models are active |
| POST | `/api/chat` | `{message, conversation_id?, mode: "agent" \| "rag" \| "chat"}` → Server-Sent Events stream |
| POST | `/api/answer` | `{question}` → `{answer, sources}` without memory (used by evals) |
| GET | `/api/search?q=...&mode=hybrid` | Shows exactly which chunks a question retrieves |
| POST | `/api/classify` | `{message}` → `{category, priority, sentiment}` as validated JSON |
| GET | `/api/conversations/{id}` | Full conversation, including the rolling summary |
| GET | `/api/tools` | Tools the agent can use, with their risk level |
| GET | `/api/tickets` | Support tickets (`?customer_email=&status=`) |
| GET / POST | `/api/actions/{id}` | See or decide (`{approve: true}`) a refund or plan change waiting for approval |
| GET | `/api/metrics?hours=24` | Calls, tokens, cost, p50/p95 latency, cache hits and guardrail events |
| DELETE | `/api/cache` | Empties the semantic answer cache |

## Demo accounts

The agent works with mock customers (reset with `uv run python -m app.services.seed --reset`):

| Email | Situation |
| --- | --- |
| priya@example.com | Pro monthly, **charged twice** 3 days ago (INV-204518 and INV-204519) |
| rahul@example.com | Team annual, 5 seats |
| maria@example.com | Pro annual, charged 10 days ago (inside the 30-day refund window) |
| alex@example.com | Free plan |
| sam@example.com | Pro monthly with a failed payment |

Try in Agent mode: *"I was charged twice this month. My email is priya@example.com"*. The agent
checks the account, finds the duplicate and asks for approval. Nothing is refunded until you
click **Approve**.

## Use the ticket MCP server from other AI apps

```bash
cd backend
uv run python -m app.mcp_server                       # stdio
npx @modelcontextprotocol/inspector uv run python -m app.mcp_server   # explore it in a browser
```

To add it to an MCP client such as Claude Desktop, add this to the client's MCP config
(use the absolute path to your `backend` folder):

```json
{
  "mcpServers": {
    "cloudnotes-tickets": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/supportpilot/backend", "python", "-m", "app.mcp_server"]
    }
  }
}
```

## Production features (weeks 9–10)

- **Cost and latency.** Every LLM call is logged to the `llm_calls` table with its purpose,
  tokens, latency and cost. Open **Metrics** in the UI (http://localhost:3000/metrics) or
  `GET /api/metrics`. Set `PRICE_INPUT_PER_M` / `PRICE_OUTPUT_PER_M` to a hosted model's
  price to see what the same traffic would cost.
- **Semantic cache.** The first question of a chat is embedded; a near-identical earlier
  question (`CACHE_SIMILARITY`) reuses its answer without calling the model.
- **Guardrails.** Card numbers and secrets are redacted before they reach the model or the
  database, obvious prompt injections are blocked, replies that leak the system prompt are
  replaced, and the agent may only touch accounts whose email the customer typed. Blocked
  events show on the metrics page.
- **Red-team eval.** `uv run python -m app.evals.redteam_eval --tag mine` runs 20 attacks and
  reports the defended rate.

### Optional: Langfuse tracing (free, self-hosted)

Langfuse shows every prompt, reply, token count and latency as a trace. Run it locally with
its own docker compose file (see langfuse.com/self-hosting), on port 3001 so it doesn't
clash with the frontend, create a project, then put its keys in `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3001
```

and install the extra: `uv sync --extra dev --extra tracing`. Prompts are redacted before
they are sent.
