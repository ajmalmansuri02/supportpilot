# What each part teaches

Read the code in this order. Each week lists the concepts, the files that implement them, and
a few things to try so you can explain the choices in an interview.

## Week 1: Your first LLM backend

**Concepts:** tokens, context window, temperature, chat roles (system / user / assistant),
streaming, the OpenAI-compatible API format.

| File | What to look at |
| --- | --- |
| `backend/app/llm.py` | One `AsyncOpenAI` client talks to Ollama, Gemini or any compatible endpoint by changing only `base_url`, key and model. `stream()` yields tokens as they arrive. Every call is timed and its token usage recorded. |
| `backend/app/config.py` | All settings come from `.env`. `LLM_PROVIDER` is the switch. |
| `backend/app/main.py` | `/api/chat` returns Server-Sent Events: one JSON event per `data:` line. |
| `frontend/lib/api.ts` | Reads the event stream with `fetch` + `ReadableStream` and parses each event. |
| `backend/app/fake_llm.py` | A deterministic offline model so tests and CI never need a real LLM. |

**Try:** switch `LLM_PROVIDER` between `ollama` and `gemini` and compare speed and quality.
Change `TEMPERATURE` to 1.0 and ask the same question three times.

## Week 2: Prompt and context engineering

**Concepts:** system prompts, few-shot examples, structured output with JSON schema, validation
and retries, conversation memory, summarising to fit the context window.

| File | What to look at |
| --- | --- |
| `backend/app/prompts/*.md` | Prompts are versioned files, not strings in code. The version is saved with every answer. |
| `backend/app/classify.py` | Pydantic model → JSON schema → `response_format`. Invalid output gets one retry with the error, then a safe fallback. |
| `backend/app/prompts/classify.md` | Few-shot examples steer the labels. |
| `backend/app/memory.py` | History lives in Postgres. Past `MEMORY_MAX_MESSAGES`, older messages are summarised by the small model and only recent ones are sent word for word. |

**Try:** `curl -X POST localhost:8000/api/classify -H 'Content-Type: application/json' -d '{"message":"I was charged twice!!"}'`.
Remove the examples from `classify.md` and see whether the labels get worse.
Set `MEMORY_MAX_MESSAGES=4`, chat for a while, then open `/api/conversations/{id}` to see the summary.
