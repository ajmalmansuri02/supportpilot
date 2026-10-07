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

## Week 3: Basic RAG

**Concepts:** embeddings, cosine similarity, chunking, vector indexes (HNSW), grounding answers
in retrieved text, citations.

| File | What to look at |
| --- | --- |
| `data/docs/*.md` | The knowledge base: 15 help-centre pages for the made-up product. |
| `backend/app/rag/chunking.py` | Heading-aware chunking with paragraph overlap. The heading path is kept with each chunk. |
| `backend/app/rag/ingest.py` | Chunk → embed → store in pgvector. Unchanged files are skipped by hash. |
| `backend/app/schema.sql` | The `chunks` table, its `vector(768)` column and the HNSW index. |
| `backend/app/prompts/rag.md` | Answer only from the excerpts, cite them, otherwise say "I don't know". Excerpts are data, not instructions. |
| `backend/app/llm.py` (`embed`) | nomic-embed-text needs `search_query:` / `search_document:` prefixes. |

**Try:** `curl "localhost:8000/api/search?q=get my money back&mode=vector"`. Notice it finds the
refund policy even though the word "refund" is not in the question.

## Week 4: Better retrieval

**Concepts:** keyword (full-text) search, hybrid search, Reciprocal Rank Fusion, cross-encoder
reranking, query rewriting for follow-up questions, relevance thresholds.

| File | What to look at |
| --- | --- |
| `backend/app/rag/search.py` | Vector, keyword and hybrid search, plus RRF fusion in about 15 lines. |
| `backend/app/rag/rerank.py` | A cross-encoder rescores the top candidates (`RERANKER=cross_encoder`). |
| `backend/app/chat.py` (`rewrite_query`) | "How much is it?" becomes "How much does the Team plan cost?" before searching. |
| `backend/app/chat.py` (`retrieve`) | If nothing is similar enough, the bot says "I don't know" without calling the model. |

**Try:** search `2FA` with `mode=vector` and `mode=keyword`. Exact terms are where keyword
search wins, which is why hybrid exists.

## Week 5: Evals

**Concepts:** golden datasets, retrieval metrics (hit rate, MRR), LLM-as-judge (correctness,
faithfulness), abstention, regression gates in CI.

| File | What to look at |
| --- | --- |
| `evals/datasets/rag_golden.jsonl` | 50 questions with reference answers and sources, plus 5 unanswerable ones. |
| `backend/app/evals/rag_eval.py` | Runs the dataset, scores it, saves results, compares runs, fails on regressions. |
| `backend/app/prompts/judge_*.md` | The grading rubrics for the LLM judge. |
| `evals/promptfooconfig.yaml` | The same idea in promptfoo, a popular eval tool. |
| `.github/workflows/ci.yml` | The eval gate that runs on every pull request. |

**Try (this is the week 5 deliverable):** run the eval with `--retrieval vector`,
`--retrieval keyword` and `--retrieval hybrid`, then `--compare` the three result files and
put the table in this README. Then tune `RAG_MIN_SIMILARITY` until `abstain_ok` is high and
`false_abstain` is low.
