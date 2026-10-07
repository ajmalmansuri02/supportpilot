"""SupportPilot API.

Run locally with:  uv run uvicorn app.main:app --reload --port 8000
Interactive docs:  http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import chat, db, memory
from app.classify import TicketClassification, classify_ticket
from app.config import get_settings
from app.llm import get_llm
from app.rag.ingest import DEFAULT_DOCS, ingest_folder
from app.rag.search import search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("supportpilot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.open_pool()
    await db.init_schema()
    if get_settings().auto_ingest:
        await _ingest_if_empty()
    yield
    await db.close_pool()


async def _ingest_if_empty() -> None:
    """First run convenience: load data/docs if the knowledge base is empty."""
    async with db.pool().connection() as conn:
        row = await (await conn.execute("SELECT count(*) AS n FROM chunks")).fetchone()
    if row["n"] == 0 and DEFAULT_DOCS.exists():
        log.info("Knowledge base is empty, ingesting %s", DEFAULT_DOCS)
        try:
            log.info("Ingested: %s", await ingest_folder(DEFAULT_DOCS, get_llm()))
        except Exception:
            log.exception("Auto-ingest failed. Is the embedding model running? "
                          "Run `uv run python -m app.rag.ingest` once it is.")


app = FastAPI(title="SupportPilot", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    mode: Literal["chat", "rag"] = "rag"


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    retrieval_mode: Literal["vector", "keyword", "hybrid"] | None = None
    reranker: Literal["none", "cross_encoder"] | None = None


class ClassifyRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


def _is_uuid(value: str | None) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@app.get("/api/health")
async def health():
    s = get_settings()
    return {"status": "ok", "provider": s.llm_provider, "chat_model": s.chat_model,
            "embed_provider": s.embed_provider, "embed_model": s.embed_model}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Stream the answer as Server-Sent Events (one JSON object per `data:` line)."""
    llm = get_llm()
    conversation_id = req.conversation_id
    if not _is_uuid(conversation_id) or not await memory.conversation_exists(conversation_id):
        conversation_id = await memory.create_conversation()

    handlers = {"chat": chat.simple_chat, "rag": chat.rag_chat}

    async def events() -> AsyncIterator[str]:
        yield sse({"type": "meta", "conversation_id": conversation_id, "mode": req.mode})
        try:
            async for event in handlers[req.mode](llm, conversation_id, req.message):
                yield sse(event)
        except Exception as exc:  # show the error in the UI instead of a dead stream
            log.exception("chat failed")
            yield sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/classify", response_model=TicketClassification)
async def classify_endpoint(req: ClassifyRequest):
    return await classify_ticket(get_llm(), req.message)


@app.get("/api/conversations/{conversation_id}")
async def conversation_endpoint(conversation_id: str):
    conv = await memory.get_conversation(conversation_id) if _is_uuid(conversation_id) else None
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    return conv


@app.get("/api/search")
async def search_endpoint(q: str, mode: Literal["vector", "keyword", "hybrid"] | None = None,
                          k: int = 5):
    """Debug retrieval: see exactly which chunks a question pulls back."""
    hits = await search(get_llm(), q, mode=mode, top_k=k)
    return [h.to_dict() for h in hits]


@app.post("/api/answer")
async def answer_endpoint(req: AnswerRequest):
    """Non-streaming RAG answer without memory, for evals and scripts (e.g. promptfoo)."""
    return await chat.answer_question(
        get_llm(), req.question, mode=req.retrieval_mode, reranker=req.reranker
    )
