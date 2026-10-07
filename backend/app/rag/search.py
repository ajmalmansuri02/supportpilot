"""Retrieval: vector, keyword and hybrid search (weeks 3-4).

- Vector search finds chunks with similar *meaning* ("get my money back" ~ "refund"),
  but can miss exact terms like invoice numbers or "2FA".
- Keyword search (Postgres full-text) nails exact terms but misses paraphrases.
- Hybrid runs both and merges the rankings with Reciprocal Rank Fusion (RRF):
  score = sum over lists of 1 / (60 + rank). RRF only uses ranks, so it doesn't matter
  that cosine similarity and ts_rank scores are on completely different scales.
- An optional cross-encoder reranker then re-reads each (question, chunk) pair together,
  which is slower but more accurate than comparing two separate embeddings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app import db
from app.config import Settings, get_settings
from app.llm import LLMClient

RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    source: str
    title: str
    heading: str
    content: str
    score: float = 0.0
    similarity: float | None = None  # cosine similarity, when vector search found it
    keyword_match: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


_SELECT = "SELECT c.id, d.source, d.title, c.heading, c.content"


async def vector_search(query_vector: list[float], limit: int) -> list[Hit]:
    async with db.pool().connection() as conn:
        rows = await (
            await conn.execute(
                f"{_SELECT}, 1 - (c.embedding <=> %s::vector) AS similarity"
                " FROM chunks c JOIN documents d ON d.id = c.document_id"
                " ORDER BY c.embedding <=> %s::vector LIMIT %s",
                (db.to_vector(query_vector), db.to_vector(query_vector), limit),
            )
        ).fetchall()
    return [
        Hit(r["id"], r["source"], r["title"], r["heading"], r["content"],
            score=float(r["similarity"]), similarity=float(r["similarity"]))
        for r in rows
    ]


async def keyword_search(query: str, limit: int) -> list[Hit]:
    # websearch_to_tsquery ANDs every word, which is too strict for chatty questions,
    # so we OR the words together and let ts_rank_cd reward chunks that match more.
    async with db.pool().connection() as conn:
        rows = await (
            await conn.execute(
                f"""WITH q AS (
                        SELECT to_tsquery('english', string_agg(quote_literal(lexeme), ' | ')) AS query
                        FROM unnest(to_tsvector('english', %s))
                    )
                    {_SELECT}, ts_rank_cd(c.tsv, q.query) AS rank
                    FROM chunks c JOIN documents d ON d.id = c.document_id, q
                    WHERE q.query IS NOT NULL AND c.tsv @@ q.query
                    ORDER BY rank DESC LIMIT %s""",
                (query, limit),
            )
        ).fetchall()
    return [
        Hit(r["id"], r["source"], r["title"], r["heading"], r["content"],
            score=float(r["rank"]), keyword_match=True)
        for r in rows
    ]


def reciprocal_rank_fusion(*rankings: list[Hit]) -> list[Hit]:
    merged: dict[int, Hit] = {}
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            seen = merged.setdefault(hit.chunk_id, hit)
            seen.keyword_match = seen.keyword_match or hit.keyword_match
            if hit.similarity is not None:
                seen.similarity = hit.similarity
    fused = sorted(merged.values(), key=lambda h: scores[h.chunk_id], reverse=True)
    for hit in fused:
        hit.score = scores[hit.chunk_id]
    return fused


async def search(
    llm: LLMClient,
    query: str,
    *,
    mode: str | None = None,
    top_k: int | None = None,
    reranker: str | None = None,
    settings: Settings | None = None,
    query_vector: list[float] | None = None,
) -> list[Hit]:
    s = settings or get_settings()
    mode = mode or s.retrieval_mode
    top_k = top_k or s.top_k
    reranker = reranker or s.reranker

    vector_hits: list[Hit] = []
    keyword_hits: list[Hit] = []
    if mode in ("vector", "hybrid"):
        if query_vector is None:
            [query_vector] = await llm.embed([query])
        vector_hits = await vector_search(query_vector, s.candidates)
    if mode in ("keyword", "hybrid"):
        keyword_hits = await keyword_search(query, s.candidates)

    if mode == "hybrid":
        hits = reciprocal_rank_fusion(vector_hits, keyword_hits)
    else:
        hits = vector_hits or keyword_hits

    if reranker == "cross_encoder" and hits:
        from app.rag.rerank import rerank

        hits = rerank(query, hits, s.reranker_model)
    return hits[:top_k]


def is_relevant(hits: list[Hit], min_similarity: float) -> bool:
    """Cheap pre-check: did retrieval find anything worth answering from?

    Uses the best cosine similarity when vector search ran. Keyword-only search has no
    comparable score, so any hit counts. The prompt still tells the model to say
    "I don't know" when the excerpts don't contain the answer.
    """
    similarities = [h.similarity for h in hits if h.similarity is not None]
    if similarities:
        return max(similarities) >= min_similarity
    return bool(hits)


def format_context(hits: list[Hit]) -> str:
    """Number each excerpt so the model can cite it as [1], [2], ..."""
    return "\n\n".join(
        f"[{i}] {h.heading} ({h.source})\n{h.content}" for i, h in enumerate(hits, start=1)
    )
