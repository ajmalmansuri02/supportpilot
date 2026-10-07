"""Cross-encoder reranking (week 4).

A bi-encoder (the embedding model) turns the question and each chunk into vectors
separately. A cross-encoder reads the question and the chunk *together* and outputs a
relevance score, so it is more accurate but too slow to run over every chunk. That is why
we only rerank the top candidates from the fast search.

Install with:  uv sync --extra rerank   (first use downloads the ~90 MB model)
"""

from __future__ import annotations

from functools import lru_cache

from app.rag.search import Hit


@lru_cache
def _model(name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("RERANKER=cross_encoder needs `uv sync --extra rerank`") from exc
    return CrossEncoder(name)


def rerank(query: str, hits: list[Hit], model_name: str) -> list[Hit]:
    scores = _model(model_name).predict([(query, f"{h.heading}\n{h.content}") for h in hits])
    for hit, score in zip(hits, scores):
        hit.score = float(score)
    return sorted(hits, key=lambda h: h.score, reverse=True)
