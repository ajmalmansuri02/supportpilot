"""Load Markdown docs into Postgres + pgvector (week 3).

    uv run python -m app.rag.ingest ../data/docs

Re-running is safe: unchanged files are skipped (by content hash), changed files are
re-chunked and re-embedded, and files that were deleted from the folder are removed.
Use --force after changing the chunking code.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
from pathlib import Path

from app import db
from app.config import REPO_ROOT, get_settings
from app.llm import LLMClient, get_llm
from app.rag.chunking import chunk_markdown, title_of

log = logging.getLogger(__name__)

DEFAULT_DOCS = REPO_ROOT / "data" / "docs"


async def ingest_folder(folder: Path, llm: LLMClient, *, force: bool = False) -> dict[str, int]:
    settings = get_settings()
    stats = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "chunks": 0}
    files = sorted(folder.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files found in {folder}")

    async with db.pool().connection() as conn:
        existing = {
            r["source"]: r["content_hash"]
            for r in await (await conn.execute("SELECT source, content_hash FROM documents")).fetchall()
        }

    for path in files:
        text = path.read_text(encoding="utf-8")
        # The hash covers the embedding settings too, so changing the model re-embeds.
        digest = hashlib.sha256(
            f"{settings.embed_model}|{settings.embed_document_prefix}|{text}".encode()
        ).hexdigest()
        if not force and existing.get(path.name) == digest:
            stats["unchanged"] += 1
            continue
        chunks = chunk_markdown(text)
        vectors = await llm.embed([c.text_for_embedding for c in chunks], kind="document")
        async with db.pool().connection() as conn, conn.transaction():
            await conn.execute("DELETE FROM documents WHERE source = %s", (path.name,))
            doc = await (
                await conn.execute(
                    "INSERT INTO documents (source, title, content_hash) VALUES (%s, %s, %s) RETURNING id",
                    (path.name, title_of(text, path.stem), digest),
                )
            ).fetchone()
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                await conn.execute(
                    "INSERT INTO chunks (document_id, chunk_index, heading, content, embedding)"
                    " VALUES (%s, %s, %s, %s, %s::vector)",
                    (doc["id"], i, chunk.heading, chunk.content, db.to_vector(vector)),
                )
        stats["updated" if path.name in existing else "added"] += 1
        stats["chunks"] += len(chunks)
        log.info("ingested %s (%d chunks)", path.name, len(chunks))

    current = {p.name for p in files}
    async with db.pool().connection() as conn:
        for source in set(existing) - current:
            await conn.execute("DELETE FROM documents WHERE source = %s", (source,))
            stats["removed"] += 1
    return stats


async def _main(folder: Path, force: bool) -> None:
    await db.open_pool()
    try:
        await db.init_schema()
        stats = await ingest_folder(folder, get_llm(), force=force)
        print(f"Done: {stats}")
    finally:
        await db.close_pool()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Ingest Markdown docs for RAG")
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--force", action="store_true", help="re-embed every file")
    args = parser.parse_args()
    asyncio.run(_main(args.folder, args.force))
