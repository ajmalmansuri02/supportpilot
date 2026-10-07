"""Postgres access: one async connection pool shared by the whole app."""

from __future__ import annotations

from pathlib import Path

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

SCHEMA = Path(__file__).with_name("schema.sql")

_pool: AsyncConnectionPool | None = None


async def open_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row, "autocommit": True},
            open=False,
        )
        await _pool.open(wait=True, timeout=30)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("Database pool not opened; call open_pool() first")
    return _pool


async def init_schema() -> None:
    async with pool().connection() as conn:
        await conn.execute(SCHEMA.read_text())


def to_vector(values: list[float]) -> str:
    """Format a Python list as a pgvector literal, e.g. '[0.1,0.2]'."""
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"
