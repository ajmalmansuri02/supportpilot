"""Tests run against the offline fake model and a real Postgres with pgvector.

Start the database first:  docker compose up -d db
"""

import os

os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("EMBED_PROVIDER", "fake")
os.environ.setdefault(
    "DATABASE_URL", "postgresql://supportpilot:supportpilot@localhost:5432/supportpilot"
)

import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings

# Tests must never pick up a developer's real provider from .env.
os.environ["LLM_PROVIDER"] = "fake"
os.environ["EMBED_PROVIDER"] = "fake"
# The fake hashed embeddings give lower similarities than a real model.
os.environ["RAG_MIN_SIMILARITY"] = "0.2"
get_settings.cache_clear()


@pytest.fixture()
def client():
    from app import llm
    from app.main import app

    llm.set_llm(None)
    with TestClient(app) as c:
        yield c


def read_sse(response) -> list[dict]:
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events
