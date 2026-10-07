"""Ticket classification with structured output (week 2).

Instead of parsing free text, we give the model a JSON schema and validate the answer
with Pydantic. If the model returns something invalid we retry once, then fall back to a
safe default, so the rest of the app can always trust the shape of the result.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.llm import LLMClient
from app.prompts import load_prompt

log = logging.getLogger(__name__)


class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "account", "feature_request", "other"]
    priority: Literal["low", "medium", "high"]
    sentiment: Literal["positive", "neutral", "negative"]


SCHEMA = TicketClassification.model_json_schema()
FALLBACK = TicketClassification(category="other", priority="medium", sentiment="neutral")


async def classify_ticket(llm: LLMClient, message: str, *, model: str | None = None,
                          prompt: str | None = None,
                          conversation_id: str | None = None) -> TicketClassification:
    settings = get_settings()
    model = model or settings.classify_model or settings.fast_model
    messages = [
        {"role": "system", "content": load_prompt(prompt or settings.classify_prompt).text},
        {"role": "user", "content": message},
    ]
    for attempt in range(2):
        result = await llm.chat(
            messages,
            model=model,
            json_schema=SCHEMA,
            temperature=0,
            purpose="classify",
            conversation_id=conversation_id,
        )
        try:
            return TicketClassification.model_validate(json.loads(_strip_fences(result.content)))
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning("classification attempt %d invalid: %s", attempt + 1, exc)
            messages.append({"role": "assistant", "content": result.content})
            messages.append({
                "role": "user",
                "content": f"That was not valid. Error: {exc}. Return only the JSON object.",
            })
    return FALLBACK


def _strip_fences(text: str) -> str:
    """Some models wrap JSON in ```json fences even when asked not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()
