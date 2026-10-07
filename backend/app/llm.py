"""One small LLM client for every provider.

Ollama, Gemini (AI Studio) and most hosted platforms all speak the OpenAI chat API,
so a single `openai` SDK client covers them; only the base URL, key and model change.
That is the "provider switch" from week 1: set LLM_PROVIDER in .env.

Every call is timed and its token usage is recorded, which is what the cost and
latency tracking in week 9 builds on.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings, get_settings

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: int = 0


@dataclass
class CallRecord:
    """What we log for every LLM call (feeds cost and latency dashboards)."""

    purpose: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    conversation_id: str | None = None
    error: str | None = None


Listener = Callable[[CallRecord], None]


def estimate_tokens(text: str) -> int:
    """Rough fallback (~4 characters per token) for providers that omit usage."""
    return max(1, len(text) // 4)


def _connection(settings: Settings, provider: str) -> tuple[str, str]:
    if provider == "ollama":
        return settings.ollama_base_url, "ollama"  # Ollama ignores the key
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("LLM_PROVIDER=gemini needs GEMINI_API_KEY in .env")
        return settings.gemini_base_url, settings.gemini_api_key
    if provider == "openai_compatible":
        if not settings.openai_compatible_base_url:
            raise RuntimeError("Set OPENAI_COMPATIBLE_BASE_URL in .env")
        return settings.openai_compatible_base_url, settings.openai_compatible_api_key or "none"
    raise ValueError(f"Unknown provider {provider!r}")


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.provider = self.settings.llm_provider
        self.embed_provider = self.settings.embed_provider
        self._listeners: list[Listener] = []
        self._fake = None
        if "fake" in (self.provider, self.embed_provider):
            from app.fake_llm import FakeLLM

            self._fake = FakeLLM(self.settings.embed_dim)
        self._chat_client = self._make_client(self.provider)
        self._embed_client = self._make_client(self.embed_provider)

    def _make_client(self, provider: str) -> AsyncOpenAI | None:
        if provider == "fake":
            return None
        base_url, key = _connection(self.settings, provider)
        return AsyncOpenAI(
            base_url=base_url, api_key=key, timeout=self.settings.llm_timeout_seconds
        )

    # -- observability hooks ---------------------------------------------------
    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def _record(self, purpose: str, model: str, usage: Usage, started: float,
                conversation_id: str | None, error: str | None = None) -> int:
        latency_ms = int((time.perf_counter() - started) * 1000)
        cost = (
            usage.input_tokens * self.settings.price_input_per_m
            + usage.output_tokens * self.settings.price_output_per_m
        ) / 1_000_000
        record = CallRecord(
            purpose=purpose, provider=self.provider, model=model,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            latency_ms=latency_ms, cost_usd=cost, conversation_id=conversation_id, error=error,
        )
        for listener in self._listeners:
            try:
                listener(record)
            except Exception:  # a broken logger must never break a chat
                log.exception("LLM call listener failed")
        return latency_ms

    # -- chat ----------------------------------------------------------------
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        purpose: str = "chat",
        conversation_id: str | None = None,
    ) -> ChatResult:
        """One complete (non-streaming) call. Supports tool calling and JSON output."""
        model = model or self.settings.chat_model
        started = time.perf_counter()
        try:
            if self._chat_client is None:
                result = self._fake.chat(messages, tools=tools, json_schema=json_schema)
            else:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": self.settings.temperature if temperature is None else temperature,
                }
                if tools:
                    kwargs["tools"] = tools
                if json_schema:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "output", "schema": json_schema},
                    }
                resp = await self._chat_client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                calls = [
                    ToolCall(id=c.id, name=c.function.name, arguments=_parse_args(c.function.arguments))
                    for c in (msg.tool_calls or [])
                ]
                usage = Usage(
                    input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
                )
                result = ChatResult(content=msg.content or "", tool_calls=calls, usage=usage)
        except Exception as exc:
            self._record(purpose, model, Usage(), started, conversation_id, error=str(exc))
            raise
        if not result.usage.input_tokens:
            result.usage = Usage(
                input_tokens=estimate_tokens(json.dumps(messages)),
                output_tokens=estimate_tokens(result.content),
            )
        result.model = model
        result.latency_ms = self._record(purpose, model, result.usage, started, conversation_id)
        return result

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        purpose: str = "chat",
        conversation_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield the answer piece by piece as the model writes it."""
        model = model or self.settings.chat_model
        started = time.perf_counter()
        usage = Usage()
        parts: list[str] = []
        try:
            if self._chat_client is None:
                for piece in self._fake.stream(messages):
                    parts.append(piece)
                    yield piece
            else:
                stream = await self._chat_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.settings.temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for chunk in stream:
                    if chunk.usage:
                        usage = Usage(
                            input_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                        )
                    if chunk.choices and chunk.choices[0].delta.content:
                        piece = chunk.choices[0].delta.content
                        parts.append(piece)
                        yield piece
        except Exception as exc:
            self._record(purpose, model, usage, started, conversation_id, error=str(exc))
            raise
        if not usage.input_tokens:
            usage = Usage(
                input_tokens=estimate_tokens(json.dumps(messages)),
                output_tokens=estimate_tokens("".join(parts)),
            )
        self._record(purpose, model, usage, started, conversation_id)

    # -- embeddings ------------------------------------------------------------
    async def embed(self, texts: list[str], *, kind: str = "query",
                    batch_size: int = 32) -> list[list[float]]:
        """Embed texts. `kind` is "query" (a question) or "document" (a chunk to search)."""
        prefix = (self.settings.embed_query_prefix if kind == "query"
                  else self.settings.embed_document_prefix)
        texts = [prefix + t for t in texts]
        if self._embed_client is None:
            return [self._fake.embed(t) for t in texts]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            kwargs: dict[str, Any] = {"model": self.settings.embed_model, "input": batch}
            if self.embed_provider == "gemini":
                kwargs["dimensions"] = self.settings.embed_dim
            resp = await self._embed_client.embeddings.create(**kwargs)
            vectors.extend(d.embedding for d in resp.data)
        return vectors


def _parse_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        return {"_raw": raw}


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def set_llm(client: LLMClient | None) -> None:
    """Swap the shared client (used by tests)."""
    global _client
    _client = client
