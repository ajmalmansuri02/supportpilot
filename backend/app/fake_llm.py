"""A deterministic, offline stand-in for a real LLM (LLM_PROVIDER=fake).

It is not smart. It exists so tests and CI can exercise the whole pipeline
(streaming, JSON output, RAG, tool calling, evals) without downloading a model
or spending money. Never use it to judge answer quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

_WORD = re.compile(r"[a-z0-9@._-]+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_STOP = {
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "my", "i",
    "how", "do", "can", "what", "it", "with", "be", "you", "your", "me", "does", "this",
}


def _tokens(text: str) -> list[str]:
    words = [w.strip("._-") for w in _WORD.findall(text.lower())]
    return [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words if w and w not in _STOP]


def _overlap(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    return len(ta & tb) / len(ta) if ta else 0.0


def _last(messages: list[dict[str, Any]], role: str) -> str:
    for m in reversed(messages):
        if m.get("role") == role and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def _system(messages: list[dict[str, Any]]) -> str:
    return "\n".join(m["content"] for m in messages if m.get("role") == "system")


INJECTION_HINTS = ("ignore previous", "ignore all", "system prompt", "you are now", "jailbreak")


class FakeLLM:
    def __init__(self, dim: int = 768):
        self.dim = dim

    # -- embeddings: hashed bag of words, good enough for keyword-ish retrieval --
    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    # -- chat ----------------------------------------------------------------------
    def stream(self, messages: list[dict[str, Any]]):
        text = self.chat(messages).content
        for i, word in enumerate(text.split(" ")):
            yield word if i == 0 else " " + word

    def chat(self, messages, tools=None, json_schema=None):
        from app.llm import ChatResult, ToolCall

        if json_schema:
            return ChatResult(content=json.dumps(self._json(messages, json_schema)))
        if tools:
            names = {t["function"]["name"] for t in tools}
            planned = self._plan(messages, names)
            done = [m for m in messages if m.get("role") == "tool"]
            if len(done) < len(planned):
                name, args = planned[len(done)]
                call = ToolCall(id=f"call_{len(done)}", name=name, arguments=args)
                return ChatResult(content="", tool_calls=[call])
            return ChatResult(content=self._final_from_tools(messages))
        return ChatResult(content=self._answer(messages))

    def _plan(self, messages, names: set[str]) -> list[tuple[str, dict]]:
        user = _last(messages, "user")
        low = user.lower()
        if any(h in low for h in INJECTION_HINTS):
            return []
        plan: list[tuple[str, dict]] = []
        email = _EMAIL.search(user)
        if re.search(r"\b(human|person|manager|escalat)", low) and "escalate_to_human" in names:
            return [("escalate_to_human", {"reason": user[:200]})]
        if email and "get_account" in names:
            plan.append(("get_account", {"email": email.group(0)}))
        if re.search(r"refund|charged twice|double charge|billing|invoice|ticket", low) and (
            "create_ticket" in names
        ):
            plan.append((
                "create_ticket",
                {
                    "subject": user[:80],
                    "description": user,
                    "category": "billing" if "refund" in low or "charge" in low else "other",
                    "priority": "high" if "twice" in low else "medium",
                    "customer_email": email.group(0) if email else None,
                },
            ))
        if not plan and "search_docs" in names:
            plan.append(("search_docs", {"query": user}))
        return plan

    def _final_from_tools(self, messages) -> str:
        user = _last(messages, "user").lower()
        if any(h in user for h in INJECTION_HINTS):
            return "I can only help with CloudNotes support questions."
        results = [m["content"] for m in messages if m.get("role") == "tool"]
        for raw in results:
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict) and data.get("results"):
                top = data["results"][0]
                return f"{_first_sentences(top.get('content', ''))} [1]"
            if isinstance(data, dict) and data.get("ticket_id"):
                return f"I've opened ticket #{data['ticket_id']} and our team will follow up."
            if isinstance(data, dict) and data.get("escalated"):
                return "I've passed this to a human agent who will reply shortly."
        return "I couldn't find enough information to answer that."

    def _answer(self, messages) -> str:
        system = _system(messages)
        user = _last(messages, "user")
        if "summar" in system.lower():
            return "Summary: " + " ".join(user.split()[:60])
        chunks = re.findall(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\n</context>)", system + "\n", re.DOTALL)
        if "<context>" in system:
            best = max(chunks, key=lambda c: _overlap(user, c[1]), default=None)
            if not best or _overlap(user, best[1]) < 0.2:
                return "I don't know based on the CloudNotes documentation."
            return f"{_first_sentences(best[1])} [{best[0]}]"
        return f"Hello! I'm SupportPilot running on the offline test model. You said: {user}"

    def _json(self, messages, schema: dict) -> dict:
        props = schema.get("properties", {})
        text = _last(messages, "user")
        low = text.lower()
        if "category" in props:
            category = "other"
            for cat, words in {
                "billing": ("charge", "refund", "invoice", "bill", "payment", "price"),
                "technical": ("error", "bug", "crash", "sync", "not working", "broken", "slow"),
                "account": ("password", "login", "log in", "sign in", "email", "delete my"),
                "feature_request": ("feature", "would be nice", "please add", "wish"),
            }.items():
                if any(w in low for w in words):
                    category = cat
                    break
            urgent = any(w in low for w in ("urgent", "asap", "twice", "down", "cannot access"))
            angry = any(w in low for w in ("angry", "terrible", "worst", "ridiculous", "!!"))
            return {
                "category": category,
                "priority": "high" if urgent else "medium",
                "sentiment": "negative" if angry or urgent else "neutral",
            }
        if "score" in props:  # LLM-as-judge for evals: score by word overlap
            score = round(min(1.0, _overlap(text.split("ANSWER:")[-1], text) + 0.1), 2)
            return {"score": score, "reason": "offline judge: word overlap"}
        return {k: _default(v) for k, v in props.items()}


def _default(prop: dict) -> Any:
    if "enum" in prop:
        return prop["enum"][0]
    return {"string": "", "number": 0, "integer": 0, "boolean": False, "array": []}.get(
        prop.get("type"), None
    )


def _first_sentences(text: str, n: int = 2) -> str:
    text = re.sub(r"^#+ .*\n", "", text.strip()).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n]).strip()
