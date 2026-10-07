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
            turn = _current_turn(messages)
            action = self._next_action(_last(messages, "user"), turn, names)
            if action:
                name, args = action
                call = ToolCall(id=f"call_{len(turn)}", name=name, arguments=args)
                return ChatResult(content="", tool_calls=[call])
            return ChatResult(content=self._final_from_tools(messages, turn))
        return ChatResult(content=self._answer(messages))

    def _next_action(self, user: str, turn: dict[str, Any], names: set[str]):
        """Pick the next tool call from simple keyword rules and earlier results."""
        low = user.lower()
        if any(h in low for h in INJECTION_HINTS):
            return None
        email = _EMAIL.search(user)
        billing = re.search(r"refund|charged twice|double charge|billing|invoice|charge", low)
        if re.search(r"\b(human|person|manager|escalat)", low) and "escalate_to_human" in names:
            return None if "escalate_to_human" in turn else ("escalate_to_human", {"reason": user[:200]})
        if email and "get_account" in names and "get_account" not in turn:
            return ("get_account", {"email": email.group(0)})
        if billing and email:
            account = turn.get("get_account") or {}
            dup = _duplicate_invoice(account.get("invoices", []))
            if dup and "issue_refund" in names and "issue_refund" not in turn:
                return ("issue_refund", {"invoice_id": dup, "customer_email": email.group(0),
                                         "reason": "Duplicate charge; refund policy allows a full refund"})
            if not dup and "create_ticket" in names and "create_ticket" not in turn:
                return ("create_ticket", {
                    "subject": user[:80], "description": user, "category": "billing",
                    "priority": "medium", "customer_email": email.group(0)})
            return None
        if not email and "search_docs" in names and "search_docs" not in turn:
            return ("search_docs", {"query": user})
        return None

    def _final_from_tools(self, messages, turn: dict[str, Any]) -> str:
        user = _last(messages, "user").lower()
        if any(h in user for h in INJECTION_HINTS):
            return "I can only help with CloudNotes support questions."
        if refund := turn.get("issue_refund"):
            if refund.get("status") == "awaiting_approval":
                return ("I found a duplicate charge on your account and asked a support agent "
                        "to approve the refund. You'll see the update here once it's reviewed.")
            return f"I couldn't request that refund: {refund.get('error')}"
        if (esc := turn.get("escalate_to_human")) and esc.get("escalated"):
            return "I've passed this to a human agent who will reply shortly."
        if (ticket := turn.get("create_ticket")) and ticket.get("ticket_id"):
            return f"I've opened ticket #{ticket['ticket_id']} and our team will follow up."
        if (docs := turn.get("search_docs")) and docs.get("results"):
            return f"{_first_sentences(docs['results'][0].get('content', ''))} [1]"
        if (account := turn.get("get_account")) and account.get("plan"):
            return f"Your account is on the {account['plan'].title()} plan."
        if "get_account" in turn:
            return "I couldn't find an account with that email. Could you check it?"
        if "@" not in user and re.search(r"charge|refund|invoice|bill", user):
            return "I can look into that. What's the email address on your CloudNotes account?"
        return "I couldn't find enough information to answer that."

    def _answer(self, messages) -> str:
        system = _system(messages)
        user = _last(messages, "user")
        if "standalone search query" in system:
            return user.rsplit("Latest message:", 1)[-1].strip()
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
        if "score" in props:  # LLM-as-judge for evals, approximated by word overlap
            answer = text.rsplit("ANSWER:", 1)[-1]
            if "REFERENCE:" in text:  # correctness: how much of the reference is in the answer
                reference = text.split("REFERENCE:", 1)[1].split("ANSWER:", 1)[0]
                score = _overlap(reference, answer)
            else:  # faithfulness: how much of the answer is in the excerpts
                excerpts = text.split("EXCERPTS:", 1)[-1].rsplit("ANSWER:", 1)[0]
                score = 1.0 if "don't know" in answer.lower() else _overlap(answer, excerpts)
            return {"score": round(score, 2), "reason": "offline judge: word overlap"}
        return {k: _default(v) for k, v in props.items()}


def _default(prop: dict) -> Any:
    if "enum" in prop:
        return prop["enum"][0]
    return {"string": "", "number": 0, "integer": 0, "boolean": False, "array": []}.get(
        prop.get("type"), None
    )


def _current_turn(messages) -> dict[str, Any]:
    """Map tool name -> parsed result for tool calls made since the last user message."""
    idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    names, results = {}, {}
    for m in messages[idx + 1:]:
        for call in m.get("tool_calls") or []:
            names[call["id"]] = call["function"]["name"]
        if m.get("role") == "tool":
            try:
                results[names.get(m["tool_call_id"], "?")] = json.loads(m["content"])
            except (json.JSONDecodeError, TypeError):
                results[names.get(m["tool_call_id"], "?")] = {}
    return results


def _duplicate_invoice(invoices: list[dict]) -> str | None:
    """Return the later of two paid invoices with the same amount on the same day."""
    paid = [i for i in invoices if i.get("status") == "paid"]
    for a in paid:
        for b in paid:
            if a is not b and a["amount_usd"] == b["amount_usd"] and a["issued_at"][:10] == b["issued_at"][:10]:
                return max(a, b, key=lambda i: i["issued_at"])["id"]
    return None


def _first_sentences(text: str, n: int = 2) -> str:
    text = re.sub(r"^#+ .*\n", "", text.strip()).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n]).strip()
