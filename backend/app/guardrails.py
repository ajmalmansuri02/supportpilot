"""Guardrails (week 10): checks that run before and after the model.

Mapped to the OWASP Top 10 for LLM applications:
- LLM01 Prompt injection      -> check_input() blocks obvious attacks before any model call;
                                 retrieved docs and tool results are labelled as data in prompts
- LLM02 Sensitive information -> redact() strips card numbers and secrets before they reach the
                                 model or the database; logs and traces get full redaction
- LLM07 System prompt leakage -> check_output() replaces replies that echo the system prompt
- LLM06 Excessive agency      -> tool risk levels, human approval, and email_allowed() so the
                                 agent only touches accounts the customer named themselves

Heuristics are cheap, fast and predictable, but attackers rephrase. Treat them as a first
layer; set GUARD_LLM=true to add a small-model classifier as a second layer, and measure
both with the red-team eval (app/evals/redteam_eval.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app import db
from app.config import get_settings
from app.llm import LLMClient
from app.prompts import load_prompt

# --- PII and secrets ---------------------------------------------------------------------

_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PASSWORD = re.compile(r"(?i)\b(password|passcode|pin|otp|cvv)\b(\s*(is|:|=)\s*)(\S+)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d -]{8,13}\d(?!\w)")


def _luhn(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _mask_cards(text: str) -> str:
    def repl(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "[CARD REDACTED]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)

    return _CARD.sub(repl, text)


def redact(text: str, *, for_logs: bool = False) -> tuple[str, list[str]]:
    """Remove secrets. Card numbers and passwords always; emails and phones only in logs,
    because the agent legitimately needs the customer's email to find their account."""
    found: list[str] = []
    out = _mask_cards(text)
    if out != text:
        found.append("card_number")
    new = _PASSWORD.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", out)
    if new != out:
        found.append("password")
    out = new
    if for_logs:
        out = _EMAIL.sub("[EMAIL]", out)
        out = _PHONE.sub("[PHONE]", out)
    return out, found


# --- prompt injection ----------------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"ignore (all|your) (instructions|rules)",
    r"disregard (all |your |the )?(previous |prior )?(instructions|rules|guidelines)",
    r"(reveal|show|print|repeat|output|tell me) (me )?(your|the) (system )?(prompt|instructions|rules)",
    r"\bsystem prompt\b",
    r"you are (now|no longer)\b",
    r"\b(developer|dan|jailbreak|god) mode\b",
    r"pretend (to be|you are)",
    r"act as (an? )?(admin|administrator|developer|system)",
    r"(new|updated) instructions?:",
    r"</?(system|instructions?)>",
    r"\brefund (every|all) (invoice|charge)s?\b",
]
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

BLOCKED_REPLY = ("I can only help with CloudNotes support questions, and I can't change how I "
                 "work. What can I help you with?")


@dataclass
class InputCheck:
    text: str                       # the message to use from now on (redacted)
    blocked: bool = False
    reason: str | None = None
    flags: list[str] = field(default_factory=list)


GUARD_SCHEMA = {
    "type": "object",
    "properties": {"is_attack": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["is_attack", "reason"],
}


async def check_input(message: str, llm: LLMClient | None = None) -> InputCheck:
    text, flags = redact(message)
    if match := _INJECTION.search(text):
        return InputCheck(text, True, f"injection pattern: {match.group(0)!r}", flags)
    if llm is not None and get_settings().guard_llm:
        result = await llm.chat(
            [{"role": "system", "content": load_prompt("guard").text},
             {"role": "user", "content": text}],
            model=get_settings().fast_model, json_schema=GUARD_SCHEMA, temperature=0,
            purpose="guard",
        )
        try:
            verdict = json.loads(result.content)
            if verdict.get("is_attack"):
                return InputCheck(text, True, f"llm guard: {verdict.get('reason', '')}", flags)
        except json.JSONDecodeError:
            pass
    return InputCheck(text, False, None, flags)


# --- output checks ---------------------------------------------------------------------

def _leak_markers() -> list[str]:
    """Distinctive lines from our prompts that should never appear in a reply."""
    markers = []
    for name in ("system", "agent", "rag"):
        for line in load_prompt(name).text.splitlines():
            line = line.strip(" -0123456789.")
            if len(line) > 40 and "don't know" not in line:
                markers.append(line[:60].lower())
    return markers


def check_output(answer: str) -> tuple[str, bool]:
    """Return (safe answer, leaked?)."""
    low = answer.lower()
    if any(marker in low for marker in _leak_markers()):
        return BLOCKED_REPLY, True
    return answer, False


# --- tool permissions -------------------------------------------------------------------

def emails_in(texts: list[str]) -> set[str]:
    return {e.lower() for t in texts for e in _EMAIL.findall(t)}


def email_allowed(email: str | None, allowed: set[str] | None) -> bool:
    """The agent may only act on accounts whose email the customer typed in this chat.

    Without real login this is a stand-in for authentication: it stops "look up
    rahul@example.com's invoices" unless Rahul's email came from the customer.
    """
    if allowed is None or not email:
        return True
    return email.strip().lower() in allowed


async def log_event(conversation_id: str | None, kind: str, detail: str) -> None:
    async with db.pool().connection() as conn:
        await conn.execute(
            "INSERT INTO guardrail_events (conversation_id, kind, detail) VALUES (%s, %s, %s)",
            (conversation_id, kind, detail[:500]),
        )
