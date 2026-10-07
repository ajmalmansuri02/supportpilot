"""Prompts live in versioned Markdown files (app/prompts/*.md), not in code.

Keeping them as files makes changes reviewable in pull requests, and the version number
is logged with every answer so an eval regression can be traced to a prompt change.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).with_name("prompts")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    text: str

    def format(self, **values: str) -> str:
        text = self.text
        for key, value in values.items():
            text = text.replace("{" + key + "}", value)
        return text


@lru_cache
def load_prompt(name: str) -> Prompt:
    raw = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    version = 0
    if raw.startswith("---"):
        _, header, raw = raw.split("---", 2)
        for line in header.strip().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "version":
                version = int(value.strip())
    return Prompt(name=name, version=version, text=raw.strip())
