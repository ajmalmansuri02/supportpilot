"""Split Markdown documents into chunks for embedding (week 3).

Why chunk at all? An embedding summarises a whole piece of text as one vector, so a long
document becomes a blurry average. Small, focused chunks match questions much better.

Strategy used here:
1. Split on Markdown headings, so a chunk never mixes two topics.
2. If a section is still too long, split it on paragraphs up to `max_chars`,
   repeating the last paragraph of the previous piece (overlap) so facts
   that span a boundary are not lost.
3. Keep the heading path ("Refund policy > Annual plans") with each chunk. It is
   embedded together with the text and shown to the model, which helps a lot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    heading: str
    content: str

    @property
    def text_for_embedding(self) -> str:
        return f"{self.heading}\n\n{self.content}"


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return (heading path, body) for every section that has text."""
    path: list[str] = []
    sections: list[tuple[str, str]] = []
    body: list[str] = []

    def flush():
        text = "\n".join(body).strip()
        if text:
            sections.append((" > ".join(path), text))
        body.clear()

    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = path[: level - 1] + [match.group(2).strip()]
        else:
            body.append(line)
    flush()
    return sections


def chunk_markdown(markdown: str, max_chars: int = 1200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, text in split_sections(markdown):
        if len(text) <= max_chars:
            chunks.append(Chunk(heading, text))
            continue
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        current: list[str] = []
        for para in paragraphs:
            if current and len("\n\n".join(current + [para])) > max_chars:
                chunks.append(Chunk(heading, "\n\n".join(current)))
                current = [current[-1]]  # overlap: carry the last paragraph forward
            current.append(para)
        if current:
            chunks.append(Chunk(heading, "\n\n".join(current)))
    return chunks


def title_of(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return fallback
