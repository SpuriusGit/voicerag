"""Markdown-aware chunking.

Splitting on a fixed character window cuts tables and code blocks in half and
measurably hurts retrieval. This splitter instead:

1. segments the document by markdown headings, keeping the heading path as
   metadata (and prepending it to the chunk text, which helps embeddings),
2. packs whole paragraphs into chunks up to ``chunk_size`` characters,
3. carries ``chunk_overlap`` characters of the previous chunk into the next one
   so a fact that straddles a boundary is still retrievable,
4. hard-splits any single paragraph that exceeds the budget on sentence marks.
"""

from __future__ import annotations

import re

from voicerag.rag.documents import Chunk, Document, stable_id

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?。？！])\s+")


def _heading_sections(text: str) -> list[tuple[str, str]]:
    """Return ``(heading_path, body)`` pairs, preserving document order."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buffer: list[str] = []
    current = ""

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current, body))
        buffer.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            stack[:] = stack[: level - 1]
            stack.append(title)
            current = " > ".join(stack)
        else:
            buffer.append(line)
    flush()
    return sections or [("", text.strip())]


def _split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    sentences = _SENTENCE_END.split(paragraph)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > limit:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    # A single sentence longer than the limit still has to be cut somewhere.
    out: list[str] = []
    for part in parts:
        if len(part) <= limit:
            out.append(part)
        else:
            out.extend(part[i : i + limit] for i in range(0, len(part), limit))
    return out


def chunk_document(
    document: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    chunks: list[Chunk] = []
    position = 0

    for heading, body in _heading_sections(document.text):
        prefix = f"{heading}\n\n" if heading else ""
        budget = max(chunk_size - len(prefix), chunk_size // 2)

        paragraphs: list[str] = []
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if not para:
                continue
            paragraphs.extend(_split_long_paragraph(para, budget) if len(para) > budget else [para])

        buffer = ""
        for para in paragraphs:
            candidate = f"{buffer}\n\n{para}".strip() if buffer else para
            if buffer and len(candidate) > budget:
                chunks.append(_make_chunk(document, prefix, buffer, heading, position))
                position += 1
                tail = buffer[-chunk_overlap:] if chunk_overlap else ""
                buffer = f"{tail}\n\n{para}".strip() if tail else para
            else:
                buffer = candidate
        if buffer.strip():
            chunks.append(_make_chunk(document, prefix, buffer, heading, position))
            position += 1

    return chunks


def _make_chunk(document: Document, prefix: str, body: str, heading: str, position: int) -> Chunk:
    text = f"{prefix}{body}".strip()
    return Chunk(
        chunk_id=stable_id(document.doc_id, str(position), text[:200]),
        doc_id=document.doc_id,
        text=text,
        source=document.source,
        position=position,
        metadata={"heading": heading, "chars": len(text)},
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in documents:
        out.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return out
