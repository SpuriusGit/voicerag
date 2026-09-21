"""Core RAG data structures shared by the loader, store and retriever."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A source file before chunking."""

    doc_id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    position: int
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "source": self.source,
            "position": self.position,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Chunk:
        return cls(
            chunk_id=raw["chunk_id"],
            doc_id=raw["doc_id"],
            text=raw["text"],
            source=raw["source"],
            position=int(raw["position"]),
            metadata=raw.get("metadata", {}),
        )


@dataclass
class ScoredChunk:
    """A chunk with its retrieval and (optionally) rerank score."""

    chunk: Chunk
    score: float
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        return self.score if self.rerank_score is None else self.rerank_score

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "position": self.chunk.position,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
        }


def stable_id(*parts: str) -> str:
    """Content-addressed id, so re-ingesting unchanged files is idempotent."""
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]
