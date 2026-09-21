"""Cross-encoder reranking.

Bi-encoder retrieval scores query and passage independently, so it ranks by
topical similarity rather than by whether the passage actually answers the
question. A cross-encoder reads the pair jointly and reorders the shortlist.

Measure the gain on your own corpus before paying for it::

    python scripts/run_ablation.py --with-rerank

The cost is latency, which is why only ``top_k`` candidates are reranked down to
``top_n`` rather than the whole corpus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from voicerag.config import resolve_device
from voicerag.rag.documents import ScoredChunk


class BaseReranker(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...


class NoopReranker(BaseReranker):
    """Keeps retrieval order — the baseline every rerank experiment is measured against."""

    name = "noop"

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        return candidates[:top_n]


class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.device = resolve_device(device)  # type: ignore[arg-type]
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=self.device, max_length=max_length)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)
        return sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))[:top_n]


def build_reranker(kind: str, model_name: str, device: str = "auto") -> BaseReranker:
    if kind == "noop":
        return NoopReranker()
    if kind == "cross_encoder":
        return CrossEncoderReranker(model_name=model_name, device=device)
    raise ValueError(f"Unknown reranker kind: {kind!r}")
