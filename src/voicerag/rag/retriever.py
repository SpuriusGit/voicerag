"""Retrieval: hybrid candidate generation followed by cross-encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass, field

from voicerag.monitoring.metrics import observe_stage, record_fusion
from voicerag.rag.bm25 import BM25Index, reciprocal_rank_fusion
from voicerag.rag.documents import ScoredChunk
from voicerag.rag.embeddings import BaseEmbedder
from voicerag.rag.reranker import BaseReranker, NoopReranker
from voicerag.rag.store import VectorStore


@dataclass
class RetrievalResult:
    query: str
    chunks: list[ScoredChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates_considered: int = 0
    fused: bool = False
    lexical_coverage: float = 0.0

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "chunks": [c.as_dict() for c in self.chunks],
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "candidates_considered": self.candidates_considered,
            "fused": self.fused,
            "lexical_coverage": round(self.lexical_coverage, 3),
        }


class Retriever:
    """Dense retrieval, optionally fused with BM25, then reranked.

    ``fusion`` decides when BM25 joins in:

    * ``lexical_gate`` (default) — fuse only when at least
      ``min_lexical_overlap`` of the query's terms appear in the index. A query
      in a language the corpus does not contain fails this test, and fusing it
      would inject a near-random ranking into RRF (docs/EXPERIMENTS.md E1b).
    * ``always`` — the unconditional behaviour, kept so the gate can be measured
      against it.
    * ``off`` — dense only.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: BaseEmbedder,
        reranker: BaseReranker | None = None,
        top_k: int = 20,
        top_n: int = 4,
        hybrid: bool = True,
        fusion: str = "lexical_gate",
        min_lexical_overlap: float = 0.35,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker or NoopReranker()
        self.top_k = top_k
        self.top_n = top_n
        self.hybrid = hybrid and fusion != "off"
        self.fusion = fusion
        self.min_lexical_overlap = min_lexical_overlap
        self._bm25 = BM25Index(store.chunks) if self.hybrid and len(store) else None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> RetrievalResult:
        k = top_k or self.top_k
        n = top_n or self.top_n
        timings: dict[str, float] = {}

        with observe_stage("embed", self.embedder.name) as t:
            query_vector = self.embedder.embed_query(query)
        timings["embed"] = t["elapsed_s"] * 1000

        coverage = 0.0
        fused = False
        with observe_stage("search", self.embedder.name) as t:
            dense = self.store.search(query_vector, top_k=k)
            candidates = dense
            if self._bm25 is not None:
                coverage = self._bm25.lexical_coverage(query)
                fused = self.fusion == "always" or coverage >= self.min_lexical_overlap
                if fused:
                    sparse = self._bm25.search(query, top_k=k)
                    candidates = reciprocal_rank_fusion(dense, sparse, top_k=k)
                record_fusion("fused" if fused else "skipped_low_lexical_overlap")
        timings["search"] = t["elapsed_s"] * 1000

        with observe_stage("rerank", self.reranker.name) as t:
            reranked = self.reranker.rerank(query, candidates, top_n=n)
        timings["rerank"] = t["elapsed_s"] * 1000

        return RetrievalResult(
            query=query,
            chunks=reranked,
            timings_ms=timings,
            candidates_considered=len(candidates),
            fused=fused,
            lexical_coverage=coverage,
        )
