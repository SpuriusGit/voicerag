"""Small BM25 implementation for hybrid retrieval.

Dense embeddings miss exact identifiers — model names, flags, error codes — which
is exactly what people ask about in a technical knowledge base. BM25 catches
those, and the two result lists are fused with Reciprocal Rank Fusion.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from voicerag.rag.documents import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9_Ѐ-ӿ]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.docs = [tokenize(c.text) for c in chunks]
        self.doc_lengths = [len(d) for d in self.docs]
        self.avg_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_freqs = [Counter(d) for d in self.docs]

        doc_freq: Counter[str] = Counter()
        for doc in self.docs:
            doc_freq.update(set(doc))
        n = max(len(self.docs), 1)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms or not self.docs:
            return []
        scores = [0.0] * len(self.docs)
        for i, freqs in enumerate(self.term_freqs):
            length = self.doc_lengths[i] or 1
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
            total = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if tf:
                    total += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / (tf + norm)
            scores[i] = total

        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [ScoredChunk(chunk=self.chunks[i], score=scores[i]) for i in ranked if scores[i] > 0]


def reciprocal_rank_fusion(
    dense: list[ScoredChunk],
    sparse: list[ScoredChunk],
    k: int = 60,
    top_k: int = 20,
) -> list[ScoredChunk]:
    """Fuse two ranked lists by rank, not by score.

    RRF avoids having to calibrate cosine similarity against BM25 scores, which
    live on completely different scales.
    """
    fused: dict[str, float] = {}
    by_id: dict[str, ScoredChunk] = {}
    for ranking in (dense, sparse):
        for rank, item in enumerate(ranking, start=1):
            cid = item.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(cid, item)

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    return [ScoredChunk(chunk=by_id[cid].chunk, score=score) for cid, score in ordered]
