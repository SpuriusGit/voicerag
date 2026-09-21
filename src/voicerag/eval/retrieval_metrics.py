"""Retrieval quality metrics.

Relevance is judged at *source-document* level: the eval set records which files
should answer a question, not which chunk, because chunk ids change every time
the chunker is re-tuned and that would make historical runs incomparable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_at_k(retrieved_sources: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1.0 if any relevant document appears in the top-k."""
    if not relevant:
        return 0.0
    return float(bool(set(retrieved_sources[:k]) & set(relevant)))


def recall_at_k(retrieved_sources: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the relevant documents found in the top-k."""
    if not relevant:
        return 0.0
    found = set(retrieved_sources[:k]) & set(relevant)
    return len(found) / len(set(relevant))


def precision_at_k(retrieved_sources: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = retrieved_sources[:k]
    if not top:
        return 0.0
    return sum(1 for s in top if s in set(relevant)) / len(top)


def reciprocal_rank(retrieved_sources: Sequence[str], relevant: Sequence[str]) -> float:
    """1/rank of the first relevant document; 0 if none was retrieved."""
    relevant_set = set(relevant)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_sources: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Binary-gain nDCG — rewards putting the right document first, not just in the list."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, source in enumerate(retrieved_sources[:k], start=1)
        if source in relevant_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def aggregate_retrieval(
    per_item: list[dict],
    ks: Sequence[int] = (1, 3, 4),
) -> dict[str, float]:
    """Average per-item metrics, skipping items with no ground-truth documents.

    ``k`` values above the pipeline's ``top_n`` are meaningless here: the report
    measures the documents that actually reached the prompt, not the raw
    candidate list, so the defaults stay at or below the shipped ``top_n``.
    """
    scored = [row for row in per_item if row.get("relevant")]
    if not scored:
        return {}
    out: dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = sum(hit_at_k(r["retrieved"], r["relevant"], k) for r in scored) / len(
            scored
        )
        out[f"recall@{k}"] = sum(
            recall_at_k(r["retrieved"], r["relevant"], k) for r in scored
        ) / len(scored)
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r["retrieved"], r["relevant"], k) for r in scored) / len(
            scored
        )
    out["mrr"] = sum(reciprocal_rank(r["retrieved"], r["relevant"]) for r in scored) / len(scored)
    return {k: round(v, 4) for k, v in out.items()}
