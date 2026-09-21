"""Retrieval ablation study.

Answers the questions the design decisions rest on, with numbers instead of
opinions: does heading-aware chunking beat a fixed window, does hybrid search
beat dense-only, does reranking pay for its latency, and what chunk size works
best on this corpus.

Each configuration is built into its own throwaway index and evaluated on the
same question set, so only one variable changes at a time.

Run:
    python scripts/run_ablation.py                     # light backends, seconds
    VOICERAG_RAG__EMBEDDER=sentence_transformers \
    VOICERAG_RAG__RERANKER=cross_encoder \
        python scripts/run_ablation.py --with-rerank   # real models, minutes
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

from voicerag.config import get_settings
from voicerag.eval.dataset import load_eval_set
from voicerag.eval.retrieval_metrics import aggregate_retrieval
from voicerag.rag.chunking import chunk_documents
from voicerag.rag.embeddings import build_embedder
from voicerag.rag.loaders import load_directory
from voicerag.rag.reranker import build_reranker
from voicerag.rag.retriever import Retriever
from voicerag.rag.store import VectorStore


@dataclass(frozen=True)
class Variant:
    name: str
    chunk_size: int
    chunk_overlap: int
    hybrid: bool
    rerank: bool
    top_k: int = 20
    top_n: int = 4


DEFAULT_VARIANTS = [
    Variant("dense-only, 800/120", 800, 120, hybrid=False, rerank=False),
    Variant("hybrid, 800/120 (shipped)", 800, 120, hybrid=True, rerank=False),
    Variant("hybrid, 400/60", 400, 60, hybrid=True, rerank=False),
    Variant("hybrid, 1600/200", 1600, 200, hybrid=True, rerank=False),
    Variant("hybrid, no overlap", 800, 0, hybrid=True, rerank=False),
]


class ModelPool:
    """Loads each model once and hands the same instance to every variant.

    Building a fresh embedder per variant leaks GPU memory: torch modules hold
    reference cycles, so the previous variant's weights are still resident when
    the next one loads, and on an 8 GB card the reranked variant then dies with
    an out-of-memory error. Only the chunking changes between variants, so the
    models never needed rebuilding in the first place.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._embedder = None
        self._rerankers: dict[str, object] = {}

    def embedder(self):
        if self._embedder is None:
            self._embedder = build_embedder(
                self.cfg.embedder, self.cfg.embedding_model, self.cfg.device
            )
        return self._embedder

    def reranker(self, kind: str):
        if kind not in self._rerankers:
            self._rerankers[kind] = build_reranker(kind, self.cfg.reranker_model, self.cfg.device)
        return self._rerankers[kind]


def free_gpu_cache() -> None:
    """Return the allocator's unused blocks between variants."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def evaluate_variant(variant: Variant, corpus: Path, items, cfg, pool: ModelPool) -> dict:
    documents = load_directory(corpus)
    chunks = chunk_documents(documents, variant.chunk_size, variant.chunk_overlap)
    embedder = pool.embedder()

    started = time.perf_counter()
    store = VectorStore(embedder.dimension, embedder.name)
    store.add(chunks, embedder.embed_passages([c.text for c in chunks]))
    index_s = time.perf_counter() - started

    reranker = pool.reranker("cross_encoder" if variant.rerank else "noop")
    retriever = Retriever(
        store=store,
        embedder=embedder,
        reranker=reranker,
        top_k=variant.top_k,
        top_n=variant.top_n,
        hybrid=variant.hybrid,
    )

    rows = []
    latencies = []
    for item in items:
        result = retriever.retrieve(item.question)
        sources: list[str] = []
        for scored in result.chunks:
            if scored.chunk.source not in sources:
                sources.append(scored.chunk.source)
        rows.append({"retrieved": sources, "relevant": list(item.relevant_sources)})
        latencies.append(sum(result.timings_ms.values()))

    metrics = aggregate_retrieval(rows, ks=(1, 3, variant.top_n))
    return {
        "variant": variant.name,
        "chunks": len(chunks),
        "chunk_size": variant.chunk_size,
        "chunk_overlap": variant.chunk_overlap,
        "hybrid": variant.hybrid,
        "rerank": variant.rerank,
        "index_build_s": round(index_s, 2),
        "mean_query_ms": round(sum(latencies) / len(latencies), 1),
        **metrics,
    }


def to_markdown(results: list[dict], top_n: int) -> str:
    columns = [
        "variant",
        "chunks",
        f"hit@{top_n}",
        "hit@1",
        "mrr",
        f"ndcg@{top_n}",
        "mean_query_ms",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in results:
        cells = []
        for column in columns:
            value = row.get(column, "-")
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/qa.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/ablation"))
    parser.add_argument(
        "--with-rerank",
        action="store_true",
        help="Add a reranked variant (downloads a cross-encoder; needs requirements-ml.txt)",
    )
    args = parser.parse_args()

    cfg = get_settings().rag
    items = [i for i in load_eval_set(args.dataset) if i.relevant_sources]
    variants = list(DEFAULT_VARIANTS)
    if args.with_rerank:
        variants.append(
            Variant("hybrid + cross-encoder rerank", 800, 120, hybrid=True, rerank=True)
        )

    print(f"Embedder: {cfg.embedder} ({cfg.embedding_model})")
    print(f"Questions with ground truth: {len(items)}\n")

    pool = ModelPool(cfg)
    results = []
    for variant in variants:
        print(f"-> {variant.name} ...", flush=True)
        results.append(evaluate_variant(variant, args.corpus, items, cfg, pool))
        free_gpu_cache()

    table = to_markdown(results, top_n=variants[0].top_n)
    print("\n" + table)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (args.out / f"ablation-{stamp}.json").write_text(
        json.dumps({"embedder": cfg.embedder, "results": results}, indent=2), encoding="utf-8"
    )
    (args.out / f"ablation-{stamp}.md").write_text(
        f"# Retrieval ablation ({cfg.embedder})\n\n{table}\n", encoding="utf-8"
    )
    print(f"\nSaved to {args.out}/ablation-{stamp}.{{json,md}}")


if __name__ == "__main__":
    main()
