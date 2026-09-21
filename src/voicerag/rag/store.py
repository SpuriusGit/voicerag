"""Persistent vector store.

Backed by a plain normalised float32 matrix (cosine == dot product). FAISS is
used automatically when installed and the corpus is large enough for the index
build to pay for itself; below that threshold a numpy matmul is faster than the
FAISS call overhead. Both paths return identical results, so tests can run
without FAISS.

Persistence is two files — ``vectors.npz`` and ``chunks.jsonl`` — which are easy
to diff, inspect and ship in a container layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voicerag.rag.documents import Chunk, ScoredChunk
from voicerag.rag.embeddings import l2_normalize

FAISS_MIN_VECTORS = 20_000


class VectorStore:
    def __init__(self, dimension: int, embedder_name: str = "unknown") -> None:
        self.dimension = dimension
        self.embedder_name = embedder_name
        self._vectors = np.zeros((0, dimension), dtype=np.float32)
        self._chunks: list[Chunk] = []
        self._faiss_index = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks vs {len(vectors)} vectors")
        if not chunks:
            return
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected dim {self.dimension}, got {vectors.shape[1]}")

        known = {c.chunk_id for c in self._chunks}
        keep = [i for i, c in enumerate(chunks) if c.chunk_id not in known]
        if not keep:
            return
        self._chunks.extend(chunks[i] for i in keep)
        self._vectors = np.vstack([self._vectors, l2_normalize(vectors[keep].astype(np.float32))])
        self._faiss_index = None  # invalidate

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[ScoredChunk]:
        if len(self) == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        query = l2_normalize(query)
        top_k = min(top_k, len(self))

        index = self._maybe_faiss()
        pairs: list[tuple[int, float]]
        if index is not None:
            scores, indices = index.search(query, top_k)
            pairs = [
                (int(i), float(s))
                for i, s in zip(indices[0].tolist(), scores[0].tolist(), strict=True)
            ]
        else:
            sims = (self._vectors @ query[0]).astype(np.float32)
            # argpartition keeps this O(n) instead of a full sort of the corpus
            top = np.argpartition(-sims, top_k - 1)[:top_k]
            top = top[np.argsort(-sims[top])]
            pairs = [(int(i), float(sims[i])) for i in top]

        return [ScoredChunk(chunk=self._chunks[i], score=float(s)) for i, s in pairs if i >= 0]

    def _maybe_faiss(self):
        if len(self) < FAISS_MIN_VECTORS:
            return None
        if self._faiss_index is not None:
            return self._faiss_index
        try:
            import faiss
        except ImportError:
            return None
        index = faiss.IndexFlatIP(self.dimension)
        index.add(self._vectors)
        self._faiss_index = index
        return index

    # ---------- persistence ----------

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "vectors.npz", vectors=self._vectors)
        with (out / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "dimension": self.dimension,
                    "embedder": self.embedder_name,
                    "count": len(self),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, path: Path | str) -> VectorStore:
        src = Path(path)
        meta_file = src / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"No index at {src}. Build one first: python scripts/ingest.py")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        store = cls(dimension=int(meta["dimension"]), embedder_name=meta.get("embedder", "unknown"))
        store._vectors = np.load(src / "vectors.npz")["vectors"].astype(np.float32)
        with (src / "chunks.jsonl").open(encoding="utf-8") as fh:
            store._chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
        if len(store._chunks) != len(store._vectors):
            raise ValueError("Corrupt index: chunk/vector count mismatch")
        return store

    def stats(self) -> dict:
        lengths = [len(c.text) for c in self._chunks]
        return {
            "chunks": len(self),
            "dimension": self.dimension,
            "embedder": self.embedder_name,
            "sources": len({c.source for c in self._chunks}),
            "mean_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
            "vectors_mb": round(self._vectors.nbytes / 1024**2, 2),
        }
