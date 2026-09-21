"""Embedding backends.

``SentenceTransformerEmbedder`` is the real one (multilingual E5 by default, so
Ukrainian questions match an English corpus). ``HashingEmbedder`` is a
dependency-free deterministic fallback that keeps unit tests and CI fast — it is
a real bag-of-character-ngrams projection, not a random stub, so retrieval tests
still assert meaningful ordering.

E5 models require the ``query:`` / ``passage:`` prefixes; forgetting them costs
several points of recall, so the prefixing is handled here rather than left to
callers.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from voicerag.config import resolve_device


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class BaseEmbedder(ABC):
    """Embeds passages and queries into a shared L2-normalised space."""

    name: str = "base"
    dimension: int = 0

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.device = resolve_device(device)  # type: ignore[arg-type]
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=self.device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self._needs_e5_prefix = "e5" in model_name.lower()

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        prepared = [f"{prefix}{t}" for t in texts] if self._needs_e5_prefix else texts
        vectors = self._model.encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query: ")


class HashingEmbedder(BaseEmbedder):
    """Hashed character 3-gram + word unigram projection.

    Deterministic, CPU-only and instant — good enough to keep the retrieval code
    path under test without downloading 500 MB of weights in CI.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.name = f"hashing-{dimension}"
        self.dimension = dimension

    def _features(self, text: str) -> list[str]:
        lowered = " ".join(text.lower().split())
        words = lowered.split()
        trigrams = [lowered[i : i + 3] for i in range(max(len(lowered) - 2, 0))]
        return words + trigrams

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for feature in self._features(text):
            digest = hashlib.md5(feature.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        return vec

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return l2_normalize(np.vstack([self._vector(t) for t in texts]))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


def build_embedder(kind: str, model_name: str, device: str = "auto") -> BaseEmbedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=model_name, device=device)
    raise ValueError(f"Unknown embedder kind: {kind!r}")
