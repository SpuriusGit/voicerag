import numpy as np
import pytest

from voicerag.rag.bm25 import BM25Index, reciprocal_rank_fusion, tokenize
from voicerag.rag.documents import Chunk, ScoredChunk
from voicerag.rag.embeddings import HashingEmbedder
from voicerag.rag.store import VectorStore


def _chunk(cid: str, text: str, source: str = "a.md", position: int = 0) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", text=text, source=source, position=position)


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder()
    first = embedder.embed_passages(["repeatable text"])
    second = embedder.embed_passages(["repeatable text"])
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_hashing_embedder_ranks_related_text_higher():
    embedder = HashingEmbedder()
    passages = embedder.embed_passages(
        ["The RTX 4060 laptop GPU has 8 GB of VRAM.", "Sourdough bread needs a starter."]
    )
    query = embedder.embed_query("how much vram does the rtx 4060 have")
    similarities = passages @ query
    assert similarities[0] > similarities[1]


def test_store_roundtrip_preserves_chunks_and_scores(tmp_path):
    embedder = HashingEmbedder()
    chunks = [_chunk("c1", "gpu memory limits"), _chunk("c2", "audio transcription")]
    store = VectorStore(embedder.dimension, embedder.name)
    store.add(chunks, embedder.embed_passages([c.text for c in chunks]))
    store.save(tmp_path / "idx")

    loaded = VectorStore.load(tmp_path / "idx")
    assert len(loaded) == 2
    assert loaded.embedder_name == embedder.name
    hit = loaded.search(embedder.embed_query("gpu memory"), top_k=1)[0]
    assert hit.chunk.chunk_id == "c1"
    assert -1.0 <= hit.score <= 1.0


def test_store_deduplicates_by_chunk_id():
    embedder = HashingEmbedder()
    chunk = _chunk("c1", "same chunk")
    store = VectorStore(embedder.dimension)
    vectors = embedder.embed_passages([chunk.text])
    store.add([chunk], vectors)
    store.add([chunk], vectors)
    assert len(store) == 1


def test_store_rejects_wrong_dimension():
    store = VectorStore(dimension=8)
    with pytest.raises(ValueError):
        store.add([_chunk("c1", "x")], np.zeros((1, 4), dtype=np.float32))


def test_store_rejects_mismatched_counts():
    store = VectorStore(dimension=4)
    with pytest.raises(ValueError):
        store.add([_chunk("c1", "x")], np.zeros((2, 4), dtype=np.float32))


def test_loading_missing_index_explains_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        VectorStore.load(tmp_path / "nope")


def test_search_on_empty_store_returns_nothing():
    assert VectorStore(dimension=4).search(np.zeros(4, dtype=np.float32)) == []


def test_bm25_finds_exact_identifier_tokens():
    chunks = [
        _chunk("c1", "Set gpu_memory_utilization to 0.85 on shared cards.", "gpu.md"),
        _chunk("c2", "Audio is downmixed to mono before transcription.", "stt.md"),
    ]
    hits = BM25Index(chunks).search("gpu_memory_utilization", top_k=2)
    assert hits[0].chunk.chunk_id == "c1"


def test_bm25_tokenizer_handles_cyrillic():
    assert "відеопам" in " ".join(tokenize("Скільки відеопам'яті потрібно?"))


def test_rrf_promotes_chunks_ranked_by_both_retrievers():
    shared = _chunk("shared", "shared text")
    dense = [ScoredChunk(_chunk("d1", "dense only"), 0.9), ScoredChunk(shared, 0.4)]
    sparse = [ScoredChunk(_chunk("s1", "sparse only"), 12.0), ScoredChunk(shared, 8.0)]
    fused = reciprocal_rank_fusion(dense, sparse, top_k=3)
    assert fused[0].chunk.chunk_id == "shared"


def test_retriever_returns_stage_timings(pipeline):
    result = pipeline.retriever.retrieve("how much vram does the rtx 4060 laptop gpu have")
    assert {"embed", "search", "rerank"} <= set(result.timings_ms)
    assert result.chunks
    assert result.chunks[0].chunk.source == "04_gpu_capacity.md"


def test_retriever_respects_top_n(pipeline):
    assert len(pipeline.retriever.retrieve("gpu", top_k=10, top_n=2).chunks) == 2
