"""Shared fixtures.

The whole suite runs on the dependency-free backends (hashing embedder, echo
LLM, stub STT) so CI needs no GPU and no model downloads, while still exercising
the real chunking, retrieval, prompt and API code paths.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    return REPO_ROOT / "data" / "corpus"


@pytest.fixture(scope="session")
def prompts_dir() -> Path:
    return REPO_ROOT / "prompts"


@pytest.fixture(scope="session")
def eval_set_path() -> Path:
    return REPO_ROOT / "data" / "eval" / "qa.jsonl"


@pytest.fixture(scope="session")
def built_index(tmp_path_factory, corpus_dir: Path) -> Path:
    """Build a real index from the repository corpus once per test session."""
    from voicerag.rag.chunking import chunk_documents
    from voicerag.rag.embeddings import HashingEmbedder
    from voicerag.rag.loaders import load_directory
    from voicerag.rag.store import VectorStore

    target = tmp_path_factory.mktemp("index")
    chunks = chunk_documents(load_directory(corpus_dir), chunk_size=800, chunk_overlap=120)
    embedder = HashingEmbedder()
    store = VectorStore(dimension=embedder.dimension, embedder_name=embedder.name)
    store.add(chunks, embedder.embed_passages([c.text for c in chunks]))
    store.save(target)
    return target


@pytest.fixture
def test_settings(built_index: Path, prompts_dir: Path, tmp_path: Path, monkeypatch):
    """Point the process-wide settings at the throwaway index and stub backends."""
    from voicerag.api import deps
    from voicerag.config import get_settings

    env = {
        "VOICERAG_RAG__EMBEDDER": "hashing",
        "VOICERAG_RAG__RERANKER": "noop",
        "VOICERAG_RAG__STORE_PATH": str(built_index),
        "VOICERAG_LLM__BACKEND": "echo",
        "VOICERAG_LLM__MODEL": "echo",
        "VOICERAG_STT__BACKEND": "stub",
        "VOICERAG_PROMPTS__DIR": str(prompts_dir),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # Run from an empty directory so a developer's local .env cannot leak in;
    # every path handed to the settings above is absolute.
    monkeypatch.chdir(tmp_path)

    get_settings.cache_clear()
    deps.reset_caches()
    settings = get_settings()
    yield settings
    get_settings.cache_clear()
    deps.reset_caches()


@pytest.fixture
def pipeline(test_settings):
    from voicerag.rag.pipeline import RAGPipeline

    return RAGPipeline.from_settings(test_settings)


@pytest.fixture
def client(test_settings):
    from fastapi.testclient import TestClient

    from voicerag.api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    from voicerag.stt.audio import save_wav, synth_tone

    return save_wav(synth_tone(1.0), tmp_path / "sample.wav")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Drop any VOICERAG_* variables inherited from the developer's shell."""
    for key in list(os.environ):
        if key.startswith("VOICERAG_"):
            monkeypatch.delenv(key, raising=False)
