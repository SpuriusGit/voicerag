"""Lazy singletons for the heavy components.

Models are loaded once on first use and shared across requests — reloading a
cross-encoder per request would dominate latency and exhaust VRAM.
"""

from __future__ import annotations

from functools import lru_cache

from voicerag.config import get_settings
from voicerag.logging_utils import get_logger
from voicerag.rag.pipeline import RAGPipeline
from voicerag.stt.base import BaseSTT
from voicerag.stt.factory import build_stt

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    settings = get_settings()
    log.info(
        "loading_pipeline",
        llm_backend=settings.llm.backend,
        llm_model=settings.llm.model,
        embedder=settings.rag.embedder,
        reranker=settings.rag.reranker,
    )
    pipeline = RAGPipeline.from_settings(settings)
    log.info("pipeline_ready", **pipeline.retriever.store.stats())
    return pipeline


@lru_cache(maxsize=1)
def get_stt() -> BaseSTT:
    settings = get_settings()
    log.info("loading_stt", backend=settings.stt.backend, model=settings.stt.model)
    return build_stt(settings.stt)


def reset_caches() -> None:
    """Used by tests and by the reload path after re-ingesting the corpus."""
    get_pipeline.cache_clear()
    get_stt.cache_clear()
