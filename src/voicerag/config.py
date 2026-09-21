"""Typed application configuration.

Every setting is overridable through environment variables using the
``VOICERAG_<SECTION>__<FIELD>`` convention, which keeps the same file working
for local runs, docker-compose and CI without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["auto", "cpu", "cuda"]


class LLMSettings(BaseModel):
    backend: Literal["ollama", "openai_compat", "echo"] = "ollama"
    model: str = "qwen2.5:3b-instruct"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_s: float = 120.0


class RAGSettings(BaseModel):
    embedder: Literal["sentence_transformers", "hashing"] = "sentence_transformers"
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker: Literal["cross_encoder", "noop"] = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = Field(20, ge=1, description="candidates fetched from the vector store")
    top_n: int = Field(4, ge=1, description="chunks kept after reranking")
    fusion: Literal["lexical_gate", "always", "off"] = Field(
        "lexical_gate",
        description=(
            "How BM25 joins dense retrieval. 'lexical_gate' fuses only when enough of "
            "the query exists in the index vocabulary, 'always' fuses unconditionally, "
            "'off' is dense-only."
        ),
    )
    min_lexical_overlap: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Share of query terms that must appear in the index before BM25 is fused. "
            "Calibrated in docs/EXPERIMENTS.md E1c; the mechanism matters more than "
            "the constant."
        ),
    )
    chunk_size: int = 800
    chunk_overlap: int = 120
    store_path: Path = Path("storage/index")
    device: Device = "auto"


class STTSettings(BaseModel):
    backend: Literal["faster_whisper", "stub"] = "faster_whisper"
    model: str = "small"
    device: Device = "auto"
    compute_type: str = "int8_float16"
    language: str = ""
    beam_size: int = 5
    vad_filter: bool = True


class PromptSettings(BaseModel):
    dir: Path = Path("prompts")
    # Pinned, never @latest: adding a prompt file must not promote it. v4 was
    # measured as worse than v3 and would otherwise have shipped on creation.
    answer: str = "rag_answer@v3"
    judge: str = "judge_faithfulness@latest"
    query_rewrite: str = "query_rewrite@latest"


class ServiceSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    max_upload_mb: int = 25


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOICERAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = LLMSettings()
    rag: RAGSettings = RAGSettings()
    stt: STTSettings = STTSettings()
    prompts: PromptSettings = PromptSettings()
    service: ServiceSettings = ServiceSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cleared in tests via ``cache_clear``)."""
    return Settings()


def resolve_device(requested: Device) -> str:
    """Translate ``auto`` into the best device actually available."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
