"""Build the configured LLM backend."""

from __future__ import annotations

from voicerag.config import LLMSettings, get_settings
from voicerag.llm.base import BaseLLM
from voicerag.llm.echo import EchoLLM


def build_llm(settings: LLMSettings | None = None) -> BaseLLM:
    cfg = settings or get_settings().llm

    if cfg.backend == "echo":
        return EchoLLM(model=cfg.model)

    if cfg.backend == "ollama":
        from voicerag.llm.ollama import OllamaLLM

        return OllamaLLM(
            model=cfg.model,
            base_url=cfg.base_url,
            timeout_s=cfg.timeout_s,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    if cfg.backend == "openai_compat":
        from voicerag.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout_s=cfg.timeout_s,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    raise ValueError(f"Unknown LLM backend: {cfg.backend!r}")
