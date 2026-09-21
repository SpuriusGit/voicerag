"""Ollama backend (local GGUF models served on http://localhost:11434)."""

from __future__ import annotations

import time

import httpx

from voicerag.llm.base import BaseLLM, ChatMessage, LLMError, LLMResponse


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
            },
        }
        started = time.perf_counter()
        try:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        latency = time.perf_counter() - started

        text = (data.get("message") or {}).get("content", "")
        if not text:
            raise LLMError(f"Ollama returned an empty completion: {data}")
        return LLMResponse(
            text=text.strip(),
            model=data.get("model", self.model),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            latency_s=latency,
            finish_reason=data.get("done_reason", "stop"),
            raw=data,
        )

    def health(self) -> bool:
        try:
            resp = self._client.get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()
