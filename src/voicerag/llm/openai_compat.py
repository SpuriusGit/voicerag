"""OpenAI-compatible backend: vLLM, TGI, llama.cpp server, LM Studio, gateways.

Deliberately uses raw ``httpx`` instead of the ``openai`` SDK — the payload is
small, and it keeps the container image free of an extra dependency tree.
"""

from __future__ import annotations

import time

import httpx

from voicerag.llm.base import BaseLLM, ChatMessage, LLMError, LLMResponse


class OpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "",
        timeout_s: float = 120.0,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s, headers=headers)

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
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI-compatible request failed: {exc}") from exc
        latency = time.perf_counter() - started

        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Malformed completion payload: {data}") from exc

        usage = data.get("usage") or {}
        return LLMResponse(
            text=(text or "").strip(),
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=latency,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    def health(self) -> bool:
        try:
            return self._client.get("/models", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()
