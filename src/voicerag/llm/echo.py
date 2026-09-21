"""Deterministic stub backend.

Keeps unit tests and CI free of model weights while still exercising the full
prompt -> pipeline -> response path. It echoes a quoted fragment of the context
so retrieval regressions still surface in tests.
"""

from __future__ import annotations

import re
import time

from voicerag.llm.base import BaseLLM, ChatMessage, LLMResponse


class EchoLLM(BaseLLM):
    def __init__(self, model: str = "echo", canned: str | None = None) -> None:
        super().__init__(model)
        self.canned = canned
        self.calls: list[list[ChatMessage]] = []

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        self.calls.append(messages)
        if self.canned is not None:
            text = self.canned
        else:
            joined = "\n".join(m.content for m in messages)
            first_sentence = next(
                (s.strip() for s in re.split(r"(?<=[.!?])\s", joined) if len(s.strip()) > 20),
                joined[:200],
            )
            text = f"[echo] {first_sentence[:300]}"
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text.split()),
            latency_s=time.perf_counter() - started,
        )
