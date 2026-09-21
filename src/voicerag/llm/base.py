"""Backend-agnostic LLM interface.

The rest of the codebase depends only on :class:`BaseLLM`, so switching from a
local Ollama build to a vLLM server (or to a stub in CI) is a config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.latency_s <= 0 or not self.completion_tokens:
            return 0.0
        return self.completion_tokens / self.latency_s


class LLMError(RuntimeError):
    """Raised when a backend is unreachable or returns an unusable response."""


class BaseLLM(ABC):
    """Minimal synchronous chat interface."""

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a completion for ``messages``."""

    def complete(self, prompt: str, **kwargs: object) -> LLMResponse:
        """Convenience wrapper for single-turn prompts."""
        return self.chat([ChatMessage(role="user", content=prompt)], **kwargs)  # type: ignore[arg-type]

    def health(self) -> bool:
        """Cheap liveness probe used by ``/health``."""
        return True

    def close(self) -> None:  # pragma: no cover - default no-op
        return None
