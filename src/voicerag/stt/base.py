"""STT interface and result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    def as_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "text": self.text,
            "avg_logprob": round(self.avg_logprob, 3),
            "no_speech_prob": round(self.no_speech_prob, 3),
        }


@dataclass
class Transcript:
    text: str
    language: str
    duration_s: float
    processing_s: float
    model: str
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def real_time_factor(self) -> float:
        """<1.0 means faster than real time — the number that decides if live use is viable."""
        return self.processing_s / self.duration_s if self.duration_s else 0.0

    @property
    def mean_confidence(self) -> float:
        """Mean segment log-prob mapped into (0, 1]; a cheap transcription-quality signal."""
        if not self.segments:
            return 0.0
        import math

        return sum(math.exp(s.avg_logprob) for s in self.segments) / len(self.segments)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "duration_s": round(self.duration_s, 2),
            "processing_s": round(self.processing_s, 2),
            "real_time_factor": round(self.real_time_factor, 3),
            "mean_confidence": round(self.mean_confidence, 3),
            "model": self.model,
            "segments": [s.as_dict() for s in self.segments],
        }


class BaseSTT(ABC):
    model: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: Path | str, language: str | None = None) -> Transcript: ...
