"""Offline STT stub for tests and CI."""

from __future__ import annotations

import time
from pathlib import Path

from voicerag.stt.audio import load_audio
from voicerag.stt.base import BaseSTT, Transcript, TranscriptSegment


class StubSTT(BaseSTT):
    def __init__(
        self, text: str = "what gpu memory does the service need", model: str = "stub"
    ) -> None:
        self.model = model
        self.text = text

    def transcribe(self, audio_path: Path | str, language: str | None = None) -> Transcript:
        started = time.perf_counter()
        clip = load_audio(audio_path)
        return Transcript(
            text=self.text,
            language=language or "en",
            duration_s=clip.duration_s,
            processing_s=time.perf_counter() - started,
            model=self.model,
            segments=[TranscriptSegment(0.0, clip.duration_s, self.text, avg_logprob=-0.2)],
        )
