"""faster-whisper backend (CTranslate2 build of Whisper).

Chosen over ``openai-whisper`` because it is ~4x faster at the same WER and fits
``small``/``medium`` into 8 GB of VRAM at int8_float16 — see docs/EXPERIMENTS.md
for the accuracy/latency table that motivated the default.
"""

from __future__ import annotations

import time
from pathlib import Path

from voicerag.config import resolve_device
from voicerag.monitoring.metrics import observe_stage, record_stt
from voicerag.stt.base import BaseSTT, Transcript, TranscriptSegment


class FasterWhisperSTT(BaseSTT):
    def __init__(
        self,
        model: str = "small",
        device: str = "auto",
        compute_type: str = "int8_float16",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model = model
        self.device = resolve_device(device)  # type: ignore[arg-type]
        # int8_float16 is a CUDA-only combination; CPU needs plain int8.
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = WhisperModel(model, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio_path: Path | str, language: str | None = None) -> Transcript:
        started = time.perf_counter()
        with observe_stage("stt", self.model):
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                language=language or None,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                condition_on_previous_text=False,  # prevents repetition loops on noisy audio
            )
            segments = [
                TranscriptSegment(
                    start_s=s.start,
                    end_s=s.end,
                    text=s.text.strip(),
                    avg_logprob=getattr(s, "avg_logprob", 0.0),
                    no_speech_prob=getattr(s, "no_speech_prob", 0.0),
                )
                for s in segments_iter
            ]
        processing_s = time.perf_counter() - started
        duration_s = float(getattr(info, "duration", 0.0))
        record_stt(self.model, duration_s, processing_s)

        return Transcript(
            text=" ".join(s.text for s in segments).strip(),
            language=getattr(info, "language", language or "unknown"),
            duration_s=duration_s,
            processing_s=processing_s,
            model=self.model,
            segments=segments,
        )
