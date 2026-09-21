from __future__ import annotations

from voicerag.config import STTSettings, get_settings
from voicerag.stt.base import BaseSTT


def build_stt(settings: STTSettings | None = None) -> BaseSTT:
    cfg = settings or get_settings().stt
    if cfg.backend == "stub":
        from voicerag.stt.stub import StubSTT

        return StubSTT()
    if cfg.backend == "faster_whisper":
        from voicerag.stt.whisper import FasterWhisperSTT

        return FasterWhisperSTT(
            model=cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            beam_size=cfg.beam_size,
            vad_filter=cfg.vad_filter,
        )
    raise ValueError(f"Unknown STT backend: {cfg.backend!r}")
