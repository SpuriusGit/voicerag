"""Audio preparation for STT.

Whisper-family models expect 16 kHz mono PCM. Browser uploads arrive as webm/ogg
opus, phone recordings as m4a — so anything that is not already a readable wav is
converted through ffmpeg, and stereo is downmixed rather than silently truncated
to the first channel.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16_000


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray  # float32, mono, in [-1, 1]
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    @property
    def rms_dbfs(self) -> float:
        """Loudness check — near-silent uploads are the most common 'STT is broken' report."""
        if not len(self.samples):
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(self.samples))))
        return 20 * np.log10(max(rms, 1e-10))


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _read_wav(path: Path) -> AudioClip:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    dtypes: dict[int, type[np.signedinteger]] = {1: np.int8, 2: np.int16, 4: np.int32}
    if width not in dtypes:
        raise AudioError(f"Unsupported PCM sample width: {width} bytes")
    dtype = dtypes[width]
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    data /= float(np.iinfo(dtype).max)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return AudioClip(samples=data, sample_rate=rate)


def _ffmpeg_to_wav(src: Path, dst: Path, sample_rate: int) -> None:
    if not has_ffmpeg():
        raise AudioError(
            f"Cannot decode {src.suffix or 'audio'}: ffmpeg is not installed "
            "(apt-get install ffmpeg, or upload 16 kHz mono wav)."
        )
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    if result.returncode != 0:
        raise AudioError(f"ffmpeg failed: {result.stderr.strip()[:400]}")


def resample_linear(clip: AudioClip, target_rate: int = TARGET_SAMPLE_RATE) -> AudioClip:
    """Cheap resampler used when ffmpeg is unavailable but the wav is readable."""
    if clip.sample_rate == target_rate or not len(clip.samples):
        return clip
    ratio = target_rate / clip.sample_rate
    new_length = int(round(len(clip.samples) * ratio))
    source_grid = np.linspace(0, len(clip.samples) - 1, num=len(clip.samples))
    target_grid = np.linspace(0, len(clip.samples) - 1, num=new_length)
    return AudioClip(
        samples=np.interp(target_grid, source_grid, clip.samples).astype(np.float32),
        sample_rate=target_rate,
    )


def load_audio(path: Path | str, sample_rate: int = TARGET_SAMPLE_RATE) -> AudioClip:
    """Decode any audio file into mono float32 at ``sample_rate``."""
    src = Path(path)
    if not src.exists():
        raise AudioError(f"Audio file not found: {src}")

    if src.suffix.lower() == ".wav":
        try:
            return resample_linear(_read_wav(src), sample_rate)
        except (wave.Error, AudioError):
            pass  # compressed data in a .wav container -> fall through to ffmpeg

    with tempfile.TemporaryDirectory() as tmp:
        converted = Path(tmp) / "audio.wav"
        _ffmpeg_to_wav(src, converted, sample_rate)
        return _read_wav(converted)


def save_wav(clip: AudioClip, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(clip.samples, -1.0, 1.0)
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(clip.sample_rate)
        wav.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return out


def synth_tone(
    duration_s: float = 1.0, freq: float = 440.0, sample_rate: int = TARGET_SAMPLE_RATE
) -> AudioClip:
    """Deterministic test signal — lets audio tests run without fixture files."""
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    return AudioClip(
        samples=(0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sample_rate=sample_rate
    )
