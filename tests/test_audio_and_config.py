import numpy as np
import pytest

from voicerag.config import LLMSettings, RAGSettings, Settings, resolve_device
from voicerag.llm.factory import build_llm
from voicerag.monitoring.resources import ResourceSampler, snapshot_resources
from voicerag.stt.audio import (
    AudioClip,
    AudioError,
    load_audio,
    resample_linear,
    save_wav,
    synth_tone,
)
from voicerag.stt.factory import build_stt


def test_wav_roundtrip_preserves_duration_and_rate(tmp_path):
    original = synth_tone(duration_s=1.5, freq=220.0)
    clip = load_audio(save_wav(original, tmp_path / "tone.wav"))
    assert clip.sample_rate == 16_000
    assert clip.duration_s == pytest.approx(1.5, abs=0.01)
    assert np.allclose(clip.samples[:100], original.samples[:100], atol=1e-3)


def test_loudness_detects_near_silence():
    assert synth_tone(0.5).rms_dbfs > -30
    silent = AudioClip(samples=np.zeros(8000, dtype=np.float32), sample_rate=16_000)
    assert silent.rms_dbfs < -100


def test_resampling_changes_length_but_not_duration():
    clip = AudioClip(samples=synth_tone(1.0, sample_rate=8000).samples, sample_rate=8000)
    resampled = resample_linear(clip, 16_000)
    assert resampled.sample_rate == 16_000
    assert len(resampled.samples) == pytest.approx(len(clip.samples) * 2, abs=2)
    assert resampled.duration_s == pytest.approx(clip.duration_s, abs=0.01)


def test_missing_audio_file_raises_audio_error(tmp_path):
    with pytest.raises(AudioError):
        load_audio(tmp_path / "nope.wav")


def test_stub_stt_reports_duration_and_confidence(tmp_path):
    from voicerag.config import STTSettings

    path = save_wav(synth_tone(2.0), tmp_path / "s.wav")

    result = build_stt(STTSettings(backend="stub")).transcribe(path)
    assert result.duration_s == pytest.approx(2.0, abs=0.05)
    assert 0 < result.mean_confidence <= 1.0
    assert result.as_dict()["real_time_factor"] >= 0


def test_settings_read_nested_environment_variables(monkeypatch):
    monkeypatch.setenv("VOICERAG_LLM__MODEL", "qwen2.5:7b")
    monkeypatch.setenv("VOICERAG_RAG__TOP_N", "9")
    settings = Settings(_env_file=None)
    assert settings.llm.model == "qwen2.5:7b"
    assert settings.rag.top_n == 9


def test_settings_reject_invalid_values():
    with pytest.raises(ValueError):
        RAGSettings(top_k=0)
    with pytest.raises(ValueError):
        LLMSettings(backend="not_a_backend")


def test_resolve_device_returns_a_concrete_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_llm_factory_builds_the_configured_backend():
    assert build_llm(LLMSettings(backend="echo", model="echo")).model == "echo"
    with pytest.raises(ValueError):
        build_llm(LLMSettings.model_construct(backend="nope", model="x"))


def test_resource_snapshot_has_memory_fields():
    snapshot = snapshot_resources().as_dict()
    assert snapshot["process_rss_mb"] > 0
    assert snapshot["system_ram_total_mb"] > 0
    assert isinstance(snapshot["gpus"], list)


def test_resource_sampler_collects_samples():
    with ResourceSampler(interval_s=0.05) as sampler:
        sum(i * i for i in range(200_000))
    assert sampler.samples
    assert sampler.summary()["peak_rss_mb"] > 0
