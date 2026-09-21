"""Prometheus metrics for pipeline latency, throughput and errors.

Latency is recorded *per stage* (embed / search / rerank / llm / stt) because a
single end-to-end number never tells you which component regressed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

# Buckets chosen for interactive RAG: sub-second retrieval, multi-second generation.
_LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0)

STAGE_LATENCY = Histogram(
    "voicerag_stage_latency_seconds",
    "Wall-clock latency of a single pipeline stage.",
    labelnames=("stage", "model"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

REQUESTS = Counter(
    "voicerag_requests_total",
    "Requests handled, by route and outcome.",
    labelnames=("route", "status"),
    registry=REGISTRY,
)

ERRORS = Counter(
    "voicerag_errors_total",
    "Unhandled errors, by stage and exception type.",
    labelnames=("stage", "error"),
    registry=REGISTRY,
)

TOKENS = Counter(
    "voicerag_llm_tokens_total",
    "Tokens reported by the LLM backend.",
    labelnames=("model", "kind"),  # kind: prompt | completion
    registry=REGISTRY,
)

AUDIO_SECONDS = Counter(
    "voicerag_stt_audio_seconds_total",
    "Total audio duration submitted to the STT backend.",
    labelnames=("model",),
    registry=REGISTRY,
)

RTF = Histogram(
    "voicerag_stt_real_time_factor",
    "STT real-time factor (processing time / audio duration); lower is better.",
    labelnames=("model",),
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 3.0),
    registry=REGISTRY,
)

FUSION_DECISIONS = Counter(
    "voicerag_fusion_decisions_total",
    "Whether BM25 was fused into the candidate list, and why not when it was skipped.",
    labelnames=("decision",),
    registry=REGISTRY,
)

GPU_MEMORY_USED = Gauge(
    "voicerag_gpu_memory_used_bytes",
    "GPU memory currently used, per device.",
    labelnames=("device",),
    registry=REGISTRY,
)

GPU_UTILIZATION = Gauge(
    "voicerag_gpu_utilization_percent",
    "GPU utilization percentage, per device.",
    labelnames=("device",),
    registry=REGISTRY,
)

PROCESS_RSS = Gauge(
    "voicerag_process_rss_bytes",
    "Resident set size of the serving process.",
    registry=REGISTRY,
)


@contextmanager
def observe_stage(stage: str, model: str = "-") -> Iterator[dict[str, float]]:
    """Time a pipeline stage and export it as a Prometheus observation.

    Yields a dict that receives ``elapsed_s`` so callers can reuse the timing in
    their API response without measuring twice.
    """
    holder: dict[str, float] = {}
    started = time.perf_counter()
    try:
        yield holder
    except Exception as exc:  # noqa: BLE001 - re-raised after accounting
        ERRORS.labels(stage=stage, error=type(exc).__name__).inc()
        raise
    finally:
        elapsed = time.perf_counter() - started
        holder["elapsed_s"] = elapsed
        STAGE_LATENCY.labels(stage=stage, model=model).observe(elapsed)


def track(route: str, status: str) -> None:
    REQUESTS.labels(route=route, status=status).inc()


def record_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    if prompt_tokens:
        TOKENS.labels(model=model, kind="prompt").inc(prompt_tokens)
    if completion_tokens:
        TOKENS.labels(model=model, kind="completion").inc(completion_tokens)


def record_stt(model: str, audio_seconds: float, processing_seconds: float) -> None:
    AUDIO_SECONDS.labels(model=model).inc(audio_seconds)
    if audio_seconds > 0:
        RTF.labels(model=model).observe(processing_seconds / audio_seconds)


def record_fusion(decision: str) -> None:
    """A rising skip rate means queries are arriving in a language the index lacks."""
    FUSION_DECISIONS.labels(decision=decision).inc()


def render_prometheus() -> bytes:
    """Serialize the registry; refreshes resource gauges first."""
    from voicerag.monitoring.resources import snapshot_resources

    snap = snapshot_resources()
    PROCESS_RSS.set(snap.process_rss_bytes)
    for gpu in snap.gpus:
        GPU_MEMORY_USED.labels(device=str(gpu.index)).set(gpu.memory_used_bytes)
        GPU_UTILIZATION.labels(device=str(gpu.index)).set(gpu.utilization_percent)
    return generate_latest(REGISTRY)
