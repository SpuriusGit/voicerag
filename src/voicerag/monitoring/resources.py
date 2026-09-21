"""GPU / RAM sampling.

Reads NVML when ``pynvml`` (shipped with torch) is importable and falls back to
parsing ``nvidia-smi`` so the same code works inside slim containers.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass(frozen=True)
class GPUStats:
    index: int
    name: str
    memory_used_bytes: int
    memory_total_bytes: int
    utilization_percent: float

    @property
    def memory_used_mb(self) -> float:
        return self.memory_used_bytes / 1024**2


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp: float
    process_rss_bytes: int
    system_ram_used_bytes: int
    system_ram_total_bytes: int
    cpu_percent: float
    gpus: list[GPUStats] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "process_rss_mb": round(self.process_rss_bytes / 1024**2, 1),
            "system_ram_used_mb": round(self.system_ram_used_bytes / 1024**2, 1),
            "system_ram_total_mb": round(self.system_ram_total_bytes / 1024**2, 1),
            "cpu_percent": self.cpu_percent,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "memory_used_mb": round(g.memory_used_mb, 1),
                    "memory_total_mb": round(g.memory_total_bytes / 1024**2, 1),
                    "utilization_percent": g.utilization_percent,
                }
                for g in self.gpus
            ],
        }


def _gpus_via_nvml() -> list[GPUStats]:
    import pynvml

    pynvml.nvmlInit()
    try:
        stats: list[GPUStats] = []
        for idx in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            stats.append(
                GPUStats(
                    index=idx,
                    name=name.decode() if isinstance(name, bytes) else str(name),
                    memory_used_bytes=int(mem.used),
                    memory_total_bytes=int(mem.total),
                    utilization_percent=float(util.gpu),
                )
            )
        return stats
    finally:
        pynvml.nvmlShutdown()


def _gpus_via_smi() -> list[GPUStats]:
    if not shutil.which("nvidia-smi"):
        return []
    query = "index,name,memory.used,memory.total,utilization.gpu"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if out.returncode != 0:
        return []
    stats: list[GPUStats] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, name, used_mb, total_mb, util = parts
        stats.append(
            GPUStats(
                index=int(idx),
                name=name,
                memory_used_bytes=int(float(used_mb) * 1024**2),
                memory_total_bytes=int(float(total_mb) * 1024**2),
                utilization_percent=float(util),
            )
        )
    return stats


def gpu_stats() -> list[GPUStats]:
    """Best-effort GPU stats; never raises, returns ``[]`` on CPU-only hosts."""
    try:
        return _gpus_via_nvml()
    except Exception:  # noqa: BLE001 - NVML missing/driver mismatch is expected
        pass
    try:
        return _gpus_via_smi()
    except Exception:  # noqa: BLE001
        return []


def snapshot_resources() -> ResourceSnapshot:
    proc = psutil.Process()
    vm = psutil.virtual_memory()
    return ResourceSnapshot(
        timestamp=time.time(),
        process_rss_bytes=proc.memory_info().rss,
        system_ram_used_bytes=vm.total - vm.available,
        system_ram_total_bytes=vm.total,
        cpu_percent=psutil.cpu_percent(interval=None),
        gpus=gpu_stats(),
    )


class ResourceSampler:
    """Background sampler used by benchmarks to report peak GPU/RAM usage.

    Usage::

        with ResourceSampler(interval_s=0.25) as sampler:
            run_workload()
        print(sampler.peak_gpu_memory_mb(), sampler.peak_rss_mb())
    """

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.samples: list[ResourceSnapshot] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ResourceSampler:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 4)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Sampling must never kill the workload it is measuring.
            with contextlib.suppress(Exception):
                self.samples.append(snapshot_resources())
            self._stop.wait(self.interval_s)

    def peak_rss_mb(self) -> float:
        if not self.samples:
            return 0.0
        return max(s.process_rss_bytes for s in self.samples) / 1024**2

    def peak_gpu_memory_mb(self) -> float:
        peak = 0.0
        for snap in self.samples:
            for gpu in snap.gpus:
                peak = max(peak, gpu.memory_used_mb)
        return peak

    def mean_gpu_utilization(self) -> float:
        values = [g.utilization_percent for s in self.samples for g in s.gpus]
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> dict:
        return {
            "samples": len(self.samples),
            "peak_rss_mb": round(self.peak_rss_mb(), 1),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb(), 1),
            "mean_gpu_utilization_percent": round(self.mean_gpu_utilization(), 1),
        }
