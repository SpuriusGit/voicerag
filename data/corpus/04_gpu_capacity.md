# GPU Capacity Planning

## Laptop cards

An RTX 4060 laptop GPU has 8 GB of VRAM. A working configuration on that card is
a 3B chat model at 4-bit (about 2.6 GB), the bge-reranker-v2-m3 cross-encoder
(about 1.1 GB), the multilingual-e5-small embedder (about 0.2 GB) and
faster-whisper small at int8_float16 (about 1.2 GB). That totals near 5.1 GB and
leaves headroom for activations at a 4k context.

Loading a 7B model alongside the reranker on the same 8 GB card causes
out-of-memory errors during generation, not at load time, which makes the
failure look intermittent.

## Server cards

A single 24 GB card such as an RTX 4090 or L4 runs a 7B model with vLLM at
roughly 40 concurrent requests before time-to-first-token exceeds one second.
Set `gpu_memory_utilization` to 0.85 rather than the default 0.90 when the same
card also hosts the reranker.

## Monitoring thresholds

Alert when GPU memory usage stays above 92 percent for five minutes, when p95
end-to-end answer latency exceeds 6 seconds, or when the STT real-time factor
exceeds 0.6. Metrics are exported in Prometheus format at /metrics and are
labelled per pipeline stage, because a single end-to-end number never identifies
which component regressed.

## CPU fallback

Everything runs on CPU without code changes, roughly 8 to 12 times slower for
generation. The embedder and BM25 stay usable on CPU; the cross-encoder does not
and should be switched to the noop reranker when no GPU is present.
