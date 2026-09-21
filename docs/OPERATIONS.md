# Operations

## Deploying

```bash
make docker-up
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5:3b-instruct
curl -fsS localhost:8000/health | jq
```

The stack is API + Ollama + Prometheus + Grafana. The API image is multi-stage,
runs as uid 10001, and has a healthcheck that hits `/health`. The index is built
on first start if the mounted volume is empty.

GPU: `docker compose -f docker/docker-compose.yml --profile gpu up -d api-gpu`.
Model weights live on a mounted volume rather than in the image — baking them in
adds gigabytes and forces a rebuild on every model change.

## What to watch

| metric | alert at | why |
| --- | --- | --- |
| `voicerag_stage_latency_seconds{stage="llm"}` p95 | > 6 s | user-visible slowness |
| `voicerag_gpu_memory_used_bytes` | > 7.5 GB on an 8 GB card | OOM during generation is imminent |
| `voicerag_stt_real_time_factor` p95 | > 0.6 | Whisper likely fell back to CPU |
| `voicerag_errors_total` rate | > 0.2/s | backend unreachable or bad input surge |

Rules ship in `docker/alerts.yml`. Latency is labelled per stage on purpose: an
end-to-end number tells you something is slow, never what.

## Runbook

### Answers are refused too often

1. `GET /health` → does `index.chunks` match the corpus? A stale index after a
   corpus update is the most common cause. Re-run `make ingest`.
2. `POST /search` with the same query. If the expected chunk is not among the
   candidates, the problem is **retrieval**: check that `index.embedder` matches
   the configured embedder. A mismatch is detected at load time and raises, but
   an index built with a *different revision* of the same model will not raise —
   it will just rank badly.
3. If the chunk is there but the answer still refuses, the problem is
   **generation**: `GET /prompts/diff?a=…&b=…` and check the temperature.

### Latency regressed

Read the per-stage histograms. `rerank` up alone → the cross-encoder is on CPU.
`llm` up while everything else is flat → longer context; check `top_n` and chunk
sizes. `embed` up → the embedder loaded on CPU.

### Out of memory

At load time the model is simply too large for the card. *During generation* the
weights fit but activations do not: lower `top_n`, shorten the context, or move
the reranker to CPU (`VOICERAG_RAG__RERANKER=noop` as the emergency stop).

### Transcripts are empty or nonsense

Check `mean_confidence` and the upload's loudness — below −45 dBFS is effectively
silence, and Whisper hallucinates text on silence. Confirm `ffmpeg` is present
in the container; without it, compressed uploads cannot be decoded and the
service answers 415.

### Backend unreachable

`/health` reports `llm.reachable: false` and `/ask` answers 503 with a request
id. Check that Ollama has the model pulled: the container starts healthy with no
models at all.

## Logs

Structured JSON with a `request_id` on every line, bound per request and
returned to the client as `X-Request-ID`:

```json
{"event": "answered", "prompt_ref": "rag_answer@v3", "prompt_sha": "142d394aa474ed8c",
 "refused": false, "sources": 4, "t_embed": 11.4, "t_rerank": 178.6, "t_llm": 1902.3,
 "request_id": "a3f2b1c4d5e6", "level": "info", "timestamp": "2026-09-21T10:02:11Z"}
```

The prompt hash on every answer is what makes "which wording produced this" a
question with an answer.

## Updating the corpus

```bash
make ingest          # rebuild; chunk ids are content-addressed, so this is idempotent
curl -X POST localhost:8000/... # (no reload endpoint: restart the service)
```

Re-ingesting unchanged files produces identical chunk ids and changes nothing.
The service caches the index at startup, so it must be restarted to pick up a
rebuilt index — deliberate, because a half-written index should never be served.
