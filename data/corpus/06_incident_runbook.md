# Incident Runbook

## The service answers "I cannot answer this from the available documents" too often

Check the index first. Call /health and confirm the chunk count matches the
corpus. An empty or stale index is the most common cause, usually after the
corpus was updated without re-running ingestion.

If the index is healthy, call /search with the same query. If the expected chunk
is absent from the candidates, the problem is retrieval: check that the embedder
named in the index metadata matches the configured one. A mismatched embedder
produces meaningless similarities rather than an error.

If the chunk is present but the answer still refuses, the problem is generation:
compare prompt versions with /prompts/diff and check the temperature.

## Latency has increased

Read the per-stage histograms at /metrics. Rerank latency growing usually means
the cross-encoder fell back to CPU after a driver reset. LLM latency growing
while other stages are flat usually means a longer context: check the retrieved
chunk sizes and the top_n setting.

## Out-of-memory during generation

Reduce top_n, lower the context window, or move the reranker to CPU. Out-of-
memory at load time means the model is simply too large for the card; out-of-
memory during generation means activations no longer fit and the configuration
is too tight.

## Transcripts are empty or nonsense

Check the reported mean confidence and the loudness of the upload. Values below
-45 dBFS indicate an almost silent recording. Confirm ffmpeg is installed in the
container; without it, compressed uploads cannot be decoded.
