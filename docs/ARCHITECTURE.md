# Architecture

## Request paths

```
POST /ask          question ─────────────────────────────┐
POST /voice-ask    audio ─▶ STT ─▶ transcript ────────────┤
                                                          ▼
                                                    Retriever
                                     ┌────────────────────┴────────────────────┐
                                     ▼                                         ▼
                          embed query (e5)                              BM25 over chunks
                                     │                                         │
                             cosine top_k=20                          lexical top_k=20
                                     └──────────── RRF (k=60) ─────────────────┘
                                                          │
                                              cross-encoder rerank → top_n=4
                                                          │
                                            PromptRegistry.get("rag_answer@latest")
                                                          │
                                                   BaseLLM.chat()
                                                          │
                                        answer + [Sn] citations + timings + usage
```

Each labelled arrow is a Prometheus histogram observation (`stage=embed|search|
rerank|llm|stt`), which is what makes a latency regression attributable.

## Layers and their contracts

| Layer | Contract | Implementations |
| --- | --- | --- |
| `llm/` | `BaseLLM.chat(messages) -> LLMResponse` | Ollama, OpenAI-compatible, echo |
| `stt/` | `BaseSTT.transcribe(path) -> Transcript` | faster-whisper, stub |
| `rag/embeddings.py` | `BaseEmbedder.embed_{passages,queries}` | sentence-transformers, hashing |
| `rag/reranker.py` | `BaseReranker.rerank(query, candidates, top_n)` | cross-encoder, noop |
| `rag/store.py` | `VectorStore.search(vector, top_k)` | numpy matmul, FAISS above 20k vectors |
| `prompts/` | `PromptRegistry.get(ref) -> PromptTemplate` | YAML files on disk |

Nothing above the interface knows which implementation it got. That is what
makes the same code run on a laptop GPU, on a vLLM server, and in CI with no
weights at all — and it is why the test suite exercises the real pipeline
instead of mocking it.

## Configuration

`Settings` (pydantic-settings) is the single source of truth. Every field is
overridable as `VOICERAG_<SECTION>__<FIELD>`, so the same image serves local
development, compose and CI:

```bash
VOICERAG_RAG__TOP_N=6 VOICERAG_LLM__MODEL=qwen2.5:7b voicerag ask "..."
```

`resolve_device("auto")` picks CUDA when torch reports it, CPU otherwise —
computed once per component, never per request.

## Index format

Two files plus metadata, deliberately inspectable:

```
storage/index/
  vectors.npz     float32 [n, dim], L2-normalised (cosine == dot product)
  chunks.jsonl    one JSON object per chunk: id, doc_id, text, source, position, heading
  meta.json       dimension, embedder name, chunk count
```

`meta.json` records which embedder built the index. Loading it with a different
embedder raises a clear error instead of silently returning meaningless
similarities — a failure mode that is otherwise very hard to diagnose, because
nothing crashes and the answers are merely wrong.

## Model lifecycle

Heavy components are cached per process (`functools.lru_cache` in `api/deps.py`)
and shared across requests. Reloading a cross-encoder per request would dominate
latency and exhaust VRAM. `reset_caches()` exists for tests and for reloading
after re-ingestion.

## Error handling

| Exception | HTTP | Meaning |
| --- | --- | --- |
| `PromptError` | 400 | Unknown prompt name or version, or a missing variable |
| `AudioError` | 415 | Undecodable upload, usually a missing ffmpeg |
| `LLMError` | 503 | Model backend unreachable or returned nothing usable |
| anything else | 500 | Logged with the request id, never leaked to the client |

Every response carries `X-Request-ID`, which is also bound to every structured
log line emitted while handling it.

## What is deliberately not here

- **Streaming.** The evaluation harness needs complete answers; streaming would
  be a serving-layer addition, not an architectural change.
- **Incremental re-indexing.** The corpus is small enough to rebuild; content
  addressed chunk ids already make re-ingestion idempotent.
- **A chunk-level relevance ground truth.** Chunk ids change whenever the
  chunker is re-tuned, which would make historical eval runs incomparable, so
  relevance is recorded per source document.
