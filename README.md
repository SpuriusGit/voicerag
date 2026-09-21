# VoiceRAG — voice-enabled RAG over local open-source models

A production-shaped retrieval-augmented question answering service you can run on
a single consumer GPU. Ask it a question by text or by voice; it transcribes,
retrieves, reranks, answers with inline citations, and refuses when the corpus
does not support an answer.

Everything is local and open-source: **faster-whisper** for speech,
**multilingual-e5** embeddings, a **bge cross-encoder** reranker, and a
**Qwen2.5-3B** chat model served by **Ollama** or any OpenAI-compatible runtime
(vLLM, TGI, llama.cpp).

```
                    ┌──────────────┐
  audio ──────────▶ │ faster-whisper│──┐
                    └──────────────┘  │  transcript
                                      ▼
  text ────────────────────────▶ ┌─────────┐   query
                                 │ retriever│──────────┐
                                 └─────────┘          │
              ┌───────────────────────┬───────────────┘
              ▼                       ▼
        dense (e5 + cosine)     sparse (BM25)
              └────────► RRF fusion ◄─┘
                            │ top_k = 20
                            ▼
                   cross-encoder rerank
                            │ top_n = 4
                            ▼
                 versioned prompt (rag_answer@v3)
                            │
                            ▼
                    LLM ──▶ answer + [S1] citations
                            │
                   Prometheus: per-stage latency, GPU/RAM, tokens, RTF
```

## Why this exists

It is a portfolio project built around the parts of an LLM engineer's job that
are easy to claim and hard to fake: measuring retrieval quality, versioning
prompts like code, watching GPU memory, and being honest about what a
fine-tune did or did not improve.

## Quick start (no GPU, no downloads, ~30 seconds)

The repository ships dependency-free stand-ins for every heavy component — a
hashed-ngram embedder, an echo LLM and a stub STT — so the full pipeline runs
before you download a single model weight.

```bash
make install          # CPU dependencies only
make demo             # ingest -> search -> evaluate, end to end
make test             # 79 tests, ~1 s
```

`make demo` prints retrieval results and a metrics table. Nothing is mocked
except the three model backends: the chunker, hybrid retriever, prompt registry,
evaluation harness and metrics are the real ones.

## Running it for real

```bash
make install-ml                                    # torch, transformers, whisper, peft
ollama pull qwen2.5:3b-instruct                    # or point at a vLLM server
cp .env.example .env                               # defaults are already sensible
make ingest                                        # build the index with e5 embeddings
make serve                                         # http://localhost:8000/docs
```

```bash
voicerag ask "How much VRAM does an RTX 4060 laptop GPU have?"
voicerag ask "Скільки відеопам'яті потрібно для моделі small?" --prompt rag_answer@v2
voicerag search "reranking" --top-n 3              # retrieval only, for debugging
voicerag transcribe recording.m4a
voicerag eval --judge                              # full quality report
voicerag bench --runs 20                           # latency + peak GPU/RAM
voicerag prompts diff rag_answer@v1 rag_answer@v3
```

### Docker

```bash
make docker-up        # api + ollama + prometheus + grafana
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5:3b-instruct
curl localhost:8000/health
```

The CPU image is a multi-stage build (~700 MB, non-root, healthchecked). The GPU
variant is `docker/Dockerfile.gpu`; run it with the `gpu` compose profile.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/ask` | Question → grounded answer with citations and per-stage timings |
| `POST` | `/voice-ask` | Audio upload → transcript → answer |
| `POST` | `/transcribe` | STT only, with real-time factor and confidence |
| `POST` | `/search` | Retrieval only — is a miss retrieval or generation? |
| `GET` | `/prompts` | Every prompt version with its content hash |
| `GET` | `/prompts/diff?a=…&b=…` | Unified diff between two prompt versions |
| `GET` | `/health` | Index stats, LLM reachability, live GPU/RAM |
| `GET` | `/metrics` | Prometheus exposition |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How much VRAM does an RTX 4060 have?","prompt_ref":"rag_answer@v3"}' | jq
```

Response shape (values illustrative — timings depend on your hardware):

```json
{
  "answer": "An RTX 4060 laptop GPU has 8 GB of VRAM. [S1]",
  "refused": false,
  "sources": [{"source": "04_gpu_capacity.md", "score": 0.71, "rerank_score": 0.94, "cited": true}],
  "prompt": {"ref": "rag_answer@v3", "sha256": "142d394aa474ed8c"},
  "timings_ms": {"embed": 11.4, "search": 2.1, "rerank": 178.6, "llm": 1902.3, "total": 2094.4},
  "usage": {"prompt_tokens": 812, "completion_tokens": 24, "tokens_per_second": 12.6}
}
```

Every answer carries the prompt reference **and its content hash**, so any logged
response can be traced back to the exact wording that produced it.

## Retrieval quality

Measured, not asserted. `python scripts/run_ablation.py` rebuilds an index per
configuration and evaluates all of them on the same 19 ground-truth questions.
Results below are from the zero-dependency embedder so anyone can reproduce them
in seconds; the ranking of the variants is what matters, not the absolute values.

| variant | chunks | hit@4 | hit@1 | MRR | nDCG@4 |
| --- | --- | --- | --- | --- | --- |
| dense only, 800/120 | 22 | 0.895 | 0.684 | 0.789 | 0.813 |
| **hybrid (dense + BM25), 800/120 — shipped** | 22 | **0.947** | **0.789** | **0.868** | **0.889** |
| hybrid, 400/60 | 36 | 0.895 | 0.789 | 0.833 | 0.849 |
| hybrid, 1600/200 | 22 | 0.947 | 0.789 | 0.868 | 0.889 |

Adding BM25 to dense retrieval is worth 5 points of hit@4 and 8 of MRR on this
corpus. Over-splitting (400-char chunks) hurts: it fragments the passage that
answers the question. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for method,
caveats and the metrics that still need a real LLM to fill in.

## Answer quality

Three layers, because each catches something the others miss:

* **deterministic** — token recall against a reference, citation validity, answer
  length. Free, stable, runs on every commit.
* **behavioural** — refusal accuracy, scored in both directions: answering an
  out-of-corpus question is a hallucination, refusing an answerable one is a
  failure. The eval set contains 4 deliberately unanswerable questions.
* **LLM-as-a-judge** — faithfulness of each answer to the retrieved context,
  graded by a versioned judge prompt at temperature 0.

```bash
voicerag eval --judge --prompt rag_answer@v2 --run-id v2
voicerag eval --judge --prompt rag_answer@v3 --run-id v3
voicerag compare runs/eval/v2.json runs/eval/v3.json
```

Each report embeds the configuration that produced it — prompt ref and hash,
models, `top_k`/`top_n`, index size — so two runs are always comparable.

## Prompts are versioned like code

```
prompts/rag_answer/v1.yaml   baseline, no grounding constraint  (deprecated)
prompts/rag_answer/v2.yaml   + citations + refusal path         (superseded)
prompts/rag_answer/v3.yaml   + language mirroring + length cap  (production)
```

New wording means a new file, never an edit. Each version carries its rationale,
a changelog and the command that evaluates it. Pin one per request with
`{"prompt_ref": "rag_answer@v2"}`, diff two with `voicerag prompts diff`.

## Monitoring

`/metrics` exposes per-stage latency histograms (`embed`, `search`, `rerank`,
`llm`, `stt`), token counters, STT real-time factor, error counters by stage and
exception type, plus live GPU memory, GPU utilisation and process RSS. Alert
rules for latency, GPU pressure and error rate ship in `docker/alerts.yml`.

Per-stage labelling is the point: a single end-to-end number never tells you
whether the reranker fell back to CPU or the context simply got longer.

```bash
voicerag bench --runs 20 --out runs/bench/latest.json   # p50/p95/max + peak VRAM
```

## LoRA fine-tuning

```bash
make finetune-data                                    # build the SFT set from the corpus
python finetune/train_lora.py --config finetune/configs/lora_qwen3b.yaml
python finetune/eval_lora.py --adapter runs/lora/qwen3b-rag-style --base Qwen/Qwen2.5-3B-Instruct
```

QLoRA, rank 16, 4-bit NF4, gradient checkpointing — sized for 8 GB. Training
examples are rendered with the *same versioned prompt the service uses at
inference*, because a format mismatch between training and serving is the
fastest way to waste a fine-tune. The training run writes a `run_card.json` with
loss, peak VRAM, wall-clock time and adapter size; the adapter is compared
against the base model before it may be promoted.
See [docs/FINETUNING.md](docs/FINETUNING.md).

## Project layout

```
src/voicerag/
  config.py          typed settings, env-overridable (VOICERAG_SECTION__FIELD)
  llm/               ollama | openai-compatible | echo, behind one interface
  stt/               faster-whisper, audio decoding/resampling, stub
  prompts/           versioned registry with content hashing and diffing
  rag/               chunking, embeddings, vector store, BM25, RRF, reranking, pipeline
  eval/              retrieval metrics, answer metrics, LLM judge, report runner
  monitoring/        Prometheus metrics, GPU/RAM sampling
  api/               FastAPI service
  cli.py             voicerag ingest | ask | search | eval | bench | serve | prompts
finetune/            dataset prep, QLoRA training, adapter eval, merge/export
scripts/             retrieval ablation study
docker/              CPU + GPU images, compose stack, Prometheus rules
docs/                architecture, experiments, evaluation, operations, fine-tuning
tests/               79 tests, no GPU or network required
```

## Design decisions worth defending

- **Hybrid retrieval over pure dense.** Embeddings miss exact identifiers —
  model names, CLI flags, error codes — which is most of what people ask a
  technical knowledge base. Fusing by rank (RRF) avoids calibrating cosine
  similarity against BM25 scores, which live on unrelated scales.
- **Heading-aware chunking.** A chunk reading "8 GB is the practical limit" is
  useless without its heading path; the path is prepended before embedding.
- **Refusal as a measured behaviour, not a hope.** The eval set contains
  unanswerable questions and the metric penalises both over-answering and
  over-refusing.
- **Stub backends as first-class code.** CI runs the real pipeline with fake
  models rather than mocking the pipeline, so a chunking or fusion regression
  fails the build.
- **Prompt hashes in every response.** Without them, "which prompt produced this
  bad answer" is unanswerable a week later.

## Known limitations

- The corpus is a small synthetic handbook (6 documents, 22 chunks). Metrics on
  it are directional; the harness, not the numbers, is the deliverable.
- Faithfulness and refusal metrics need a real local LLM; with the echo backend
  they are structurally zero. `recorded: null` in the prompt files means exactly
  that — no number has been measured yet on that axis, and none was invented.
- The LoRA scripts are written for a GPU box and validated by dry-run and config
  parsing here; no adapter has been trained in this repository.
- No streaming responses, no multi-tenant auth, no incremental re-indexing.

## Requirements

Python 3.10+. CPU-only for the test suite and the demo; an 8 GB GPU for the full
stack (see `data/corpus/04_gpu_capacity.md` for the memory budget the defaults
are built around). `ffmpeg` is needed to decode compressed audio uploads.

## License

MIT.
