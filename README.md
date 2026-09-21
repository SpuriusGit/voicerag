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

On that weak embedder, BM25 fusion is worth 5 points of hit@4 and over-splitting
(400-char chunks) clearly hurts.

### Then the same ablation with the real embedder reversed the first result

`multilingual-e5-small`, RTX 4060 Laptop, same 19 questions
(`runs/ablation/ablation-20260921-140004.json`):

| variant | hit@4 | hit@1 | MRR | nDCG@4 | mean query |
| --- | --- | --- | --- | --- | --- |
| **dense only, 800/120** | 1.000 | **1.000** | **1.000** | 0.980 | **6 ms** |
| hybrid, 800/120 (shipped) | 1.000 | 0.947 | 0.974 | 0.981 | 6 ms |
| hybrid, 400/60 | 1.000 | 0.842 | 0.921 | 0.942 | 8 ms |
| hybrid + cross-encoder rerank | 1.000 | **1.000** | **1.000** | **1.000** | 703 ms |

**hit@4 is saturated** — every variant scores 1.000, so the headline metric can
no longer rank anything and the eval set has to get harder. But `hit@1` still
discriminated, and it said hybrid retrieval was *worse* than plain dense, by
exactly one question:

> **q18** — *"Яка температура використовується для відповідей RAG?"*
> dense ranks the correct chunk first; hybrid ranks an unrelated one first.

The corpus is English and the question is Ukrainian, so BM25 has almost no
vocabulary in common with the query, its ranking is close to noise, and RRF —
which weighs purely by rank — gives that noise an equal vote.

### So the fusion step was gated, and re-measured

BM25 now joins only when at least 35% of the query's terms exist in the index
(`lexical_coverage`). No language detector: a query in a language the corpus
does not contain scores near zero by construction. The threshold was chosen
after measuring the statistic across all 23 questions — English spans 0.38–1.00,
Ukrainian 0.00–0.30 — not by intuition.

| variant | fused | hit@1 | MRR | nDCG@4 | mean query |
| --- | --- | --- | --- | --- | --- |
| dense only | 0 / 19 | 1.000 | 1.000 | 0.980 | 6.2 ms |
| hybrid, always fuse | 19 / 19 | 0.947 | 0.974 | 0.981 | 6.2 ms |
| **hybrid + lexical gate — shipped** | 16 / 19 | **1.000** | **1.000** | **1.000** | **6.1 ms** |
| gated hybrid + cross-encoder | 16 / 19 | 1.000 | 1.000 | 1.000 | 671 ms |

The gate skips exactly the three Ukrainian questions and beats both alternatives
on every metric at no latency cost. So the earlier result did not mean "hybrid
retrieval is a bad idea" — it meant "fusing unconditionally is a bad idea", a
distinction visible only because the failure was traced to a named question
instead of averaged away.

It also deflates the reranker: once BM25 stops breaking q18, the cross-encoder
has nothing left to fix — identical metrics for **665 ms, 42% of end-to-end
latency**. Measuring a component against a broken baseline flatters it.

Only the chunk-size finding replicates across both embedders: 400/60 is worst
in both. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for method and caveats.

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
voicerag eval --judge --judge-model qwen2.5:7b-instruct --prompt rag_answer@v3 --run-id v3
voicerag compare runs/eval/real-v1.json runs/eval/real-v2.json runs/eval/real-v3.json
```

The generator is graded by a **different** model than the one being evaluated —
a model judging its own answers is biased in its own favour, and the CLI warns
when no separate grader is given.

### Measured: what each prompt version actually changed

Generator `qwen2.5:3b-instruct` (q4, Ollama), judge `qwen2.5:7b-instruct`,
production retrieval stack, RTX 4060 Laptop 8 GB, 23 questions.

| metric | v1 baseline | v2 + citations/refusal | v3 production |
| --- | --- | --- | --- |
| hit@4 (retrieval) | 1.000 | 1.000 | 1.000 |
| **refusal accuracy** | 0.826 | 0.913 | **1.000** |
| out-of-corpus refused | **0 / 4** | 4 / 4 | 4 / 4 |
| citation rate | 0.000 | 0.737 | **0.842** |
| citation validity | n/a | 1.000 | 0.941 |
| faithfulness (judge) | 0.783 | *1.000* | 0.935 |
| mean answer length | 114.7 words | 29.9 | **24.9** |
| mean latency | 8915 ms | 3357 ms | **2427 ms** |

Two findings worth more than the table:

1. **v1 hallucinated exactly as predicted.** With no grounding constraint it
   answered all four unanswerable questions from parametric memory — including
   a flat *"The capital of Australia is Canberra."* Retrieval was identical
   across all three versions, so the failure is purely generation-side. This is
   what the `out_of_corpus` rows in the eval set exist to catch.
2. **v2 scores a perfect 1.000 faithfulness and is still the worse prompt.**
   It over-refuses answerable questions (multi-hop refusal accuracy 0.667), and
   the judge scores a refusal as fully supported — so refusing more *raises*
   faithfulness. Read alone, that metric would have promoted the wrong version.
   Faithfulness and refusal accuracy only mean something together.
3. **A fourth version was written, measured and rejected.** `v4` showed the
   citation format by example instead of describing it. It worked: zero
   malformed tags, validity 1.000 against v3's 0.941 — and citation rate
   collapsed from 0.895 to 0.37, because the model stopped attributing on more
   than half the questions. It is kept in the repository as
   `status: rejected` with its numbers. Writing the fix also exposed that the
   defect had been misdiagnosed and that one of the two citation metrics was
   measuring nothing ([E7](docs/EXPERIMENTS.md)).

Each report embeds the configuration that produced it — prompt ref and hash,
generator, judge, `top_k`/`top_n`, index size — so two runs are always
comparable. Raw reports: `runs/eval/real-v{1,2,3}.json`.

## Prompts are versioned like code

```
prompts/rag_answer/v1.yaml   baseline, no grounding constraint  (deprecated)
prompts/rag_answer/v2.yaml   + citations + refusal path         (superseded)
prompts/rag_answer/v3.yaml   + language mirroring + length cap  (production)
prompts/rag_answer/v4.yaml   + citation format by example       (rejected: see E7)
```

The shipped version is **pinned**, never `@latest` — adding a file must not
promote it, and v4 would otherwise have gone live the moment it was written.

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

Measured on an RTX 4060 Laptop, full production stack, 20 runs after warm-up
(`runs/bench/real.json`):

| stage | mean | p50 | p95 | share |
| --- | --- | --- | --- | --- |
| embed | 23 ms | 22 | 48 | 1% |
| search | 1 ms | 0 | 1 | <1% |
| rerank | 656 ms | 648 | 884 | **42%** |
| llm | 887 ms | 803 | 1307 | 57% |
| **total** | **1567 ms** | 1489 | 2142 | |

Peak GPU 7041 MB of 8188, peak RSS 1060 MB. Retrieval proper is 1.5% of the
request; the cross-encoder costs almost as much as generation itself. That is
the kind of thing an end-to-end average hides and a per-stage histogram makes
impossible to miss.

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

- **Hybrid retrieval, but only where BM25 can see the query.** Embeddings miss
  exact identifiers (model names, CLI flags, error codes), which is much of what
  people ask a technical knowledge base, and fusing by rank (RRF) avoids
  calibrating cosine similarity against BM25 scores. Measurement then showed
  unconditional fusion *hurting* on cross-lingual queries, so fusion is gated on
  lexical coverage — and the gated version beats both plain dense and
  always-fuse. The design survived contact with the data by being narrowed, not
  by being defended.
- **Heading-aware chunking.** A chunk reading "8 GB is the practical limit" is
  useless without its heading path; the path is prepended before embedding. The
  chunk-size result is the one that replicated across both embedders.
- **Refusal as a measured behaviour, not a hope.** The eval set contains
  unanswerable questions and the metric penalises both over-answering and
  over-refusing.
- **Stub backends as first-class code.** CI runs the real pipeline with fake
  models rather than mocking the pipeline, so a chunking or fusion regression
  fails the build.
- **Prompt hashes in every response.** Without them, "which prompt produced this
  bad answer" is unanswerable a week later.

## Known limitations

- The corpus is a small synthetic handbook (6 documents, 22 chunks). With a real
  embedder the retrieval metrics saturate at 1.000, so they can no longer rank
  configurations — the eval set needs to get harder before it can. The harness,
  not the numbers, is the deliverable.
- **Answer metrics are noisier than they look.** Generation runs at temperature
  0.1, not 0. Running one configuration twice moved citation rate by 0.158 —
  three questions — and faithfulness by up to 0.044. So differences below ~0.1
  on citation rate and ~0.05 on faithfulness are not interpretable here, the
  v2↔v3 faithfulness gap included. Retrieval metrics are deterministic and
  unaffected. Fix for next time: temperature 0 and mean ± spread over three
  runs. Measured in [E6](docs/EXPERIMENTS.md); a single run would have shipped
  a noise artefact as a finding.
- The judge is a 7B model grading a 3B one. It is independent of the generator,
  but it is not a human, and it scores refusals generously — see the v2 result.
  Use it for relative comparisons, never as an accuracy figure.
- v3 leaves 3 of 19 answerable questions uncited and one of those carries a
  malformed `[1]` tag. Prompt v4 fixed the format and made the attribution
  worse, so it was rejected; the open problem is citation *frequency*, not
  format.
- The LoRA scripts are written for a GPU box and validated by dry-run and config
  parsing here; no adapter has been trained in this repository.
- No streaming responses, no multi-tenant auth, no incremental re-indexing.

## Requirements

Python 3.10+. CPU-only for the test suite and the demo; an 8 GB GPU for the full
stack (see `data/corpus/04_gpu_capacity.md` for the memory budget the defaults
are built around). `ffmpeg` is needed to decode compressed audio uploads.

## License

MIT.
