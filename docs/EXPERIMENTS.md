# Experiments

An experiment that is not reproducible is an opinion. Every number here is
produced by a script in this repository, and every script prints the
configuration that produced it.

## E1 — Retrieval ablation

**Question.** Does hybrid retrieval beat dense-only, and what chunk size fits
this corpus?

**Method.** `python scripts/run_ablation.py` builds a separate index per
configuration from the same corpus, runs the same 19 ground-truth questions
through each, and scores the source documents that reach the prompt (`top_n=4`).
One variable changes at a time.

**Setup.** 6 documents, hashed-ngram embedder (so the run needs no downloads and
anyone can reproduce it), `top_k=20`, `top_n=4`, no reranker.

| variant | chunks | hit@4 | hit@1 | MRR | nDCG@4 | mean query ms |
| --- | --- | --- | --- | --- | --- | --- |
| dense only, 800/120 | 22 | 0.895 | 0.684 | 0.789 | 0.813 | 0.1 |
| **hybrid, 800/120 (shipped)** | 22 | **0.947** | **0.789** | **0.868** | **0.889** | 0.1 |
| hybrid, 400/60 | 36 | 0.895 | 0.789 | 0.833 | 0.849 | 0.2 |
| hybrid, 1600/200 | 22 | 0.947 | 0.789 | 0.868 | 0.889 | 0.1 |
| hybrid, no overlap | 22 | 0.947 | 0.789 | 0.868 | 0.889 | 0.1 |

**Findings.**

1. Adding BM25 to dense retrieval is worth **+0.05 hit@4 and +0.08 MRR**. The
   questions it rescues are the ones containing literal identifiers
   (`gpu_memory_utilization`, `condition_on_previous_text`) — exactly the class
   of query where embedding similarity is weakest.
2. Splitting at 400 characters **hurts** (−0.05 hit@4) while producing 64% more
   chunks. It fragments the passage that answers the question across two chunks,
   neither of which is individually convincing.
3. 1600/200 and 800/120 are identical here **only because this corpus is small**:
   every heading section already fits inside 1600 characters, so the larger
   budget changes nothing. This is a property of the corpus, not evidence that
   chunk size does not matter.
4. Overlap makes no difference at this corpus size for the same reason. It is
   kept at 120 because sections *will* exceed the budget on a real corpus, and
   the cost of overlap is a few percent of index size.

**Caveats.** 19 questions is a small sample: a single item is worth 0.053 of
hit@4, so differences below ~0.05 are noise. The hashed embedder is weaker than
e5 in absolute terms; re-run with
`VOICERAG_RAG__EMBEDDER=sentence_transformers` for production numbers.

## E2 — Cross-encoder reranking

**Question.** Does reranking justify roughly 180 ms of added latency?

**Status: not run in this repository.** It requires downloading
`BAAI/bge-reranker-v2-m3` (~2 GB) and a GPU to be meaningful. The experiment is
implemented and one command away:

```bash
make install-ml
VOICERAG_RAG__EMBEDDER=sentence_transformers python scripts/run_ablation.py --with-rerank
```

**What to look for.** Reranking should move `hit@1` far more than `hit@4` — its
job is ordering the shortlist, not finding new documents. If `hit@1` does not
improve, the retriever is already returning the right chunk first and the
reranker is pure latency.

## E3 — Prompt versions

**Question.** What did each prompt revision actually change?

**Status: partially measurable without an LLM.** Retrieval metrics are identical
across prompt versions by construction (the prompt does not influence
retrieval). Refusal accuracy, faithfulness and citation validity need a real
generator; with the echo backend they are structurally zero.

```bash
voicerag eval --judge --prompt rag_answer@v1 --run-id v1
voicerag eval --judge --prompt rag_answer@v2 --run-id v2
voicerag eval --judge --prompt rag_answer@v3 --run-id v3
voicerag compare runs/eval/v1.json runs/eval/v2.json runs/eval/v3.json
```

**Hypotheses being tested** (recorded before running, which is the point):

| version | change | expected effect | metric that would show it |
| --- | --- | --- | --- |
| v1 → v2 | numbered sources, mandatory refusal sentence | fewer answers to the 4 out-of-corpus questions | `refusal_accuracy` |
| v1 → v2 | inline `[Sn]` citations | citations appear and point at provided sources | `citation_rate`, `citation_validity` |
| v2 → v3 | answer in the question's language | the 4 Ukrainian questions answered in Ukrainian | manual review of `runs/eval/*.json` |
| v2 → v3 | ~120-word cap | shorter answers, no faithfulness loss | `mean_answer_words`, `faithfulness` |

If v3 lowers faithfulness relative to v2, the length cap is too aggressive and
v2 stays in production. The prompt files carry `recorded: null` until these runs
happen — a deliberately visible reminder that no number has been invented.

## E4 — Latency and memory budget

```bash
voicerag bench --runs 20 --out runs/bench/latest.json
```

Reports p50/p95/max per stage plus peak GPU memory, peak process RSS and mean
GPU utilisation, sampled in a background thread during the run.

**Status: the harness runs; no GPU numbers are recorded here**, because the
measurements would describe one laptop and mislead anyone else. The budget the
defaults are designed against is documented in `data/corpus/04_gpu_capacity.md`.

**What the stage split is for.** When p95 rises, one histogram moves:

| stage regressed | most likely cause |
| --- | --- |
| `rerank` | cross-encoder fell back to CPU after a driver reset |
| `llm` (others flat) | longer context — check `top_n` and chunk sizes |
| `embed` | embedder loaded on CPU, or batch size collapsed to 1 |
| `stt` | larger Whisper model, or VAD disabled on long silences |

## E5 — LoRA fine-tuning

**Status: not trained here.** The pipeline is complete and validated by
`python finetune/train_lora.py --dry-run`, but no adapter exists in this
repository: the corpus yields 28 weak-supervision examples, which is far too few
to produce an adapter worth reporting. Claiming a result from it would be worse
than reporting none.

The method, the configuration rationale and the promotion criteria are in
[FINETUNING.md](FINETUNING.md).

## Reproducing everything

```bash
make demo                      # pipeline + evaluation, no downloads
make ablation                  # E1
make bench                     # E4 harness
make install-ml && make finetune-data   # E5 inputs
```
