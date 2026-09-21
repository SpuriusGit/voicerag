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

**Method.** All three versions evaluated on the same 23 questions with the same
retrieval stack, so the prompt is the only variable. Generator
`qwen2.5:3b-instruct` (q4, Ollama); grader `qwen2.5:7b-instruct` — deliberately
a different model, because one grading its own output is biased in its favour.

```bash
for V in v1 v2 v3; do
  voicerag eval --judge --judge-model qwen2.5:7b-instruct \
      --prompt "rag_answer@$V" --run-id "real-$V"
done
voicerag compare runs/eval/real-v1.json runs/eval/real-v2.json runs/eval/real-v3.json
```

**Setup.** 2026-09-21, RTX 4060 Laptop 8 GB, `multilingual-e5-small` + BM25 +
`bge-reranker-v2-m3`, `top_k=20`, `top_n=4`, 22 chunks.

| metric | v1 | v2 | v3 |
| --- | --- | --- | --- |
| hit@4 / MRR | 1.000 | 1.000 | 1.000 |
| refusal accuracy | 0.826 | 0.913 | **1.000** |
| out-of-corpus refused | 0 / 4 | 4 / 4 | 4 / 4 |
| citation rate | 0.000 | 0.737 | **0.842** |
| citation validity | 0.000 | 0.737 | **0.842** |
| faithfulness (judge) | 0.783 | *1.000* | 0.935 |
| token recall | 0.648 | 0.553 | 0.587 |
| mean answer words | 114.7 | 29.9 | **24.9** |
| mean latency | 8915 ms | 3357 ms | **2427 ms** |
| run duration | 354 s | 218 s | 180 s |

**Hypotheses, recorded before the run, and what actually happened:**

| change | expected | outcome |
| --- | --- | --- |
| v1 → v2: numbered sources + refusal sentence | fewer answers to the 4 unanswerable questions | **confirmed** — 0/4 refused becomes 4/4 |
| v1 → v2: inline `[Sn]` citations | citations appear and resolve | **confirmed** — 0.000 → 0.737 |
| v2 → v3: answer in the question's language | the Ukrainian questions answered in Ukrainian | **not confirmed** — see below |
| v2 → v3: ~120-word cap | shorter answers, no faithfulness loss | **confirmed on length** (29.9 → 24.9 words), faithfulness discussed below |

**Findings.**

1. **v1 hallucinated exactly as predicted.** It answered every out-of-corpus
   question from parametric memory, most bluntly *"The capital of Australia is
   Canberra."* Retrieval was identical to v2 and v3, so nothing about the
   failure was retrieval-side. Its 114.7-word answers also made it 3.7× slower
   than v3 — a grounding constraint bought accuracy *and* latency.

2. **A perfect faithfulness score picked the wrong prompt.** v2 scores 1.000,
   v3 scores 0.935, and v3 is still the better prompt: v2 gets there by
   refusing questions the corpus *does* answer (multi-hop refusal accuracy
   0.667; it refused "Яка температура використовується для відповідей RAG?",
   which is stated verbatim in the corpus). The judge scores a refusal as fully
   supported, so refusing more raises faithfulness. This is the concrete reason
   `refusal_accuracy` is scored in both directions and read alongside
   faithfulness — either one alone promotes the wrong version.

3. **The language rule changed nothing and was kept anyway.** v3 answers all
   three answerable Ukrainian questions in Ukrainian — but so did v1, without
   being told to. v2's only English answer to a Ukrainian question was the
   mandated refusal sentence, which is English by design. The hypothesis is not
   confirmed: this model mirrors language on its own, and the rule is insurance
   against a model that does not, not a measured improvement. Recording it as
   "not confirmed" costs nothing; pretending it worked would poison every later
   comparison.

4. **Open defect: citation format drift.** v3's citation validity is 0.842,
   not 1.000, because the model sometimes emits `[1]` instead of `[S1]` —
   visible in q18. The metric is doing its job; the fix belongs in a v4 that
   shows the tag format by example rather than describing it.

**Caveats.** 23 questions: one item moves a mean by 0.043, so the v2↔v3
faithfulness difference (0.065) is barely outside noise and should not be quoted
as a precise effect. The v1→v3 refusal difference (0/4 versus 4/4) is not
subtle. The judge is a 7B model, independent of the generator but not a human.

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
