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
hit@4, so differences below ~0.05 are noise.

### E1b — the same ablation with the production embedder

Re-run on 2026-09-21 with `multilingual-e5-small` on an RTX 4060 Laptop
(`runs/ablation/ablation-20260921-140004.json`). **It reverses finding 1.**

| variant | chunks | hit@4 | hit@1 | MRR | nDCG@4 | mean query ms |
| --- | --- | --- | --- | --- | --- | --- |
| **dense only, 800/120** | 22 | 1.000 | **1.000** | **1.000** | 0.980 | **6.0** |
| hybrid, 800/120 (shipped) | 22 | 1.000 | 0.947 | 0.974 | 0.981 | 5.9 |
| hybrid, 400/60 | 36 | 1.000 | 0.842 | 0.921 | 0.942 | 7.8 |
| hybrid, 1600/200 | 22 | 1.000 | 0.947 | 0.974 | 0.981 | 7.4 |
| hybrid, no overlap | 22 | 1.000 | 0.947 | 0.974 | 0.981 | 7.1 |
| hybrid + cross-encoder rerank | 22 | 1.000 | **1.000** | **1.000** | **1.000** | 702.8 |

1. **hit@4 is saturated.** Every variant reaches 1.000, so the headline metric
   can no longer rank configurations. `hit@1`, MRR and nDCG still discriminate,
   and they are what the rest of this section uses. The real conclusion is that
   the eval set is too easy and has to grow before it can settle anything else.

2. **Hybrid retrieval is worse than dense-only here — the opposite of E1.**
   With a strong multilingual embedder, fusing BM25 costs 0.053 of hit@1. That
   is exactly one question, and it is identifiable:

   > **q18** — *"Яка температура використовується для відповідей RAG?"*
   > dense ranks `01_model_serving.md > Generation parameters` first (correct);
   > hybrid ranks `05_finetuning_policy.md > Evaluation` first (wrong).

   The mechanism is clear. The corpus is English, the question is Ukrainian, so
   BM25 has almost no lexical overlap to work with and its ranking is close to
   noise. RRF weights both lists equally by rank, so that noise gets an equal
   vote and displaces a correct dense hit. **BM25 does not do cross-lingual
   retrieval, and fusing it unconditionally injects noise on every non-English
   query** — a defect this corpus is small enough to make visible as a single
   named item rather than a vague average.

3. **Chunk size replicates across embedders.** 400/60 is the worst variant in
   both runs (hit@1 0.842 here, hit@4 0.895 in E1). Splitting mid-passage hurts
   regardless of how good the embedder is — the one finding here solid enough
   to act on.

4. Overlap and 1600/200 still change nothing, for the reason given above: every
   heading section already fits the budget. A property of the corpus, not of
   chunking.

**What this means for the shipped default.** `hybrid: true` is currently on, and
on this corpus it is not earning its place. It is kept — with this result
recorded against it — because the corpus is 6 English documents and hybrid
retrieval's whole purpose is exact-identifier matching on queries in the corpus
language, which 19 questions cannot settle either way. The honest next step is
to gate fusion on query language (or require a minimum BM25 score before it
contributes) and re-measure, not to flip a default on one Ukrainian question.

## E2 — Cross-encoder reranking

**Question.** Does reranking justify its latency?

**Method.** The last row of E1b plus the per-stage benchmark in E4.

| | hit@1 | nDCG@4 | rerank stage latency |
| --- | --- | --- | --- |
| hybrid, no reranker | 0.947 | 0.981 | — |
| hybrid + `bge-reranker-v2-m3` | **1.000** | **1.000** | **656 ms** (p95 884 ms) |
| dense only, no reranker | **1.000** | 0.980 | — (6 ms total query) |

**Finding: on this corpus, no.** The cross-encoder behaves exactly as theory
predicts — it moves `hit@1` and nDCG, not `hit@4`, because its job is ordering a
shortlist rather than finding new documents. But the one question it rescues is
q18, the same question BM25 broke. Dense-only retrieval already answers it
correctly in 6 ms; the reranker spends **656 ms — 42% of end-to-end latency
(E4)** — repairing damage the fusion step caused.

That is not an argument that reranking is useless. It is an argument that
**this corpus cannot demonstrate its value**, because retrieval is already
perfect without it. A reranker earns 656 ms when the retriever is genuinely
imperfect — a larger, more confusable corpus. Until such a corpus exists here,
the number to quote is the cost, not a benefit.

**Note on the ablation's 702.8 ms** versus the benchmark's 656 ms: the ablation
mean includes loading the cross-encoder on the first query. E4 warms up first
and is the number to trust.

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

**Measured.** 2026-09-21, RTX 4060 Laptop 8 GB, 20 runs after 3 warm-up runs,
full production stack (`multilingual-e5-small` + BM25 + `bge-reranker-v2-m3`,
`qwen2.5:3b-instruct` q4 on Ollama). Raw: `runs/bench/real.json`.

| stage | mean | p50 | p95 | max | share of total |
| --- | --- | --- | --- | --- | --- |
| embed | 23 ms | 22 | 48 | 48 | 1% |
| search | 1 ms | 0 | 1 | 1 | <1% |
| **rerank** | **656 ms** | 648 | 884 | 884 | **42%** |
| llm | 887 ms | 803 | 1307 | 1307 | 57% |
| **total** | **1567 ms** | 1489 | 2142 | 2142 | |

Peak GPU memory **7041 MB** of 8188, peak process RSS 1060 MB, mean GPU
utilisation 54.7%.

**Findings.**

1. **Reranking costs as much as generation.** 656 ms against 887 ms for the LLM
   itself. On a 3B model the cross-encoder is not a rounding error on top of
   generation — it is half the latency budget. Combined with E2, where it buys
   nothing this corpus can show, it is the first thing to drop under a latency
   target.
2. **Retrieval proper is free.** Embedding and vector search together are 24 ms,
   1.5% of the request. Optimising them would be optimising noise.
3. **Memory is genuinely tight.** 7041 MB peak of 8188 leaves ~1.1 GB of head
   room with the 3B generator resident. That is the measurement behind the
   guidance in `data/corpus/04_gpu_capacity.md` — and behind the crash below.

**A real failure this produced.** The first attempt at E1b died with
`torch.OutOfMemoryError` while loading the cross-encoder. Cause: the ablation
script built a fresh embedder per variant, torch modules hold reference cycles,
so five variants' weights were still resident when the sixth loaded, on top of
1.47 GB Ollama was holding. Fixed by loading each model once
(`ModelPool` in `scripts/run_ablation.py`) and emptying the allocator cache
between variants. The metrics were unchanged by the fix — verified by re-running
the light-backend ablation and diffing — so this was a memory bug, not a
correctness bug.

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
