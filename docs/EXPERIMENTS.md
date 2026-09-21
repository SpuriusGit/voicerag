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

**What this means for the shipped default.** The next step is not to flip a
default on one question, but to fix the mechanism: stop fusing BM25 when BM25
has nothing to fuse. That is E1c.

### E1c — gating fusion on lexical coverage

**Question.** If BM25 only hurts when it has no vocabulary in common with the
query, can that condition be detected cheaply and used to skip fusion?

**Design, chosen by measurement rather than intuition.** Two candidate
statistics were computed over all 23 questions before anything was implemented:

| statistic | English questions | Ukrainian questions | separates? |
| --- | --- | --- | --- |
| `lexical_coverage` — share of query terms present in the index | 0.38 – 1.00 | 0.00 – 0.30 | **yes**, gap of 0.08 |
| IDF-weighted signal — share of discriminative mass matched | 0.07 – 0.63 | 0.00 – 0.21 | no, ranges overlap |

The IDF-weighted version is the more sophisticated idea and it is the one that
fails: `q20` ("What is the capital of Australia?") scores 0.07 because its words
are common ones, which is not the property being detected. Plain coverage
answers the actual question — *does BM25 recognise these words at all* — and a
query in a language the corpus does not contain scores near zero by
construction, with no language detector required.

The threshold is `0.35`, sitting in the observed gap. **It is calibrated on 23
questions with 0.08 of margin, so it is a reasonable default, not a constant
anyone should trust blindly** — it is exposed as
`VOICERAG_RAG__MIN_LEXICAL_OVERLAP`. The mechanism is the durable part.

**Result** (`runs/ablation/ablation-20260921-141216.json`, same setup as E1b):

| variant | queries fused | hit@1 | MRR | nDCG@4 | mean query |
| --- | --- | --- | --- | --- | --- |
| dense only | 0 / 19 | 1.000 | 1.000 | 0.980 | 6.2 ms |
| hybrid, always fuse | 19 / 19 | 0.947 | 0.974 | 0.981 | 6.2 ms |
| **hybrid + lexical gate (shipped)** | 16 / 19 | **1.000** | **1.000** | **1.000** | **6.1 ms** |
| gated hybrid + cross-encoder | 16 / 19 | 1.000 | 1.000 | 1.000 | 671.4 ms |

The gate skips exactly the three Ukrainian questions with ground truth and fuses
the other sixteen. It recovers the hit@1 that unconditional fusion lost **and**
keeps the ranking quality that BM25 contributes: nDCG@4 1.000, against 0.980 for
dense-only and 0.981 for always-fuse. It is the best non-reranked configuration
on every metric, at no measurable latency cost.

So the E1b result did not mean "hybrid retrieval is a bad idea". It meant
"fusing unconditionally is a bad idea". The distinction is only visible because
the failure was traced to a named question instead of averaged away.

`voicerag_fusion_decisions_total{decision=...}` counts the skips in production:
a rising skip rate means queries are arriving in a language the index does not
contain, which is a content problem, not a retrieval bug.

## E2 — Cross-encoder reranking

**Question.** Does reranking justify its latency?

**Method.** The last row of E1b plus the per-stage benchmark in E4.

| | hit@1 | nDCG@4 | rerank stage latency |
| --- | --- | --- | --- |
| hybrid, always fuse, no reranker | 0.947 | 0.981 | — |
| hybrid, always fuse + `bge-reranker-v2-m3` | 1.000 | 1.000 | 656 ms (p95 884 ms) |
| **hybrid + lexical gate, no reranker** | **1.000** | **1.000** | **— (6 ms total query)** |
| gated hybrid + `bge-reranker-v2-m3` | 1.000 | 1.000 | 665 ms |

**Finding: on this corpus, no — and after E1c, emphatically no.** The
cross-encoder behaves exactly as theory predicts: it moves `hit@1` and nDCG,
not `hit@4`, because its job is ordering a shortlist rather than finding new
documents. But the single question it rescued was q18 — the one unconditional
BM25 fusion had broken. Once the gate stops BM25 from breaking it, the reranker
has nothing left to fix: gated retrieval and gated-plus-reranked retrieval
produce **identical** metrics (1.000 / 1.000 / 1.000), and the reranker's
contribution is 665 ms of latency for a measurably empty difference.

That is not an argument that reranking is useless in general. It is an argument
that **this corpus cannot demonstrate its value**, because retrieval is already
perfect without it. A reranker earns 650 ms when the retriever is genuinely
imperfect — a larger, more confusable corpus. Until such a corpus exists here,
the number to quote for it is its cost.

It also illustrates a trap worth naming: before E1c, the reranker looked like it
was buying +0.053 hit@1. It was actually paying to undo damage from an earlier
stage. Measuring a component against a broken baseline flatters it.

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

## E6 — How much of a difference is real?

**Question.** The gate (E1c) improved retrieval. Did it change answer quality —
and more importantly, could this harness even tell?

**Why it had to be asked.** Generation runs at temperature 0.1, not 0. It is
nearly deterministic, which is not deterministic. Before attributing any
answer-metric difference to a change, the size of run-to-run noise has to be
known. So each configuration was run twice, everything else identical.

| run | fusion | citation rate | faithfulness | token recall | words |
| --- | --- | --- | --- | --- | --- |
| `real-v3` | always | 0.842 | 0.935 | 0.587 | 24.9 |
| `real-v3-repeat` | always | 0.842 | 0.913 | 0.582 | 25.1 |
| `real-v3-gated` | gate | 0.737 | 0.891 | 0.588 | 25.6 |
| `real-v3-gated-2` | gate | 0.895 | 0.935 | 0.580 | 25.4 |

**Finding: the harness cannot resolve this difference, and neither can anyone
else at this sample size.**

- Citation rate varies by **0.158 between two runs of the same configuration**
  (gated: 0.737 then 0.895) — three questions. Both always-fuse runs happened to
  land on 0.842, which at n=2 says nothing about stability.
- Faithfulness varies by **0.022 across identical always-fuse runs** and spans
  0.891–0.935 within the gated pair.
- Refusal accuracy was 1.000 in all four runs — the one answer metric that was
  stable here, because it measures a behaviour the prompt pins down rather than
  a wording choice.

Taken alone, `real-v3-gated` looked like the gate had cost 2 citations and 0.044
faithfulness. The second gated run scored *higher* than both always-fuse runs.
The first reading was noise, and a single run would have shipped it as a finding.

**Consequences, applied backwards as well as forwards.**

1. Answer-metric differences below roughly **0.1 on citation rate** and **0.05
   on faithfulness** are not interpretable at 19–23 questions and 1–2 runs.
2. This retroactively qualifies E3: the v2↔v3 faithfulness difference of 0.065
   sits inside that band and must not be quoted as an effect. The v1↔v3 refusal
   difference (0/4 versus 4/4 out-of-corpus) sits far outside it and stands.
3. The E1c retrieval result is unaffected — retrieval is deterministic given a
   fixed index, so its numbers repeat exactly.
4. The cheap fix for a future run is temperature 0 plus a fixed seed for
   evaluation, and reporting mean ± spread over at least three runs rather than
   a single number. Not done here; recorded as the next thing to do.

The uncomfortable version of this finding: **most of the answer-quality numbers
elsewhere in this document are single-run measurements**, and this section is
the reason to treat the small ones as indicative rather than settled.

## Reproducing everything

```bash
make demo                      # pipeline + evaluation, no downloads
make ablation                  # E1
make bench                     # E4 harness
make install-ml && make finetune-data   # E5 inputs
```
