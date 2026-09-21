# Evaluation

## The dataset

`data/eval/qa.jsonl`, 23 questions over the 6-document corpus:

| type | n | purpose |
| --- | --- | --- |
| `factual` | 16 | single-passage answers, 3 of them in Ukrainian |
| `multi_hop` | 3 | require combining two sections or two documents |
| `out_of_corpus` | 4 | must be refused — the hallucination probe |

Each row records `relevant_sources` at **document** level, not chunk level.
Chunk ids change whenever the chunker is re-tuned; anchoring ground truth to
chunks would silently invalidate every historical run.

```json
{"id": "q01", "question": "How much VRAM does an RTX 4060 laptop GPU have?",
 "reference": "8 GB.", "relevant_sources": ["04_gpu_capacity.md"],
 "type": "factual", "language": "en"}
```

## Retrieval metrics

| metric | what it answers |
| --- | --- |
| `hit@k` | did any correct document reach the prompt at all? |
| `recall@k` | for multi-document questions, how many did we find? |
| `MRR` | how high did the first correct document rank? |
| `nDCG@k` | rank-weighted credit — rewards first place over fourth |

Items with no ground truth (`out_of_corpus`) are excluded from retrieval
averages rather than scored as zero: there is no correct document to retrieve,
so counting them would drag the metric down for the wrong reason.

Default `k` values stay at or below the pipeline's `top_n`, because the report
measures the documents that actually reached the prompt, not the raw candidate
list.

## Answer metrics

**Deterministic** (no LLM, run on every commit):

- `token_recall` — share of reference content words present in the answer.
  Blunt: it cannot distinguish a paraphrase from a miss. It catches gross
  regressions, nothing subtler.
- `citation_rate` — share of answerable questions that carry any `[Sn]` tag.
- `citation_validity` — share of tags pointing at a source actually provided.
  A model inventing `[S9]` when four sources were given is fabricating
  provenance, which is worse than not citing at all.
- `mean_answer_words`, `mean_latency_ms`.

**Behavioural:**

- `refusal_accuracy` — scored in **both** directions. Answering an
  `out_of_corpus` question is a hallucination; refusing an answerable one is an
  unnecessary failure. A model that refuses everything scores 4/23, not 1.0.

**LLM-as-a-judge:**

- `faithfulness` — 1.0 / 0.5 / 0.0 for fully / partially / not supported by the
  retrieved context, graded by `judge_faithfulness@v1` at temperature 0 with a
  fixed JSON schema. A correct refusal scores 1.0.

### Judging the judge

The judge is the noisiest metric here and is treated accordingly:

- it is versioned like any other prompt, so a scoring change is visible in git;
- it runs at temperature 0, so repeated runs are comparable;
- unparsable output scores 0.0 and is recorded as such rather than skipped;
- out-of-range scores are clamped, so a confused judge cannot skew the mean;
- it judges support-by-context only, explicitly *not* whether the answer matches
  its own knowledge.

Use it for **relative** comparisons between configurations. Do not quote its
absolute value as accuracy.

## Reports

```bash
voicerag eval --judge --run-id nightly
```

Writes `runs/eval/nightly.json` (full per-item detail) and `runs/eval/nightly.md`
(a readable summary). Every report embeds:

```json
{"prompt": "rag_answer@v3 (142d394aa474ed8c)", "llm": "qwen2.5:3b-instruct",
 "embedder": "intfloat/multilingual-e5-small", "reranker": "BAAI/bge-reranker-v2-m3",
 "hybrid": true, "top_k": 20, "top_n": 4, "chunks_indexed": 22}
```

Two reports are compared with `voicerag compare a.json b.json`, which prints the
markdown table that goes into `EXPERIMENTS.md`.

## Regression guard in CI

`tests/test_eval.py::test_retrieval_quality_on_the_real_corpus_meets_the_baseline`
runs the full harness on the real corpus with the light backends and fails the
build if `hit@4` drops below 0.85 or MRR below 0.75. It is a floor, not a
target: it catches a broken chunker or a broken fusion, and it costs about a
second.

## Honest limits

- 23 questions is small. One item is worth 0.043 of a per-item mean, so
  differences under ~0.05 are noise.
- The corpus is synthetic and written by one person, which makes the questions
  more answerable than real user questions.
- Answer-quality numbers depend on the local LLM. Nothing in this repository
  reports a faithfulness number that has not been measured — the prompt files
  say `recorded: null` where no run exists.
