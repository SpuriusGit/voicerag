# Data

## `corpus/` — synthetic knowledge base

Six markdown documents written **for this project** as the RAG corpus: an
internal handbook for an LLM serving platform (model choices, retrieval design,
STT pipeline, GPU capacity, fine-tuning policy, incident runbook).

The documents are realistic in shape and internally consistent — headings,
cross-references, concrete numbers — because that is what a retriever has to
cope with. **The figures inside them are illustrative content, not measurements
taken from this repository.** They exist to be retrieved and cited. Measurements
that *were* taken here live in [../docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md),
and every one of them names the script that produced it.

## `eval/qa.jsonl` — evaluation set

23 questions with reference answers and document-level ground truth, including
4 deliberately unanswerable ones and 4 in Ukrainian. Format and rationale:
[../docs/EVALUATION.md](../docs/EVALUATION.md).

## `finetune/` — generated, not committed as source

Built by `make finetune-data` from the corpus. Regenerate rather than edit.

## `audio/` — drop recordings here

Empty by default. `voicerag transcribe data/audio/yours.m4a` accepts wav, mp3,
m4a, ogg, webm and flac (compressed formats need `ffmpeg`).
