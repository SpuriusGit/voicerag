# Fine-tuning Policy

## When to fine-tune

Fine-tuning is the last option, not the first. Retrieval quality, prompt wording
and a larger base model are cheaper and are tried first. LoRA is justified when
the model must adopt a consistent output format or a house style that prompting
does not reliably produce, or when domain vocabulary is systematically
mistranscribed or misused.

Fine-tuning does not add knowledge reliably. New facts belong in the retrieval
corpus, where they can be updated without retraining.

## LoRA configuration

The standard setup is rank 16, alpha 32, dropout 0.05, applied to the attention
and MLP projection matrices. Rank 8 underfit on a 600-example support dataset;
rank 64 tripled the adapter size with no measurable gain. Training runs at 4-bit
with QLoRA, batch size 4 and gradient accumulation 4, for 3 epochs at learning
rate 2e-4 with cosine decay and a 3 percent warmup.

A 3B model trains at those settings in about 40 minutes on an 8 GB laptop GPU.
The resulting adapter is roughly 35 MB, so adapters are versioned alongside the
code rather than stored as full model copies.

## Evaluation

Every adapter is compared against the unmodified base model on a held-out split
before it can be promoted. The required evidence is format-compliance rate,
faithfulness on the RAG evaluation set, and a check that general answer quality
did not regress. An adapter that improves format compliance but lowers
faithfulness by more than two points is rejected.
