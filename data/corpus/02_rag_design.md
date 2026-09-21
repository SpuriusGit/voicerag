# Retrieval Design

## Chunking

Documents are split along markdown headings first, then packed into chunks of at
most 800 characters with 120 characters of overlap. The heading path is
prepended to each chunk before embedding, which gives isolated paragraphs the
context they need: a chunk reading only "8 GB is the practical limit" is useless
without the heading "GPU Capacity > Laptop cards".

Fixed-width splitting at the same size scored 0.68 hit@4 against 0.81 for
heading-aware splitting on the evaluation set.

## Embeddings

The default embedding model is multilingual-e5-small, 384 dimensions. It is
multilingual, so Ukrainian questions retrieve English passages without a
translation step. E5 models require the "query:" and "passage:" prefixes;
omitting them cost 6 points of hit@4 in an early experiment. bge-m3 scored two
points higher but is six times larger and pushed retrieval latency above the
300 ms budget on CPU.

## Hybrid search

Dense retrieval alone misses exact identifiers such as model names, CLI flags
and error codes. BM25 is run in parallel over the same chunks and the two ranked
lists are merged with Reciprocal Rank Fusion at k=60. Fusing by rank avoids
having to calibrate cosine similarity against BM25 scores, which are on
unrelated scales.

## Reranking

The retriever fetches 20 candidates and a bge-reranker-v2-m3 cross-encoder
reorders them down to the 4 chunks that reach the prompt. Reranking raised
hit@4 from 0.74 to 0.89 and cut hallucinated answers roughly in half, at a cost
of about 180 ms per query on the GPU. Reranking more than 20 candidates did not
improve the metric further and only added latency.
