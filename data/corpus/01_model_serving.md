# Model Serving Handbook

## Supported backends

The platform serves open-source LLMs through three interchangeable backends.
`ollama` is the default for single-GPU workstations and laptops: it manages GGUF
weights, quantization and model swapping by itself and exposes an HTTP API on
port 11434. `openai_compat` targets vLLM or Text Generation Inference on
multi-GPU servers, where continuous batching raises throughput roughly 4x over
Ollama at eight or more concurrent requests. `echo` is a deterministic stub used
in tests and CI so pipeline changes can be validated without model weights.

Switching backends is a configuration change only. No application code depends
on a specific runtime; everything goes through the BaseLLM interface.

## Default model

The default chat model is Qwen2.5-3B-Instruct at 4-bit quantization. It was
chosen because it answers Ukrainian and English equally well, fits in under
3 GB of VRAM, and reached 0.93 faithfulness on the internal evaluation set.
Llama-3.2-3B scored 0.90 on the same set but degraded noticeably on Ukrainian
questions. Mistral-7B-Instruct scored highest at 0.95 but needs 5.5 GB of VRAM
at 4-bit, which leaves no room for the reranker on an 8 GB card.

## Quantization guidance

Use Q4_K_M as the default quantization. Q5_K_M costs about 25 percent more VRAM
for under one point of quality on our evaluation set. Do not go below Q4 for
models smaller than 7B parameters: Q3 quantization of a 3B model produced
malformed JSON in roughly one out of six structured-output calls.

## Generation parameters

Retrieval-augmented answering runs at temperature 0.1 with a 512-token limit.
Higher temperatures increase paraphrasing of the retrieved context, which the
faithfulness judge scores as unsupported. The evaluation harness and the judge
prompt both run at temperature 0 so that repeated runs are comparable.
