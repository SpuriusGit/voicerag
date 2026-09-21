"""End-to-end RAG: retrieve -> render a versioned prompt -> generate -> cite."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from voicerag.config import Settings, get_settings
from voicerag.llm.base import BaseLLM
from voicerag.llm.factory import build_llm
from voicerag.monitoring.metrics import observe_stage, record_tokens
from voicerag.prompts.registry import PromptRegistry
from voicerag.rag.documents import ScoredChunk
from voicerag.rag.embeddings import build_embedder
from voicerag.rag.reranker import build_reranker
from voicerag.rag.retriever import Retriever
from voicerag.rag.store import VectorStore

REFUSAL = "I cannot answer this from the available documents."
_CITATION = re.compile(r"\[S(\d+)\]")


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[ScoredChunk]
    cited_indices: list[int] = field(default_factory=list)
    prompt_ref: str = ""
    prompt_sha: str = ""
    model: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    usage: dict[str, float] = field(default_factory=dict)

    @property
    def is_refusal(self) -> bool:
        return REFUSAL.lower() in self.answer.lower()

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.is_refusal,
            "sources": [
                {**c.as_dict(), "cited": i + 1 in self.cited_indices}
                for i, c in enumerate(self.sources)
            ],
            "prompt": {"ref": self.prompt_ref, "sha256": self.prompt_sha},
            "model": self.model,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "usage": self.usage,
        }


class RAGPipeline:
    """Wires the configured components together; built once per process."""

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        prompts: PromptRegistry,
        answer_prompt_ref: str = "rag_answer@latest",
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompts = prompts
        self.answer_prompt_ref = answer_prompt_ref

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RAGPipeline:
        cfg = settings or get_settings()
        store = VectorStore.load(cfg.rag.store_path)
        embedder = build_embedder(cfg.rag.embedder, cfg.rag.embedding_model, cfg.rag.device)
        if embedder.dimension != store.dimension:
            raise ValueError(
                f"Embedder {embedder.name} produces {embedder.dimension}-d vectors but the index "
                f"at {cfg.rag.store_path} was built with {store.dimension}-d "
                f"({store.embedder_name}). Re-run scripts/ingest.py."
            )
        retriever = Retriever(
            store=store,
            embedder=embedder,
            reranker=build_reranker(cfg.rag.reranker, cfg.rag.reranker_model, cfg.rag.device),
            top_k=cfg.rag.top_k,
            top_n=cfg.rag.top_n,
            fusion=cfg.rag.fusion,
            min_lexical_overlap=cfg.rag.min_lexical_overlap,
        )
        return cls(
            retriever=retriever,
            llm=build_llm(cfg.llm),
            prompts=PromptRegistry(Path(cfg.prompts.dir)),
            answer_prompt_ref=cfg.prompts.answer,
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        top_n: int | None = None,
        prompt_ref: str | None = None,
    ) -> RAGAnswer:
        retrieval = self.retriever.retrieve(question, top_k=top_k, top_n=top_n)
        template = self.prompts.get(prompt_ref or self.answer_prompt_ref)

        if not retrieval.chunks:
            return RAGAnswer(
                question=question,
                answer=REFUSAL,
                sources=[],
                prompt_ref=template.ref,
                prompt_sha=template.sha256,
                model=self.llm.model,
                timings_ms=retrieval.timings_ms,
            )

        messages = template.render(
            question=question,
            chunks=[
                {"source": c.chunk.source, "text": c.chunk.text, "score": c.final_score}
                for c in retrieval.chunks
            ],
        )

        with observe_stage("llm", self.llm.model) as t:
            response = self.llm.chat(messages)
        timings = {**retrieval.timings_ms, "llm": t["elapsed_s"] * 1000}
        timings["total"] = sum(timings.values())
        record_tokens(self.llm.model, response.prompt_tokens, response.completion_tokens)

        return RAGAnswer(
            question=question,
            answer=response.text,
            sources=retrieval.chunks,
            cited_indices=sorted({int(m) for m in _CITATION.findall(response.text)}),
            prompt_ref=template.ref,
            prompt_sha=template.sha256,
            model=response.model,
            timings_ms=timings,
            usage={
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "tokens_per_second": round(response.tokens_per_second, 1),
            },
        )

    def build_context(self, chunks: list[ScoredChunk]) -> str:
        """Flatten chunks the way the judge prompt expects them."""
        return "\n\n".join(
            f"[S{i}] ({c.chunk.source})\n{c.chunk.text}" for i, c in enumerate(chunks, start=1)
        )
