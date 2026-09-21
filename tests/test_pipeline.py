from voicerag.llm.base import ChatMessage
from voicerag.llm.echo import EchoLLM
from voicerag.rag.pipeline import REFUSAL, RAGPipeline
from voicerag.rag.retriever import Retriever


def test_answer_returns_sources_and_prompt_provenance(pipeline):
    result = pipeline.answer("How much VRAM does an RTX 4060 laptop GPU have?")
    assert result.sources
    assert result.prompt_ref.startswith("rag_answer@v")
    assert len(result.prompt_sha) == 16
    assert result.timings_ms["total"] > 0


def test_answer_uses_the_pinned_prompt_version(pipeline):
    assert pipeline.answer("gpu memory", prompt_ref="rag_answer@v2").prompt_ref == "rag_answer@v2"


def test_prompt_sent_to_the_llm_contains_retrieved_context(pipeline, prompts_dir):
    spy = EchoLLM(canned="answer [S1]")
    pipeline.llm = spy
    pipeline.answer("How much VRAM does an RTX 4060 laptop GPU have?")

    sent = "\n".join(m.content for m in spy.calls[0])
    assert "8 GB of VRAM" in sent
    assert "[S1]" in sent


def test_citations_are_parsed_back_out_of_the_answer(pipeline):
    pipeline.llm = EchoLLM(canned="Eight gigabytes [S1] and also [S3].")
    result = pipeline.answer("vram")
    assert result.cited_indices == [1, 3]
    assert result.as_dict()["sources"][0]["cited"] is True


def test_refusal_is_detected(pipeline):
    pipeline.llm = EchoLLM(canned=REFUSAL)
    result = pipeline.answer("What is the capital of Australia?")
    assert result.is_refusal
    assert result.as_dict()["refused"] is True


def test_empty_retrieval_refuses_without_calling_the_llm(pipeline):
    spy = EchoLLM()
    pipeline.llm = spy
    pipeline.retriever = Retriever(
        store=pipeline.retriever.store,
        embedder=pipeline.retriever.embedder,
        top_k=1,
        top_n=0,
        hybrid=False,
    )
    pipeline.retriever.retrieve = lambda *a, **k: _empty_result()  # type: ignore[assignment]
    result = pipeline.answer("anything")
    assert result.answer == REFUSAL
    assert spy.calls == []


def _empty_result():
    from voicerag.rag.retriever import RetrievalResult

    return RetrievalResult(query="anything", chunks=[], timings_ms={"embed": 0.1})


def test_mismatched_embedder_dimension_is_reported_clearly(test_settings, monkeypatch):
    import pytest

    monkeypatch.setattr(
        "voicerag.rag.pipeline.build_embedder",
        lambda *a, **k: _FakeEmbedder(dimension=7),
    )
    with pytest.raises(ValueError, match="Re-run scripts/ingest.py|re-run"):
        RAGPipeline.from_settings(test_settings)


class _FakeEmbedder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.name = "fake"

    def embed_passages(self, texts):  # pragma: no cover - never reached
        raise NotImplementedError

    def embed_queries(self, texts):  # pragma: no cover - never reached
        raise NotImplementedError


def test_build_context_numbers_sources_for_the_judge(pipeline):
    result = pipeline.answer("gpu memory")
    context = pipeline.build_context(result.sources)
    assert "[S1] (" in context
    assert context.count("[S") == len(result.sources)


def test_echo_llm_reports_usage():
    response = EchoLLM().chat([ChatMessage(role="user", content="a somewhat longer question here")])
    assert response.completion_tokens > 0
    assert response.total_tokens >= response.completion_tokens
