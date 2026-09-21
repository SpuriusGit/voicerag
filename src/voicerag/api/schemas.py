"""Request/response models. These are the service's public contract."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=100, description="candidates before reranking")
    top_n: int | None = Field(None, ge=1, le=20, description="chunks kept after reranking")
    prompt_ref: str | None = Field(
        None,
        description="Pin a prompt version, e.g. 'rag_answer@v2'. Defaults to the configured one.",
        examples=["rag_answer@v3"],
    )


class SourceOut(BaseModel):
    chunk_id: str
    source: str
    position: int
    text: str
    score: float
    rerank_score: float | None = None
    cited: bool = False


class PromptInfo(BaseModel):
    ref: str
    sha256: str


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    sources: list[SourceOut]
    prompt: PromptInfo
    model: str
    timings_ms: dict[str, float]
    usage: dict[str, float]
    request_id: str


class TranscriptOut(BaseModel):
    text: str
    language: str
    duration_s: float
    processing_s: float
    real_time_factor: float
    mean_confidence: float
    model: str


class VoiceAskResponse(AskResponse):
    transcript: TranscriptOut


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=100)
    top_n: int | None = Field(None, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    chunks: list[SourceOut]
    timings_ms: dict[str, float]
    candidates_considered: int


class HealthResponse(BaseModel):
    status: str
    version: str
    llm: dict
    index: dict
    stt: dict
    resources: dict


class PromptOut(BaseModel):
    name: str
    version: int
    ref: str
    sha256: str
    description: str
    variables: list[str]
    is_latest: bool
    metadata: dict


class ErrorResponse(BaseModel):
    detail: str
    request_id: str
