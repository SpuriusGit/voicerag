"""FastAPI application: text Q&A, voice Q&A, search, prompts, health, metrics."""

from __future__ import annotations

import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from voicerag import __version__
from voicerag.api.deps import get_pipeline, get_stt
from voicerag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    PromptOut,
    SearchRequest,
    SearchResponse,
    VoiceAskResponse,
)
from voicerag.config import get_settings
from voicerag.llm.base import LLMError
from voicerag.logging_utils import configure_logging, get_logger, get_request_id, new_request_id
from voicerag.monitoring.metrics import render_prometheus, track
from voicerag.monitoring.resources import snapshot_resources
from voicerag.prompts.registry import PromptError
from voicerag.rag.pipeline import RAGPipeline
from voicerag.stt.audio import AudioError
from voicerag.stt.base import BaseSTT

log = get_logger(__name__)

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".opus", ".mp4"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.service.log_level)
    log.info("service_starting", version=__version__, llm=settings.llm.model)
    yield
    log.info("service_stopped")


app = FastAPI(
    title="VoiceRAG",
    version=__version__,
    description=(
        "Voice-enabled RAG over local open-source models: faster-whisper STT, "
        "hybrid retrieval with cross-encoder reranking, versioned prompts, "
        "Prometheus latency/GPU metrics."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = new_request_id()
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        track(request.url.path, "error")
        log.exception("unhandled_error", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": rid},
            headers={"X-Request-ID": rid},
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    track(request.url.path, str(response.status_code))
    response.headers["X-Request-ID"] = rid
    log.info(
        "request",
        path=request.url.path,
        method=request.method,
        status=response.status_code,
        duration_ms=round(elapsed_ms, 2),
    )
    return response


@app.exception_handler(PromptError)
async def _prompt_error(_request: Request, exc: PromptError) -> JSONResponse:
    return JSONResponse(
        status_code=400, content={"detail": str(exc), "request_id": get_request_id()}
    )


@app.exception_handler(LLMError)
async def _llm_error(_request: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(
        status_code=503, content={"detail": str(exc), "request_id": get_request_id()}
    )


@app.exception_handler(AudioError)
async def _audio_error(_request: Request, exc: AudioError) -> JSONResponse:
    return JSONResponse(
        status_code=415, content={"detail": str(exc), "request_id": get_request_id()}
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    settings = get_settings()
    index_stats: dict = {"loaded": False}
    llm_info: dict = {
        "backend": settings.llm.backend,
        "model": settings.llm.model,
        "reachable": False,
    }
    status = "ok"
    try:
        pipeline = get_pipeline()
        index_stats = {"loaded": True, **pipeline.retriever.store.stats()}
        llm_info["reachable"] = pipeline.llm.health()
        if not llm_info["reachable"]:
            status = "degraded"
    except Exception as exc:  # noqa: BLE001 - health must answer even when broken
        status = "degraded"
        index_stats["error"] = str(exc)

    return HealthResponse(
        status=status,
        version=__version__,
        llm=llm_info,
        index=index_stats,
        stt={"backend": settings.stt.backend, "model": settings.stt.model},
        resources=snapshot_resources().as_dict(),
    )


@app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
def metrics() -> Response:
    return Response(content=render_prometheus(), media_type="text/plain; version=0.0.4")


@app.get("/prompts", response_model=list[PromptOut], tags=["prompts"])
def list_prompts(pipeline: RAGPipeline = Depends(get_pipeline)) -> list[PromptOut]:
    """Every prompt version the service can serve, with its content hash."""
    return [PromptOut(**row) for row in pipeline.prompts.catalog()]


@app.get("/prompts/diff", response_class=PlainTextResponse, tags=["prompts"])
def diff_prompts(a: str, b: str, pipeline: RAGPipeline = Depends(get_pipeline)) -> str:
    """Unified diff between two prompt versions, e.g. ``?a=rag_answer@v2&b=rag_answer@v3``."""
    return pipeline.prompts.diff(a, b) or "(identical)"


@app.post("/search", response_model=SearchResponse, tags=["rag"])
def search(payload: SearchRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> SearchResponse:
    """Retrieval only — useful for debugging whether a miss is retrieval or generation."""
    result = pipeline.retriever.retrieve(payload.query, top_k=payload.top_k, top_n=payload.top_n)
    return SearchResponse(**result.as_dict())


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(payload: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> AskResponse:
    answer = pipeline.answer(
        payload.question,
        top_k=payload.top_k,
        top_n=payload.top_n,
        prompt_ref=payload.prompt_ref,
    )
    log.info(
        "answered",
        prompt_ref=answer.prompt_ref,
        prompt_sha=answer.prompt_sha,
        refused=answer.is_refusal,
        sources=len(answer.sources),
        **{f"t_{k}": round(v, 1) for k, v in answer.timings_ms.items()},
    )
    return AskResponse(**answer.as_dict(), request_id=get_request_id())


@app.post("/voice-ask", response_model=VoiceAskResponse, tags=["rag", "stt"])
async def voice_ask(
    file: UploadFile = File(..., description="wav/mp3/m4a/ogg/webm recording"),
    language: str | None = Form(None, description="ISO code; omit to auto-detect"),
    prompt_ref: str | None = Form(None),
    pipeline: RAGPipeline = Depends(get_pipeline),
    stt: BaseSTT = Depends(get_stt),
) -> VoiceAskResponse:
    """Transcribe a recording, then answer it from the indexed corpus."""
    settings = get_settings()
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(
            415, f"Unsupported audio format {suffix!r}; use one of {sorted(AUDIO_SUFFIXES)}"
        )

    payload = await file.read()
    max_bytes = settings.service.max_upload_mb * 1024**2
    if len(payload) > max_bytes:
        raise HTTPException(413, f"Upload exceeds {settings.service.max_upload_mb} MB")
    if not payload:
        raise HTTPException(400, "Empty upload")

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / f"upload{suffix}"
        audio_path.write_bytes(payload)
        transcript = stt.transcribe(audio_path, language=language or settings.stt.language or None)

    if not transcript.text.strip():
        raise HTTPException(422, "No speech detected in the uploaded audio")

    log.info(
        "transcribed",
        language=transcript.language,
        duration_s=round(transcript.duration_s, 2),
        rtf=round(transcript.real_time_factor, 3),
        confidence=round(transcript.mean_confidence, 3),
    )
    answer = pipeline.answer(transcript.text, prompt_ref=prompt_ref)
    return VoiceAskResponse(
        **answer.as_dict(),
        transcript=transcript.as_dict(),
        request_id=get_request_id(),
    )


@app.post("/transcribe", tags=["stt"])
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    stt: BaseSTT = Depends(get_stt),
) -> dict:
    """STT only, with per-request real-time factor and confidence."""
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(415, f"Unsupported audio format {suffix!r}")
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "Empty upload")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(payload)
        return stt.transcribe(path, language=language).as_dict()
