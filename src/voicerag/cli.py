"""``voicerag`` command line: ingest, ask, search, transcribe, eval, bench, prompts."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from voicerag import __version__
from voicerag.config import get_settings
from voicerag.logging_utils import configure_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Voice-enabled RAG over local open-source LLM/STT models.",
)
console = Console()


@app.callback()
def _root(
    log_level: str = typer.Option("WARNING", "--log-level", help="DEBUG/INFO/WARNING/ERROR"),
    pretty_logs: bool = typer.Option(True, help="Human-readable logs instead of JSON"),
) -> None:
    configure_logging(log_level, json_logs=not pretty_logs)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"voicerag {__version__}")


@app.command()
def ingest(
    corpus: Annotated[Path, typer.Option(help="Directory of .md/.txt documents")] = Path(
        "data/corpus"
    ),
    out: Annotated[Path | None, typer.Option(help="Index output directory")] = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedder: str | None = typer.Option(None, help="sentence_transformers | hashing"),
) -> None:
    """Chunk a corpus, embed it and write a persistent index."""
    from voicerag.rag.chunking import chunk_documents
    from voicerag.rag.embeddings import build_embedder
    from voicerag.rag.loaders import load_directory
    from voicerag.rag.store import VectorStore

    cfg = get_settings().rag
    target = out or cfg.store_path
    started = time.perf_counter()

    documents = load_directory(corpus)
    if not documents:
        console.print(f"[red]No documents found under {corpus}[/red]")
        raise typer.Exit(code=1)

    chunks = chunk_documents(
        documents,
        chunk_size or cfg.chunk_size,
        chunk_overlap if chunk_overlap is not None else cfg.chunk_overlap,
    )
    emb = build_embedder(embedder or cfg.embedder, cfg.embedding_model, cfg.device)
    console.print(f"Embedding {len(chunks)} chunks with [cyan]{emb.name}[/cyan] ...")
    vectors = emb.embed_passages([c.text for c in chunks])

    store = VectorStore(dimension=emb.dimension, embedder_name=emb.name)
    store.add(chunks, vectors)
    store.save(target)

    table = Table(title=f"Index written to {target}", show_header=False)
    for key, value in store.stats().items():
        table.add_row(str(key), str(value))
    table.add_row("documents", str(len(documents)))
    table.add_row("elapsed_s", f"{time.perf_counter() - started:.2f}")
    console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer from the corpus")],
    prompt_ref: str | None = typer.Option(None, "--prompt", help="e.g. rag_answer@v2"),
    top_n: int | None = None,
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Answer a question with the full RAG pipeline."""
    from voicerag.rag.pipeline import RAGPipeline

    result = RAGPipeline.from_settings().answer(question, top_n=top_n, prompt_ref=prompt_ref)
    if as_json:
        console.print_json(json.dumps(result.as_dict(), ensure_ascii=False))
        return

    console.print(f"\n[bold]{result.answer}[/bold]\n")
    table = Table(title="Sources")
    table.add_column("#")
    table.add_column("source")
    table.add_column("score", justify="right")
    table.add_column("cited", justify="center")
    for i, scored in enumerate(result.sources, start=1):
        table.add_row(
            f"S{i}",
            scored.chunk.source,
            f"{scored.final_score:.3f}",
            "yes" if i in result.cited_indices else "",
        )
    console.print(table)
    timings = "  ".join(f"{k}={v:.0f}ms" for k, v in result.timings_ms.items())
    console.print(
        f"[dim]{result.prompt_ref} ({result.prompt_sha})  {result.model}  {timings}[/dim]"
    )


@app.command()
def search(
    query: str,
    top_k: int | None = None,
    top_n: int | None = None,
) -> None:
    """Retrieval only — inspect what the generator would receive."""
    from voicerag.rag.pipeline import RAGPipeline

    result = RAGPipeline.from_settings().retriever.retrieve(query, top_k=top_k, top_n=top_n)
    table = Table(title=f"Top {len(result.chunks)} of {result.candidates_considered} candidates")
    table.add_column("source")
    table.add_column("heading")
    table.add_column("score", justify="right")
    table.add_column("rerank", justify="right")
    for scored in result.chunks:
        table.add_row(
            scored.chunk.source,
            str(scored.chunk.metadata.get("heading", ""))[:40],
            f"{scored.score:.3f}",
            "-" if scored.rerank_score is None else f"{scored.rerank_score:.3f}",
        )
    console.print(table)
    console.print(
        "[dim]" + "  ".join(f"{k}={v:.0f}ms" for k, v in result.timings_ms.items()) + "[/dim]"
    )


@app.command()
def transcribe(audio: Path, language: str | None = None) -> None:
    """Transcribe an audio file and print the transcript with STT quality signals."""
    from voicerag.stt.factory import build_stt

    result = build_stt().transcribe(audio, language=language)
    console.print(f"\n[bold]{result.text}[/bold]\n")
    console.print(
        f"[dim]lang={result.language}  duration={result.duration_s:.1f}s  "
        f"rtf={result.real_time_factor:.2f}  confidence={result.mean_confidence:.2f}[/dim]"
    )


@app.command("voice-ask")
def voice_ask(audio: Path, language: str | None = None) -> None:
    """Transcribe a recording and answer it from the corpus."""
    from voicerag.rag.pipeline import RAGPipeline
    from voicerag.stt.factory import build_stt

    transcript = build_stt().transcribe(audio, language=language)
    console.print(f"[cyan]heard:[/cyan] {transcript.text}")
    result = RAGPipeline.from_settings().answer(transcript.text)
    console.print(f"\n[bold]{result.answer}[/bold]\n")


@app.command("eval")
def run_eval(
    dataset: Path = Path("data/eval/qa.jsonl"),
    out: Path = Path("runs/eval"),
    prompt_ref: str | None = typer.Option(None, "--prompt"),
    judge: bool = typer.Option(False, "--judge", help="Enable LLM-as-a-judge faithfulness scoring"),
    run_id: str | None = None,
    limit: int | None = None,
) -> None:
    """Run the offline evaluation harness and write a JSON + markdown report."""
    from voicerag.eval.answer_metrics import FaithfulnessJudge
    from voicerag.eval.dataset import load_eval_set
    from voicerag.eval.runner import evaluate
    from voicerag.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings()
    items = load_eval_set(dataset)[: limit or None]
    grader = (
        FaithfulnessJudge(pipeline.llm, pipeline.prompts, get_settings().prompts.judge)
        if judge
        else None
    )
    console.print(f"Evaluating {len(items)} items ...")
    report = evaluate(pipeline, items, judge=grader, prompt_ref=prompt_ref, run_id=run_id)
    path = report.save(out)

    table = Table(title=f"{report.run_id}  ({report.duration_s:.1f}s)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in {**report.retrieval, **report.answers}.items():
        table.add_row(key, f"{value:.3f}")
    console.print(table)
    console.print(f"[green]Report:[/green] {path}")


@app.command()
def compare(reports: list[Path]) -> None:
    """Print a side-by-side table of two or more evaluation reports."""
    from voicerag.eval.runner import compare_reports, load_report

    console.print(compare_reports([load_report(p) for p in reports]))


@app.command()
def bench(
    question: str = "How much VRAM does an RTX 4060 laptop GPU have?",
    runs: int = 10,
    warmup: int = 2,
    out: Path | None = None,
) -> None:
    """Measure per-stage latency and peak GPU/RAM for a repeated question."""
    from voicerag.monitoring.resources import ResourceSampler
    from voicerag.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings()
    for _ in range(warmup):
        pipeline.answer(question)

    stage_times: dict[str, list[float]] = {}
    with ResourceSampler(interval_s=0.25) as sampler:
        for _ in range(runs):
            result = pipeline.answer(question)
            for stage, value in result.timings_ms.items():
                stage_times.setdefault(stage, []).append(value)

    table = Table(title=f"Latency over {runs} runs (ms)")
    for column in ("stage", "mean", "p50", "p95", "max"):
        table.add_column(column, justify="right" if column != "stage" else "left")
    summary: dict[str, dict[str, float]] = {}
    for stage, values in stage_times.items():
        ordered = sorted(values)
        p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
        stats = {
            "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": p95,
            "max": max(values),
        }
        summary[stage] = stats
        table.add_row(stage, *[f"{stats[k]:.0f}" for k in ("mean", "p50", "p95", "max")])
    console.print(table)
    console.print(sampler.summary())

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "question": question,
                    "runs": runs,
                    "latency_ms": summary,
                    "resources": sampler.summary(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"[green]Saved:[/green] {out}")


prompts_app = typer.Typer(help="Inspect the versioned prompt registry.", no_args_is_help=True)
app.add_typer(prompts_app, name="prompts")


@prompts_app.command("list")
def prompts_list() -> None:
    from voicerag.prompts.registry import PromptRegistry

    table = Table(title="Prompt registry")
    for column in ("ref", "sha256", "status", "description"):
        table.add_column(column)
    for row in PromptRegistry(get_settings().prompts.dir).catalog():
        marker = " [green](latest)[/green]" if row["is_latest"] else ""
        table.add_row(
            row["ref"] + marker,
            row["sha256"],
            str(row["metadata"].get("status", "-")),
            row["description"].strip().split("\n")[0][:60],
        )
    console.print(table)


@prompts_app.command("show")
def prompts_show(ref: str) -> None:
    from voicerag.prompts.registry import PromptRegistry

    template = PromptRegistry(get_settings().prompts.dir).get(ref)
    console.print(f"[bold]{template.ref}[/bold]  sha={template.sha256}")
    console.print("\n[cyan]system:[/cyan]\n" + template.system)
    console.print("\n[cyan]user:[/cyan]\n" + template.user)
    console.print(f"\n[dim]metadata: {json.dumps(template.metadata, ensure_ascii=False)}[/dim]")


@prompts_app.command("diff")
def prompts_diff(a: str, b: str) -> None:
    from voicerag.prompts.registry import PromptRegistry

    console.print(PromptRegistry(get_settings().prompts.dir).diff(a, b) or "(identical)")


@app.command()
def serve(
    host: str | None = None,
    port: int | None = None,
    reload: bool = False,
) -> None:
    """Run the FastAPI service."""
    import uvicorn

    cfg = get_settings().service
    uvicorn.run(
        "voicerag.api.main:app",
        host=host or cfg.host,
        port=port or cfg.port,
        reload=reload,
        log_level=cfg.log_level.lower(),
    )


if __name__ == "__main__":
    app()
