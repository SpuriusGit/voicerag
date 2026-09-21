"""Offline evaluation harness.

Runs the full pipeline over the eval set, records per-item results and writes a
JSON report plus a markdown table. Every report embeds the configuration that
produced it — prompt ref and hash, models, top_k/top_n — so two runs can be
compared without guessing what changed.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from voicerag import __version__
from voicerag.eval.answer_metrics import (
    FaithfulnessJudge,
    aggregate_answers,
    citation_validity,
    has_citation,
    refusal_correct,
    token_recall,
)
from voicerag.eval.dataset import EvalItem, load_eval_set
from voicerag.eval.retrieval_metrics import aggregate_retrieval
from voicerag.logging_utils import get_logger
from voicerag.monitoring.resources import ResourceSampler
from voicerag.rag.pipeline import RAGPipeline

log = get_logger(__name__)


@dataclass
class EvalReport:
    run_id: str
    created_at: str
    config: dict
    retrieval: dict[str, float] = field(default_factory=dict)
    answers: dict[str, float] = field(default_factory=dict)
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    items: list[dict] = field(default_factory=list)
    duration_s: float = 0.0

    def save(self, directory: Path | str) -> Path:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.run_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / f"{self.run_id}.md").write_text(self.to_markdown(), encoding="utf-8")
        return path

    def to_markdown(self) -> str:
        lines = [
            f"# Evaluation run `{self.run_id}`",
            "",
            f"- created: {self.created_at}",
            f"- items: {len(self.items)}  |  duration: {self.duration_s:.1f}s",
            "",
            "## Configuration",
            "",
            "| key | value |",
            "| --- | --- |",
            *[f"| {k} | `{v}` |" for k, v in self.config.items()],
            "",
            "## Retrieval",
            "",
            "| metric | value |",
            "| --- | --- |",
            *[f"| {k} | {v:.3f} |" for k, v in self.retrieval.items()],
            "",
            "## Answers",
            "",
            "| metric | value |",
            "| --- | --- |",
            *[f"| {k} | {v:.3f} |" for k, v in self.answers.items()],
        ]
        if self.by_type:
            lines += [
                "",
                "## By question type",
                "",
                "| type | n | hit@4 | refusal_acc | faithfulness |",
                "| --- | --- | --- | --- | --- |",
            ]
            nan = float("nan")
            for qtype, stats in sorted(self.by_type.items()):
                lines.append(
                    f"| {qtype} | {int(stats.get('n', 0))} "
                    f"| {stats.get('hit@4', nan):.3f} "
                    f"| {stats.get('refusal_accuracy', nan):.3f} "
                    f"| {stats.get('faithfulness', nan):.3f} |"
                )
        if self.resources:
            lines += [
                "",
                "## Resources",
                "",
                "```json",
                json.dumps(self.resources, indent=2),
                "```",
            ]
        return "\n".join(lines) + "\n"


def evaluate(
    pipeline: RAGPipeline,
    items: list[EvalItem],
    judge: FaithfulnessJudge | None = None,
    prompt_ref: str | None = None,
    run_id: str | None = None,
    sample_resources: bool = True,
) -> EvalReport:
    started = time.perf_counter()
    rid = run_id or datetime.now(timezone.utc).strftime("eval-%Y%m%d-%H%M%S")
    retriever = pipeline.retriever
    template = pipeline.prompts.get(prompt_ref or pipeline.answer_prompt_ref)

    sampler = ResourceSampler(interval_s=0.5) if sample_resources else None
    if sampler:
        sampler.start()

    per_item: list[dict] = []
    for index, item in enumerate(items, start=1):
        result = pipeline.answer(item.question, prompt_ref=template.ref)
        retrieved_sources: list[str] = []
        for scored in result.sources:
            if scored.chunk.source not in retrieved_sources:
                retrieved_sources.append(scored.chunk.source)

        row = {
            "id": item.id,
            "type": item.type,
            "language": item.language,
            "question": item.question,
            "answer": result.answer,
            "reference": item.reference,
            "retrieved": retrieved_sources,
            "relevant": list(item.relevant_sources),
            "expects_refusal": item.expects_refusal,
            "refusal_correct": refusal_correct(result.answer, item.expects_refusal),
            "token_recall": token_recall(result.answer, item.reference),
            "has_citation": has_citation(result.answer),
            "citation_validity": citation_validity(result.answer, len(result.sources)),
            "answer_words": float(len(result.answer.split())),
            "latency_ms": result.timings_ms.get("total", 0.0),
            "faithfulness": None,
            "judge_reason": "",
        }

        if judge is not None:
            verdict = judge.score(
                question=item.question,
                context=pipeline.build_context(result.sources),
                answer=result.answer,
            )
            row["faithfulness"] = verdict.score
            row["judge_reason"] = verdict.reason

        per_item.append(row)
        log.info(
            "eval_item", n=f"{index}/{len(items)}", id=item.id, refusal_ok=row["refusal_correct"]
        )

    if sampler:
        sampler.stop()

    report = EvalReport(
        run_id=rid,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config={
            "voicerag_version": __version__,
            "prompt": f"{template.ref} ({template.sha256})",
            "llm": pipeline.llm.model,
            "embedder": retriever.embedder.name,
            "reranker": retriever.reranker.name,
            "hybrid": retriever.hybrid,
            "top_k": retriever.top_k,
            "top_n": retriever.top_n,
            "chunks_indexed": len(retriever.store),
            "judge": "on" if judge else "off",
            "python": platform.python_version(),
        },
        retrieval=aggregate_retrieval(per_item),
        answers=aggregate_answers(per_item),
        by_type=_by_type(per_item),
        resources=sampler.summary() if sampler else {},
        items=per_item,
        duration_s=time.perf_counter() - started,
    )
    return report


def _by_type(per_item: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for qtype in sorted({r["type"] for r in per_item}):
        rows = [r for r in per_item if r["type"] == qtype]
        stats: dict[str, float] = {"n": float(len(rows))}
        stats.update(aggregate_retrieval(rows, ks=(4,)))
        stats.update(aggregate_answers(rows))
        out[qtype] = stats
    return out


def compare_reports(
    reports: list[EvalReport],
    keys: tuple[str, ...] = (
        "hit@4",
        "mrr",
        "refusal_accuracy",
        "citation_validity",
        "faithfulness",
        "mean_latency_ms",
    ),
) -> str:
    """Side-by-side markdown table — the artefact that goes into docs/EXPERIMENTS.md."""
    header = "| metric | " + " | ".join(r.run_id for r in reports) + " |"
    divider = "| --- " * (len(reports) + 1) + "|"
    lines = [header, divider]
    for key in keys:
        cells = []
        for report in reports:
            value = report.retrieval.get(key, report.answers.get(key))
            cells.append("-" if value is None else f"{value:.3f}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_report(path: Path | str) -> EvalReport:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalReport(**raw)


__all__ = ["EvalReport", "compare_reports", "evaluate", "load_eval_set", "load_report"]
