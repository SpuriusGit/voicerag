"""Evaluation dataset loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    reference: str
    relevant_sources: tuple[str, ...] = ()
    type: str = "factual"
    language: str = "en"
    metadata: dict = field(default_factory=dict)

    @property
    def expects_refusal(self) -> bool:
        return self.type == "out_of_corpus"


def load_eval_set(path: Path | str) -> list[EvalItem]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Eval set not found: {src}")
    items: list[EvalItem] = []
    for line_no, line in enumerate(src.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{src}:{line_no}: invalid JSON: {exc}") from exc
        items.append(
            EvalItem(
                id=raw.get("id", f"item-{line_no}"),
                question=raw["question"],
                reference=raw.get("reference", ""),
                relevant_sources=tuple(raw.get("relevant_sources", ())),
                type=raw.get("type", "factual"),
                language=raw.get("language", "en"),
                metadata=raw.get("metadata", {}),
            )
        )
    return items
