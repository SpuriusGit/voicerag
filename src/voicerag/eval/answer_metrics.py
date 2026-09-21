"""Answer quality metrics.

Three layers, deliberately:

* deterministic overlap metrics — free, stable, catch gross regressions;
* refusal / citation checks — verify the behavioural contract of the prompt;
* LLM-as-a-judge faithfulness — the only one that catches a fluent answer that
  is not supported by the retrieved context.

The judge is the expensive and noisiest layer, so it runs at temperature 0 with a
fixed JSON schema, and its prompt is versioned like any other.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from voicerag.llm.base import BaseLLM
from voicerag.prompts.registry import PromptRegistry
from voicerag.rag.pipeline import REFUSAL

_WORD = re.compile(r"[a-z0-9Ѐ-ӿ]+")
_CITATION = re.compile(r"\[S(\d+)\]")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "it",
    "that",
    "this",
    "with",
    "at",
    "by",
    "be",
    "as",
}


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def token_recall(answer: str, reference: str) -> float:
    """Share of reference content words present in the answer.

    A blunt instrument — it cannot tell a paraphrase from a miss — but it is free
    and deterministic, so it runs on every commit while the judge runs nightly.
    """
    ref = _tokens(reference)
    if not ref:
        return 0.0
    return len(ref & _tokens(answer)) / len(ref)


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower() in answer.lower().strip()


def refusal_correct(answer: str, expects_refusal: bool) -> float:
    """1.0 when refusal behaviour matches expectation — both ways.

    Answering an out-of-corpus question is a hallucination; refusing an
    answerable one is an unnecessary failure. Both are counted here.
    """
    return float(is_refusal(answer) == expects_refusal)


def citation_validity(answer: str, n_sources: int) -> float:
    """Share of [Sn] tags that point at a source that was actually provided."""
    tags = [int(t) for t in _CITATION.findall(answer)]
    if not tags:
        return 0.0
    return sum(1 for t in tags if 1 <= t <= n_sources) / len(tags)


def has_citation(answer: str) -> float:
    return float(bool(_CITATION.search(answer)))


@dataclass
class JudgeVerdict:
    score: float
    reason: str
    raw: str = ""


class FaithfulnessJudge:
    """LLM-as-a-judge grader over (context, question, answer)."""

    def __init__(
        self,
        llm: BaseLLM,
        prompts: PromptRegistry,
        prompt_ref: str = "judge_faithfulness@latest",
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.template = prompts.get(prompt_ref)

    def score(self, question: str, context: str, answer: str) -> JudgeVerdict:
        messages = self.template.render(question=question, context=context, answer=answer)
        response = self.llm.chat(messages, temperature=0.0, max_tokens=200)
        return self._parse(response.text)

    @staticmethod
    def _parse(text: str) -> JudgeVerdict:
        match = _JSON_OBJECT.search(text)
        if not match:
            return JudgeVerdict(score=0.0, reason="unparsable judge output", raw=text)
        try:
            payload = json.loads(match.group(0))
            score = float(payload.get("score", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return JudgeVerdict(score=0.0, reason="unparsable judge output", raw=text)
        # Clamp to the documented scale so a confused judge cannot skew the mean.
        score = min(max(score, 0.0), 1.0)
        return JudgeVerdict(score=score, reason=str(payload.get("reason", ""))[:300], raw=text)


def aggregate_answers(per_item: list[dict]) -> dict[str, float]:
    if not per_item:
        return {}

    def mean(key: str, rows: list[dict]) -> float:
        values = [r[key] for r in rows if r.get(key) is not None]
        return sum(values) / len(values) if values else 0.0

    answerable = [r for r in per_item if not r["expects_refusal"]]
    out = {
        "refusal_accuracy": mean("refusal_correct", per_item),
        "token_recall": mean("token_recall", answerable),
        "citation_rate": mean("has_citation", answerable),
        "citation_validity": mean("citation_validity", answerable),
        "mean_answer_words": mean("answer_words", per_item),
        "mean_latency_ms": mean("latency_ms", per_item),
    }
    judged = [r for r in per_item if r.get("faithfulness") is not None]
    if judged:
        out["faithfulness"] = mean("faithfulness", judged)
    return {k: round(v, 4) for k, v in out.items()}
