import pytest

from voicerag.eval.answer_metrics import (
    FaithfulnessJudge,
    citation_validity,
    has_citation,
    refusal_correct,
    token_recall,
)
from voicerag.eval.dataset import load_eval_set
from voicerag.eval.retrieval_metrics import (
    aggregate_retrieval,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from voicerag.eval.runner import compare_reports, evaluate
from voicerag.llm.echo import EchoLLM
from voicerag.rag.pipeline import REFUSAL


def test_hit_and_recall_at_k():
    retrieved = ["a.md", "b.md", "c.md"]
    assert hit_at_k(retrieved, ["c.md"], 3) == 1.0
    assert hit_at_k(retrieved, ["c.md"], 2) == 0.0
    assert recall_at_k(retrieved, ["a.md", "z.md"], 3) == 0.5
    assert precision_at_k(retrieved, ["a.md"], 2) == 0.5


def test_reciprocal_rank_and_ndcg_reward_earlier_positions():
    assert reciprocal_rank(["x", "y", "gold"], ["gold"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], ["gold"]) == 0.0
    assert ndcg_at_k(["gold", "x"], ["gold"], 2) == 1.0
    assert ndcg_at_k(["x", "gold"], ["gold"], 2) < 1.0


def test_items_without_ground_truth_are_excluded_from_retrieval_metrics():
    rows = [
        {"retrieved": ["a.md"], "relevant": ["a.md"]},
        {"retrieved": ["b.md"], "relevant": []},  # out-of-corpus item
    ]
    assert aggregate_retrieval(rows, ks=(1,))["hit@1"] == 1.0
    assert aggregate_retrieval([rows[1]], ks=(1,)) == {}


def test_refusal_accuracy_penalises_both_directions():
    assert refusal_correct(REFUSAL, expects_refusal=True) == 1.0
    assert refusal_correct("Canberra.", expects_refusal=True) == 0.0
    assert refusal_correct(REFUSAL, expects_refusal=False) == 0.0


def test_citation_metrics():
    assert has_citation("8 GB [S1]") == 1.0
    assert citation_validity("uses [S1] and [S7]", n_sources=4) == 0.5
    assert citation_validity("no tags", n_sources=4) == 0.0


def test_token_recall_ignores_stopwords():
    assert token_recall("The card has 8 GB of VRAM", "8 GB VRAM") == 1.0
    assert token_recall("unrelated sentence", "8 GB VRAM") == 0.0


def test_judge_parses_json_and_clamps_out_of_range_scores():
    assert FaithfulnessJudge._parse('{"score": 0.5, "reason": "partly"}').score == 0.5
    assert FaithfulnessJudge._parse('{"score": 7}').score == 1.0
    assert FaithfulnessJudge._parse("not json at all").score == 0.0


def test_judge_scores_through_a_stub_llm(pipeline):
    judge = FaithfulnessJudge(
        EchoLLM(canned='{"score": 1.0, "reason": "supported by S1"}'), pipeline.prompts
    )
    verdict = judge.score("q", "[S1] context", "answer [S1]")
    assert verdict.score == 1.0
    assert "supported" in verdict.reason


def test_eval_set_is_well_formed(eval_set_path):
    items = load_eval_set(eval_set_path)
    assert len(items) >= 20
    assert {i.id for i in items}.__len__() == len(items), "duplicate ids"
    assert any(i.expects_refusal for i in items), "no out-of-corpus items"
    for item in items:
        if item.expects_refusal:
            assert not item.relevant_sources
        else:
            assert item.relevant_sources


def test_eval_references_point_at_files_that_exist(eval_set_path, corpus_dir):
    names = {p.name for p in corpus_dir.glob("*.md")}
    for item in load_eval_set(eval_set_path):
        assert set(item.relevant_sources) <= names, item.id


def test_evaluate_produces_a_report_with_config_provenance(pipeline, eval_set_path, tmp_path):
    items = load_eval_set(eval_set_path)[:6]
    report = evaluate(pipeline, items, run_id="unit", sample_resources=False)

    assert len(report.items) == 6
    assert "hit@4" in report.retrieval
    assert "refusal_accuracy" in report.answers
    assert "rag_answer@v" in report.config["prompt"]
    assert report.config["chunks_indexed"] > 0

    saved = report.save(tmp_path)
    assert saved.exists()
    assert (tmp_path / "unit.md").read_text().startswith("# Evaluation run")


def test_retrieval_quality_on_the_real_corpus_meets_the_baseline(pipeline, eval_set_path):
    """Guardrail: a chunking or fusion change that drops hit@4 fails CI."""
    items = load_eval_set(eval_set_path)
    report = evaluate(pipeline, items, run_id="guardrail", sample_resources=False)
    assert report.retrieval["hit@4"] >= 0.85
    assert report.retrieval["mrr"] >= 0.75


def test_compare_reports_renders_a_markdown_table(pipeline, eval_set_path):
    items = load_eval_set(eval_set_path)[:4]
    a = evaluate(pipeline, items, run_id="run-a", sample_resources=False)
    b = evaluate(pipeline, items, run_id="run-b", sample_resources=False)
    table = compare_reports([a, b])
    assert "run-a" in table and "run-b" in table and "hit@4" in table
