import pytest

from voicerag.prompts.registry import PromptError, PromptRegistry


@pytest.fixture
def registry(prompts_dir):
    return PromptRegistry(prompts_dir)


def test_all_prompt_versions_load(registry):
    assert "rag_answer" in registry.names()
    versions = registry.versions("rag_answer")
    # Contiguous from 1: a gap means a version file was deleted rather than
    # superseded, which breaks the audit trail every recorded run depends on.
    assert versions == list(range(1, len(versions) + 1))
    assert len(versions) >= 3


def test_latest_resolves_to_highest_version(registry):
    assert registry.get("rag_answer@latest").version == max(registry.versions("rag_answer"))
    assert registry.get("rag_answer").version == registry.get("rag_answer@latest").version


def test_pinned_version_is_honoured(registry):
    assert registry.get("rag_answer@v2").version == 2


def test_unknown_prompt_and_version_raise(registry):
    with pytest.raises(PromptError):
        registry.get("does_not_exist@v1")
    with pytest.raises(PromptError):
        registry.get("rag_answer@v99")
    with pytest.raises(PromptError):
        registry.get("rag_answer@banana")


def test_hash_is_stable_and_version_specific(registry):
    v2, v3 = registry.get("rag_answer@v2"), registry.get("rag_answer@v3")
    assert v2.sha256 == registry.get("rag_answer@v2").sha256
    assert v2.sha256 != v3.sha256


def test_render_produces_system_and_user_messages(registry):
    messages = registry.get("rag_answer@v3").render(
        question="What GPU?",
        chunks=[{"source": "gpu.md", "text": "The card has 8 GB."}],
    )
    assert [m.role for m in messages] == ["system", "user"]
    assert "[S1] (gpu.md)" in messages[1].content
    assert "What GPU?" in messages[1].content


def test_missing_variable_raises_instead_of_rendering_blank(registry):
    with pytest.raises(PromptError):
        registry.get("rag_answer@v3").render(question="only question")


def test_production_prompt_enforces_grounding(registry):
    system = registry.get("rag_answer@latest").system.lower()
    assert "only from the numbered sources" in system
    assert "i cannot answer this from the available documents" in system


def test_diff_between_versions_is_non_empty(registry):
    diff = registry.diff("rag_answer@v1", "rag_answer@v3")
    assert "---" in diff and "+++" in diff


def test_catalog_marks_exactly_one_latest_per_prompt(registry):
    catalog = registry.catalog()
    for name in registry.names():
        rows = [r for r in catalog if r["name"] == name]
        assert sum(1 for r in rows if r["is_latest"]) == 1
