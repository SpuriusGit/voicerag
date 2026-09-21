from voicerag.rag.chunking import chunk_document, chunk_documents
from voicerag.rag.documents import Document


def _doc(text: str) -> Document:
    return Document(doc_id="d1", text=text, source="demo.md")


def test_heading_path_is_tracked_and_prepended():
    doc = _doc("# Top\n\nintro text here\n\n## Child\n\nchild body text\n\n### Leaf\n\nleaf body")
    chunks = chunk_document(doc, chunk_size=400, chunk_overlap=0)

    headings = [c.metadata["heading"] for c in chunks]
    assert "Top" in headings
    assert "Top > Child" in headings
    assert "Top > Child > Leaf" in headings
    leaf = next(c for c in chunks if c.metadata["heading"] == "Top > Child > Leaf")
    assert leaf.text.startswith("Top > Child > Leaf")


def test_sibling_heading_does_not_inherit_previous_branch():
    doc = _doc("# Top\n\n## A\n\nbody a\n\n## B\n\nbody b")
    headings = {c.metadata["heading"] for c in chunk_document(doc, 400, 0)}
    assert "Top > B" in headings
    assert not any(h.startswith("Top > A > ") for h in headings)


def test_chunks_respect_size_budget():
    doc = _doc("# T\n\n" + "\n\n".join(f"sentence number {i} here." for i in range(200)))
    for chunk in chunk_document(doc, chunk_size=300, chunk_overlap=50):
        # the heading prefix is added on top of the body budget
        assert len(chunk.text) <= 300 + len("T\n\n") + 50


def test_overlap_carries_context_forward():
    doc = _doc("# T\n\n" + "\n\n".join(f"paragraph {i} " + "w " * 40 for i in range(6)))
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=80)
    assert len(chunks) > 1
    overlaps = [
        any(chunks[i].text[-40:].strip() in chunks[i + 1].text for i in range(len(chunks) - 1))
    ]
    assert any(overlaps)


def test_oversized_single_paragraph_is_split():
    doc = _doc("# T\n\n" + "x" * 5000)
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=0)
    assert len(chunks) > 5


def test_chunk_ids_are_stable_across_runs():
    doc = _doc("# T\n\nsome stable body text for hashing")
    first = chunk_document(doc, 400, 0)
    second = chunk_document(doc, 400, 0)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_real_corpus_chunks_are_non_empty(corpus_dir):
    from voicerag.rag.loaders import load_directory

    chunks = chunk_documents(load_directory(corpus_dir), 800, 120)
    assert len(chunks) > 10
    assert all(c.text.strip() for c in chunks)
    assert {c.source for c in chunks} == {p.name for p in corpus_dir.glob("*.md")}
