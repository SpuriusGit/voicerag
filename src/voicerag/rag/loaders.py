"""Corpus loading from the local filesystem."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from voicerag.rag.documents import Document, stable_id

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


def load_directory(
    root: Path | str,
    suffixes: Iterable[str] = TEXT_SUFFIXES,
    encoding: str = "utf-8",
) -> list[Document]:
    """Read every supported file under ``root`` into a :class:`Document`."""
    return list(iter_directory(root, suffixes, encoding))


def iter_directory(
    root: Path | str,
    suffixes: Iterable[str] = TEXT_SUFFIXES,
    encoding: str = "utf-8",
) -> Iterator[Document]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root_path}")
    wanted = {s.lower() for s in suffixes}

    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in wanted:
            continue
        text = path.read_text(encoding=encoding, errors="replace").strip()
        if not text:
            continue
        relative = path.relative_to(root_path).as_posix()
        yield Document(
            doc_id=stable_id(relative, text),
            text=text,
            source=relative,
            metadata={"path": str(path), "bytes": path.stat().st_size},
        )
