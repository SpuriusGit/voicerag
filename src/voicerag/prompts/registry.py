"""Versioned prompt registry.

Prompts live in ``prompts/<name>/v<N>.yaml`` and are treated like code: every
change is a new file, never an in-place edit. Each rendered prompt carries a
``sha256`` of its template, which is logged next to every answer — so any
recorded experiment can be traced back to the exact wording that produced it.

Reference syntax: ``rag_answer@v2`` (pinned) or ``rag_answer@latest``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined, TemplateError

from voicerag.llm.base import ChatMessage

_VERSION_FILE = re.compile(r"^v(\d+)\.ya?ml$")
_ENV = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


class PromptError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    system: str
    user: str
    description: str = ""
    variables: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict, repr=False)
    path: Path | None = field(default=None, repr=False)

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"

    @property
    def sha256(self) -> str:
        """Stable fingerprint of the template text (not of the rendered output)."""
        payload = f"{self.name}|{self.version}|{self.system}|{self.user}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def render(self, **variables: object) -> list[ChatMessage]:
        missing = [v for v in self.variables if v not in variables]
        if missing:
            raise PromptError(f"{self.ref}: missing variables {missing}")
        try:
            system = _ENV.from_string(self.system).render(**variables).strip()
            user = _ENV.from_string(self.user).render(**variables).strip()
        except TemplateError as exc:
            raise PromptError(f"{self.ref}: render failed: {exc}") from exc

        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=user))
        return messages

    @classmethod
    def from_file(cls, path: Path) -> PromptTemplate:
        match = _VERSION_FILE.match(path.name)
        if not match:
            raise PromptError(f"{path}: filename must look like v1.yaml")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "user" not in raw:
            raise PromptError(f"{path}: 'user' section is required")

        declared = raw.get("version")
        version = int(match.group(1))
        if declared is not None and int(declared) != version:
            raise PromptError(
                f"{path}: version field ({declared}) disagrees with filename (v{version})"
            )
        return cls(
            name=raw.get("name") or path.parent.name,
            version=version,
            system=str(raw.get("system", "")),
            user=str(raw["user"]),
            description=str(raw.get("description", "")),
            variables=tuple(raw.get("variables", ()) or ()),
            metadata={
                k: v
                for k, v in raw.items()
                if k not in {"name", "version", "system", "user", "description", "variables"}
            },
            path=path,
        )


class PromptRegistry:
    """Loads every prompt version found under ``root`` and resolves references."""

    def __init__(self, root: Path | str = "prompts") -> None:
        self.root = Path(root)
        self._cache: dict[str, dict[int, PromptTemplate]] = {}
        self._load()

    def _load(self) -> None:
        if not self.root.is_dir():
            raise PromptError(f"Prompt directory not found: {self.root}")
        for prompt_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            versions: dict[int, PromptTemplate] = {}
            for file in sorted(prompt_dir.iterdir()):
                if not _VERSION_FILE.match(file.name):
                    continue
                template = PromptTemplate.from_file(file)
                versions[template.version] = template
            if versions:
                self._cache[prompt_dir.name] = versions

    def names(self) -> list[str]:
        return sorted(self._cache)

    def versions(self, name: str) -> list[int]:
        if name not in self._cache:
            raise PromptError(f"Unknown prompt {name!r}; available: {self.names()}")
        return sorted(self._cache[name])

    def get(self, ref: str) -> PromptTemplate:
        """Resolve ``name@vN`` / ``name@latest`` / ``name`` to a template."""
        name, _, version_part = ref.partition("@")
        name = name.strip()
        if name not in self._cache:
            raise PromptError(f"Unknown prompt {name!r}; available: {self.names()}")
        versions = self._cache[name]

        if version_part in ("", "latest"):
            return versions[max(versions)]
        normalized = version_part.lstrip("vV")
        if not normalized.isdigit():
            raise PromptError(f"Bad prompt reference {ref!r}; expected name@v2 or name@latest")
        version = int(normalized)
        if version not in versions:
            raise PromptError(f"{name}: version v{version} not found (have {sorted(versions)})")
        return versions[version]

    def diff(self, ref_a: str, ref_b: str) -> str:
        """Unified diff between two prompt versions — used in prompt code review."""
        import difflib

        a, b = self.get(ref_a), self.get(ref_b)
        return "\n".join(
            difflib.unified_diff(
                f"{a.system}\n---\n{a.user}".splitlines(),
                f"{b.system}\n---\n{b.user}".splitlines(),
                fromfile=a.ref,
                tofile=b.ref,
                lineterm="",
            )
        )

    def catalog(self) -> list[dict]:
        """Machine-readable listing, exposed at ``GET /prompts``."""
        return [
            {
                "name": name,
                "version": tpl.version,
                "ref": tpl.ref,
                "sha256": tpl.sha256,
                "description": tpl.description,
                "variables": list(tpl.variables),
                "is_latest": tpl.version == max(versions),
                "metadata": tpl.metadata,
            }
            for name, versions in sorted(self._cache.items())
            for tpl in (versions[v] for v in sorted(versions))
        ]
