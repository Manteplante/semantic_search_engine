"""How the embedder is applied: read search.yaml and check it against model.yaml.

model.yaml says *which* model; search.yaml says how big the chunks are, which
sections to skip, and how many search hits to return. Every key is validated
here so a typo is an error with the valid keys listed, not a default silently used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config
from .config import ConfigError, settings
from .model import ModelSpec

SECTIONS = {"corpus", "search"}
FIELDS = {
    "corpus": {
        "chunk_tokens",
        "overlap_tokens",
        "skip_sections",
        "drop_lines",
        "min_section_tokens",
        "min_paragraph_tokens",
    },
    "search": {
        "top_k",
        "passages_per_document",
        "paragraphs",
        "rerank_candidates",
        "min_relevance",
        "batch_size",
    },
}
# Room for the embedder's special tokens and the "passage: " prefix.
PREFIX_MARGIN = 16


@dataclass(frozen=True)
class Options:
    chunk_tokens: int
    overlap_tokens: int
    skip_sections: tuple[str, ...]
    drop_lines: tuple[re.Pattern[str], ...]
    min_section_tokens: int
    min_paragraph_tokens: int
    top_k: int
    passages_per_document: int
    paragraphs: int
    rerank_candidates: int
    min_relevance: float
    batch_size: int


def section(data: dict[str, Any], name: str, path: Path) -> dict[str, Any]:
    block = data.get(name) or {}
    if not isinstance(block, dict):
        raise ConfigError(f"{path.name}: `{name}:` should be a mapping of settings")
    unknown = sorted(set(block) - FIELDS[name])
    if unknown:
        raise ConfigError(
            f"{path.name}: unknown {name} key {unknown[0]!r}. "
            f"Valid keys: {', '.join(sorted(FIELDS[name]))}"
        )
    return block


def patterns(found: Any, path: Path) -> tuple[re.Pattern[str], ...]:
    """`corpus.drop_lines`, compiled case-insensitively; a bad regex names itself."""
    compiled = []
    for pattern in found or ():
        try:
            compiled.append(re.compile(str(pattern), re.IGNORECASE))
        except re.error as problem:
            raise ConfigError(
                f"{path.name}: `corpus.drop_lines` pattern {pattern!r} is not a valid regex "
                f"({problem})"
            ) from problem
    return tuple(compiled)


def options(embedder: ModelSpec, path: Path | None = None) -> Options:
    """search.yaml, checked against the limits of the embedder that will run it."""
    path = path or settings().root / "search.yaml"
    data = config.read_yaml(path, SECTIONS, "section")
    corpus = section(data, "corpus", path)
    search = section(data, "search", path)
    found = Options(
        chunk_tokens=int(corpus.get("chunk_tokens", 400)),
        overlap_tokens=int(corpus.get("overlap_tokens", 50)),
        skip_sections=tuple(str(name) for name in corpus.get("skip_sections") or ()),
        drop_lines=patterns(corpus.get("drop_lines"), path),
        min_section_tokens=int(corpus.get("min_section_tokens", 0)),
        min_paragraph_tokens=int(corpus.get("min_paragraph_tokens", 0)),
        top_k=int(search.get("top_k", 5)),
        passages_per_document=int(search.get("passages_per_document", 2)),
        paragraphs=int(search.get("paragraphs", 5)),
        rerank_candidates=int(search.get("rerank_candidates", 50)),
        min_relevance=float(search.get("min_relevance", 0.0)),
        batch_size=int(search.get("batch_size", 32)),
    )
    if not 0 <= found.overlap_tokens < found.chunk_tokens:
        raise ConfigError(f"{path.name}: `corpus.overlap_tokens` must be below `chunk_tokens`")
    if found.chunk_tokens + PREFIX_MARGIN > embedder.max_length:
        raise ConfigError(
            f"{path.name}: `corpus.chunk_tokens` ({found.chunk_tokens}) does not fit the "
            f"embedder's max_length ({embedder.max_length}) - keep it at most "
            f"{embedder.max_length - PREFIX_MARGIN}"
        )
    if found.min_section_tokens < 0:
        raise ConfigError(f"{path.name}: `corpus.min_section_tokens` must be 0 or more")
    if found.min_paragraph_tokens < 0:
        raise ConfigError(f"{path.name}: `corpus.min_paragraph_tokens` must be 0 or more")
    if found.top_k < 1:
        raise ConfigError(f"{path.name}: `search.top_k` must be at least 1")
    if found.passages_per_document < 1:
        raise ConfigError(f"{path.name}: `search.passages_per_document` must be at least 1")
    if found.paragraphs < 1:
        raise ConfigError(f"{path.name}: `search.paragraphs` must be at least 1")
    if found.rerank_candidates < 0:
        raise ConfigError(f"{path.name}: `search.rerank_candidates` must be 0 (off) or more")
    if not 0 <= found.min_relevance <= 1:
        raise ConfigError(f"{path.name}: `search.min_relevance` is a 0-1 score")
    return found
