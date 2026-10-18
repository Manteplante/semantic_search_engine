"""Semantic search: embed every corpus chunk once, then rank chunks by cosine similarity.

The index is `embeddings.npy` in search/output/ (one L2-normalised float16 row per
chunk), the `chunks` and `paragraphs` tables in the document database (ingestion/db.py;
chunk n is row n of the matrix), and `index.json` recording which embedder built it.
Ranking is a single matrix product over a memory-mapped matrix - no vector database
needed - and only the best rows' text is then read from the database.

Each document's vectors are also cached in `vectors/<doc_id>-<key>.npy`, where the key
hashes exactly what was embedded (embedder, pooling, prefix and chunk texts). `make index`
re-embeds only documents whose key changed, then joins the cached blocks.

Cosine similarity orders passages well but cannot say "none of these fit", so
the best candidates go through a reranker (a cross-encoder reading question and
passage together), whose 0-1 relevance takes a cutoff. Hits are then grouped by
document, so one long document cannot fill every slot.

`make search-paragraph` uses the same first stage, then reranks the whole
paragraphs those chunks cover, and shows each in full with the PDF page it is on.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConfigError
from .corpus import Section, passage_header
from .model import ModelSpec

EMBEDDINGS = "embeddings.npy"
INDEX_INFO = "index.json"
VECTORS = "vectors"  # per-document cache
VECTOR_DTYPE = np.float16  # halves disk and memory; cosine ranking barely notices
SCORE_BLOCK = 65536  # rows scored at once, so a float16 matrix is never copied whole


def passage_text(row: dict) -> str:
    """What gets embedded for a chunk: document title and section heading, then the text.

    The header tells the model which document a passage is from, even when the
    passage itself never names its subject.
    """
    header = passage_header(row.get("title"), row.get("section"))
    return f"{header}\n{row['text']}" if header else row["text"]


def cite(row: dict) -> str:
    """How a document is cited: title (author, year)."""
    who = ", ".join(str(part) for part in (row.get("author"), row.get("year")) if part)
    name = row.get("title") or row["document"]
    return f"{name} ({who})" if who else name


def describe(row: dict) -> str:
    """How a chunk is cited: title (author, year) - section."""
    return f"{cite(row)} - {row['section']}"


def pages(row: dict) -> str:
    """ "PDF p. 12", "PDF p. 12-13", or "" for markdown parsed before page markers."""
    first, last = row.get("page"), row.get("last_page")
    if first is None:
        return ""
    return f"PDF p. {first}" if last in (None, first) else f"PDF p. {first}-{last}"


def cite_paragraph(row: dict) -> str:
    """Where a paragraph is: collection · section · PDF page(s)."""
    parts = [row["collection"], row.get("section") or "(title page)"]
    return " · ".join([*parts, pages(row)] if pages(row) else parts)


def pdf_path(row: dict, data_dir: Path) -> Path:
    """The PDF the paragraph was parsed from: data/<collection>/<doc_id>.pdf."""
    return data_dir / row["collection"] / f"{row['document']}.pdf"


def embed(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    prefix: str = "",
    max_length: int = 512,
    batch_size: int = 32,
    pooling: str = "mean",
) -> np.ndarray:
    """L2-normalised embeddings, one row per text: the mean of the token vectors, or
    (`pooling="last"`) the vector of each text's last real token."""
    import torch

    rows = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
        if pooling == "last":
            pooled = last_token(hidden, encoded["attention_mask"])
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        rows.append(pooled.float().cpu().numpy())
    return np.concatenate(rows) if rows else np.zeros((0, 0), dtype=np.float32)


def last_token(hidden: Any, mask: Any) -> Any:
    """Each row's vector at its last non-padding position, for left or right padding."""
    import torch

    if bool(mask[:, -1].all()):  # left padding: every text ends at the last position
        return hidden[:, -1]
    last = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def rerank(
    question: str,
    texts: list[str],
    tokenizer: Any,
    model: Any,
    max_length: int = 512,
    batch_size: int = 32,
) -> np.ndarray:
    """Relevance of each text to the question, 0-1: the cross-encoder's logit through a sigmoid."""
    import torch

    scores = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            [question] * len(batch),
            batch,
            padding=True,
            truncation="only_second",
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            logits = model(**encoded).logits.view(-1)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


def keep_relevant(hits: list[tuple[dict, float]], min_relevance: float) -> list[tuple[dict, float]]:
    """Reranked (row, score) hits, best first, without those under the cutoff."""
    ranked = sorted(hits, key=lambda hit: -hit[1])
    return [(row, score) for row, score in ranked if score >= min_relevance]


def top_k(matrix: np.ndarray, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
    """(row index, cosine score) of the k best rows; both sides are already normalised."""
    if matrix.size == 0:
        return []
    vector = vector.astype(np.float32)
    scores = np.concatenate(
        [
            matrix[start : start + SCORE_BLOCK].astype(np.float32) @ vector
            for start in range(0, matrix.shape[0], SCORE_BLOCK)
        ]
    )
    best = np.argsort(-scores)[:k]
    return [(int(index), float(scores[index])) for index in best]


def group_by_document(
    hits: list[tuple[dict, float]], k: int, per_document: int
) -> list[tuple[float, list[tuple[dict, float]]]]:
    """The k best documents from best-first (row, score) hits, each with its best passages.

    A document ranks by its best chunk. Returns (best score, [(row, score), ...]) per document.
    """
    found: dict[str, list[tuple[dict, float]]] = {}
    for row, score in hits:
        key = f"{row['collection']}/{row['document']}"
        if key not in found:
            if len(found) == k:
                continue
            found[key] = []
        if len(found[key]) < per_document:
            found[key].append((row, score))
    return [(passages[0][1], passages) for passages in found.values()]


def inspect_report(
    sections: list[Section],
    rows: list[dict],
    prefix: str,
    count_tokens: Callable[[str], int],
    paragraphs: Sequence[dict] = (),
) -> list[str]:
    """`make inspect`: every string exactly as the embedder reads it, the paragraphs
    `make search-paragraph` can show, then what was left out."""
    lines = []
    for row in rows:
        text = prefix + passage_text(row)
        lines += [f"--- {row['id']}  ({count_tokens(text)} tokens)", text, ""]
    for row in paragraphs:
        where = pages(row) or "no page"
        lines += [
            f"--- {row['id']}  {where}  ({count_tokens(row['text'])} tokens)",
            row["text"],
            "",
        ]
    for section in sections:
        if section.reason is not None:
            snippet = " ".join(section.text.split())[:80]
            lines.append(
                f"[dropped: {section.reason}] {section.heading or '(title page)'}: {snippet}"
            )
    return lines


def vector_key(embedder: ModelSpec, texts: Sequence[str]) -> str:
    """Hash of exactly what a document's vectors depend on: change any part and they're redone."""
    payload = [embedder.id, embedder.pooling, embedder.passage_prefix, embedder.max_length]
    blob = json.dumps([*payload, list(texts)], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def cached_vectors(
    cache_dir: Path, document: str, key: str, texts: Sequence[str], embed_texts: Callable
) -> tuple[np.ndarray, bool]:
    """A document's chunk vectors, from the cache if its key matches; (vectors, embedded now)."""
    path = cache_dir / f"{document}-{key}.npy"
    if path.exists():
        return np.load(path), False
    vectors = np.asarray(embed_texts(list(texts)) if texts else np.zeros((0, 0)), VECTOR_DTYPE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale in cache_dir.glob(f"{document}-*.npy"):
        if cached_document(stale) == document:  # not another id that merely starts like it
            stale.unlink()
    np.save(path, vectors)
    return vectors, True


def cached_document(path: Path) -> str:
    """The doc_id a cache file belongs to: its name minus the trailing -<key>."""
    return path.stem.rsplit("-", 1)[0]


def prune_vectors(cache_dir: Path, keep: set[str]) -> None:
    """Drop cached vectors of documents no longer in the index (`keep`: their doc_ids)."""
    for path in cache_dir.glob("*.npy"):
        if cached_document(path) not in keep:
            path.unlink()


def join_vectors(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """One matrix from per-document blocks, in order; documents without chunks add nothing."""
    blocks = [block for block in blocks if len(block)]
    return np.concatenate(blocks) if blocks else np.zeros((0, 0), dtype=VECTOR_DTYPE)


def save_index(output_dir: Path, embeddings: np.ndarray, embedder: ModelSpec) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / EMBEDDINGS, embeddings.astype(VECTOR_DTYPE))
    info = {"embedder": embedder.id, "pooling": embedder.pooling, "rows": int(len(embeddings))}
    (output_dir / INDEX_INFO).write_text(json.dumps(info, indent=2) + "\n")


def load_index(output_dir: Path, embedder: ModelSpec) -> np.ndarray:
    """The saved matrix, memory-mapped, refusing one built differently than model.yaml says."""
    info_path = output_dir / INDEX_INFO
    if not info_path.exists():
        raise ConfigError("No search index yet. Run: make index")
    info = json.loads(info_path.read_text())
    if info.get("embedder") != embedder.id:
        raise ConfigError(
            f"The search index was built with {info.get('embedder')!r} but model.yaml now "
            f"names {embedder.id!r}. Run: make index"
        )
    if info.get("pooling", "mean") != embedder.pooling:
        raise ConfigError(
            f"The search index was built with {info.get('pooling', 'mean')!r} pooling but "
            f"model.yaml now says {embedder.pooling!r}. Run: make index"
        )
    return np.load(output_dir / EMBEDDINGS, mmap_mode="r")
