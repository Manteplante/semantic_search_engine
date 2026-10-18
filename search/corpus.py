"""Parsed documents -> section-aware chunks with metadata, the unit search works on.

Docling writes every heading as `##`, so a document arrives as a flat list of
sections. Front and back matter (table of contents, foreword, references) is
dropped by heading name from search.yaml, noise lines (captions, table rows,
title-page boilerplate) by the `drop_lines` patterns, and sections left too
short to mean anything by `min_section_tokens`. Each remaining section is
packed into chunks of at most `chunk_tokens` tokens - title and section heading
included, since they are embedded with the chunk - overlapping a little so a
sentence cut at a boundary is still found. The document metadata from ingestion's
database (title, authors, year) is joined on by doc_id - the markdown file's name - so
every chunk knows which document it is from.

Chunks are what search recalls; paragraphs are what `make search-paragraph`
shows. Each section is also kept as its paragraphs, with the PDF pages they are
on (from the parser's page markers), and every chunk lists the paragraphs it
covers. Docling cuts paragraphs at page breaks, so a block that stops mid-sentence
is joined to the one that finishes it.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from parsing.pdf_parser import PAGE_MARKER

CountTokens = Callable[[str], int]

HEADING = re.compile(r"^#{1,6}\s+(.*)$")
NUMBERING = re.compile(r"^\d+(\.\d+)*\.?\s*")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
DOT_LEADER = re.compile(r"\.{5,}")  # table-of-contents lines: "Method ........ 12"
IMAGE_PLACEHOLDER = "<!-- image -->"
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # PDF debris, e.g. NUL bytes in Docling output
PAGE = re.compile("^" + re.escape(PAGE_MARKER).replace(r"\{\}", r"(\d+)") + "$")
SENTENCE_STOPS = '.!?:;»"”)'  # a block ending in none of these was cut mid-sentence
SCRAP_WORDS = 6  # shorter blocks, or ones not starting with a letter, look like footnote scraps

# Why split_sections() left a section out; `make inspect` prints these.
SKIPPED = "skipped heading"
TOO_SHORT = "too short"


@dataclass(frozen=True)
class Document:
    """One parsed document on disk: parsing/output/<collection>/<doc_id>.md."""

    path: Path
    collection: str

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def where(self) -> str:
        return f"{self.collection}/{self.stem}"


def find_documents(input_dir: Path) -> list[Document]:
    return [
        Document(path=path, collection=path.parent.name)
        for path in sorted(input_dir.glob("*/*.md"))
    ]


def normalise_heading(heading: str) -> str:
    return NUMBERING.sub("", heading.strip()).strip().lower()


def is_noise(line: str, drop: Sequence[re.Pattern[str]]) -> bool:
    if IMAGE_PLACEHOLDER in line or DOT_LEADER.search(line):
        return True
    return any(pattern.search(line.strip()) for pattern in drop)


@dataclass(frozen=True)
class Paragraph:
    """A paragraph and the PDF pages it is on; None when the markdown has no page markers."""

    text: str
    page: int | None = None
    last_page: int | None = None


@dataclass(frozen=True)
class Section:
    heading: str
    paragraphs: tuple[Paragraph, ...]
    reason: str | None = None  # why it is left out of search; None when it is kept

    @property
    def text(self) -> str:
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)


def continues(previous: str, following: str) -> bool:
    """`previous` stops mid-sentence (usually at a page break) and `following` finishes it."""
    return previous[-1] not in SENTENCE_STOPS and following[0].islower()


def is_scrap(text: str) -> bool:
    """A footnote line or fragment (", 17-18", "84 See …", "Sloterdijk, Rules"), not body text."""
    return not text[0].isalpha() or len(text.split()) < SCRAP_WORDS


def continued(found: list[Paragraph], following: str) -> int | None:
    """Index of the paragraph `following` finishes, or None.

    Docling puts a page's footnotes between a paragraph cut at the page break and
    its continuation, so look back past footnote scraps to the body text first;
    only when there is none does the block right before count.
    """
    for index in range(len(found) - 1, -1, -1):
        if not is_scrap(found[index].text):
            if continues(found[index].text, following):
                return index
            break
    if found and continues(found[-1].text, following):
        return len(found) - 1
    return None


def paragraphs(
    lines: Iterable[tuple[str, int | None]], drop: Sequence[re.Pattern[str]] = ()
) -> list[Paragraph]:
    """Blank-line separated blocks of (line, page), minus noise, with page-break cuts rejoined.

    A paragraph rejoined past footnote scraps keeps its place; the scraps follow it.
    """
    blocks: list[list[tuple[str, int | None]]] = [[]]
    for line, page in lines:
        text = html.unescape(CONTROL.sub("", line)).strip()
        if not text:
            if blocks[-1]:
                blocks.append([])
        elif not is_noise(line, drop):
            blocks[-1].append((text, page))
    found: list[Paragraph] = []
    for block in blocks:
        if not block:
            continue
        text = "\n".join(line for line, _ in block)
        index = continued(found, text)
        if index is None:
            found.append(Paragraph(text, block[0][1], block[-1][1]))
        else:
            joined = found[index]
            found[index] = Paragraph(f"{joined.text} {text}", joined.page, block[-1][1])
    return found


def sort_sections(
    markdown: str,
    skip: Iterable[str] = (),
    drop: Sequence[re.Pattern[str]] = (),
    min_tokens: int = 0,
    count_tokens: CountTokens | None = None,
) -> list[Section]:
    """One Section per `#` heading, with a reason when it is left out of search.

    Text before the first heading (the title page) gets the heading "". Sections
    that are empty once noise lines are gone vanish without a reason. Page markers
    set the page of the lines after them and never become text.
    """
    skipped = {normalise_heading(name) for name in skip}
    count = count_tokens or (lambda text: len(text.split()))
    page = None
    raw: list[tuple[str, list[tuple[str, int | None]]]] = [("", [])]
    for line in markdown.splitlines():
        marker = PAGE.match(line.strip())
        heading = HEADING.match(line)
        if marker:
            page = int(marker.group(1))
            raw[-1][1].append(("", page))  # a page starts a new block; continues() rejoins
        elif heading:
            raw.append((html.unescape(CONTROL.sub("", heading.group(1)).strip()), []))
        else:
            raw[-1][1].append((line, page))
    found = []
    for heading, lines in raw:
        section = Section(heading, tuple(paragraphs(lines, drop)))
        if not section.paragraphs:
            continue
        if normalise_heading(heading) in skipped:
            section = Section(heading, section.paragraphs, SKIPPED)
        elif count(section.text) < min_tokens:
            section = Section(heading, section.paragraphs, TOO_SHORT)
        found.append(section)
    return found


def split_sections(
    markdown: str,
    skip: Iterable[str] = (),
    drop: Sequence[re.Pattern[str]] = (),
    min_tokens: int = 0,
    count_tokens: CountTokens | None = None,
) -> list[tuple[str, str]]:
    """(heading, text) for every section worth searching - see sort_sections()."""
    return [
        (section.heading, section.text)
        for section in sort_sections(markdown, skip, drop, min_tokens, count_tokens)
        if section.reason is None
    ]


def passage_header(title: str | None, section: str | None) -> str:
    """Document title and section heading: embedded above every chunk, so each names its source."""
    return "\n".join(part for part in (title, section) if part)


def split_long(text: str, count_tokens: CountTokens, size: int) -> list[str]:
    """A run of words longer than `size` tokens, cut into pieces that each fit."""
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for word in text.split():
        cost = count_tokens(word)
        if current and used + cost > size:
            pieces.append(" ".join(current))
            current, used = [], 0
        current.append(word)
        used += cost
    if current:
        pieces.append(" ".join(current))
    return pieces


def sentences(text: str, count_tokens: CountTokens, size: int) -> list[tuple[str, int]]:
    """Sentences with their token counts, none longer than `size`."""
    found = []
    for sentence in SENTENCE_END.split(" ".join(text.split())):
        if not sentence:
            continue
        cost = count_tokens(sentence)
        if cost <= size:
            found.append((sentence, cost))
        else:
            found += [
                (piece, count_tokens(piece)) for piece in split_long(sentence, count_tokens, size)
            ]
    return found


def packed(current: list[tuple[str, int, int]]) -> tuple[str, tuple[int, ...]]:
    """A chunk's text and the indices of the paragraphs its sentences come from."""
    return " ".join(s for s, _, _ in current), tuple(sorted({n for _, _, n in current}))


def chunk_paragraphs(
    texts: Sequence[str], count_tokens: CountTokens, size: int, overlap: int = 0
) -> list[tuple[str, tuple[int, ...]]]:
    """Pack whole sentences of consecutive paragraphs into chunks of at most `size` tokens.

    Each new chunk starts with the last `overlap` tokens' worth of sentences from
    the one before. Token counts are summed per sentence, which is close enough
    to the tokenizer's count of the joined text to stay under a model's limit.
    Every chunk comes with the indices (into `texts`) of the paragraphs it covers.
    """
    chunks: list[tuple[str, tuple[int, ...]]] = []
    current: list[tuple[str, int, int]] = []  # (sentence, tokens, paragraph index)
    for number, text in enumerate(texts):
        for sentence, cost in sentences(text, count_tokens, size):
            if current and sum(c for _, c, _ in current) + cost > size:
                chunks.append(packed(current))
                tail: list[tuple[str, int, int]] = []
                for previous in reversed(current):
                    if sum(c for _, c, _ in tail) + previous[1] > overlap:
                        break
                    tail.insert(0, previous)
                current = tail if sum(c for _, c, _ in tail) + cost <= size else []
            current.append((sentence, cost, number))
    if current:
        chunks.append(packed(current))
    return chunks


def chunk(text: str, count_tokens: CountTokens, size: int, overlap: int = 0) -> list[str]:
    """chunk_paragraphs() for a single text, without the paragraph indices."""
    return [piece for piece, _ in chunk_paragraphs([text], count_tokens, size, overlap)]


def document_metadata(row: dict) -> dict:
    """What every chunk of a document carries about it, from its database row."""
    return {
        "title": row.get("title"),
        "author": "; ".join(row.get("authors") or []) or None,
        "year": row.get("year"),
    }


def load_metadata(rows: Iterable[dict]) -> dict[str, dict]:
    """Document database rows as chunk metadata, keyed like the parsed files: by doc_id."""
    return {row["doc_id"]: document_metadata(row) for row in rows}


def document_corpus(
    document: Document,
    metadata: dict[str, str] | None,
    count_tokens: CountTokens,
    size: int,
    overlap: int,
    skip: Iterable[str],
    drop: Sequence[re.Pattern[str]] = (),
    min_tokens: int = 0,
    min_paragraph_tokens: int = 0,
) -> tuple[list[dict], list[dict]]:
    """(chunk rows, paragraph rows) for one document, each carrying its metadata.

    The header (title + section) is embedded with each chunk, so it comes out of
    the `size` budget - the embedded text stays within what the model can read.
    A header longer than half of `size` (no real title is) only gets half, so
    chunks never shrink to a few tokens.

    Paragraphs shorter than `min_paragraph_tokens` (footnote scraps, stray
    captions) still feed the chunks but get no row: they are never shown alone.
    A chunk's `paragraphs` lists the ids of the paragraph rows it covers.
    """
    metadata = metadata or {}
    title = " ".join(metadata["title"].split()) if metadata.get("title") else None
    where = document.where
    document_fields = {
        "document": document.stem,
        "collection": document.collection,
        "author": metadata.get("author"),
        "year": metadata.get("year"),
        "title": title,
    }
    chunks: list[dict] = []
    rows: list[dict] = []
    markdown = document.path.read_text(encoding="utf-8")
    for section in sort_sections(markdown, skip, drop, min_tokens, count_tokens):
        if section.reason is not None:
            continue
        ids: list[str | None] = []
        for paragraph in section.paragraphs:
            if count_tokens(paragraph.text) < min_paragraph_tokens:
                ids.append(None)
                continue
            ids.append(f"{where}#p{len(rows)}")
            rows.append(
                {"id": ids[-1], **document_fields, "section": section.heading}
                | {"page": paragraph.page, "last_page": paragraph.last_page, "text": paragraph.text}
            )
        budget = max(size - count_tokens(passage_header(title, section.heading)), size // 2)
        texts = [paragraph.text for paragraph in section.paragraphs]
        for piece, covered in chunk_paragraphs(
            texts, count_tokens, budget, min(overlap, budget // 2)
        ):
            chunks.append(
                {"id": f"{where}#{len(chunks)}", **document_fields, "section": section.heading}
                | {"text": piece, "paragraphs": [ids[n] for n in covered if ids[n]]}
            )
    return chunks, rows


def document_rows(*args, **kwargs) -> list[dict]:
    """Only the chunk rows of document_corpus()."""
    return document_corpus(*args, **kwargs)[0]
