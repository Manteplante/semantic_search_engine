"""Command line for the search stage: `uv run python -m search <command>`.

The Makefile's Search targets call these. Only `get` touches the network.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from ingestion import db

from . import config, hardware
from .config import ConfigError, settings
from .corpus import (
    Document,
    document_corpus,
    document_metadata,
    find_documents,
    load_metadata,
    sort_sections,
)
from .model import download, load, load_tokenizer, specs
from .options import Options, options
from .search import (
    VECTORS,
    cached_vectors,
    cite,
    cite_paragraph,
    embed,
    group_by_document,
    inspect_report,
    join_vectors,
    keep_relevant,
    load_index,
    passage_text,
    pdf_path,
    prune_vectors,
    rerank,
    save_index,
    top_k,
    vector_key,
)


def token_counter(tokenizer):
    # verbose=False: counting a long paragraph is fine, it is never fed to the model whole.
    return lambda text: len(tokenizer(text, add_special_tokens=False, verbose=False).input_ids)


def corpus_for(
    document: Document, metadata: dict | None, count_tokens, chosen: Options
) -> tuple[list[dict], list[dict]]:
    """(chunk rows, paragraph rows) for one document, as search.yaml says to cut it."""
    return document_corpus(
        document,
        metadata,
        count_tokens,
        chosen.chunk_tokens,
        chosen.overlap_tokens,
        chosen.skip_sections,
        chosen.drop_lines,
        chosen.min_section_tokens,
        chosen.min_paragraph_tokens,
    )


def run_hardware(args):
    print("\n".join(hardware.report()))


def run_get(args):
    for found in specs().values():
        if found.downloaded and not args.force:
            print(f"[GET] {found.role}: {found.id} already in {found.path}")
            continue
        print(f"[GET] {found.role}: downloading {found.id} ...")
        print(f"[GET] saved to {download(found)}")


def parsed_documents(input_dir) -> list[tuple[dict, Document]]:
    """Database rows of the parsed documents, with their markdown on disk."""
    with db.engine().connect() as conn:
        rows = db.pending(conn, "parsed")
    found = []
    for row in rows:
        path = input_dir / row["collection"] / f"{row['doc_id']}.md"
        if path.exists():
            found.append((row, Document(path=path, collection=row["collection"])))
        else:
            print(f"[INDEX] missing {path} - run make parse FORCE=1 to rebuild it")
    return found


def run_index(args):
    where = settings()
    embedder = specs()["embedder"]
    chosen = options(embedder)
    documents = parsed_documents(where.input_dir)
    if not documents:
        raise ConfigError("No parsed documents in the database. Run: make ingest, make parse")
    tokenizer = load_tokenizer(embedder)
    count_tokens = token_counter(tokenizer)
    model = None

    def embed_texts(texts):
        nonlocal model
        if model is None:  # weights load only when some document actually needs embedding
            model = load(embedder)[1]
        return embed(
            texts,
            tokenizer,
            model,
            prefix=embedder.passage_prefix,
            max_length=embedder.max_length,
            batch_size=chosen.batch_size,
            pooling=embedder.pooling,
        )

    cache = where.output_dir / VECTORS
    rows, paragraphs, blocks, embedded = [], [], [], 0
    for row, document in documents:
        new, new_paragraphs = corpus_for(document, document_metadata(row), count_tokens, chosen)
        texts = [passage_text(chunk) for chunk in new]
        vectors, fresh = cached_vectors(
            cache, row["doc_id"], vector_key(embedder, texts), texts, embed_texts
        )
        embedded += fresh
        paged = "" if any(item["page"] for item in new_paragraphs) else " (no page markers)"
        print(
            f"[INDEX] {document.where}: {len(new)} chunks, "
            f"{len(new_paragraphs)} paragraphs{paged}{' (embedded)' if fresh else ''}"
        )
        rows += new
        paragraphs += new_paragraphs
        blocks.append(vectors)
    prune_vectors(cache, {row["doc_id"] for row, _ in documents})
    with db.engine().begin() as conn:
        db.replace_index(conn, rows, paragraphs)
    save_index(where.output_dir, join_vectors(blocks), embedder)
    print(
        f"[INDEX] {len(rows)} chunks from {len(documents)} documents, {embedded} embedded now with "
        f"{embedder.id}; saved to {where.output_dir}"
    )


def matches(wanted: str, stem: str, metadata: dict | None) -> bool:
    """`make inspect T=...`: part of the doc_id, an author or the title, any case."""
    fields = [stem, *((metadata or {}).get(key) or "" for key in ("author", "title"))]
    return any(wanted.lower() in str(field).lower() for field in fields)


def run_inspect(args):
    where = settings()
    embedder = specs()["embedder"]
    chosen = options(embedder)
    with db.engine().connect() as conn:
        metadata = load_metadata(db.all_documents(conn))
    documents = [
        document
        for document in find_documents(where.input_dir)
        if matches(args.document, document.stem, metadata.get(document.stem))
    ]
    if not documents:
        raise ConfigError(f"No parsed document in {where.input_dir} matches {args.document!r}")
    count_tokens = token_counter(load_tokenizer(embedder))
    for document in documents:
        found = metadata.get(document.stem)
        sections = sort_sections(
            document.path.read_text(encoding="utf-8"),
            chosen.skip_sections,
            chosen.drop_lines,
            chosen.min_section_tokens,
            count_tokens,
        )
        rows, paragraphs = corpus_for(document, found, count_tokens, chosen)
        print(f"[INSPECT] {document.where}: {len(rows)} chunks, {len(paragraphs)} paragraphs")
        report = inspect_report(sections, rows, embedder.passage_prefix, count_tokens, paragraphs)
        print("\n".join(report))


def recall(question: str, chosen: Options) -> list[tuple[dict, float]]:
    """First stage: chunks by cosine similarity, best first - `rerank_candidates` of them,
    or every chunk when reranking is off."""
    where = settings()
    embedder = specs()["embedder"]
    matrix = load_index(where.output_dir, embedder)
    with db.engine().connect() as conn:
        if db.count_chunks(conn) != matrix.shape[0]:
            raise ConfigError("The chunks table and embeddings.npy disagree. Run: make index")
    tokenizer, model = load(embedder)
    vector = embed(
        [question],
        tokenizer,
        model,
        embedder.query_prefix,
        embedder.max_length,
        pooling=embedder.pooling,
    )[0]
    candidates = top_k(matrix, vector, chosen.rerank_candidates or matrix.shape[0])
    with db.engine().connect() as conn:
        rows = db.chunk_rows(conn, [index for index, _ in candidates])
    return [(row, score) for row, (_, score) in zip(rows, candidates, strict=True)]


def relevant(question: str, rows: list[dict], chosen: Options) -> list[tuple[dict, float]]:
    """Second stage: rows the reranker scores at least `min_relevance`, best first."""
    reranker = specs()["reranker"]
    tokenizer, model = load(reranker)
    texts = [passage_text(row) for row in rows]
    scores = rerank(question, texts, tokenizer, model, reranker.max_length, chosen.batch_size)
    hits = [(row, float(score)) for row, score in zip(rows, scores, strict=True)]
    return keep_relevant(hits, chosen.min_relevance)


def find_hits(question: str, count: int | None) -> list[tuple[float, list[tuple[dict, float]]]]:
    """The best documents for a question, each with its best passages - see group_by_document."""
    chosen = options(specs()["embedder"])
    ranked = recall(question, chosen)
    if chosen.rerank_candidates:
        ranked = relevant(question, [row for row, _ in ranked], chosen)
    return group_by_document(ranked, count or chosen.top_k, chosen.passages_per_document)


def find_paragraphs(question: str, count: int | None) -> list[tuple[dict, float]]:
    """The most relevant whole paragraphs: those the recalled chunks cover, reranked."""
    chosen = options(specs()["embedder"])
    if not chosen.rerank_candidates:
        raise ConfigError(
            "search-paragraph needs the reranker: set search.rerank_candidates above 0"
        )
    chunks = recall(question, chosen)
    ids = list(dict.fromkeys(name for row, _ in chunks for name in row["paragraphs"] or []))
    with db.engine().connect() as conn:
        rows = db.paragraph_rows(conn, ids)
    return relevant(question, rows, chosen)[: count or chosen.paragraphs]


def no_match() -> None:
    cutoff = options(specs()["embedder"]).min_relevance
    print(
        f"[SEARCH] no passage scored above min_relevance {cutoff} - "
        "lower it in search.yaml to see weaker matches"
    )


def run_paragraph(args):
    where = settings()
    found = find_paragraphs(args.query, args.top_k)
    if not found:
        no_match()
    for number, (row, score) in enumerate(found, start=1):
        pdf = pdf_path(row, where.data_dir)
        shown = pdf.relative_to(where.root) if pdf.is_relative_to(where.root) else pdf
        print(f"[{number}] {score:.3f}  {cite(row)}")
        print(f"    {cite_paragraph(row)}")
        print(f"    {shown}" + ("" if pdf.exists() else " (missing)"))
        print(textwrap.indent(textwrap.fill(row["text"], width=96), "    "))
        print()


def run_search(args):
    found = find_hits(args.query, args.top_k)
    if not found:
        no_match()
    for number, (best, passages) in enumerate(found, start=1):
        print(f"[{number}] {best:.3f}  {cite(passages[0][0])}")
        for row, score in passages:
            snippet = " ".join(row["text"].split())[:200]
            print(f"    {score:.3f}  {row['section'] or '(title page)'}\n        {snippet}...")
        print()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="search", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("hardware", help="show GPU/CPU and the device models will use")
    get = commands.add_parser("get", help="download the models in model.yaml")
    get.add_argument("--force", action="store_true", help="download again even if present")
    commands.add_parser("index", help="chunk parsed documents and build the search index")
    inspect = commands.add_parser("inspect", help="print exactly what the embedder reads")
    inspect.add_argument("document", help="part of a doc_id, an author or a title")
    search = commands.add_parser("search", help="semantic search over the corpus")
    search.add_argument("query")
    search.add_argument("--top-k", type=int)
    paragraph = commands.add_parser(
        "paragraph", help="the most relevant whole paragraphs, with document, section and PDF page"
    )
    paragraph.add_argument("query")
    paragraph.add_argument("--top-k", type=int, help="paragraphs to show (search.paragraphs)")
    return root


COMMANDS = {
    "hardware": run_hardware,
    "get": run_get,
    "index": run_index,
    "inspect": run_inspect,
    "search": run_search,
    "paragraph": run_paragraph,
}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config.reset()
    try:
        COMMANDS[args.command](args)
    except ConfigError as problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    except Exception as problem:
        advice = hardware.as_advice(problem)
        if advice is None:
            raise
        print(f"error: {advice}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
