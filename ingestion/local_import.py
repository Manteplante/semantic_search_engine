"""`make ingest`: register the PDFs under data/<collection>/ in the document database.

Drop PDFs into data/<collection>/<doc_id>.pdf and run this; `make parse` then picks them
up. The file name (without .pdf) is the doc_id, the key every later stage joins on, so it
must be a safe file name: letters, digits, `.`, `_` and `-`. Re-running is cheap: a PDF
whose sha256 matches the stored one is left alone; a changed one gets a fresh parse.

A source that fetches documents over the network writes the same rows (db.upsert) and
files (data/<collection>/<doc_id>.pdf), using ingestion/fetch.py for the HTTP side.
"""

import hashlib
import re
import sys

from ingestion import db
from search.config import settings

# The id becomes a file name in every stage, so anything else is refused.
SAFE_NAME = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,79}$")


def safe_id(identifier):
    if not isinstance(identifier, str) or not SAFE_NAME.match(identifier):
        raise ValueError(f"unexpected document id {identifier!r}")
    return identifier


def pdf_path(data_dir, record):
    """data/<collection>/<doc_id>.pdf"""
    return data_dir / safe_id(record["collection"]) / f"{safe_id(record['doc_id'])}.pdf"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def find_pdfs(data_dir):
    """(collection, pdf) for every data/<collection>/*.pdf, sorted."""
    return [(path.parent.name, path) for path in sorted(data_dir.glob("*/*.pdf"))]


def register(engine, collection, pdf):
    """Store or refresh one PDF's row; returns "new", "changed" or "unchanged"."""
    record = {"doc_id": safe_id(pdf.stem), "collection": safe_id(collection)}
    sha = file_sha256(pdf)
    with engine.connect() as conn:
        stored = db.get(conn, record["doc_id"])
    if stored and stored.get("pdf_sha256") == sha and stored.get("collection") == collection:
        return "unchanged"
    record |= {
        "file_name": pdf.name,
        "file_size": pdf.stat().st_size,
        "pdf_sha256": sha,
        "parse_error": None,  # a new file gets a fresh parse attempt
    }
    with engine.begin() as conn:
        db.upsert(conn, record)
    return "changed" if stored else "new"


def main():
    data_dir = settings().data_dir
    engine = db.engine()
    found = find_pdfs(data_dir)
    print(f"[INGEST] {len(found)} PDFs under {data_dir}, into {db.shown_url()}")
    counts = {}
    for collection, pdf in found:
        try:
            status = register(engine, collection, pdf)
        except ValueError as problem:
            print(f"[SKIP] {pdf.relative_to(data_dir)}: {problem} - rename the file or folder")
            continue
        counts[status] = counts.get(status, 0) + 1
        if status != "unchanged":
            print(f"[INGEST] {status}: {pdf.relative_to(data_dir)}")
    print(f"[INGEST] {', '.join(f'{n} {s}' for s, n in counts.items()) or 'nothing to do'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
