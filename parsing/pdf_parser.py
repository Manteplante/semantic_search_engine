"""Convert every registered PDF to markdown with docling.

data/<collection>/<doc_id>.pdf -> parsing/output/<collection>/<doc_id>.md

The work list comes from the document database (ingestion/db.py), not from the folder:
only the newest PARSE_LIMIT documents per collection are considered (default 20; 0 = all),
and of those only the ones whose PDF changed since their markdown was built. A PDF
docling cannot read is recorded with its error and not retried until its file changes,
or PARSE_FORCE=1, which redoes everything. Markdown (not plain text) is kept because its
headings let the search stage chunk it section by section.

OCR is off by default: on born-digital PDFs OCR only re-reads text that is already
there, yet it made one 130-page document run past 10 minutes instead of 8 seconds.
PARSE_OCR=1 turns it on for scanned PDFs. Docling stops any PDF after PARSE_TIMEOUT
seconds (default 300), and a PDF it cannot convert completely is recorded as a parse
error rather than saved half-done.

Each PDF is converted in a fresh worker process. Docling's memory grows from document
to document (one long document peaks at 4-7 GB, and a single process reached 8 GB after
~30 ordinary ones), so a new process per PDF keeps it bounded. A worker that runs out of
memory or overruns its deadline is killed and that PDF recorded as failed; the run goes on.

Each PDF page starts with a `<!-- page N -->` marker (N = page in the PDF, as a
viewer shows it), so search can cite the page a paragraph is on.
"""

import hashlib
import multiprocessing
import os
import time
from pathlib import Path

from ingestion import db
from search.config import settings

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "parsing" / "output"
DEFAULT_LIMIT_PER_COLLECTION = 20  # 0 means no limit
PAGE_MARKER = "<!-- page {} -->"  # an HTML comment, so the markdown still renders cleanly
MAX_ERROR = 500  # characters of a docling error kept in the database
DEFAULT_TIMEOUT = 300  # seconds docling may spend on one PDF before giving up
GRACE = 120  # seconds a worker gets on top of that (loading models) before it is killed


def targets(rows, data_dir=DATA_DIR, output_dir=OUTPUT_DIR, limit=0):
    """(document row, pdf, markdown path) per document, mirroring <collection>/.

    With a limit, only the newest `limit` documents of each collection are kept.
    """
    ordered = sorted(rows, key=lambda row: (row["collection"], -(row["year"] or 0), row["doc_id"]))
    per_collection = {}
    found = []
    for row in ordered:
        collection = row["collection"]
        per_collection[collection] = per_collection.get(collection, 0) + 1
        if limit and per_collection[collection] > limit:
            continue
        relative = Path(collection) / row["doc_id"]
        found.append(
            (row, data_dir / relative.with_suffix(".pdf"), output_dir / relative.with_suffix(".md"))
        )
    return found


def needs_parse(row, force=False):
    """Whether the markdown is behind the PDF; a recorded failure waits for a new file."""
    if force:
        return True
    return row["parse_error"] is None and row["parsed_sha256"] != row["pdf_sha256"]


def paged_markdown(document):
    """The document as markdown, each page led by its PAGE_MARKER.

    Exported page by page because docling's own page-break placeholder carries no
    number, and counting placeholders drifts on blank pages. Docling assigns an
    item to its first page only, so nothing is exported twice.
    """
    pages = [
        f"{PAGE_MARKER.format(number)}\n\n{document.export_to_markdown(page_no=number)}".rstrip()
        for number in sorted(document.pages)
    ]
    return "\n\n".join(pages) + "\n"


def get_limit_per_collection():
    return int(os.environ.get("PARSE_LIMIT", DEFAULT_LIMIT_PER_COLLECTION))


def get_force():
    return os.environ.get("PARSE_FORCE", "") not in ("", "0")


def get_ocr():
    return os.environ.get("PARSE_OCR", "") not in ("", "0")


def get_timeout():
    return int(os.environ.get("PARSE_TIMEOUT") or DEFAULT_TIMEOUT)


def conversion_problem(result, seconds, timeout):
    """Why a docling result can't be saved, or None if it converted completely.

    A timed-out document comes back as "partial_success" with only its first pages, so
    anything short of "success" is a failure: a half document would quietly lose its text.
    """
    status = getattr(result.status, "value", result.status)
    if status == "success":
        return None
    errors = [getattr(error, "error_message", str(error)) for error in result.errors or []]
    if not errors and seconds >= timeout:
        errors = [f"timed out after {timeout}s (PARSE_TIMEOUT)"]
    return f"docling {status}: " + ("; ".join(errors[:3]) or "no error given")


def make_converter(ocr, timeout):
    """Docling's converter with OCR on or off and a per-document timeout."""
    from docling.datamodel.base_models import InputFormat  # slow imports, only when needed
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=ocr, document_timeout=timeout)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def convert_one(pdf, ocr, timeout):
    """(markdown, None) for a complete conversion, else (None, why). Runs in a worker."""
    start = time.time()
    try:
        result = make_converter(ocr, timeout).convert(pdf, raises_on_error=False)
    except Exception as e:
        return None, str(e) or type(e).__name__
    problem = conversion_problem(result, time.time() - start, timeout)
    return (None, problem) if problem else (paged_markdown(result.document), None)


def worker(target, args, send):
    send.send(target(*args))
    send.close()


def run_isolated(target, args, deadline, context="spawn"):
    """target(*args) in its own process, which is killed after `deadline` seconds.

    Returns what target returned, or (None, why) when the process died - the kernel's
    OOM killer, a crash - or had to be killed. "spawn" starts a clean interpreter, so
    nothing (CUDA, docling's caches) is inherited from this process.
    """
    ctx = multiprocessing.get_context(context)
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=worker, args=(target, args, send), daemon=True)
    process.start()
    send.close()  # the worker holds the only writer, so its death closes the pipe
    try:
        if not receive.poll(deadline):
            process.kill()
            return None, f"killed after {deadline}s without finishing (PARSE_TIMEOUT)"
        return receive.recv()
    except EOFError:
        process.join()
        return None, f"the worker died (exit code {process.exitcode}), most likely out of memory"
    finally:
        process.join()
        receive.close()


def record_error(engine, row, pdf, message):
    print(f"[PARSE] Error on {pdf.name}: {message}")
    with engine.begin() as conn:
        db.update(conn, row["doc_id"], parse_error=message[:MAX_ERROR])


def main():
    force = get_force()
    limit = get_limit_per_collection()
    ocr = get_ocr()
    timeout = get_timeout()
    engine = db.engine()
    with engine.connect() as conn:
        downloaded = db.pending(conn, "downloaded")
    where = settings()  # honours PROJECT_ROOT, like every other stage
    todo = [
        (row, pdf, md)
        for row, pdf, md in targets(downloaded, where.data_dir, where.input_dir, limit)
        if needs_parse(row, force)
    ]
    scope = f"newest {limit} per collection" if limit else "all"
    print(f"[PARSE] {len(todo)} PDFs to convert ({scope}; LIMIT=0 for all)")
    print(f"[PARSE] OCR {'on' if ocr else 'off (PARSE_OCR=1 for scans)'}, {timeout}s per PDF")
    if not todo:
        return

    for row, pdf, md in todo:
        if not pdf.exists():
            print(f"[PARSE] Missing {pdf.relative_to(where.root)} - restore it, then make ingest")
            continue
        start = time.time()
        markdown, problem = run_isolated(convert_one, (str(pdf), ocr, timeout), timeout + GRACE)
        if problem:
            record_error(engine, row, pdf, problem)
            continue
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(markdown, encoding="utf-8")
        with engine.begin() as conn:
            db.update(
                conn,
                row["doc_id"],
                parsed_sha256=row["pdf_sha256"],
                md_sha256=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                parse_error=None,
            )
        print(f"[PARSE] {md.relative_to(where.root)} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
