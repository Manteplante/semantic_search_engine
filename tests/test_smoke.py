import json
from pathlib import Path

import app.backend
import ingestion.db
import ingestion.export
import ingestion.fetch
import ingestion.local_import
import parsing.pdf_parser
import search.__main__
import search.config
import search.corpus
import search.hardware
import search.model
import search.options
import search.search


def test_layers_import():
    assert app.backend
    assert ingestion.db
    assert ingestion.export
    assert ingestion.fetch
    assert ingestion.local_import
    assert parsing.pdf_parser
    assert search.__main__
    assert search.config
    assert search.corpus
    assert search.hardware
    assert search.model
    assert search.options
    assert search.search


def test_data_dir_ignores_generated_files():
    gitignore = Path("data/.gitignore").read_text()
    for pattern in ("*.pdf", "*.part", "*.csv", "*.parquet", "*.sqlite", "*.sqlite-journal"):
        assert pattern in gitignore


def test_parsing_output_ignores_generated_files():
    gitignore = Path("parsing/output/.gitignore").read_text()
    for pattern in ("*.md", "*.txt", "*.json"):
        assert pattern in gitignore


def test_search_output_ignores_generated_files():
    gitignore = Path("search/output/.gitignore").read_text()
    for pattern in ("*.jsonl", "*.npy", "*.json"):
        assert pattern in gitignore


def test_model_weights_and_secrets_are_ignored():
    gitignore = Path(".gitignore").read_text().splitlines()
    assert "models_local/" in gitignore
    assert ".env" in gitignore
    assert "!.env.example" in gitignore


def test_parse_force_defaults_off(monkeypatch):
    monkeypatch.delenv("PARSE_FORCE", raising=False)
    assert parsing.pdf_parser.get_force() is False
    monkeypatch.setenv("PARSE_FORCE", "1")
    assert parsing.pdf_parser.get_force() is True


def test_parse_limit_defaults_to_twenty(monkeypatch):
    monkeypatch.delenv("PARSE_LIMIT", raising=False)
    assert parsing.pdf_parser.get_limit_per_collection() == 20


def test_parse_limit_reads_env_override(monkeypatch):
    monkeypatch.setenv("PARSE_LIMIT", "0")
    assert parsing.pdf_parser.get_limit_per_collection() == 0


def document_row(doc_id, collection="b-reports", year=2020, **fields):
    return {"doc_id": doc_id, "collection": collection, "year": year} | fields


def test_parse_limit_keeps_newest_per_collection(tmp_path):
    rows = [
        document_row("old-00000001", year=2019),
        document_row("new-00000002", year=2024),
        document_row("mid-00000003", year=2022),
        document_row("only-0000004", collection="a-papers"),
        document_row("noyear-00005", collection="a-papers", year=None),
    ]

    def names(limit):
        return [
            row["doc_id"] for row, _, _ in parsing.pdf_parser.targets(rows, tmp_path, limit=limit)
        ]

    assert names(1) == ["only-0000004", "new-00000002"]
    assert names(2) == ["only-0000004", "noyear-00005", "new-00000002", "mid-00000003"]
    assert len(names(0)) == 5


def test_parse_targets_are_named_by_doc_id(tmp_path):
    row = document_row("report-2024-001")
    [(found, pdf, md)] = parsing.pdf_parser.targets([row], tmp_path / "data", tmp_path / "out")
    assert found is row
    assert pdf == tmp_path / "data" / "b-reports" / "report-2024-001.pdf"
    assert md == tmp_path / "out" / "b-reports" / "report-2024-001.md"


def test_needs_parse_follows_the_pdf_hash_and_waits_after_a_failure():
    needs_parse = parsing.pdf_parser.needs_parse
    fresh = {"pdf_sha256": "a", "parsed_sha256": None, "parse_error": None}
    assert needs_parse(fresh)
    assert not needs_parse(fresh | {"parsed_sha256": "a"})
    assert needs_parse(fresh | {"parsed_sha256": "old"})  # the PDF changed
    assert not needs_parse(fresh | {"parse_error": "broken PDF"})
    assert needs_parse(fresh | {"parsed_sha256": "a"}, force=True)


def test_parse_ocr_defaults_off_and_timeout_to_300(monkeypatch):
    monkeypatch.delenv("PARSE_OCR", raising=False)
    monkeypatch.delenv("PARSE_TIMEOUT", raising=False)
    assert parsing.pdf_parser.get_ocr() is False
    assert parsing.pdf_parser.get_timeout() == 300
    monkeypatch.setenv("PARSE_OCR", "1")
    monkeypatch.setenv("PARSE_TIMEOUT", "60")
    assert parsing.pdf_parser.get_ocr() is True
    assert parsing.pdf_parser.get_timeout() == 60


def test_only_a_complete_conversion_is_saved():
    from types import SimpleNamespace

    problem = parsing.pdf_parser.conversion_problem

    def result(status, *messages):
        errors = [SimpleNamespace(error_message=message) for message in messages]
        return SimpleNamespace(status=SimpleNamespace(value=status), errors=errors)

    assert problem(result("success"), 5, 300) is None
    assert problem(result("partial_success"), 301, 300) == (
        "docling partial_success: timed out after 300s (PARSE_TIMEOUT)"
    )
    assert problem(result("failure", "broken xref"), 2, 300) == "docling failure: broken xref"


def _double(value):
    return value * 2, None


def _hang(value):
    import time

    time.sleep(30)


def _die(value):
    import os
    import signal

    os.kill(os.getpid(), signal.SIGKILL)  # what the kernel's OOM killer does


def test_run_isolated_returns_the_result_of_the_worker():
    assert parsing.pdf_parser.run_isolated(_double, (21,), 30, context="fork") == (42, None)


def test_run_isolated_kills_a_worker_past_its_deadline():
    import time

    start = time.time()
    markdown, problem = parsing.pdf_parser.run_isolated(_hang, (1,), 0.5, context="fork")
    assert markdown is None and "killed after 0.5s" in problem
    assert time.time() - start < 10


def test_run_isolated_survives_a_worker_killed_for_memory():
    markdown, problem = parsing.pdf_parser.run_isolated(_die, (1,), 30, context="fork")
    assert markdown is None and "exit code -9" in problem and "out of memory" in problem


def test_paged_markdown_marks_every_pdf_page():
    class FakeDocument:  # docling's DoclingDocument, as far as paged_markdown uses it
        pages = {3: None, 1: None, 2: None}

        def export_to_markdown(self, page_no):
            return {1: "## Title", 2: "", 3: "Text."}[page_no]

    # page 2 is blank: it keeps its marker, so later pages keep their numbers
    assert parsing.pdf_parser.paged_markdown(FakeDocument()) == (
        "<!-- page 1 -->\n\n## Title\n\n<!-- page 2 -->\n\n<!-- page 3 -->\n\nText.\n"
    )


def test_parsing_notebook_is_valid_and_unexecuted():
    notebook = json.loads(Path("parsing/docling_prototype.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    # Keep outputs out of version control so notebook diffs stay readable.
    assert not any(cell.get("outputs") for cell in notebook["cells"])
