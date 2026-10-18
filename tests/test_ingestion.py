"""Ingestion without the network: the document database, the HTTP helpers and local import.
Every database here is a SQLite file in tmp_path."""

import re
from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

from ingestion import db, fetch, local_import
from search.config import ConfigError

ID = "report-2024-001"


@pytest.fixture(autouse=True)
def tmp_database(tmp_path, monkeypatch):
    """Anything that falls back to DATABASE_URL lands in tmp_path, never data/."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'documents.sqlite'}")


@pytest.fixture
def engine(tmp_path):
    return db.engine(f"sqlite:///{tmp_path / 'test.sqlite'}")


def document(doc_id=ID, **fields):
    return {"doc_id": doc_id, "collection": "example"} | fields


# database


def test_upsert_inserts_then_updates_only_given_columns(engine):
    with engine.begin() as conn:
        db.upsert(conn, document(title="Coast", authors=["Jane Doe"], pdf_sha256="abc"))
        db.upsert(conn, document(title="Coast and sea", year=2024))
        found = db.get(conn, ID)
        assert db.all_documents(conn) == [found]
    assert found["title"] == "Coast and sea"
    assert found["year"] == 2024
    assert found["authors"] == ["Jane Doe"]  # JSON list survives
    assert found["pdf_sha256"] == "abc"  # not in the second upsert, so kept


def test_pending_follows_stage_status(engine):
    with engine.begin() as conn:
        db.upsert(conn, document("a-00000001", pdf_sha256="x", year=2020))
        db.upsert(conn, document("b-00000002", pdf_sha256="y", year=2024))
        db.upsert(conn, document("c-00000003"))
        db.update(conn, "a-00000001", md_sha256="m")
        assert [row["doc_id"] for row in db.pending(conn, "downloaded")] == [
            "b-00000002",  # newest first
            "a-00000001",
        ]
        assert [row["doc_id"] for row in db.pending(conn, "parsed")] == ["a-00000001"]


def test_untrusted_text_is_stored_as_data_not_sql(engine):
    evil = "x'); DROP TABLE documents;--"
    tags = ['"quoted"', "it's", "; DELETE FROM documents"]
    with engine.begin() as conn:
        db.upsert(conn, document(title=evil, tags=tags))
        db.upsert(conn, document("other-0000001", title="Robert'); DROP TABLE chunks;--"))
    with engine.connect() as conn:
        assert db.get(conn, ID)["title"] == evil
        assert db.get(conn, ID)["tags"] == tags
        assert len(db.all_documents(conn)) == 2
    assert {"documents", "chunks", "paragraphs"} <= set(inspect(engine).get_table_names())


def test_lookups_by_untrusted_value_find_nothing_instead_of_everything(engine):
    with engine.begin() as conn:
        db.upsert(conn, document())
        assert db.get(conn, "' OR '1'='1") is None
        assert db.paragraph_rows(conn, ["' OR '1'='1"]) == []


def test_unknown_stage_or_column_never_reaches_sql(engine):
    with engine.begin() as conn:
        with pytest.raises(ConfigError, match="unknown stage"):
            db.pending(conn, "parsed; DROP TABLE documents")
        with pytest.raises(ConfigError, match="no column"):
            db.upsert(conn, document(**{"title = 'x'; --": "y"}))
        with pytest.raises(ConfigError, match="no column"):
            db.update(conn, ID, colour="red")


def test_postgres_upsert_binds_every_value():
    evil = "x'); DROP TABLE documents;--"
    statement = postgresql.insert(db.documents).values(doc_id=ID, title=evil)
    statement = statement.on_conflict_do_update(
        index_elements=["doc_id"], set_={"title": statement.excluded.title}
    )
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT (doc_id) DO UPDATE" in sql
    assert "DROP TABLE" not in sql and "%(title" in sql  # the value travels as a parameter
    assert set(db.UPSERTS) == {"sqlite", "postgresql"}


def test_no_sql_is_built_from_strings():
    source = Path(db.__file__).read_text()
    pattern = r"(execute|text)\(\s*f?[\"']|\.format\(|%\s*\("
    assert not re.search(pattern, source), "build SQL with SQLAlchemy Core, never from strings"


def test_password_never_shown():
    shown = db.shown_url("postgresql+psycopg://reader:s3cret@db.example.org:5432/documents")
    assert "s3cret" not in shown and "reader" in shown


def test_default_database_lives_in_data(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    from search import config

    config.reset()
    assert db.database_url() == f"sqlite:///{tmp_path / 'data' / 'documents.sqlite'}"
    config.reset()


def test_search_index_rows_come_back_in_the_order_asked(engine, monkeypatch):
    monkeypatch.setattr(db, "BATCH", 2)  # several IN (...) batches
    chunk = {"document": ID, "section": "S", "text": "t", "paragraphs": ["p1"]}
    with engine.begin() as conn:
        db.replace_index(
            conn,
            [chunk | {"id": f"c{n}"} for n in range(5)],
            [{"id": f"p{n}", "document": ID, "text": "x", "page": n} for n in range(3)],
        )
        assert db.count_chunks(conn) == 5
        assert [row["id"] for row in db.chunk_rows(conn, [4, 0, 2])] == ["c4", "c0", "c2"]
        assert db.chunk_rows(conn, [0])[0]["paragraphs"] == ["p1"]
        assert [row["page"] for row in db.paragraph_rows(conn, ["p2", "p0", "nope"])] == [2, 0]
        db.replace_index(conn, [chunk | {"id": "only"}], [])  # replaces, never appends
        assert db.count_chunks(conn) == 1


# HTTP helpers


class FakeResponse:
    def __init__(self, payload=None, blocks=()):
        self.payload = payload
        self.blocks = blocks

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        yield from self.blocks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_session_retries_rate_limits_and_server_errors():
    adapter = fetch.SESSION.get_adapter("https://example.org")
    assert adapter.max_retries.total == 5
    assert {429, 503} <= set(adapter.max_retries.status_forcelist)


def test_get_json_passes_params_and_a_timeout(monkeypatch):
    asked = {}

    def fake_get(self, url, **kwargs):
        asked.update(kwargs, url=url)
        return FakeResponse({"ok": True})

    monkeypatch.setattr(fetch.requests.Session, "get", fake_get)
    assert fetch.get_json("https://example.org/api", params={"q": "x"}) == {"ok": True}
    assert asked["params"] == {"q": "x"} and asked["timeout"] == fetch.TIMEOUT


def test_download_to_hashes_and_leaves_no_partial_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fetch.requests.Session, "get", lambda self, url, **kw: FakeResponse(blocks=[b"%P", b"DF"])
    )
    target = tmp_path / "example" / "doc.pdf"
    sha = fetch.download_to("https://example.org/doc.pdf", target)
    assert target.read_bytes() == b"%PDF"
    assert sha == local_import.file_sha256(target)
    assert not target.with_name("doc.pdf.part").exists()


# local import


@pytest.mark.parametrize(
    "identifier", ["../../etc/passwd", "a/b-12345678", "", None, "id with spaces", ".hidden"]
)
def test_identifiers_that_are_not_safe_file_names_are_refused(identifier):
    with pytest.raises(ValueError, match="unexpected document id"):
        local_import.safe_id(identifier)


def test_pdf_path_is_collection_then_doc_id(tmp_path):
    record = document()
    assert local_import.pdf_path(tmp_path, record) == tmp_path / "example" / f"{ID}.pdf"


def test_register_stores_new_skips_unchanged_and_refreshes_changed(engine, tmp_path):
    pdf = tmp_path / "data" / "example" / f"{ID}.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF one")
    assert local_import.find_pdfs(tmp_path / "data") == [("example", pdf)]
    assert local_import.register(engine, "example", pdf) == "new"
    with engine.begin() as conn:
        db.update(conn, ID, parse_error="broken", title="Kept")
    assert local_import.register(engine, "example", pdf) == "unchanged"
    with engine.connect() as conn:
        assert db.get(conn, ID)["parse_error"] == "broken"  # unchanged file: no retry
    pdf.write_bytes(b"%PDF two")
    assert local_import.register(engine, "example", pdf) == "changed"
    with engine.connect() as conn:
        stored = db.get(conn, ID)
    assert stored["parse_error"] is None  # a new file gets a fresh parse attempt
    assert stored["title"] == "Kept"  # metadata a source wrote is not wiped
    assert stored["pdf_sha256"] == local_import.file_sha256(pdf)
    assert stored["file_size"] == len(b"%PDF two")


def test_register_refuses_an_unsafe_file_name(engine, tmp_path):
    pdf = tmp_path / "example" / "my report.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    with pytest.raises(ValueError):
        local_import.register(engine, "example", pdf)
