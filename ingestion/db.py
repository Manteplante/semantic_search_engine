"""The document database: one row per document, plus the text of the search index.

SQLite by default (data/documents.sqlite). DATABASE_URL in .env points it at any database
SQLAlchemy can reach - e.g. postgresql+psycopg://user:password@host/dbname - with no code
change. This module is the only place that talks to the database.

Security rules, kept by tests:
- SQL is built by SQLAlchemy Core and every value is a bound parameter. No f-strings,
  `%` or `.format` ever build SQL.
- Anything that changes a query's shape (a stage name, a column name) is looked up in a
  fixed dict or checked against the table's own columns, never pasted into SQL.
- Text from a source (titles, abstracts, tags) is untrusted and only ever stored as a value.
- The URL can hold a password, so it is only shown through `shown_url()`.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    Connection,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    make_url,
    select,
)
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from search.config import ConfigError, settings

DB_FILE = "documents.sqlite"
BATCH = 500  # rows per IN (...) lookup or bulk insert; far below SQLite's variable limit

metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("doc_id", String(80), primary_key=True),  # also the PDF and markdown file name
    Column("collection", String(64), nullable=False),  # its folder under data/
    Column("title", Text),
    Column("authors", JSON),
    Column("year", Integer),
    Column("language", String(16)),
    Column("abstract", Text),
    Column("tags", JSON),
    Column("source_url", Text),  # where it was fetched from, if anywhere
    Column("source_modified", String(40)),  # the source's change stamp, to skip re-fetches
    Column("file_name", Text),
    Column("file_size", BigInteger),
    Column("license", Text),
    Column("pdf_sha256", String(64)),
    Column("parsed_sha256", String(64)),  # the pdf_sha256 the markdown was built from
    Column("md_sha256", String(64)),
    Column("parse_error", Text),
)


def document_fields() -> tuple[Column, ...]:
    """Columns every chunk and paragraph carries (a column belongs to one table, so new ones)."""
    return (
        Column("document", String(80), nullable=False, index=True),
        Column("collection", String(64)),
        Column("title", Text),
        Column("author", Text),
        Column("year", Integer),
        Column("section", Text),
    )


# The search index's text, rebuilt by `make index`. `position` is the row in embeddings.npy.
chunks = Table(
    "chunks",
    metadata,
    Column("position", Integer, primary_key=True, autoincrement=False),
    Column("id", String(255), nullable=False, unique=True),
    *document_fields(),
    Column("text", Text, nullable=False),
    Column("paragraphs", JSON),
)
paragraphs = Table(
    "paragraphs",
    metadata,
    Column("id", String(255), primary_key=True),
    *document_fields(),
    Column("page", Integer),
    Column("last_page", Integer),
    Column("text", Text, nullable=False),
)

# Every stage `pending()` may be asked for. A name not in here is an error, so no caller
# input ever reaches the query.
STAGES = {
    "downloaded": documents.c.pdf_sha256.is_not(None),
    "parsed": documents.c.md_sha256.is_not(None),
}

# Upserts are dialect-specific SQL; these are the databases it is written for.
UPSERTS = {"sqlite": sqlite_insert, "postgresql": postgresql_insert}


def database_url() -> str:
    # Read raw, not through config.env(): a password may contain `#`.
    return os.environ.get("DATABASE_URL") or "sqlite:///" + str(settings().data_dir / DB_FILE)


def shown_url(url: str | None = None) -> str:
    """The URL with any password masked - the only form that may be printed."""
    return make_url(url or database_url()).render_as_string(hide_password=True)


def engine(url: str | None = None) -> Engine:
    """The database at `url` (default: DATABASE_URL, else data/documents.sqlite), tables created."""
    return _engine(url or database_url())


@functools.lru_cache(maxsize=8)
def _engine(url: str) -> Engine:
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    created = create_engine(parsed)
    metadata.create_all(created)
    return created


def checked(table: Table, row: dict[str, Any]) -> dict[str, Any]:
    """The row, if every key is a column of the table; column names never come from input."""
    unknown = sorted(set(row) - set(table.c.keys()))
    if unknown:
        raise ConfigError(f"table {table.name!r} has no column {unknown[0]!r}")
    return row


def batches(items: Sequence[Any]) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), BATCH):
        yield items[start : start + BATCH]


def upsert(conn: Connection, record: dict[str, Any]) -> None:
    """Insert a document, or update just the given columns of the stored one."""
    record = checked(documents, record)
    if not record.get("doc_id"):
        raise ConfigError("a document record needs its doc_id")
    dialect_insert = UPSERTS.get(conn.dialect.name)
    if dialect_insert is None:
        raise ConfigError(
            f"database {conn.dialect.name!r} is not supported: use SQLite or Postgres"
        )
    statement = dialect_insert(documents).values(**record)
    changes = {name: statement.excluded[name] for name in record if name != "doc_id"}
    if changes:
        statement = statement.on_conflict_do_update(index_elements=["doc_id"], set_=changes)
    else:
        statement = statement.on_conflict_do_nothing(index_elements=["doc_id"])
    conn.execute(statement)


def update(conn: Connection, doc_id: str, **fields: Any) -> None:
    """Set some columns of one stored document, e.g. its parse status."""
    values = checked(documents, fields)
    conn.execute(documents.update().where(documents.c.doc_id == doc_id).values(**values))


def get(conn: Connection, doc_id: str) -> dict[str, Any] | None:
    query = select(documents).where(documents.c.doc_id == doc_id)
    found = conn.execute(query).mappings().first()
    return dict(found) if found else None


def pending(conn: Connection, stage: str) -> list[dict[str, Any]]:
    """Documents that have reached a stage (see STAGES), by collection then newest first."""
    condition = STAGES.get(stage)
    if condition is None:
        raise ConfigError(f"unknown stage {stage!r}. Valid stages: {', '.join(STAGES)}")
    query = select(documents).where(condition)
    return [dict(row) for row in conn.execute(ordered(query)).mappings()]


def all_documents(conn: Connection) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(ordered(select(documents))).mappings()]


def ordered(query: Any) -> Any:
    return query.order_by(documents.c.collection, documents.c.year.desc(), documents.c.doc_id)


def replace_index(
    conn: Connection, chunk_rows: Iterable[dict[str, Any]], paragraph_rows: Iterable[dict[str, Any]]
) -> None:
    """Swap in a new search index; chunk n is row n of embeddings.npy."""
    conn.execute(delete(chunks))
    conn.execute(delete(paragraphs))
    numbered = [checked(chunks, {"position": n, **row}) for n, row in enumerate(chunk_rows)]
    for batch in batches(numbered):
        conn.execute(insert(chunks), list(batch))
    for batch in batches([checked(paragraphs, dict(row)) for row in paragraph_rows]):
        conn.execute(insert(paragraphs), list(batch))


def count_chunks(conn: Connection) -> int:
    return conn.execute(select(func.count()).select_from(chunks)).scalar_one()


def chunk_rows(conn: Connection, positions: Sequence[int]) -> list[dict[str, Any]]:
    """The chunks at these embeddings.npy rows, in the order asked for."""
    found = {}
    for batch in batches([int(position) for position in positions]):
        query = select(chunks).where(chunks.c.position.in_(batch))
        found |= {row["position"]: dict(row) for row in conn.execute(query).mappings()}
    return [found[int(position)] for position in positions if int(position) in found]


def paragraph_rows(conn: Connection, ids: Sequence[str]) -> list[dict[str, Any]]:
    """The paragraphs with these ids, in the order asked for."""
    found = {}
    for batch in batches(list(ids)):
        query = select(paragraphs).where(paragraphs.c.id.in_(batch))
        found |= {row["id"]: dict(row) for row in conn.execute(query).mappings()}
    return [found[name] for name in ids if name in found]
