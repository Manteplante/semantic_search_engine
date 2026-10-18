"""`make export`: the document database as one file, for a spreadsheet or pandas.

Writes data/documents.csv, or data/documents.parquet with FORMAT=parquet (that needs pyarrow,
which is not a dependency yet: `uv add pyarrow`). In CSV, lists (authors, tags) are
joined with "; "; Parquet keeps them as lists.
"""

import os
import sys

import pandas as pd

from ingestion import db
from search.config import ConfigError, settings

FORMATS = ("csv", "parquet")
LIST_SEPARATOR = "; "


def get_format():
    chosen = (os.environ.get("EXPORT_FORMAT") or "csv").strip().lower()
    if chosen not in FORMATS:
        raise ConfigError(f"unknown export FORMAT {chosen!r}. Valid: {', '.join(FORMATS)}")
    return chosen


def as_frame(rows, flat_lists=False):
    """Document rows as a DataFrame with every database column, in table order."""
    frame = pd.DataFrame(rows, columns=list(db.documents.c.keys()))
    if flat_lists:
        for column in ("authors", "tags"):
            frame[column] = frame[column].map(
                lambda value: LIST_SEPARATOR.join(value) if isinstance(value, list) else value
            )
    return frame


def main():
    try:
        chosen = get_format()
        if chosen == "parquet":
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                raise ConfigError("FORMAT=parquet needs pyarrow. Run: uv add pyarrow") from None
    except ConfigError as problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    with db.engine().connect() as conn:
        rows = db.all_documents(conn)
    target = settings().data_dir / f"documents.{chosen}"
    if chosen == "csv":
        as_frame(rows, flat_lists=True).to_csv(target, index=False)
    else:
        as_frame(rows).to_parquet(target, index=False)
    print(f"[EXPORT] {len(rows)} documents -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
