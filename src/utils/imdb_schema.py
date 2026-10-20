"""Schema-aware loading for the real IMDb CSVs used by the Join Order Benchmark.

The IMDb CSVs shipped in ``imdb.tgz`` are **headerless**, so a header-based
``csv.DictReader`` would mis-key every row. JOB's ``schema.sql`` carries the
column order for each table; :func:`parse_schema_sql` recovers it and
:func:`load_relation_rows` reads a headerless CSV into ``list[dict]`` keyed by
the real column names (which the parsed JOB joins and filters reference).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.utils.types import AttributeName, RelationName

if TYPE_CHECKING:
    from pathlib import Path

_CREATE_TABLE = re.compile(
    r"CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);", re.IGNORECASE | re.DOTALL
)


def parse_schema_sql(schema_sql: str) -> dict[RelationName, list[AttributeName]]:
    """Parse ``CREATE TABLE`` statements into ``{table: [columns in order]}``."""
    schema: dict[RelationName, list[AttributeName]] = {}
    for match in _CREATE_TABLE.finditer(schema_sql):
        table = RelationName(match.group(1))
        columns: list[AttributeName] = []
        for raw_line in match.group(2).splitlines():
            line = raw_line.strip().rstrip(",").strip()
            if not line:
                continue
            first = line.split()[0]
            if first.upper() in {"PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT", "CHECK"}:
                continue
            columns.append(AttributeName(first))
        schema[table] = columns
    return schema


def load_relation_rows(
    csv_path: Path, columns: list[AttributeName], limit: int | None = None
) -> list[dict[str, str]]:
    """Read a headerless IMDb CSV into rows keyed by ``columns`` (schema order).

    A field count mismatch raises, so a wrong schema fails loudly instead of
    silently mis-aligning columns. ``limit`` caps the number of rows read (for
    memory-bounded local runs); a cap makes cardinalities a subsample estimate,
    so callers must report it.

    The dump follows the PostgreSQL CSV convention JOB documents for loading it
    (``QUOTE '"' ESCAPE '\\'``), in which a backslash escapes the next character
    **only inside a quoted field**. Both halves of that rule matter, and Python's
    :mod:`csv` can express neither on its own:

    * ``movie_info.csv`` contains ``"\\"Hatikva\\" neighborhood, Tel Aviv,
      Israel"``. Reading it with the CSV default ends the field at ``\\"``, so the
      embedded commas split one value into three and a 5-column row becomes 7.
    * ``title.csv`` contains the unquoted title ``\\Frag'ile\\`` followed by a
      delimiter. Reading it with ``escapechar='\\\\'`` swallows that delimiter, so
      two fields merge and a 12-column row becomes 11.

    Passing ``escapechar`` fixes the first and causes the second, because
    :mod:`csv` applies the escape everywhere while PostgreSQL applies it only
    within quotes. DuckDB's CSV reader implements the PostgreSQL rule, so it is
    used here as a parser (not as an engine); ``all_varchar`` keeps values as
    written and SQL ``NULL`` is mapped back to the empty string, preserving the
    representation the rest of the pipeline expects.
    """
    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        msg = (
            "duckdb is required to read the IMDb CSVs (PostgreSQL-style "
            "quoting); install it with `pip install duckdb`"
        )
        raise ModuleNotFoundError(msg) from exc

    quoted = str(csv_path).replace("'", "''")
    select = (
        "SELECT * FROM read_csv("
        + ", ".join(
            (
                f"'{quoted}'",
                "header=false",
                "all_varchar=true",
                "delim=','",
                "quote='\"'",
                "escape='\\'",
            )
        )
        + ")"
    )
    if limit is not None:
        select += f" LIMIT {limit}"

    width = len(columns)
    rows: list[dict[str, str]] = []
    for line_no, fields in enumerate(duckdb.sql(select).fetchall(), start=1):
        if len(fields) != width:
            raise ValueError(
                f"{csv_path.name} line {line_no}: expected {width} fields "
                f"for schema {columns}, got {len(fields)}"
            )
        rows.append(
            {
                column: "" if value is None else str(value)
                for column, value in zip(columns, fields, strict=True)
            }
        )
    return rows


def load_schema(data_dir: Path) -> dict[RelationName, list[AttributeName]]:
    """Load and parse ``schema.sql`` from a prepared JOB data directory."""
    schema_path = data_dir / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"schema.sql not found in {data_dir}; run prepare_real_job first"
        )
    return parse_schema_sql(schema_path.read_text(encoding="utf-8"))
