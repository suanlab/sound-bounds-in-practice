"""Load the six JOB-light tables of IMDb into the injection PostgreSQL.

Creates database ``imdb`` (if missing) with the JOB schema for ``title``,
``cast_info``, ``movie_companies``, ``movie_info``, ``movie_info_idx`` and
``movie_keyword`` (from ``data/job_full/schema.sql``), streams the CSVs from the
host with ``COPY ... (FORMAT csv, ESCAPE '\\')`` -- the PostgreSQL quoting the
dump uses, which also maps empty fields to NULL -- adds an index on every
``movie_id`` foreign key (``title.id`` is the primary key), and ANALYZEs.

Usage::

    python -m experiments.pg_injection.load_imdb
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import psycopg2

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TABLES = (
    "title",
    "cast_info",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
)


def create_statement(schema: str, table: str) -> str:
    """The ``CREATE TABLE`` statement for one table from the JOB schema file."""
    match = re.search(rf"CREATE TABLE {table} \(.*?\);", schema, flags=re.S)
    if match is None:
        msg = f"{table} not found in schema"
        raise ValueError(msg)
    return match.group(0)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default="host=127.0.0.1 port=55432 user=postgres password=postgres",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/job_full"))
    args = parser.parse_args()

    admin = psycopg2.connect(args.dsn + " dbname=template1")
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'imdb'")
        if cursor.fetchone() is None:
            cursor.execute("CREATE DATABASE imdb")
    admin.close()

    schema = (args.data_dir / "schema.sql").read_text()
    connection = psycopg2.connect(args.dsn + " dbname=imdb")
    connection.set_client_encoding("UTF8")
    connection.autocommit = True
    cursor = connection.cursor()
    for table in TABLES:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        cursor.execute(create_statement(schema, table))
        # Stream raw bytes: the dump is UTF-8 and the host locale may not be.
        with (args.data_dir / f"{table}.csv").open("rb") as handle:
            cursor.copy_expert(
                f"COPY {table} FROM STDIN WITH (FORMAT csv, ESCAPE '\\')", handle
            )
        cursor.execute(f"SELECT count(*) FROM {table}")
        row = cursor.fetchone()
        logger.info("%s: %s rows", table, row[0] if row else "?")
        if table != "title":
            cursor.execute(f"CREATE INDEX ON {table} (movie_id)")
    cursor.execute("ANALYZE")
    logger.info("imdb loaded and analysed")


if __name__ == "__main__":
    main()
