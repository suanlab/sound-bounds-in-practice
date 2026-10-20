"""PostgreSQL's own estimate of every join it sizes, inside the full query.

``native_estimates`` read each sub-plan's estimate from a separate ``EXPLAIN``
of that sub-plan as a query of its own. Inside the full query PostgreSQL sizes
the same set of relations differently (equivalence classes add implied join
clauses), so injecting those stand-alone values did not reproduce the native
plan on 56 of 146 STATS-CEB queries. This script reads the in-context values
instead, through two hooks of the STATS-CEB benchmark patch compiled into the
server:

* ``print_sub_queries`` logs, for every call that sizes a join, the two input
  relations (so the call's relation set is their union);
* ``debug_card_est`` with ``ml_joinest_enabled`` logs the planner's own
  estimate at that call and then *replaces* it with the next value of
  ``ml_joinest_fname``.

The replacement would perturb every later call, so the values are recovered
one at a time: a pass that feeds back the native values learned so far leaves
the planner's state exactly native up to the first unknown call, whose own
estimate it logs. A relation set's first call is the one that sets the join
relation's size (later calls size parameterized paths). Planning only.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.native_estimates_incontext
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import psycopg2

from experiments.pg_injection.inject_plans import WORKLOAD_PATHS
from experiments.pg_injection.subset_counts import (
    count_sql,
    load_subqueries,
    with_dbname,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EST_FILE = "/tmp/incontext_join_est.txt"
RECORD = "join_est_record_job.txt"
OLD_NEW = "old_new_join_est.txt"
RELOPT = re.compile(r"RELOPTINFO \(([^)]*)\): rows=")
PLACEHOLDER = 1.0
FEED_SLACK = 4096
# CARD_EST_QUERY_NUM in benchmark.patch; its fscanf loop has no bound check.
PATCH_BUFFER = 10_000


def _size(cursor: Any, name: str) -> int:
    cursor.execute("SELECT size FROM pg_stat_file(%s, true)", (name,))
    row = cursor.fetchone()
    return 0 if row is None or row[0] is None else int(row[0])


def _tail(cursor: Any, name: str, offset: int) -> str:
    cursor.execute("SELECT pg_read_file(%s, %s, 100000000, true)", (name, offset))
    row = cursor.fetchone()
    return "" if row is None or row[0] is None else str(row[0])


def one_pass(
    cursor: Any, sql: str, feed: list[float]
) -> tuple[list[float], list[frozenset[str]]]:
    """Plan ``sql`` feeding ``feed`` to the first calls; log every call."""
    if len(feed) + FEED_SLACK > PATCH_BUFFER:
        msg = f"{len(feed)} known estimates would overflow the patch buffer"
        raise RuntimeError(msg)
    values = ", ".join(f"{v:.17g}" for v in [*feed, *([PLACEHOLDER] * FEED_SLACK)])
    cursor.execute(f"COPY (SELECT unnest(ARRAY[{values}]::float8[])) TO '{EST_FILE}'")
    record_at, old_new_at = _size(cursor, RECORD), _size(cursor, OLD_NEW)
    cursor.execute("SET join_est_no = 0")
    cursor.execute("EXPLAIN " + sql)
    cursor.fetchall()
    natives = [
        float(line.split(":")[0])
        for line in _tail(cursor, OLD_NEW, old_new_at).splitlines()
        if line.strip()
    ]
    rels = RELOPT.findall(_tail(cursor, RECORD, record_at))
    if len(rels) != 2 * len(natives):
        msg = f"{len(rels)} logged inputs for {len(natives)} join calls"
        raise RuntimeError(msg)
    sets = [
        frozenset(rels[2 * i].split()) | frozenset(rels[2 * i + 1].split())
        for i in range(len(natives))
    ]
    return natives, sets


def in_context_estimates(cursor: Any, sql: str) -> dict[frozenset[str], float]:
    """Native estimate of every join relation PostgreSQL sizes for ``sql``."""
    known: list[float] = []
    while True:
        natives, sets = one_pass(cursor, sql, known)
        if len(natives) < len(known):
            msg = f"a pass made {len(natives)} join calls after {len(known)} were known"
            raise RuntimeError(msg)
        if len(natives) > len(known) + FEED_SLACK:
            msg = f"{len(natives)} join calls exceed the estimate file"
            raise RuntimeError(msg)
        if len(known) == len(natives):
            break
        known.append(natives[len(known)])
    out: dict[frozenset[str], float] = {}
    for relset, value in zip(sets, known, strict=True):
        out.setdefault(relset, value)  # first call sets the join relation's size
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default="host=127.0.0.1 port=55432 dbname=stats "
        "user=postgres password=postgres",
    )
    parser.add_argument("--workload", choices=sorted(WORKLOAD_PATHS), default="stats")
    parser.add_argument("--queries", type=str, default="", help="e.g. 1,27")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v2/native_incontext_estimations.csv"),
    )
    args = parser.parse_args()

    estimates_dir, subplan_file, query_file, dbname = WORKLOAD_PATHS[args.workload]
    del estimates_dir
    dsn = with_dbname(args.dsn, dbname)
    lines = [raw for raw in query_file.read_text().splitlines() if raw.strip()]
    queries = {
        i + 1: count_sql(line.split("||", 1)[-1].strip().rstrip(";"))
        for i, line in enumerate(lines)
    }
    wanted = {int(q) for q in args.queries.split(",") if q} or set(queries)
    subplans: dict[int, list[str]] = {}
    for row in load_subqueries(subplan_file):
        tables = " ".join(a.upper() for a in row["aliases"])
        subplans.setdefault(row["query_id"], []).append(tables)

    connection = psycopg2.connect(dsn)
    connection.autocommit = True
    cursor = connection.cursor()
    for setting in (
        "print_sub_queries = on",
        "debug_card_est = on",
        "ml_joinest_enabled = on",
        f"ml_joinest_fname = '{EST_FILE}'",
    ):
        cursor.execute(f"SET {setting}")
    out_lines = ["QueryID,Tables,native_Estimate"]
    missing = 0
    for query_id in sorted(wanted):
        estimates = in_context_estimates(cursor, queries[query_id])
        for tables in subplans.get(query_id, []):
            key = frozenset(t.lower() for t in tables.split())
            if key not in estimates:
                missing += 1
                continue
            out_lines.append(f"{query_id},{tables},{estimates[key]}")
        logger.info("q%d: %d join relations sized", query_id, len(estimates))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n")
    logger.info(
        "%d sub-plan estimates (%d sub-plans never sized by the planner) -> %s",
        len(out_lines) - 1,
        missing,
        args.output,
    )


if __name__ == "__main__":
    main()
