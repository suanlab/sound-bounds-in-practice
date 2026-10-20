"""PostgreSQL's own (native) estimate for every sub-plan of the STATS-CEB queries.

The artifact's ``postgres_subquery_estimations.csv`` came from another
PostgreSQL instance, and planning on it is far worse than this instance's native
planning. To guard the estimates *this* optimizer actually uses, read them back:
``EXPLAIN (FORMAT JSON) SELECT COUNT(*) ...`` of each sub-plan query, taking the
row estimate of the node below the aggregate. Planning only -- no execution.
Injecting these values unchanged must reproduce the native plan; that is
checked in ``inject_plans.py`` (policy ``native_raw`` against ``native``).

Writes a CSV in the artifact's format (``QueryID,Tables,native_Estimate``).

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.native_estimates
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import psycopg2

from experiments.pg_injection.subset_counts import (
    WORKLOADS,
    count_sql,
    load_subqueries,
    with_dbname,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=(
            "host=127.0.0.1 port=55432 dbname=stats user=postgres password=postgres"
        ),
    )
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="stats")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/native_subquery_estimations.csv"),
    )
    args = parser.parse_args()

    preset = WORKLOADS[args.workload]
    if args.workload != "stats":
        args.dsn = with_dbname(args.dsn, preset["dbname"])
    connection = psycopg2.connect(args.dsn)
    connection.autocommit = True
    cursor = connection.cursor()
    lines = ["QueryID,Tables,native_Estimate"]
    for row in load_subqueries(preset["subqueries"]):
        cursor.execute("EXPLAIN (FORMAT JSON) " + count_sql(row["sql"]))
        fetched = cursor.fetchone()
        if fetched is None:
            msg = f"no EXPLAIN output for q{row['query_id']} {row['aliases']}"
            raise RuntimeError(msg)
        plan = fetched[0][0]["Plan"]
        node = plan["Plans"][0] if plan.get("Node Type") == "Aggregate" else plan
        tables = " ".join(a.upper() for a in row["aliases"])
        lines.append(f"{row['query_id']},{tables},{float(node['Plan Rows'])}")
    args.output.write_text("\n".join(lines) + "\n")
    logger.info("%d sub-plan estimates -> %s", len(lines) - 1, args.output)


if __name__ == "__main__":
    main()
