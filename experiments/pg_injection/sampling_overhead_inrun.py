"""Sampling cost against query execution, timed in one run.

``sampling_overhead`` divides sampled-count times from one run by native
execution times from another, taken under different host load. This script
times both back to back, per query and under the same load gate: PostgreSQL
executes its own plan for the query, then the in-engine sampled count of every
sub-plan of that query (``TABLESAMPLE BERNOULLI`` on the largest relation, as in
``subset_counts``). Each query alternates the two ``--reps`` times (once when
the native plan takes longer than 5 s) and reports the median of each.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.sampling_overhead_inrun
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import psycopg2

from experiments.pg_injection.inject_plans import QUERIES, wait_for_quiet
from experiments.pg_injection.subset_counts import (
    FROM_ITEM,
    WORKLOADS,
    _enough_disk,
    _read_jsonl,
    connect,
    count_sql,
    load_subqueries,
    safe_count,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def time_native(
    cursor: Any, dsn: str, sql: str, timeout_ms: int
) -> tuple[Any, float, bool]:
    """Execute the query with PostgreSQL's own plan; (cursor, ms, censored)."""
    cursor.execute(f"SET statement_timeout = {timeout_ms}")
    start = time.perf_counter()
    try:
        cursor.execute(sql)
        cursor.fetchall()
    except psycopg2.errors.QueryCanceled:
        cursor.connection.rollback()
        return cursor, float(timeout_ms), True
    except psycopg2.OperationalError:
        logger.warning("backend crashed on the native plan; reconnecting")
        _, cursor = connect(dsn)
        return cursor, float(timeout_ms), True
    return cursor, 1000.0 * (time.perf_counter() - start), False


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default="host=127.0.0.1 port=55432 dbname=stats "
        "user=postgres password=postgres",
    )
    parser.add_argument("--rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--native-timeout-ms", type=int, default=300_000)
    parser.add_argument("--sample-timeout-ms", type=int, default=120_000)
    parser.add_argument("--max-load", type=float, default=60.0)
    parser.add_argument("--max-wait-s", type=float, default=1800.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v2/sampling_overhead_inrun.jsonl"),
    )
    args = parser.parse_args()

    preset = WORKLOADS["stats"]
    by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in load_subqueries(preset["subqueries"]):
        by_query[row["query_id"]].append(row)
    lines = [raw for raw in QUERIES.read_text().splitlines() if raw.strip()]
    queries = {
        i + 1: count_sql(line.split("||", 1)[-1].strip().rstrip(";")) + ";"
        for i, line in enumerate(lines)
    }
    done = (
        {r["query_id"] for r in _read_jsonl(args.output)}
        if args.output.exists()
        else set()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    _, cursor = connect(args.dsn)
    cursor.execute("SELECT relname, reltuples FROM pg_class WHERE relkind = 'r'")
    reltuples = {name.lower(): float(n) for name, n in cursor.fetchall()}

    with args.output.open("a") as sink:
        for query_id, sql in sorted(queries.items()):
            if query_id in done or query_id not in by_query:
                continue
            if not _enough_disk(args.output):
                logger.error("less than 10 GB free on the output disk; stopping")
                break
            load, high = wait_for_quiet(args.max_load, args.max_wait_s)
            samples = []
            for row in by_query[query_id]:
                tables = {
                    a.lower(): t.lower() for t, a in FROM_ITEM.findall(row["sql"])
                }
                largest = max(row["aliases"], key=lambda a: reltuples[tables[a]])
                samples.append(count_sql(row["sql"], (largest, args.rate, args.seed)))
            native_ms: list[float] = []
            sample_ms: list[float] = []
            native_censored = sample_censored = False
            for rep in range(args.reps):
                cursor, ms, censored = time_native(
                    cursor, args.dsn, sql, args.native_timeout_ms
                )
                native_ms.append(ms)
                native_censored |= censored
                total = 0.0
                for sample_sql in samples:
                    cursor, value, elapsed, _ = safe_count(
                        cursor, args.dsn, sample_sql, args.sample_timeout_ms
                    )
                    total += elapsed
                    sample_censored |= value is None
                sample_ms.append(total)
                if rep == 0 and ms > 5_000:
                    break  # long query: time it once
            record = {
                "query_id": query_id,
                "n_subplans": len(samples),
                "rate": args.rate,
                "native_ms": native_ms,
                "sample_total_ms": sample_ms,
                "native_median_ms": median(native_ms),
                "sample_median_ms": median(sample_ms),
                "native_censored": native_censored,
                "sample_censored": sample_censored,
                "loadavg_1m_start": load,
                "high_load": high,
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            logger.info(
                "q%d: native %.0f ms, %d samples %.0f ms (%.2fx)",
                query_id,
                record["native_median_ms"],
                len(samples),
                record["sample_median_ms"],
                record["sample_median_ms"] / max(record["native_median_ms"], 1e-9),
            )


if __name__ == "__main__":
    main()
