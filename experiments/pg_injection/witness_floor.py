"""Witness floors: is proving a sub-join large cheaper than sampling it? (H7)

A sound lower bound need not come from a sample: any ``k`` result tuples of the
exact join witness ``|S| >= k``, and ``SELECT COUNT(*) FROM (SELECT 1 ... LIMIT
k)`` stops as soon as it has them. For every sub-join in the hindsight-optimal
certifying sets (``price_of_certification``), with ``k`` the sound sampled
count the certificate used (so the witnessed floor certifies exactly as the
sample did), this times the witness query and, back to back, the sampled count
of the same sub-join (r = 0.10, seed 0), each with ``timed_count`` (two runs,
the second timed; 120 s timeout). Deviation from the frozen prediction, which
spoke of floors "computed from bounds only": the floor values are the sampled
counts, so that both routes certify the same queries. Exploratory.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.witness_floor
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from experiments.pg_injection.inject_plans import _read_jsonl, wait_for_quiet
from experiments.pg_injection.subset_counts import (
    SUBQUERIES,
    connect,
    count_sql,
    load_subqueries,
    timed_count,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def witness_sql(sql: str, k: int) -> str:
    """Count of at most ``k`` result tuples of the exact join."""
    body = re.sub(r"^\s*SELECT\s+\*\s+FROM", "SELECT 1 FROM", sql, flags=re.I)
    return f"SELECT COUNT(*) FROM ({body} LIMIT {k}) AS witness"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=(
            "host=127.0.0.1 port=55432 dbname=stats user=postgres password=postgres"
        ),
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("results/plan_certificate/price_of_certification.json"),
    )
    parser.add_argument(
        "--samples", type=Path, default=Path("results/pg_injection/subset_counts.jsonl")
    )
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--max-load", type=float, default=60.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v3/witness_floor.jsonl"),
    )
    args = parser.parse_args()

    sql_of = {
        (r["query_id"], tuple(r["aliases"])): r["sql"]
        for r in load_subqueries(SUBQUERIES)
    }
    samples = {
        (r["query_id"], tuple(sorted(r["aliases"]))): r
        for r in _read_jsonl(args.samples)
    }
    targets = [
        (row["query_id"], tuple(aliases))
        for row in json.loads(args.oracle.read_text())["per_query"]
        for aliases in row["oracle_subjoins"]
    ]
    done = set()
    if args.output.exists():
        done = {(r["query_id"], tuple(r["aliases"])) for r in _read_jsonl(args.output)}
    _, cursor = connect(args.dsn)
    with args.output.open("a") as sink:
        for key in targets:
            if key in done:
                continue
            sample = samples[key]
            k = int(sample["lower_sound"])
            load, _ = wait_for_quiet(args.max_load, 1800.0)
            witnessed, witness_ms = timed_count(
                cursor, witness_sql(sql_of[key], k), args.timeout_ms
            )
            sampled, sample_ms = timed_count(
                cursor,
                count_sql(sql_of[key], (sample["sampled_alias"], 0.10, 0)),
                args.timeout_ms,
            )
            if witnessed is not None and witnessed != k:
                msg = f"{key}: witnessed {witnessed} < floor {k}, which is impossible"
                raise AssertionError(msg)
            record = {
                "query_id": key[0],
                "aliases": list(key[1]),
                "floor": k,
                "witness_ms": witness_ms,
                "witness_censored": witnessed is None,
                "sample_ms": sample_ms,
                "sample_censored": sampled is None,
                "loadavg_1m_start": load,
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            logger.info(
                "q%d %s floor %d: witness %.0f ms, sample %.0f ms",
                key[0],
                key[1],
                k,
                witness_ms,
                sample_ms,
            )


if __name__ == "__main__":
    main()
