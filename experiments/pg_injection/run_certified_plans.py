"""Execute the join orders that selective sampling certifies, against native plans.

For every STATS-CEB query that ``selective_sampling.py`` certifies within its
target, run (interleaved, ``--reps`` times, under the load gate) PostgreSQL's
native plan and the certified left-deep join order forced with a
``pg_hint_plan`` ``Leading(a b c ...)`` hint, which fixes the join order and
leaves join methods and inner/outer choices to PostgreSQL. End to end, the
certified path costs the measured in-engine sampling time plus its execution.
Post-hoc evaluation of the offline design.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.run_certified_plans
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from experiments.pg_injection.inject_plans import (
    QUERIES,
    PlanCrashedError,
    _enough_disk,
    _read_jsonl,
    plan_of,
    reconnect,
    run_interleaved,
    wait_for_quiet,
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
    parser.add_argument(
        "--selective",
        type=Path,
        default=Path("results/plan_certificate/selective_sampling.json"),
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=300_000)
    parser.add_argument("--max-load", type=float, default=60.0)
    parser.add_argument("--max-wait-s", type=float, default=1800.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/certified_plans.jsonl"),
    )
    args = parser.parse_args()

    queries = {
        i + 1: line.split("||", 1)[1].strip()
        for i, line in enumerate(
            raw for raw in QUERIES.read_text().splitlines() if raw.strip()
        )
    }
    rows = json.loads(args.selective.read_text())["per_query"]
    done = (
        {r["query_id"] for r in _read_jsonl(args.output)}
        if args.output.exists()
        else set()
    )
    cursor = reconnect(args.dsn)
    with args.output.open("a") as sink:
        for row in rows:
            if not row["selective_certified"] or row["query_id"] in done:
                continue
            if not _enough_disk(args.output):
                logger.error("less than 10 GB free; stopping")
                break
            sql = queries[row["query_id"]]
            leading = f"/*+ Leading({' '.join(row['selective_order'])}) */ "
            native_digest, _ = plan_of(cursor, sql)
            forced_digest, forced_plan = plan_of(cursor, leading + sql)
            load_start, high_load = wait_for_quiet(args.max_load, args.max_wait_s)
            pending = {"native": sql, "certified": leading + sql}
            timings: dict[str, dict[str, float | bool | int]] = {}
            while pending:
                try:
                    timings.update(
                        run_interleaved(cursor, pending, args.timeout_ms, args.reps)
                    )
                    pending = {}
                except PlanCrashedError as crash:  # out of memory: a failed plan
                    logger.warning("q%d: %s", row["query_id"], crash)
                    timings[crash.digest] = {
                        "ms": float(args.timeout_ms),
                        "censored": True,
                        "reps": 1,
                    }
                    pending = {k: s for k, s in pending.items() if k != crash.digest}
                    cursor = reconnect(args.dsn)
            record = {
                "query_id": row["query_id"],
                "order": row["selective_order"],
                "same_plan_as_native": native_digest == forced_digest,
                "native_ms": timings["native"]["ms"],
                "native_censored": timings["native"]["censored"],
                "certified_ms": timings["certified"]["ms"],
                "certified_censored": timings["certified"]["censored"],
                "sampling_ms": row["selective_ms"],
                "reps": {k: v["reps"] for k, v in timings.items()},
                "certified_plan": forced_plan,
                "loadavg_1m_start": load_start,
                "loadavg_1m_end": os.getloadavg()[0],
                "high_load": high_load,
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            logger.info(
                "q%d: native %.0f ms, certified %.0f ms (+%.0f ms sampling)%s",
                row["query_id"],
                record["native_ms"],
                record["certified_ms"],
                record["sampling_ms"],
                " [same plan]" if record["same_plan_as_native"] else "",
            )


if __name__ == "__main__":
    main()
