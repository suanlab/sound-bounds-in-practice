"""Execute a few points of the tightness sweep (H3 confirmation in time).

``tightness_sweep`` scores plans by true ``C_out`` without running them. This
runs the plans of selected policies -- by default, for our instance's own
in-context estimates, the raw estimates, the true sizes, and both synthetic
guards at ``k = 2`` and ``k = 8`` -- with the protocol of the execution run:
every distinct plan of a query three times, interleaved, 300 s timeout, a
query starting only when the one-minute load is below 60 -- except that each
statement's temporary files are capped (``TEMP_FILE_LIMIT``); a plan that hits
the cap fails and is recorded as censored.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.execute_tightness_points \
        --native-csv results/pg_injection/v2/native_incontext_estimations.csv \
        --output results/pg_injection/v3/tightness_points_stats.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import (
    WORKLOAD_PATHS,
    Key,
    PlanCrashedError,
    _enough_disk,
    _key,
    _read_jsonl,
    hint_text,
    load_estimates,
    plan_of,
    reconnect,
    run_interleaved,
    wait_for_quiet,
)
from experiments.pg_injection.subset_counts import count_sql, with_dbname
from experiments.pg_injection.tightness_sweep import policies

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# The server's data directory shares the host's root disk; cap each statement's
# temporary files so a spilling plan fails (and is recorded as censored) before
# it can fill that disk. The v3 execution run had no such cap.
TEMP_FILE_LIMIT = "2GB"


def capped_cursor(dsn: str) -> Any:
    """A hinted cursor whose temporary files are capped."""
    cursor = reconnect(dsn)
    cursor.execute(f"SET temp_file_limit = '{TEMP_FILE_LIMIT}'")
    return cursor


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=(
            "host=127.0.0.1 port=55432 dbname=stats user=postgres password=postgres"
        ),
    )
    parser.add_argument("--workload", choices=sorted(WORKLOAD_PATHS), default="stats")
    parser.add_argument("--native-csv", type=Path, required=True)
    parser.add_argument("--policies", default="raw,truth,bi_k2,bi_k8")
    parser.add_argument("--timeout-ms", type=int, default=300_000)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-load", type=float, default=60.0)
    parser.add_argument("--max-wait-s", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    estimates_dir, subplan_file, query_file, dbname = WORKLOAD_PATHS[args.workload]
    if args.workload != "stats":
        args.dsn = with_dbname(args.dsn, dbname)
    wanted = args.policies.split(",")
    truth = load_estimates("truecardinality", directory=estimates_dir)
    real = load_estimates("lpbound", directory=estimates_dir)
    native = load_estimates("native", args.native_csv, directory=estimates_dir)
    subplans: dict[int, list[Key]] = {}
    for line in subplan_file.read_text().splitlines()[1:]:
        qid, tables, _ = line.split("|", 2)
        subplans.setdefault(int(qid), []).append(_key(qid, tables))
    queries = {
        i + 1: count_sql(line.split("||", 1)[-1].strip().rstrip(";")) + ";"
        for i, line in enumerate(
            raw for raw in query_file.read_text().splitlines() if raw.strip()
        )
    }
    done = set()
    if args.output.exists():
        done = {r["query_id"] for r in _read_jsonl(args.output)}
    cursor = capped_cursor(args.dsn)
    with args.output.open("a") as sink:
        for query_id in sorted(set(queries) - done):
            if not _enough_disk(args.output):
                logger.error("less than 10 GB free on the output disk; stopping")
                break
            keys = subplans.get(query_id, [])
            if not keys or not all(
                k in native and k in truth and k in real for k in keys
            ):
                continue
            values = policies(
                keys, {k: native[k] for k in keys}, {k: truth[k] for k in keys}, real
            )
            chosen: dict[str, str] = {}
            sqls: dict[str, str] = {}
            for name in wanted:
                sql = hint_text(values[name]) + queries[query_id]
                digest, _ = plan_of(cursor, sql)
                chosen[name] = digest
                sqls.setdefault(digest, sql)
            load_start, high_load = wait_for_quiet(args.max_load, args.max_wait_s)
            timings: dict[str, dict[str, Any]] = {}
            pending = dict(sqls)
            while pending:
                try:
                    timings.update(
                        run_interleaved(
                            cursor,
                            pending,
                            args.timeout_ms,
                            args.reps,
                            always_reps=True,
                        )
                    )
                    pending = {}
                except PlanCrashedError as crash:
                    logger.warning("q%d: %s; reconnecting", query_id, crash)
                    timings[crash.digest] = {
                        "ms": float(args.timeout_ms),
                        "censored": True,
                    }
                    pending = {d: s for d, s in pending.items() if d != crash.digest}
                    cursor = capped_cursor(args.dsn)
            record = {
                "query_id": query_id,
                "policy_plan": chosen,
                "plan_ms": {d: t["ms"] for d, t in timings.items()},
                "plan_censored": {d: t["censored"] for d, t in timings.items()},
                "loadavg_1m_start": load_start,
                "loadavg_1m_end": os.getloadavg()[0],
                "high_load": high_load,
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            logger.info(
                "q%d: %d distinct plans, %s",
                query_id,
                len(sqls),
                {n: round(timings[d]["ms"]) for n, d in chosen.items()},
            )


if __name__ == "__main__":
    main()
