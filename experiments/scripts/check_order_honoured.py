"""Precondition check: does the engine actually execute the join order we give it?

The runtime experiment only measures what it claims to measure if the engine
executes the FROM-clause order rather than one of its own choosing. That is an
assumption about each engine, and assumptions about engines are exactly what
this paper argues should be tested rather than believed.

The test is a manipulation check on the *spread* across orders. For a query we
time every left-deep permutation of its aliases. An engine that reorders
internally runs the same physical plan whatever we write, so all permutations
must take about the same time and the spread collapses towards 1. An engine
that honours the order produces a wide spread, because some orders really are
much worse than others.

The evidence is one-sided and reported as such: a *large* spread proves the
engine honours the order, since an engine that ignored it could not produce
one. A small spread on a single query proves nothing --- the orders may simply
be equally good --- so several queries are tried and the largest spread per
engine is what counts.

Every permutation must also return the same count. That is what checks the
``CROSS JOIN`` rewrite used for SQLite against the comma-join form used for
DuckDB: same query, same answer, different execution order.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.check_order_honoured
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import permutations
from pathlib import Path
from typing import Any

from experiments.scripts.plan_regret import time_limit
from experiments.scripts.plan_runtime import (
    ENGINES,
    connect,
    index_join_columns,
    query_deadline,
    render_sql,
    time_order,
)
from src.utils.job_sql_parser import parse_job_sql

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# A spread at or above this on any query proves the engine honours the order.
MIN_SPREAD = 3.0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("external/LpBound")
    parser.add_argument(
        "--queries",
        type=Path,
        default=base / "benchmarks/workloads/stats/statsQueries.sql",
    )
    parser.add_argument("--data-dir", type=Path, default=base / "data/datasets/stats")
    parser.add_argument("--n-queries", type=int, default=3)
    parser.add_argument("--n-aliases", type=int, default=4)
    parser.add_argument("--max-orders", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--order-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_regret/order_honoured.json"),
    )
    args = parser.parse_args()

    lines = [
        line.strip() for line in args.queries.read_text().splitlines() if line.strip()
    ]

    candidates: list[int] = []
    for index, sql in enumerate(lines, start=1):
        parsed = parse_job_sql(sql, str(index))
        if parsed.fully_parsed and len(parsed.aliases) == args.n_aliases:
            candidates.append(index)
        if len(candidates) == args.n_queries:
            break
    if not candidates:
        msg = f"no fully parsed query with {args.n_aliases} aliases"
        raise RuntimeError(msg)

    engines: dict[str, Any] = {}
    for engine in ENGINES:
        con = connect(args.data_dir, engine)
        per_query: list[dict[str, Any]] = []
        for index in candidates:
            sql = lines[index - 1]
            parsed = parse_job_sql(sql, str(index))
            if engine == "sqlite":
                index_join_columns(con, parsed)  # type: ignore[arg-type]
            aliases = sorted(parsed.aliases)
            timings: list[float] = []
            counts: set[int] = set()
            censored = 0
            for order in list(permutations(aliases))[: args.max_orders]:
                try:
                    with (
                        time_limit(args.order_timeout_s + 5.0),
                        query_deadline(con, args.order_timeout_s),
                    ):
                        seconds, count = time_order(
                            con, render_sql(sql, order, parsed, engine), args.repeats
                        )
                except Exception:
                    # An order too slow to finish is recorded at the timeout, so
                    # the spread it implies is a lower bound, never an estimate.
                    timings.append(args.order_timeout_s)
                    censored += 1
                    continue
                timings.append(seconds)
                counts.add(count)
            spread = max(timings) / min(timings)
            per_query.append(
                {
                    "query_id": index,
                    "n_orders": len(timings),
                    "ms_fastest": min(timings) * 1000.0,
                    "ms_slowest": max(timings) * 1000.0,
                    "spread": spread,
                    "spread_is_lower_bound": censored > 0,
                    "orders_censored_at_timeout": censored,
                    "counts_agree": len(counts) == 1,
                    "count": sorted(counts)[0] if counts else None,
                }
            )
            logger.info(
                "%-7s q%-3d %2d orders  %8.1f ms .. %9.1f ms  spread %6.1fx  %s",
                engine,
                index,
                len(timings),
                min(timings) * 1000.0,
                max(timings) * 1000.0,
                spread,
                ("counts agree" if len(counts) == 1 else "COUNTS DISAGREE")
                + (f", {censored} censored" if censored else ""),
            )
        max_spread = max(q["spread"] for q in per_query)
        engines[engine] = {
            "max_spread": max_spread,
            "order_honoured": max_spread >= MIN_SPREAD,
            "all_counts_agree": all(q["counts_agree"] for q in per_query),
            "per_query": per_query,
        }
        logger.info(
            "%-7s -> max spread %.1fx across %d queries: %s\n",
            engine,
            max_spread,
            len(per_query),
            "ORDER HONOURED" if max_spread >= MIN_SPREAD else "NOT DEMONSTRATED",
        )

    payload = {
        "experiment": "join_order_honoured_manipulation_check",
        "note": (
            "Every left-deep permutation of a query's aliases is timed. An "
            "engine that reorders internally would run one plan regardless and "
            "show no spread; a large spread therefore proves the FROM order is "
            "honoured. A small spread is not evidence of the converse. Orders "
            "that exceed the timeout are recorded at the timeout, so a spread "
            "marked as a lower bound understates the real one."
        ),
        "min_spread_required": MIN_SPREAD,
        "queries": candidates,
        "n_aliases": args.n_aliases,
        "engines": engines,
        "all_honoured": all(e["order_honoured"] for e in engines.values()),
        "all_counts_agree": all(e["all_counts_agree"] for e in engines.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("-> %s", args.output)


if __name__ == "__main__":
    main()
