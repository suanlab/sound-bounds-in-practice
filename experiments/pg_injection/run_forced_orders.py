"""Execute forced left-deep join orders against PostgreSQL's own plans.

Supersedes ``run_certified_plans``, whose ``pg_hint_plan`` ``Leading`` hints
were not honoured in our build (see ``forced_order``). For every STATS-CEB query
with 3-7 relations it times, interleaved and under the same load gate:

* ``native``    -- PostgreSQL's own plan;
* ``native_lo`` -- the join order of that plan, forced the same way as the
  others, when the native plan is left-deep (a control for the cost of forcing);
* ``cout_opt``  -- the left-deep order with the smallest true ``C_out``;
* ``certified`` -- the order certified within 2x by the joint certificate
  (LpBound x in-engine sample, r = 0.10), on certified queries only.

Every forced plan is checked with ``EXPLAIN``: its join nodes must cover exactly
the prefixes of the order.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.run_forced_orders
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

from experiments.pg_injection.forced_order import UNFORCE, forced_join_sql
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
from experiments.scripts.plan_certificate_joint import left_deep_orders

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

JOINS = {"Hash Join", "Merge Join", "Nested Loop"}


def join_sets(node: dict[str, Any], out: list[frozenset[str]]) -> set[str]:
    """Collect the alias set of every join node (post-order); return the node's."""
    aliases: set[str] = set()
    if "Alias" in node:
        aliases.add(node["Alias"])
    for child in node.get("Plans", []):
        aliases |= join_sets(child, out)
    if node.get("Node Type") in JOINS:
        out.append(frozenset(aliases))
    return aliases


def left_deep_order(plan: dict[str, Any]) -> list[str] | None:
    """The join order of a plan if it is left-deep (nested join sets), else None."""
    sets: list[frozenset[str]] = []
    join_sets(plan, sets)
    chain = sorted(set(sets), key=len)
    if not chain or len(chain[0]) != 2:
        return None
    if any(len(b) != len(a) + 1 or not a < b for a, b in itertools.pairwise(chain)):
        return None
    order = sorted(chain[0])
    for a, b in itertools.pairwise(chain):
        order += sorted(b - a)
    return order


def honours(plan: dict[str, Any], order: list[str]) -> bool:
    """Whether the plan's join nodes are exactly the prefixes of ``order``."""
    sets: list[frozenset[str]] = []
    join_sets(plan, sets)
    want = {frozenset(order[:k]) for k in range(2, len(order) + 1)}
    return set(sets) == want


def cout_optimal(record: dict[str, Any]) -> list[str]:
    """The left-deep order with the smallest true C_out (final result included)."""
    truth = {frozenset(s["aliases"]): float(s["truth"]) for s in record["subjoins"]}
    index = dict.fromkeys(truth, 0)

    orders = [tuple(o) for o in left_deep_orders(record["aliases"], index)]

    def cost(order: tuple[str, ...]) -> float:
        return sum(truth[frozenset(order[: i + 1])] for i in range(1, len(order)))

    best: tuple[str, ...] = min(orders, key=cost)
    return list(best)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default="host=127.0.0.1 port=55432 dbname=stats "
        "user=postgres password=postgres",
    )
    parser.add_argument(
        "--sizes",
        type=Path,
        default=Path("results/plan_guard/subjoin_sizes_maxdeg_fixed.jsonl"),
    )
    parser.add_argument(
        "--selective",
        type=Path,
        default=Path("results/plan_certificate/selective_sampling_v2.json"),
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=300_000)
    parser.add_argument("--max-load", type=float, default=60.0)
    parser.add_argument("--max-wait-s", type=float, default=1800.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v2/forced_orders.jsonl"),
    )
    args = parser.parse_args()

    lines = [raw for raw in QUERIES.read_text().splitlines() if raw.strip()]
    queries = {i + 1: line.split("||", 1)[1].strip() for i, line in enumerate(lines)}
    records = {
        r["query_id"]: r
        for r in (json.loads(x) for x in args.sizes.read_text().splitlines() if x)
    }
    selective = {
        r["query_id"]: r for r in json.loads(args.selective.read_text())["per_query"]
    }
    done = (
        {r["query_id"] for r in _read_jsonl(args.output)}
        if args.output.exists()
        else set()
    )
    cursor = reconnect(args.dsn)
    with args.output.open("a") as sink:
        for query_id in sorted(selective):
            record = records.get(query_id)
            if query_id in done or record is None or record["n_aliases"] < 3:
                continue
            if not _enough_disk(args.output):
                logger.error("less than 10 GB free; stopping")
                break
            native_sql = UNFORCE + queries[query_id]
            _, native_plan = plan_of(cursor, native_sql)
            orders: dict[str, list[str]] = {"cout_opt": cout_optimal(record)}
            native_order = left_deep_order(native_plan)
            if native_order is not None:
                orders["native_lo"] = native_order
            if selective[query_id]["selective_certified"]:
                orders["certified"] = selective[query_id]["selective_order"]
            sqls = {"native": native_sql}
            plans = {"native": native_plan}
            for name, order in orders.items():
                sql = forced_join_sql(queries[query_id], order)
                _, plan = plan_of(cursor, sql)
                cursor.execute(UNFORCE)
                if not honours(plan, order):
                    msg = f"q{query_id}: forced order {order} not honoured ({name})"
                    raise RuntimeError(msg)
                sqls[name], plans[name] = sql, plan
            load_start, high_load = wait_for_quiet(args.max_load, args.max_wait_s)
            pending = dict(sqls)
            timings: dict[str, dict[str, Any]] = {}
            while pending:
                try:
                    timings.update(
                        run_interleaved(
                            cursor,
                            pending,
                            args.timeout_ms,
                            args.reps,
                            always_reps=True,  # every plan timed --reps times
                        )
                    )
                    pending = {}
                except PlanCrashedError as crash:  # out of memory: a failed plan
                    logger.warning("q%d: %s", query_id, crash)
                    timings[crash.digest] = {
                        "ms": float(args.timeout_ms),
                        "censored": True,
                        "reps": 1,
                    }
                    pending = {k: s for k, s in pending.items() if k != crash.digest}
                    cursor = reconnect(args.dsn)
                cursor.execute(UNFORCE)
            out = {
                "query_id": query_id,
                "n_aliases": record["n_aliases"],
                "orders": orders,
                "ms": {k: t["ms"] for k, t in timings.items()},
                "censored": {k: t["censored"] for k, t in timings.items()},
                "reps": {k: t["reps"] for k, t in timings.items()},
                "plans": plans,
                "sampling_ms": selective[query_id]["selective_ms"],
                "loadavg_1m_start": load_start,
                "high_load": high_load,
            }
            sink.write(json.dumps(out) + "\n")
            sink.flush()
            logger.info(
                "q%d: %s",
                query_id,
                {k: round(v["ms"]) for k, v in timings.items()},
            )


if __name__ == "__main__":
    main()
