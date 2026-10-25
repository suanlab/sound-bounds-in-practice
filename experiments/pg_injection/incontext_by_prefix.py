"""PostgreSQL's in-context sub-plan estimates without the benchmark patch.

The PostgreSQL 13.1 harness reads the planner's estimate of every join sub-plan
inside the full query through hooks of the STATS-CEB benchmark patch
(``native_estimates_incontext``). Newer PostgreSQL versions lack that patch. A
join relation's row estimate does not depend on how it is joined, so it can be
read from any plan that builds it: forcing a left-deep order whose prefix is the
sub-plan (explicit joins, ``join_collapse_limit = 1``, ``forced_order``) makes
the join node of that prefix carry the planner's estimate for it, and every
other prefix of the same order yields one more sub-plan. This covers every
sub-plan with few ``EXPLAIN`` calls and writes the artifact's CSV format.

``EXPLAIN`` reports estimates rounded to whole rows, so on PostgreSQL 13.1 the
values are compared with the hook values after rounding (``--compare``).

Result (2026-10-20, PostgreSQL 13.1, STATS-CEB): exact for all 912 two-relation
sub-plans, but only 1,525 of 2,577 overall. The premise is wrong for three or
more relations: the planner sizes a join relation from the first split its
search builds, and a forced order builds it from another split, which can
differ by two orders of magnitude. Use ``incontext_by_joinrel_log`` instead,
which reproduces all 2,577 (and all 696 on JOB-light).

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.incontext_by_prefix \
        --workload stats --output results/pg_injection/v3/prefix_estimates_stats.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from experiments.pg_injection.forced_order import UNFORCE, forced_join_sql
from experiments.pg_injection.inject_plans import (
    WORKLOAD_PATHS,
    Key,
    _key,
    load_estimates,
    plan_of,
    reconnect,
)
from experiments.pg_injection.run_forced_orders import JOINS
from experiments.pg_injection.subset_counts import count_sql, with_dbname

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def order_with_prefix(
    target: frozenset[str], aliases: list[str], connected: set[frozenset[str]]
) -> list[str] | None:
    """A left-deep order whose prefixes are connected and include ``target``."""
    order: list[str] = []
    for goal in (target, frozenset(aliases)):
        while len(order) < len(goal):
            nxt = next(
                (
                    a
                    for a in sorted(goal - set(order))
                    if not order or frozenset([*order, a]) in connected
                ),
                None,
            )
            if nxt is None:
                return None
            order.append(nxt)
    return order


def join_estimates(node: dict[str, Any], out: dict[frozenset[str], float]) -> set[str]:
    """Plan Rows of every join node, keyed by the aliases it joins."""
    aliases: set[str] = set()
    if "Alias" in node:
        aliases.add(node["Alias"])
    for child in node.get("Plans", []):
        aliases |= join_estimates(child, out)
    if node.get("Node Type") in JOINS:
        out[frozenset(aliases)] = float(node["Plan Rows"])
    return aliases


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
    parser.add_argument("--queries", default="", help="e.g. 1,2,10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="hook-based in-context CSV to compare with (PostgreSQL 13.1)",
    )
    args = parser.parse_args()

    _, subplan_file, query_file, dbname = WORKLOAD_PATHS[args.workload]
    dsn = with_dbname(args.dsn, dbname)
    subplans: dict[int, list[Key]] = {}
    for line in subplan_file.read_text().splitlines()[1:]:
        qid, tables, _ = line.split("|", 2)
        subplans.setdefault(int(qid), []).append(_key(qid, tables))
    lines = [raw for raw in query_file.read_text().splitlines() if raw.strip()]
    queries = {
        i + 1: count_sql(line.split("||", 1)[-1].strip().rstrip(";")) + ";"
        for i, line in enumerate(lines)
    }
    wanted = {int(q) for q in args.queries.split(",") if q} or set(subplans)
    cursor = reconnect(dsn)
    rows: list[tuple[int, tuple[str, ...], float]] = []
    n_explain = 0
    missing: list[Key] = []
    for query_id in sorted(wanted):
        keys = subplans.get(query_id, [])
        connected = {frozenset(k[1]) for k in keys}
        sizes = sorted(connected, key=len)
        aliases = sorted(sizes[-1]) if sizes else []
        found: dict[frozenset[str], float] = {}
        for key in sorted(keys, key=lambda k: -len(k[1])):
            target = frozenset(key[1])
            if target in found:
                continue
            order = order_with_prefix(target, aliases, connected)
            if order is None:
                missing.append(key)
                continue
            _, plan = plan_of(cursor, forced_join_sql(queries[query_id], order))
            cursor.execute(UNFORCE)
            n_explain += 1
            estimates: dict[frozenset[str], float] = {}
            join_estimates(plan, estimates)
            found.update(estimates)
            if target not in found:
                missing.append(key)
        for key in keys:
            if frozenset(key[1]) in found:
                rows.append((query_id, key[1], found[frozenset(key[1])]))
    lines_out = ["QueryID,Tables,native_Estimate"] + [
        f"{q},{' '.join(a.upper() for a in t)},{v:.17g}" for q, t, v in rows
    ]
    args.output.write_text("\n".join(lines_out) + "\n")
    summary: dict[str, Any] = {
        "n_subplans": sum(len(subplans.get(q, [])) for q in wanted),
        "n_estimated": len(rows),
        "n_missing": len(missing),
        "n_explain": n_explain,
    }
    if args.compare is not None:
        hooks = load_estimates("native", args.compare, directory=Path("."))
        pairs = [(v, hooks[(q, t)]) for q, t, v in rows if (q, t) in hooks]
        summary["n_compared"] = len(pairs)
        summary["n_equal_after_rounding"] = sum(
            abs(v - max(round(h), 1)) <= 1 for v, h in pairs
        )
        summary["max_abs_diff"] = max(abs(v - max(round(h), 1)) for v, h in pairs)
    logger.info("%s", json.dumps(summary))
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
