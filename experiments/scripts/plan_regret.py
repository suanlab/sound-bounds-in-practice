"""Plan-level regret: does guarding *subplan* estimates change the chosen plan?

This is the plan-level experiment the paper otherwise lacks, and it is a direct
measurement rather than a surrogate. For each real STATS-CEB query we:

  1. compute the **true** cardinality of every connected sub-join (exact
     count-based hash join),
  2. compute, for the same sub-joins, an optimistic CEG estimate and the
     max-degree upper bound $B(S)$,
  3. let a textbook left-deep optimizer choose a join order by minimising the
     sum of *estimated* intermediate sizes, once with raw estimates and once
     with each subplan estimate passed through the guard, and
  4. score both chosen orders by their **true** cost --- the sum of their true
     intermediate sizes --- against the best order under true sizes.

``plan_regret = true_cost(chosen) / true_cost(optimal)``.

Why this is not the surrogate we refused to report: plan regret is *not* a
monotone function of the final q-error. Two estimates with identical q-error on
the full query can order sub-joins differently and therefore land on plans with
very different true costs; conversely a large q-error that preserves the
ordering costs nothing. The mapping estimate -> plan -> true cost is what we
measure here.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.plan_regret --max-aliases 4
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import itertools
import json
import logging
import signal
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

from src.core.ceg import pessimistic_bound
from src.core.query import Query
from src.utils.job_eval import build_alias_rows, build_job_query
from src.utils.job_sql_parser import parse_job_sql
from src.utils.join_counting import count_join
from src.utils.types import ALL_CEG_CONFIGS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.utils.job_sql_parser import ParsedJobQuery

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class QueryTimeoutError(TimeoutError):
    """Raised when a per-query computation exceeds its budget."""


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Abort the enclosed block after ``seconds`` (SIGALRM, main thread only)."""

    def _handler(signum: int, frame: object) -> None:
        raise QueryTimeoutError

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _connected(subset: tuple[str, ...], parsed: ParsedJobQuery) -> bool:
    """Whether the aliases in ``subset`` form a connected join sub-graph."""
    if len(subset) == 1:
        return True
    members = set(subset)
    seen = {subset[0]}
    frontier = [subset[0]]
    while frontier:
        node = frontier.pop()
        for join in parsed.joins:
            pair = {join.left_relation, join.right_relation}
            if node in pair:
                other = (pair - {node}).pop()
                if other in members and other not in seen:
                    seen.add(other)
                    frontier.append(other)
    return seen == members


def _sub_query(full: Query, subset: tuple[str, ...]) -> Query | None:
    """Restrict a CEG query to ``subset``; None if it is not a valid query."""
    relations = {name: full.relations[name] for name in subset}
    joins = frozenset(
        j
        for j in full.joins
        if j.left_relation in subset and j.right_relation in subset
    )
    try:
        return Query(relations=relations, joins=joins)
    except ValueError:
        return None


def _left_deep_orders(
    aliases: list[str], parsed: ParsedJobQuery
) -> list[tuple[str, ...]]:
    """All alias permutations whose every prefix is a connected sub-join."""
    orders: list[tuple[str, ...]] = []
    for perm in itertools.permutations(aliases):
        if all(_connected(perm[: i + 1], parsed) for i in range(1, len(perm))):
            orders.append(perm)
    return orders


def _order_cost(order: tuple[str, ...], sizes: dict[frozenset[str], float]) -> float:
    """Sum of intermediate sizes over the prefixes of a left-deep order."""
    return sum(sizes[frozenset(order[: i + 1])] for i in range(1, len(order)))


def evaluate_query(
    parsed: ParsedJobQuery,
    rows_by_relation: dict[str, list[dict[str, str]]],
    tau: float,
) -> list[dict[str, Any]]:
    """Return one record per CEG configuration for this query."""
    alias_rows = build_alias_rows(parsed, rows_by_relation)
    aliases = sorted(parsed.aliases)
    full = build_job_query(parsed, rows_by_relation)

    true_sizes: dict[frozenset[str], float] = {}
    est_sizes: dict[str, dict[frozenset[str], float]] = {
        c.name: {} for c in ALL_CEG_CONFIGS
    }
    bound_sizes: dict[frozenset[str], float] = {}

    from src.estimators.baseline_ceg import BaselineCEGEstimator

    estimator = BaselineCEGEstimator()
    for size in range(2, len(aliases) + 1):
        for subset in itertools.combinations(aliases, size):
            if not _connected(subset, parsed):
                continue
            key = frozenset(subset)
            joins = frozenset(
                j
                for j in parsed.joins
                if j.left_relation in subset and j.right_relation in subset
            )
            true_sizes[key] = max(
                count_join({a: alias_rows[a] for a in subset}, key, joins), 1.0
            )
            sub = _sub_query(full, subset)
            if sub is None:
                return []
            bound_sizes[key] = max(pessimistic_bound(sub), 1.0)
            per_config = estimator.estimate_all_configs(sub)
            for config in ALL_CEG_CONFIGS:
                est_sizes[config.name][key] = max(per_config[config.name].estimate, 1.0)

    orders = _left_deep_orders(aliases, parsed)
    if not orders or any(frozenset(o) not in true_sizes for o in [tuple(aliases)]):
        return []
    optimal_cost = min(_order_cost(o, true_sizes) for o in orders)

    records: list[dict[str, Any]] = []
    for config in ALL_CEG_CONFIGS:
        raw = est_sizes[config.name]
        guarded = {
            k: (bound_sizes[k] if raw[k] > tau * bound_sizes[k] else raw[k])
            for k in raw
        }
        picked_raw = min(orders, key=lambda o: _order_cost(o, raw))
        picked_guard = min(orders, key=lambda o: _order_cost(o, guarded))
        records.append(
            {
                "query_id": parsed.query_id,
                "n_aliases": len(aliases),
                "config": config.name,
                "n_orders": len(orders),
                "regret_no_guard": _order_cost(picked_raw, true_sizes) / optimal_cost,
                "regret_guarded": _order_cost(picked_guard, true_sizes) / optimal_cost,
                "plan_changed": picked_raw != picked_guard,
            }
        )
    return records


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


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
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--max-aliases", type=int, default=4)
    parser.add_argument("--query-timeout-s", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/plan_regret/plan_regret.json")
    )
    args = parser.parse_args()

    queries = [
        line.strip() for line in args.queries.read_text().splitlines() if line.strip()
    ]
    cache: dict[str, list[dict[str, str]]] = {}
    records: list[dict[str, Any]] = []
    skipped: list[str] = []

    for index, sql in enumerate(queries, start=1):
        parsed = parse_job_sql(sql, str(index))
        if not parsed.fully_parsed or not 3 <= len(parsed.aliases) <= args.max_aliases:
            continue
        try:
            with time_limit(args.query_timeout_s):
                for table in parsed.base_relations:
                    if table not in cache:
                        with (args.data_dir / f"{table}.csv").open(newline="") as fh:
                            cache[table] = list(csv.DictReader(fh))
                new = evaluate_query(
                    parsed, {t: cache[t] for t in parsed.base_relations}, args.tau
                )
        except (QueryTimeoutError, KeyError, ValueError) as exc:
            skipped.append(f"{index}:{type(exc).__name__}")
            continue
        if not new:
            skipped.append(f"{index}:degenerate")
            continue
        records.extend(new)
        logger.info(
            "  q%-3d aliases=%d orders=%d  regret no-guard=%.2f guarded=%.2f",
            index,
            new[0]["n_aliases"],
            new[0]["n_orders"],
            median(r["regret_no_guard"] for r in new),
            median(r["regret_guarded"] for r in new),
        )
        if args.limit and len({r["query_id"] for r in records}) >= args.limit:
            break

    raw = [r["regret_no_guard"] for r in records]
    guarded = [r["regret_guarded"] for r in records]
    payload = {
        "experiment": "plan_regret_left_deep",
        "dataset": "real STATS-CEB",
        "note": (
            "Direct measurement: true intermediate cardinalities score the plans "
            "an optimizer picks from guarded vs unguarded subplan estimates. Not a "
            "monotone transform of final-query q-error."
        ),
        "tau": args.tau,
        "n_queries": len({r["query_id"] for r in records}),
        "n_pairs": len(records),
        "skipped": skipped,
        "no_guard": {
            "regret_median": median(raw) if raw else float("nan"),
            "regret_p95": _percentile(raw, 0.95),
            "regret_max": max(raw) if raw else float("nan"),
            "optimal_plan_rate": sum(1 for v in raw if v <= 1.0 + 1e-9) / len(raw)
            if raw
            else float("nan"),
        },
        "guarded": {
            "regret_median": median(guarded) if guarded else float("nan"),
            "regret_p95": _percentile(guarded, 0.95),
            "regret_max": max(guarded) if guarded else float("nan"),
            "optimal_plan_rate": sum(1 for v in guarded if v <= 1.0 + 1e-9)
            / len(guarded)
            if guarded
            else float("nan"),
        },
        "plan_changed_rate": sum(1 for r in records if r["plan_changed"]) / len(records)
        if records
        else float("nan"),
        "per_pair": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    logger.info(
        "\n%d queries, %d (query,config) pairs\n"
        "  no guard : regret median %.3f  p95 %.2f  max %.1f  optimal-plan %.1f%%\n"
        "  guarded  : regret median %.3f  p95 %.2f  max %.1f  optimal-plan %.1f%%\n"
        "  plan changed by the guard on %.1f%% of pairs -> %s",
        payload["n_queries"],
        payload["n_pairs"],
        payload["no_guard"]["regret_median"],
        payload["no_guard"]["regret_p95"],
        payload["no_guard"]["regret_max"],
        100 * payload["no_guard"]["optimal_plan_rate"],
        payload["guarded"]["regret_median"],
        payload["guarded"]["regret_p95"],
        payload["guarded"]["regret_max"],
        100 * payload["guarded"]["optimal_plan_rate"],
        100 * payload["plan_changed_rate"],
        args.output,
    )


if __name__ == "__main__":
    main()
