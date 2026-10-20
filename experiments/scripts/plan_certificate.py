r"""Plan certificates from sound cardinality brackets (pre-registered C4).

Pre-registration: ``docs/c4_preregistration.md`` (frozen 2026-09-03). Every
definition below is the one registered there; the numbers this script writes
are read only after the soundness checks pass.

The question is the decidable form of "when does a certificate help at the plan
level". The cost model $C_{out}$ is monotone in the intermediate cardinalities,
so a sound bracket $[L(S), B(S)]$ on every sub-join $S$ gives every left-deep
plan $p$ a sound cost interval $[C(p, L), C(p, B)]$. From those intervals, with
no estimator involved:

* a plan whose lower cost exceeds some plan's upper cost is never optimal
  (**dominance pruning**);
* any surviving plan's regret is at most
  $G(Q) = \max_{p \in S} C(p, B) / \min_{p'} C(p', L)$ (**certified regret**);
* one survivor means the bracket has **certified the plan**.

We measure how often today's producers (LpBound, SafeBound, max-degree; the
paper's sampling lower certificate) achieve each of these on STATS-CEB, using
the LpBound artifact's sub-query truth and bounds. We claim no new estimator and
no new bound.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.plan_certificate
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

from experiments.scripts.bootstrap_ci import cluster_bootstrap, zero_event_upper_bound
from experiments.scripts.make_lower_column_stats import load_tables
from experiments.scripts.plan_regret import (
    QueryTimeoutError,
    _connected,
    _left_deep_orders,
    _order_cost,
    _sub_query,
    time_limit,
)
from src.core.ceg import pessimistic_bound
from src.estimators.lower_certificate import aggregate_lower_certificate
from src.utils.job_eval import build_alias_rows, build_job_query
from src.utils.job_sql_parser import parse_job_sql
from src.utils.join_counting import count_join

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.utils.job_sql_parser import ParsedJobQuery

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

Subset = frozenset[str]
Sizes = dict[Subset, float]

UPPERS = ("maxdeg", "safebound", "lpbound")
LOWERS = ("one", "sample0.10", "sample0.25")
SAMPLE_RATES = {"sample0.10": 0.10, "sample0.25": 0.25}
ESTIMATORS = ("duckdb", "postgres")
# Relative shortfall at or below this is floating-point rounding of an integer
# bound (pre-registration amendment 2); anything larger is a real violation.
ROUNDING_TOLERANCE = 1e-9
USEFUL_G = 2.0
MIN_ALIASES = 3
MAX_ALIASES = 7


class SoundnessCheckError(RuntimeError):
    """A pre-registered invariant failed: a code bug, never a finding."""


def load_artifact_column(path: Path, name: str) -> dict[tuple[int, Subset], float]:
    """Read one artifact sub-query file into ``{(query_id, aliases): value}``.

    Alias strings are normalised to a lower-case frozenset because the artifact
    files do not agree on alias order.
    """
    column = f"{name}_Estimate"
    values: dict[tuple[int, Subset], float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["QueryID"]), frozenset(row["Tables"].lower().split()))
            values[key] = float(row[column])
    return values


def repair_rounding(bound: float, truth: float) -> tuple[float, bool]:
    """Ceil a bound that undershoots an integer truth by rounding only.

    Returns the (possibly repaired) bound and whether the shortfall was a real
    violation, i.e. larger than :data:`ROUNDING_TOLERANCE` relative.
    """
    if bound >= truth:
        return bound, False
    if (truth - bound) / truth <= ROUNDING_TOLERANCE:
        return float(math.ceil(bound)), False
    return bound, True


def connected_subsets(aliases: list[str], parsed: ParsedJobQuery) -> list[Subset]:
    """Every connected sub-join with at least two aliases."""
    return [
        frozenset(subset)
        for size in range(2, len(aliases) + 1)
        for subset in itertools.combinations(aliases, size)
        if _connected(subset, parsed)
    ]


def evaluate_bracket(
    orders: list[tuple[str, ...]],
    truth: Sizes,
    lower: Sizes,
    upper: Sizes,
    estimates: dict[str, Sizes],
) -> dict[str, Any]:
    """Plan intervals, pruning, certified regret and estimator picks for one bracket.

    The caller guarantees ``lower <= truth <= upper`` on every subset (flagged
    queries never reach this function's checks), and the checks below re-derive
    what that soundness implies; a failure is raised, not recorded.
    """
    cost_lo = {o: _order_cost(o, lower) for o in orders}
    cost_hi = {o: _order_cost(o, upper) for o in orders}
    cost_true = {o: _order_cost(o, truth) for o in orders}
    optimal = min(cost_true.values())
    min_hi = min(cost_hi.values())
    min_lo = min(cost_lo.values())
    survivors = [o for o in orders if cost_lo[o] <= min_hi]
    bound_g = max(cost_hi[o] for o in survivors) / min_lo
    bound_g_all = max(cost_hi.values()) / min_lo
    minimax = min(orders, key=lambda o: (cost_hi[o], o))
    minimax_bound = min_hi / min_lo

    def regret(order: tuple[str, ...]) -> float:
        return cost_true[order] / optimal

    # Pre-registered invariants (docs/c4_preregistration.md, "Checks").
    optimal_orders = [o for o in orders if cost_true[o] == optimal]
    if not any(o in survivors for o in optimal_orders):
        raise SoundnessCheckError("a true-optimal plan was pruned")
    if any(regret(o) > bound_g * (1 + 1e-12) for o in survivors):
        raise SoundnessCheckError("a survivor's regret exceeds G(Q)")
    if any(regret(o) > bound_g_all * (1 + 1e-12) for o in orders):
        raise SoundnessCheckError("a plan's regret exceeds G_all(Q)")
    if regret(minimax) > minimax_bound * (1 + 1e-12):
        raise SoundnessCheckError("the minimax plan's regret exceeds its bound")

    per_estimator: dict[str, dict[str, Any]] = {}
    for name, sizes in estimates.items():
        pick_free = min(orders, key=lambda o: (_order_cost(o, sizes), o))
        pick_pruned = min(survivors, key=lambda o: (_order_cost(o, sizes), o))
        if regret(pick_pruned) > bound_g * (1 + 1e-12):
            raise SoundnessCheckError(f"{name}: pruned pick's regret exceeds G(Q)")
        per_estimator[name] = {
            "regret_free": regret(pick_free),
            "regret_pruned": regret(pick_pruned),
            "picked_survivor": pick_free in survivors,
        }

    width = max(math.log(upper[s] / lower[s]) for s in truth)
    return {
        "n_plans": len(orders),
        "n_survivors": len(survivors),
        "pruning_rate": 1.0 - len(survivors) / len(orders),
        "G": bound_g,
        "G_all": bound_g_all,
        "certified": len(survivors) == 1,
        "useful": bound_g <= USEFUL_G,
        "width": width,
        "minimax_regret": regret(minimax),
        "minimax_bound": minimax_bound,
        "estimators": per_estimator,
    }


def sample_lower_sizes(
    subsets: list[Subset],
    alias_rows: dict[str, list[dict[str, str]]],
    parsed: ParsedJobQuery,
    rate: float,
    budget_s: float,
    seed: int,
) -> tuple[Sizes, int]:
    """Sound sampling lower certificate per sub-join; censored ones get 1."""
    sizes: Sizes = {}
    censored = 0
    for subset in subsets:
        joins = frozenset(
            j
            for j in parsed.joins
            if j.left_relation in subset and j.right_relation in subset
        )
        rows = {a: alias_rows[a] for a in subset}
        try:
            with time_limit(budget_s):
                cert = aggregate_lower_certificate(rows, subset, joins, rate, [seed])
            sizes[subset] = max(cert.sound_lower_bound, 1.0)
        except QueryTimeoutError:
            sizes[subset] = 1.0
            censored += 1
    return sizes, censored


def evaluate_query(
    parsed: ParsedJobQuery,
    rows_by_relation: dict[str, list[dict[str, str]]],
    artifact: dict[str, dict[tuple[int, Subset], float]],
    *,
    sample_budget_s: float,
    check_budget_s: float,
    seed: int,
) -> dict[str, Any]:
    """All nine brackets for one query; see the pre-registration for fields."""
    query_id = int(parsed.query_id)
    aliases = sorted(parsed.aliases)
    subsets = connected_subsets(aliases, parsed)
    alias_rows = build_alias_rows(parsed, rows_by_relation)
    full = build_job_query(parsed, rows_by_relation)

    truth: Sizes = {
        s: max(artifact["truecardinality"][(query_id, s)], 1.0) for s in subsets
    }

    # Cross-check the artifact truth with our own exact count where it fits the budget.
    cross_checked = 0
    disagreements: list[dict[str, Any]] = []
    for subset in subsets:
        joins = frozenset(
            j
            for j in parsed.joins
            if j.left_relation in subset and j.right_relation in subset
        )
        try:
            with time_limit(check_budget_s):
                ours = max(
                    count_join({a: alias_rows[a] for a in subset}, subset, joins), 1.0
                )
        except QueryTimeoutError:
            continue
        cross_checked += 1
        if ours != truth[subset]:
            disagreements.append(
                {"aliases": sorted(subset), "ours": ours, "artifact": truth[subset]}
            )

    uppers: dict[str, Sizes] = {"maxdeg": {}, "safebound": {}, "lpbound": {}}
    upper_violations: dict[str, list[list[str]]] = {u: [] for u in UPPERS}
    for subset in subsets:
        sub = _sub_query(full, tuple(sorted(subset)))
        if sub is None:
            msg = f"q{query_id}: sub-join {sorted(subset)} is not a valid query"
            raise RuntimeError(msg)
        raw = {
            "maxdeg": max(pessimistic_bound(sub), 1.0),
            "safebound": max(artifact["safebound"][(query_id, subset)], 1.0),
            "lpbound": max(artifact["lpbound"][(query_id, subset)], 1.0),
        }
        for name, value in raw.items():
            repaired, violated = repair_rounding(value, truth[subset])
            uppers[name][subset] = repaired
            if violated:
                upper_violations[name].append(sorted(subset))

    lowers: dict[str, Sizes] = {"one": dict.fromkeys(subsets, 1.0)}
    censored: dict[str, int] = {}
    for name, rate in SAMPLE_RATES.items():
        lowers[name], censored[name] = sample_lower_sizes(
            subsets, alias_rows, parsed, rate, sample_budget_s, seed
        )
    lower_violations: dict[str, list[list[str]]] = {
        name: [sorted(s) for s in subsets if sizes[s] > truth[s]]
        for name, sizes in lowers.items()
    }

    estimates: dict[str, Sizes] = {
        name: {s: max(artifact[name][(query_id, s)], 1.0) for s in subsets}
        for name in ESTIMATORS
    }
    orders = _left_deep_orders(aliases, parsed)

    brackets: dict[str, dict[str, Any]] = {}
    for upper_name in UPPERS:
        for lower_name in LOWERS:
            key = f"{upper_name}x{lower_name}"
            flagged = bool(upper_violations[upper_name] or lower_violations[lower_name])
            record: dict[str, Any] = {"flagged": flagged}
            if not flagged:
                record.update(
                    evaluate_bracket(
                        orders, truth, lowers[lower_name], uppers[upper_name], estimates
                    )
                )
            brackets[key] = record

    return {
        "query_id": query_id,
        "n_aliases": len(aliases),
        "n_subjoins": len(subsets),
        "n_plans": len(orders),
        "truth_cross_checked": cross_checked,
        "truth_disagreements": disagreements,
        "upper_violations": upper_violations,
        "lower_violations": lower_violations,
        "sample_censored": censored,
        "brackets": brackets,
    }


def _bracket_rows(per_query: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Flatten one bracket's unflagged records, one row per query."""
    rows: list[dict[str, Any]] = []
    for query in per_query:
        record = query["brackets"][key]
        if record["flagged"]:
            continue
        row = {k: v for k, v in record.items() if k != "estimators"}
        row["query_id"] = query["query_id"]
        for name, values in record["estimators"].items():
            for field, value in values.items():
                row[f"{name}_{field}"] = value
        rows.append(row)
    return rows


def _share(field: str) -> Callable[[list[dict[str, Any]]], float]:
    return lambda rows: sum(1.0 for r in rows if r[field]) / len(rows)


def _share_not(field: str) -> Callable[[list[dict[str, Any]]], float]:
    return lambda rows: sum(1.0 for r in rows if not r[field]) / len(rows)


def _mean(field: str) -> Callable[[list[dict[str, Any]]], float]:
    return lambda rows: sum(float(r[field]) for r in rows) / len(rows)


def _median(field: str) -> Callable[[list[dict[str, Any]]], float]:
    return lambda rows: float(median(float(r[field]) for r in rows))


def _p95(field: str) -> Callable[[list[dict[str, Any]]], float]:
    def statistic(rows: list[dict[str, Any]]) -> float:
        ordered = sorted(float(r[field]) for r in rows)
        position = 0.95 * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return statistic


def _paired_share_difference(
    per_query: list[dict[str, Any]], key_a: str, key_b: str, field: str
) -> list[dict[str, Any]]:
    """Per-query rows for share(field | key_a) - share(field | key_b)."""
    rows: list[dict[str, Any]] = []
    for query in per_query:
        a, b = query["brackets"][key_a], query["brackets"][key_b]
        if a["flagged"] or b["flagged"]:
            continue
        rows.append(
            {"query_id": query["query_id"], "diff": float(a[field]) - float(b[field])}
        )
    return rows


def summarise(
    per_query: list[dict[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    """Per-bracket aggregates with query-clustered bootstrap intervals."""
    summary: dict[str, Any] = {}
    for upper_name in UPPERS:
        for lower_name in LOWERS:
            key = f"{upper_name}x{lower_name}"
            rows = _bracket_rows(per_query, key)
            n_flagged = sum(1 for q in per_query if q["brackets"][key]["flagged"])
            entry: dict[str, Any] = {"n_queries": len(rows), "n_flagged": n_flagged}
            if not rows:
                summary[key] = entry
                continue
            stats: dict[str, Callable[[list[dict[str, Any]]], float]] = {
                "certified_share": _share("certified"),
                "useful_share": _share("useful"),
                "mean_pruning_rate": _mean("pruning_rate"),
                "median_G": _median("G"),
                "p95_G": _p95("G"),
                "median_width": _median("width"),
                "median_minimax_regret": _median("minimax_regret"),
                "p95_minimax_regret": _p95("minimax_regret"),
            }
            for name in ESTIMATORS:
                stats[f"{name}_share_pick_pruned_away"] = _share_not(
                    f"{name}_picked_survivor"
                )
                stats[f"{name}_median_regret_free"] = _median(f"{name}_regret_free")
                stats[f"{name}_median_regret_pruned"] = _median(f"{name}_regret_pruned")
                stats[f"{name}_p95_regret_free"] = _p95(f"{name}_regret_free")
                stats[f"{name}_p95_regret_pruned"] = _p95(f"{name}_regret_pruned")
            for name, statistic in stats.items():
                entry[name] = cluster_bootstrap(
                    rows, statistic, replicates=replicates, seed=seed
                )
            for field in ("certified", "useful"):
                if not any(r[field] for r in rows):
                    entry[f"{field}_share"]["zero_event_upper_95"] = (
                        zero_event_upper_bound(len(rows))
                    )
            summary[key] = entry

    diff_rows = _paired_share_difference(
        per_query, "lpboundxsample0.25", "lpboundxone", "certified"
    )
    summary["paired_certified_share_lpbound_sample0.25_minus_one"] = (
        cluster_bootstrap(diff_rows, _mean("diff"), replicates=replicates, seed=seed)
        if diff_rows
        else {}
    )
    return summary


def decide(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered decision rule to the primary bracket."""
    primary = summary.get("lpboundxsample0.25", {})
    paired = summary.get("paired_certified_share_lpbound_sample0.25_minus_one", {})
    if "certified_share" not in primary:
        return {"primary_bracket": "lpboundxsample0.25", "evaluable": False}
    return {
        "primary_bracket": "lpboundxsample0.25",
        "evaluable": True,
        "certificates_non_trivial": primary["certified_share"]["ci_low"] > 0.10,
        "pruning_non_trivial": primary["mean_pruning_rate"]["ci_low"] > 0.50,
        "lower_certificate_matters": bool(paired)
        and (paired["ci_low"] > 0.0 or paired["ci_high"] < 0.0),
    }


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
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=base / "results/evaluation_time/subquery_estimations/stats",
    )
    parser.add_argument("--sample-budget-s", type=float, default=30.0)
    parser.add_argument("--check-budget-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N queries")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/plan_certificate.json"),
    )
    args = parser.parse_args()

    artifact = {
        name: load_artifact_column(
            args.artifact_dir / f"{name}_subquery_estimations.csv", name
        )
        for name in ("truecardinality", "safebound", "lpbound", *ESTIMATORS)
    }
    lines = [
        line.strip() for line in args.queries.read_text().splitlines() if line.strip()
    ]
    if args.limit:
        lines = lines[: args.limit]

    cache: dict[str, list[dict[str, str]]] = {}
    per_query: list[dict[str, Any]] = []
    excluded: dict[str, list[int]] = {
        "unparsed": [],
        "n<=2": [],
        "n>7": [],
        "missing_truth": [],
    }
    for index, sql in enumerate(lines, start=1):
        parsed = parse_job_sql(sql, str(index))
        if not parsed.fully_parsed:
            excluded["unparsed"].append(index)
            continue
        aliases = sorted(parsed.aliases)
        if len(aliases) < MIN_ALIASES:
            excluded["n<=2"].append(index)
            continue
        if len(aliases) > MAX_ALIASES:
            excluded["n>7"].append(index)
            continue
        if any(
            (index, s) not in artifact["truecardinality"]
            for s in connected_subsets(aliases, parsed)
        ):
            excluded["missing_truth"].append(index)
            continue
        rows = load_tables(args.data_dir, set(parsed.base_relations), cache)
        record = evaluate_query(
            parsed,
            rows,
            artifact,
            sample_budget_s=args.sample_budget_s,
            check_budget_s=args.check_budget_s,
            seed=args.seed,
        )
        per_query.append(record)
        primary = record["brackets"]["lpboundxsample0.25"]
        logger.info(
            "q%-3d n=%d plans=%-4d lpbound x s0.25: %s",
            index,
            record["n_aliases"],
            record["n_plans"],
            "flagged"
            if primary["flagged"]
            else (
                f"survivors {primary['n_survivors']:4d}  G {primary['G']:9.2f}  "
                f"width {primary['width']:.2f}"
                + ("  CERTIFIED" if primary["certified"] else "")
            ),
        )

    summary = summarise(per_query, args.replicates, args.seed)
    payload = {
        "experiment": "plan_certificate",
        "preregistration": "docs/c4_preregistration.md (frozen 2026-09-03)",
        "note": (
            "Sound plan-cost intervals from sound cardinality brackets under the "
            "C_out cost model. Dominance pruning, certified regret bound G(Q), "
            "single-survivor certification, and the regret of real estimators "
            "against G(Q). Cost-model-level statements only; no timing."
        ),
        "settings": {
            "uppers": list(UPPERS),
            "lowers": list(LOWERS),
            "estimators": list(ESTIMATORS),
            "useful_G": USEFUL_G,
            "rounding_tolerance": ROUNDING_TOLERANCE,
            "sample_budget_s": args.sample_budget_s,
            "check_budget_s": args.check_budget_s,
            "seed": args.seed,
            "replicates": args.replicates,
        },
        "n_queries_total": len(lines),
        "n_queries_eligible": len(per_query),
        "excluded": excluded,
        "decision": decide(summary),
        "summary": summary,
        "per_query": per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("decision: %s", json.dumps(payload["decision"]))
    logger.info("-> %s", args.output)


if __name__ == "__main__":
    main()
