"""Plan-level guard ablation on STATS-CEB: gate the sub-join estimates, re-plan.

The cardinality-level ablation (``guard_policy_ablation``) shows that a
bidirectional guard --- LpBound above, a sampling certificate below --- cuts
the p95 q-error of external proposals by three orders of magnitude. The
canonical plan-level run (``plan_regret``) gates only from above, only on the
repository's own nine CEG configurations, and finds p95 plan regret unchanged
(1.58 either way). Two questions were left open by the second review panel:

* **B-W3** -- does adding the *lower* gate at plan level change the plan?
* **B-W4** -- does the guard change the plan when it is applied to the
  sub-join estimates a real optimizer would use (PostgreSQL, DuckDB,
  FactorJoin, BayesCard, from the LpBound artifact) rather than to CEG?

This script answers both on the same cost model as ``plan_regret`` and
``plan_certificate``: every connected sub-join of every eligible STATS-CEB
query (3--7 aliases) gets a true size (artifact, cross-checked in
``plan_certificate``), three upper certificates (max-degree CEG_M, LpBound,
SafeBound), one sampling certificate (sound ``L = S`` and unbiased
``L_hat = S / r`` at r = 0.10, one draw, seed 0), the nine CEG estimates and
four external estimates. A *proposal* (one estimator) is turned into a plan by
minimising left-deep C_out over its guarded sub-join sizes; regret is the true
cost of that plan over the true optimum. Policies mirror the cardinality-level
ones exactly: ``upper`` replaces ``e`` by ``U`` when ``e > tau * U``;
``lower`` replaces ``e`` by ``L`` when ``e < L / tau``; ``bi`` applies both.

The per-sub-join sizes are persisted (``subjoin_sizes.json``) so the policy
evaluation is a cheap offline pass and other guard variants can be replayed
without re-sampling. This experiment was **not** pre-registered; it is a
follow-up to the panel's objection and is reported as such.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.plan_guard_bidirectional
    PYTHONHASHSEED=0 python -m experiments.scripts.plan_guard_bidirectional \\
        --evaluate-only   # replay policies on the persisted sizes
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from experiments.scripts.bootstrap_ci import (
    cluster_bootstrap,
    percentile,
    zero_event_upper_bound,
)
from experiments.scripts.make_lower_column_stats import load_tables
from experiments.scripts.plan_certificate import (
    MAX_ALIASES,
    MIN_ALIASES,
    connected_subsets,
    load_artifact_column,
    repair_rounding,
)
from experiments.scripts.plan_regret import (
    QueryTimeoutError,
    _left_deep_orders,
    _order_cost,
    _sub_query,
    time_limit,
)
from src.core.ceg import pessimistic_bound
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.estimators.lower_certificate import aggregate_lower_certificate
from src.utils.job_eval import build_alias_rows, build_job_query
from src.utils.job_sql_parser import parse_job_sql
from src.utils.types import ALL_CEG_CONFIGS

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.utils.job_sql_parser import ParsedJobQuery

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

Subset = frozenset[str]
Sizes = dict[Subset, float]
Rows = list[dict[str, Any]]

UPPERS = ("maxdeg", "lpbound", "safebound")
EXTERNAL = ("duckdb", "postgres", "factorjoin", "bayescard")
# FactorJoin and BayesCard cover only 1,596 of the artifact's 2,577 sub-joins;
# they are evaluated on the queries where every sub-join has an estimate.
EXTERNAL_REQUIRED = ("duckdb", "postgres")
CEG_NAMES = tuple(c.name for c in ALL_CEG_CONFIGS)
SAMPLE_RATE = 0.10
# (label, upper certificate or None, lower floor or None).
POLICIES: tuple[tuple[str, str | None, str | None], ...] = (
    ("none", None, None),
    ("upper_maxdeg", "maxdeg", None),
    ("upper_lpbound", "lpbound", None),
    ("lower_sound", None, "sound"),
    ("lower_point", None, "point"),
    ("bi_maxdeg_sound", "maxdeg", "sound"),
    ("bi_lpbound_sound", "lpbound", "sound"),
    ("bi_maxdeg_point", "maxdeg", "point"),
    ("bi_lpbound_point", "lpbound", "point"),
)


# --------------------------------------------------------------------------- #
# Phase A: per-sub-join sizes.
# --------------------------------------------------------------------------- #
def collect_query(
    parsed: ParsedJobQuery,
    rows_by_relation: dict[str, list[dict[str, str]]],
    artifact: dict[str, dict[tuple[int, Subset], float]],
    *,
    rate: float,
    budget_s: float,
    seed: int,
) -> dict[str, Any]:
    """Truth, certificates and every estimate for each connected sub-join."""
    query_id = int(parsed.query_id)
    aliases = sorted(parsed.aliases)
    subsets = connected_subsets(aliases, parsed)
    alias_rows = build_alias_rows(parsed, rows_by_relation)
    full = build_job_query(parsed, rows_by_relation)
    estimator = BaselineCEGEstimator()

    subjoins: list[dict[str, Any]] = []
    for subset in subsets:
        truth = max(artifact["truecardinality"][(query_id, subset)], 1.0)
        sub = _sub_query(full, tuple(sorted(subset)))
        if sub is None:
            msg = f"q{query_id}: sub-join {sorted(subset)} is not a valid query"
            raise RuntimeError(msg)
        uppers = {
            "maxdeg": max(pessimistic_bound(sub), 1.0),
            "lpbound": max(artifact["lpbound"][(query_id, subset)], 1.0),
            "safebound": max(artifact["safebound"][(query_id, subset)], 1.0),
        }
        upper_violated = {}
        for name, value in uppers.items():
            uppers[name], upper_violated[name] = repair_rounding(value, truth)

        joins = frozenset(
            j
            for j in parsed.joins
            if j.left_relation in subset and j.right_relation in subset
        )
        started = perf_counter()
        try:
            with time_limit(budget_s):
                cert = aggregate_lower_certificate(
                    {a: alias_rows[a] for a in subset}, subset, joins, rate, [seed]
                )
            sound, point, censored = cert.sound_lower_bound, cert.point_estimate, False
        except QueryTimeoutError:
            sound, point, censored = 0.0, 0.0, True
        sample_ms = (perf_counter() - started) * 1000.0

        per_config = estimator.estimate_all_configs(sub)
        subjoins.append(
            {
                "aliases": sorted(subset),
                "truth": truth,
                "upper": uppers,
                "upper_violated": upper_violated,
                "sample": {
                    "rate": rate,
                    "sound": max(sound, 1.0),
                    "point": max(point, 1.0),
                    "censored": censored,
                    "ms": sample_ms,
                    "sound_valid": (not censored) and sound <= truth,
                },
                "ceg": {
                    name: max(per_config[name].estimate, 1.0) for name in CEG_NAMES
                },
                "external": {
                    name: (
                        max(artifact[name][(query_id, subset)], 1.0)
                        if (query_id, subset) in artifact[name]
                        else None
                    )
                    for name in EXTERNAL
                },
            }
        )
    return {
        "query_id": query_id,
        "n_aliases": len(aliases),
        "aliases": aliases,
        "n_subjoins": len(subsets),
        "n_plans": len(_left_deep_orders(aliases, parsed)),
        "subjoins": subjoins,
    }


def _eligible(
    lines: list[str], artifact: dict[str, dict[tuple[int, Subset], float]]
) -> tuple[list[ParsedJobQuery], dict[str, list[int]]]:
    """Same eligibility rule as ``plan_certificate`` plus complete estimates."""
    excluded: dict[str, list[int]] = {
        "unparsed": [],
        "n<=2": [],
        "n>7": [],
        "missing_artifact": [],
    }
    parsed_queries: list[ParsedJobQuery] = []
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
        subsets = connected_subsets(aliases, parsed)
        if any(
            (index, s) not in artifact[name]
            for name in ("truecardinality", "lpbound", "safebound", *EXTERNAL_REQUIRED)
            for s in subsets
        ):
            excluded["missing_artifact"].append(index)
            continue
        parsed_queries.append(parsed)
    return parsed_queries, excluded


def collect_sizes(args: argparse.Namespace) -> dict[str, Any]:
    """Phase A over all eligible queries, resuming from the JSONL if present."""
    artifact = {
        name: load_artifact_column(
            args.artifact_dir / f"{name}_subquery_estimations.csv", name
        )
        for name in ("truecardinality", "lpbound", "safebound", *EXTERNAL)
    }
    lines = [
        line.strip() for line in args.queries.read_text().splitlines() if line.strip()
    ]
    if args.limit:
        lines = lines[: args.limit]
    queries, excluded = _eligible(lines, artifact)

    done: dict[int, dict[str, Any]] = {}
    if args.progress.exists():
        for line in args.progress.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done[int(record["query_id"])] = record
        logger.info("resuming: %d queries already collected", len(done))
    args.progress.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, list[dict[str, str]]] = {}
    per_query: list[dict[str, Any]] = []
    for parsed in queries:
        query_id = int(parsed.query_id)
        if query_id in done:
            per_query.append(done[query_id])
            continue
        rows = load_tables(args.data_dir, set(parsed.base_relations), cache)
        started = perf_counter()
        record = collect_query(
            parsed,
            rows,
            artifact,
            rate=args.rate,
            budget_s=args.sample_budget_s,
            seed=args.seed,
        )
        per_query.append(record)
        with args.progress.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        censored = sum(1 for s in record["subjoins"] if s["sample"]["censored"])
        logger.info(
            "q%-3d n=%d subjoins=%-3d plans=%-4d censored=%d  %.0fs",
            query_id,
            record["n_aliases"],
            record["n_subjoins"],
            record["n_plans"],
            censored,
            perf_counter() - started,
        )

    return {
        "experiment": "plan_guard_bidirectional_sizes",
        "date": "2026-09-09",
        "script": "experiments/scripts/plan_guard_bidirectional.py",
        "note": (
            "Per-sub-join truth (LpBound artifact; cross-checked against our own "
            "exact count on 1,505/1,560 sub-joins in plan_certificate.json with "
            "zero disagreements), upper certificates (repair_rounding applied), "
            "one-draw sampling certificate and every proposal's estimate. Not "
            "pre-registered."
        ),
        "sample": {"rate": args.rate, "seed": args.seed, "draws": 1},
        "sample_budget_s": args.sample_budget_s,
        "n_queries": len(per_query),
        "excluded": excluded,
        "per_query": per_query,
    }


# --------------------------------------------------------------------------- #
# Phase B: policies at plan level.
# --------------------------------------------------------------------------- #
def _gate(
    raw: float, upper: float | None, lower: float | None, tau: float
) -> tuple[float, bool, bool]:
    """Apply the cardinality-level gates; returns (final, upper_fired, lower_fired)."""
    final = raw
    up = low = False
    if upper is not None and final > tau * upper:
        final, up = upper, True
    if lower is not None and final < lower / tau:
        final, low = lower, True
    return final, up, low


def evaluate_query(record: dict[str, Any], parsed: ParsedJobQuery, tau: float) -> Rows:
    """One row per (proposal, policy) for this query."""
    aliases = record["aliases"]
    orders = _left_deep_orders(aliases, parsed)
    subs = record["subjoins"]
    keys = [frozenset(s["aliases"]) for s in subs]
    truth: Sizes = {k: s["truth"] for k, s in zip(keys, subs, strict=True)}
    cost_true = {o: _order_cost(o, truth) for o in orders}
    optimal = min(cost_true.values())

    proposals: dict[str, Sizes] = {}
    for name in CEG_NAMES:
        proposals[name] = {k: s["ceg"][name] for k, s in zip(keys, subs, strict=True)}
    for name in EXTERNAL:
        if all(s["external"][name] is not None for s in subs):
            proposals[name] = {
                k: s["external"][name] for k, s in zip(keys, subs, strict=True)
            }

    def plan_under(
        raw: Sizes, upper_name: str | None, lower_name: str | None
    ) -> tuple[tuple[str, ...], int, int]:
        sizes: Sizes = {}
        fired_up = fired_low = 0
        for k, s in zip(keys, subs, strict=True):
            upper = s["upper"][upper_name] if upper_name else None
            lower = None
            if lower_name and not s["sample"]["censored"]:
                lower = s["sample"][lower_name]
            sizes[k], up, low = _gate(raw[k], upper, lower, tau)
            fired_up += up
            fired_low += low
        return (
            min(orders, key=lambda o: (_order_cost(o, sizes), o)),
            fired_up,
            fired_low,
        )

    rows: Rows = []
    for proposal, raw in proposals.items():
        plan_none, _, _ = plan_under(raw, None, None)
        regret_none = cost_true[plan_none] / optimal
        for label, upper_name, lower_name in POLICIES:
            plan, fired_up, fired_low = plan_under(raw, upper_name, lower_name)
            regret = cost_true[plan] / optimal
            rows.append(
                {
                    "query_id": record["query_id"],
                    "proposal": proposal,
                    "group": "ceg" if proposal in CEG_NAMES else proposal,
                    "policy": label,
                    "regret": regret,
                    "regret_none": regret_none,
                    "delta": regret - regret_none,
                    "optimal": regret <= 1.0 + 1e-12,
                    "plan_changed": plan != plan_none,
                    "upper_fired": fired_up,
                    "lower_fired": fired_low,
                    "n_subjoins": len(keys),
                }
            )
    return rows


def _load_parsed(queries: Path) -> dict[int, ParsedJobQuery]:
    """Parsed queries by 1-based index, for the connectivity checks of phase B."""
    lines = [line.strip() for line in queries.read_text().splitlines() if line.strip()]
    parsed_by_id: dict[int, ParsedJobQuery] = {}
    for index, sql in enumerate(lines, start=1):
        parsed = parse_job_sql(sql, str(index))
        if parsed.fully_parsed:
            parsed_by_id[index] = parsed
    return parsed_by_id


def _median_of(field: str) -> Callable[[Rows], float]:
    def statistic(rows: Rows) -> float:
        return percentile([float(r[field]) for r in rows], 0.5)

    return statistic


def _p95_of(field: str) -> Callable[[Rows], float]:
    def statistic(rows: Rows) -> float:
        return percentile([float(r[field]) for r in rows], 0.95)

    return statistic


def _share_of(field: str) -> Callable[[Rows], float]:
    def statistic(rows: Rows) -> float:
        return sum(1 for r in rows if r[field]) / len(rows) if rows else math.nan

    return statistic


def _p95_delta(rows: Rows) -> float:
    return _p95_of("regret")(rows) - _p95_of("regret_none")(rows)


def _median_delta(rows: Rows) -> float:
    return _median_of("regret")(rows) - _median_of("regret_none")(rows)


def summarise_policies(rows: Rows, replicates: int, seed: int) -> dict[str, Any]:
    """Per (group, policy) regret statistics with query-clustered intervals."""
    groups = ["ceg", *EXTERNAL, "external_pooled"]
    out: dict[str, Any] = {}
    for group in groups:
        members = [
            r
            for r in rows
            if (
                r["group"] in EXTERNAL
                if group == "external_pooled"
                else r["group"] == group
            )
        ]
        out[group] = {}
        for label, _, _ in POLICIES:
            sub = [r for r in members if r["policy"] == label]
            n = len(sub)
            worsened = sum(1 for r in sub if r["delta"] > 1e-12)
            improved = sum(1 for r in sub if r["delta"] < -1e-12)
            entry: dict[str, Any] = {
                "n_pairs": n,
                "n_queries": len({r["query_id"] for r in sub}),
                "regret_median": cluster_bootstrap(
                    sub, _median_of("regret"), replicates=replicates, seed=seed
                ),
                "regret_p95": cluster_bootstrap(
                    sub, _p95_of("regret"), replicates=replicates, seed=seed
                ),
                "regret_max": max(r["regret"] for r in sub),
                "optimal_rate": cluster_bootstrap(
                    sub, _share_of("optimal"), replicates=replicates, seed=seed
                ),
                "plan_changed_rate": sum(1 for r in sub if r["plan_changed"]) / n,
                "improved": improved,
                "worsened": worsened,
                "worsened_upper_95": (
                    zero_event_upper_bound(n) if worsened == 0 else None
                ),
                "subjoin_upper_fire_rate": sum(r["upper_fired"] for r in sub)
                / sum(r["n_subjoins"] for r in sub),
                "subjoin_lower_fire_rate": sum(r["lower_fired"] for r in sub)
                / sum(r["n_subjoins"] for r in sub),
            }
            if label != "none":
                entry["p95_delta_vs_none"] = cluster_bootstrap(
                    sub, _p95_delta, replicates=replicates, seed=seed
                )
                entry["median_delta_vs_none"] = cluster_bootstrap(
                    sub, _median_delta, replicates=replicates, seed=seed
                )
            out[group][label] = entry
    return out


def evaluate(sizes: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Phase B over the persisted sizes."""
    parsed_by_id = _load_parsed(args.queries)
    rows: Rows = []
    for record in sizes["per_query"]:
        rows.extend(evaluate_query(record, parsed_by_id[record["query_id"]], args.tau))
    summary = summarise_policies(rows, args.replicates, args.bootstrap_seed)

    subjoins = [s for q in sizes["per_query"] for s in q["subjoins"]]
    censored = sum(1 for s in subjoins if s["sample"]["censored"])
    completed = [s for s in subjoins if not s["sample"]["censored"]]
    sound_ratio = percentile(
        [s["sample"]["sound"] / s["truth"] for s in completed], 0.5
    )
    upper_viol = {u: sum(1 for s in subjoins if s["upper_violated"][u]) for u in UPPERS}
    return {
        "experiment": "plan_guard_bidirectional",
        "date": "2026-09-09",
        "script": "experiments/scripts/plan_guard_bidirectional.py",
        "preregistered": False,
        "purpose": (
            "Panel-2 items B-W3 (lower gate at plan level) and B-W4 (guard the "
            "sub-join estimates an optimizer actually uses). Left-deep C_out cost "
            "model as in plan_regret / plan_certificate; tau as in the "
            "cardinality-level ablation."
        ),
        "tau": args.tau,
        "sample": sizes["sample"],
        "n_queries": sizes["n_queries"],
        "n_subjoins": len(subjoins),
        "sample_censored_subjoins": censored,
        "sound_over_truth_median_completed": sound_ratio,
        "sound_valid_all_completed": all(s["sample"]["sound_valid"] for s in completed),
        "upper_violations": upper_viol,
        "excluded": sizes["excluded"],
        "policies": [
            {"label": label, "upper": up, "lower": low} for label, up, low in POLICIES
        ],
        "proposals": {"ceg": list(CEG_NAMES), "external": list(EXTERNAL)},
        "summary": summary,
        "per_pair": rows,
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
    parser.add_argument("--rate", type=float, default=SAMPLE_RATE)
    parser.add_argument("--sample-budget-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N queries")
    parser.add_argument(
        "--progress",
        type=Path,
        default=Path("results/plan_guard/subjoin_sizes.jsonl"),
        help="per-query resume log for phase A",
    )
    parser.add_argument(
        "--sizes",
        type=Path,
        default=Path("results/plan_guard/subjoin_sizes.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_guard/plan_guard_bidirectional.json"),
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="skip phase A and replay the policies on --sizes",
    )
    args = parser.parse_args()

    if args.evaluate_only:
        sizes = json.loads(args.sizes.read_text())
    else:
        sizes = collect_sizes(args)
        args.sizes.parent.mkdir(parents=True, exist_ok=True)
        args.sizes.write_text(json.dumps(sizes, indent=1) + "\n")
        logger.info("sizes -> %s", args.sizes)

    payload = evaluate(sizes, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")

    for group, policies in payload["summary"].items():
        logger.info("--- %s", group)
        for label, entry in policies.items():
            logger.info(
                "%-18s n=%-4d median %.3f [%.3f, %.3f]  p95 %.3f [%.3f, %.3f]  "
                "max %.2f  optimal %.1f%%  changed %.1f%%  +%d/-%d  fire up %.1f%% "
                "low %.1f%%",
                label,
                entry["n_pairs"],
                entry["regret_median"]["point"],
                entry["regret_median"]["ci_low"],
                entry["regret_median"]["ci_high"],
                entry["regret_p95"]["point"],
                entry["regret_p95"]["ci_low"],
                entry["regret_p95"]["ci_high"],
                entry["regret_max"],
                100.0 * entry["optimal_rate"]["point"],
                100.0 * entry["plan_changed_rate"],
                entry["improved"],
                entry["worsened"],
                100.0 * entry["subjoin_upper_fire_rate"],
                100.0 * entry["subjoin_lower_fire_rate"],
            )
    logger.info("-> %s", args.output)


if __name__ == "__main__":
    main()
