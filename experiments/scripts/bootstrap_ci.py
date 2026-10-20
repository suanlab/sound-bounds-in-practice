"""Query-clustered bootstrap confidence intervals for the paper's headline numbers.

Every headline statistic in this paper is computed over rows that are *not*
independent. A single query contributes one row per proposal (six on average in
the STATS-CEB utility table) and one row per CEG configuration in the plan
experiments, so rows within a query share the same true cardinality, the same
bound, and the same data. Treating them as independent observations would make
any interval far too narrow.

This script therefore resamples **queries**, not rows: each replicate draws the
observed number of distinct query ids with replacement and takes every row
belonging to a drawn query. That is the standard cluster bootstrap, and it is
the right unit here because the query is what was sampled from the benchmark.

Two of the statistics deserve their interval more than the others. The
two-engine aggregate ratio is, by the paper's own account, decided by a single
query on each engine; the interval is what shows that. And the real-JOB cost
ratio is a median over the 53 templates that finished, which is a
feasibility-selected subset --- the interval quantifies sampling error only, not
that selection (see ``censoring_bounds.py`` for the selection effect).

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.bootstrap_ci
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

Rows = list[dict[str, Any]]


# ``run_job_real`` default ``--lower-seeds 1 2 3``; ``job_full.json`` does not
# record it, so it is pinned here and in docs/REPRODUCE.md.
JOB_LOWER_DRAWS = 3


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of a sample."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def cluster_bootstrap(
    rows: Rows,
    statistic: Callable[[Rows], float],
    *,
    cluster_key: str = "query_id",
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile CI for ``statistic`` under resampling of whole clusters."""
    by_cluster: dict[Any, Rows] = defaultdict(list)
    for row in rows:
        by_cluster[row[cluster_key]].append(row)
    clusters = list(by_cluster.values())
    rng = random.Random(seed)

    draws: list[float] = []
    for _ in range(replicates):
        sample: Rows = []
        for _ in range(len(clusters)):
            sample.extend(clusters[rng.randrange(len(clusters))])
        value = statistic(sample)
        if value == value:  # drop NaN replicates
            draws.append(value)

    dropped = replicates - len(draws)
    if dropped > 0.01 * replicates:
        # The interval then describes only the resamples where the statistic is
        # defined, a different population; callers should remove the NaNs.
        logger.warning(
            "cluster_bootstrap: %d of %d replicates were NaN and dropped",
            dropped,
            replicates,
        )
    return {
        "point": statistic(rows),
        "ci_low": percentile(draws, 0.025),
        "ci_high": percentile(draws, 0.975),
        "n_rows": len(rows),
        "n_clusters": len(clusters),
        "replicates": len(draws),
        "replicates_dropped": dropped,
    }


def _ratio(numerator: str, denominator: str) -> Callable[[Rows], float]:
    def statistic(rows: Rows) -> float:
        bottom = sum(r[denominator] for r in rows)
        return sum(r[numerator] for r in rows) / bottom if bottom else float("nan")

    return statistic


def _median_of(field: str) -> Callable[[Rows], float]:
    return lambda rows: median(r[field] for r in rows) if rows else float("nan")


def _p95_of(field: str) -> Callable[[Rows], float]:
    return lambda rows: percentile([r[field] for r in rows], 0.95)


def _share_slower(rows: Rows) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r["speedup_guarded_over_no_guard"] < 1.0) / len(rows)


def _tail_over_share(rows: Rows) -> float:
    """Over-estimate share of the worst 5% of pairs by unguarded q-error."""
    if not rows:
        return float("nan")
    ordered = sorted(rows, key=lambda r: -r["qerror"])
    tail = ordered[: max(1, round(0.05 * len(ordered)))]
    return sum(1 for r in tail if r["direction"] > 0) / len(tail)


def load_tail_rows(path: Path, keep: set[tuple[str, str]] | None = None) -> Rows:
    """Unguarded q-error and error direction per (query, proposal) pair.

    The table carries one row per (query, proposal, certificate); the proposal's
    own error does not depend on which certificate is paired with it, so rows
    are de-duplicated to (query, proposal). Keeping the duplicates would weight
    a proposal by how many certificates it was paired with.

    Passing ``keep`` restricts the rows to a given set of (query, proposal)
    pairs, so the tail is computed over exactly the pairs the paper reports
    (those whose lower certificate completed within budget).
    """
    seen: set[tuple[str, str]] = set()
    rows: Rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["query_id"], row["proposal"])
            if key in seen or (keep is not None and key not in keep):
                continue
            try:
                qerror = float(row["proposal_qerror"])
                direction = float(row["direction"])
            except (KeyError, ValueError):
                continue
            if qerror != qerror:
                continue
            seen.add(key)
            rows.append(
                {
                    "query_id": row["query_id"],
                    "qerror": qerror,
                    "direction": direction,
                }
            )
    return rows


def zero_event_upper_bound(trials: int, level: float = 0.95) -> float:
    """One-sided upper bound on a rate after observing zero events.

    A bootstrap of a zero count resamples zeros and returns zero, which is not
    an interval --- it is the absence of one. The exact (Clopper--Pearson) bound
    is ``1 - (1-level)**(1/n)``, the familiar rule of three for 95%.
    """
    return 1.0 - (1.0 - level) ** (1.0 / trials) if trials > 0 else float("nan")


def load_utility_rows(path: Path) -> Rows:
    """One row per (query, proposal, certificate) with direction and firing."""
    rows: Rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                direction = float(row["direction"])
            except (KeyError, ValueError):
                continue
            rows.append(
                {
                    "query_id": row["query_id"],
                    "certificate": row["certificate"],
                    "direction": direction,
                    "fired": row["guard_fired"] == "True",
                }
            )
    return rows


def _fire_rate(certificate: str, direction: float) -> Callable[[Rows], float]:
    """Firing rate of one certificate among errors in one direction."""

    def statistic(rows: Rows) -> float:
        subset = [
            r
            for r in rows
            if r["certificate"] == certificate and r["direction"] == direction
        ]
        if not subset:
            return float("nan")
        return sum(1 for r in subset if r["fired"]) / len(subset)

    return statistic


def _direction_share(certificate: str) -> Callable[[Rows], float]:
    """Share of a certificate's rows that are under-estimates."""

    def statistic(rows: Rows) -> float:
        subset = [r for r in rows if r["certificate"] == certificate]
        if not subset:
            return float("nan")
        return sum(1 for r in subset if r["direction"] < 0) / len(subset)

    return statistic


def _max_of(field: str) -> Callable[[Rows], float]:
    """Worst case. See the warning on :func:`_max_ratio` before using the CI."""
    return lambda rows: max(r[field] for r in rows) if rows else float("nan")


def _tight_fire(certificate: str) -> Callable[[Rows], float]:
    """Firing rate of one certificate in the tightness ablation."""

    def statistic(rows: Rows) -> float:
        subset = [r for r in rows if r["certificate"] == certificate]
        if not subset:
            return float("nan")
        return sum(1 for r in subset if r["fired"]) / len(subset)

    return statistic


def _tight_fire_gain(tighter: str, baseline: str) -> Callable[[Rows], float]:
    """Percentage-point gain in firing rate, paired within each replicate.

    The pre-registered claim is a *difference* between two certificates
    evaluated on identical (query, proposal) pairs, so it must be resampled as a
    difference rather than as two independent rates.
    """

    def statistic(rows: Rows) -> float:
        return 100.0 * (_tight_fire(tighter)(rows) - _tight_fire(baseline)(rows))

    return statistic


def _share_above_one(field: str) -> Callable[[Rows], float]:
    return lambda rows: sum(1.0 for r in rows if float(r[field]) > 1.0) / len(rows)


def _p95_difference(left: str, right: str) -> Callable[[Rows], float]:
    def statistic(rows: Rows) -> float:
        return _p95_of(left)(rows) - _p95_of(right)(rows)

    return statistic


def _max_ratio(numerator: str, denominator: str) -> Callable[[Rows], float]:
    """Ratio of two worst cases computed on the *same* resampled pairs.

    Comparing two independently bootstrapped maxima is needlessly weak: the two
    policies are evaluated on identical pairs, so the comparison should be
    paired. This recomputes both maxima inside each replicate and takes their
    ratio, which is what the paper's ``2754 \to 150`` claim actually asserts.
    """

    def statistic(rows: Rows) -> float:
        if not rows:
            return float("nan")
        bottom = max(r[denominator] for r in rows)
        return max(r[numerator] for r in rows) / bottom if bottom else float("nan")

    return statistic


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/bootstrap/bootstrap_ci.json")
    )
    args = parser.parse_args()
    results = Path("results")

    policy = json.loads((results / "utility_table/guard_policy.json").read_text())
    duckdb = json.loads((results / "plan_regret/plan_runtime.json").read_text())
    sqlite = json.loads((results / "plan_regret/plan_runtime_sqlite.json").read_text())
    job = json.loads((results / "job_full/job_full.json").read_text())
    reported_pairs = {
        (str(r["query_id"]), str(r["proposal"])) for r in policy["per_pair"]
    }
    tail_rows = load_tail_rows(
        results / "utility_table/real_stats_bidirectional.csv", reported_pairs
    )

    jobs: list[tuple[str, Rows, Callable[[Rows], float]]] = []
    for name in (
        "no_guard",
        "upper_only",
        "lower_only",
        "bidirectional",
        "lower_only_sound",
        "bidirectional_sound",
    ):
        jobs.append(
            (
                f"guard policy / {name} / median q-error",
                policy["per_pair"],
                _median_of(f"qerror_{name}"),
            )
        )
        jobs.append(
            (
                f"guard policy / {name} / p95 q-error",
                policy["per_pair"],
                _p95_of(f"qerror_{name}"),
            )
        )
    for engine, payload in (("duckdb", duckdb), ("sqlite", sqlite)):
        jobs.append(
            (
                f"plan runtime / {engine} / aggregate guarded-over-unguarded",
                payload["per_pair"],
                _ratio("ms_guarded", "ms_no_guard"),
            )
        )
        jobs.append(
            (
                f"plan runtime / {engine} / median speedup",
                payload["per_pair"],
                _median_of("speedup_guarded_over_no_guard"),
            )
        )
        jobs.append(
            (
                f"plan runtime / {engine} / share of pairs slower",
                payload["per_pair"],
                _share_slower,
            )
        )
    jobs.append(
        (
            "real JOB / median lower-certificate cost ratio (3 draws, as run)",
            job["per_query"],
            _median_of("cost_ratio"),
        )
    )
    # ``run_job_real`` defaults to ``--lower-seeds 1 2 3`` and times all three
    # sub-sampled joins; STATS-CEB (``make_lower_column_stats``) uses one draw.
    # The certificate cost is linear in draws (``results/real_job/
    # c3_cost_scale.json``), so the per-draw ratio is the comparable quantity.
    per_draw_rows = [
        {**r, "cost_ratio_per_draw": r["cost_ratio"] / JOB_LOWER_DRAWS}
        for r in job["per_query"]
    ]
    jobs.append(
        (
            "real JOB / median lower-certificate cost ratio (per draw, derived)",
            per_draw_rows,
            _median_of("cost_ratio_per_draw"),
        )
    )
    jobs.append(
        (
            "real JOB / share of templates with per-draw cost ratio above 1",
            per_draw_rows,
            _share_above_one("cost_ratio_per_draw"),
        )
    )
    # The like-for-like quantity, measured rather than derived: the one-draw
    # certificate re-timed on the same 53 templates (``verify_draw_cost``).
    one_draw = json.loads(
        (results / "job_full/job_full_1draw_verification.json").read_text()
    )
    jobs.append(
        (
            "real JOB / median lower-certificate cost ratio (1 draw, measured)",
            one_draw["per_template"],
            _median_of("ratio_1draw"),
        )
    )
    jobs.append(
        (
            "real JOB / share of templates with 1-draw cost ratio above 1",
            one_draw["per_template"],
            _share_above_one("ratio_1draw"),
        )
    )
    jobs.append(
        (
            "real JOB / three-draw over one-draw certificate time (linearity)",
            [r for r in one_draw["per_template"] if r["c3_ms"] is not None],
            _median_of("c3_over_c1"),
        )
    )
    jobs.append(
        (
            "tail composition / over-estimate share of worst 5%",
            tail_rows,
            _tail_over_share,
        )
    )

    utility_rows = load_utility_rows(
        results / "utility_table/real_stats_bidirectional.csv"
    )
    for certificate, direction, label in (
        ("lpbound", 1.0, "LpBound fires on over-estimates"),
        ("safebound", 1.0, "SafeBound fires on over-estimates"),
        ("sample_lb", -1.0, "sample-LB fires on under-estimates"),
    ):
        jobs.append(
            (f"direction / {label}", utility_rows, _fire_rate(certificate, direction))
        )
    jobs.append(
        (
            "direction / under-estimate share of errors",
            utility_rows,
            _direction_share("lpbound"),
        )
    )

    tight = json.loads((results / "utility_table/tightness.json").read_text())
    for certificate in ("max_degree", "safebound", "lpbound"):
        jobs.append(
            (
                f"tightness / {certificate} / fire rate",
                tight["per_pair"],
                _tight_fire(certificate),
            )
        )
    jobs.append(
        (
            "tightness / SafeBound minus max-degree (pp, paired)",
            tight["per_pair"],
            _tight_fire_gain("safebound", "max_degree"),
        )
    )

    cost_rows = [
        {"query_id": r["query_id"], "cost_ratio": float(r["cost_ratio"])}
        for r in csv.DictReader(
            (results / "utility_table/real_stats_bidirectional.csv").open(newline="")
        )
        if r["certificate"] == "sample_lb" and r["cost_ratio"] not in ("", "nan")
    ]
    jobs.append(
        (
            "cost / STATS-CEB lower-certificate cost ratio",
            cost_rows,
            _median_of("cost_ratio"),
        )
    )

    regret = json.loads((results / "plan_regret/plan_regret_a6.json").read_text())
    for policy_name in ("no_guard", "guarded"):
        jobs.append(
            (
                f"plan regret / {policy_name} / worst case",
                regret["per_pair"],
                _max_of(f"regret_{policy_name}"),
            )
        )
        jobs.append(
            (
                f"plan regret / {policy_name} / p95",
                regret["per_pair"],
                _p95_of(f"regret_{policy_name}"),
            )
        )
    jobs.append(
        (
            "plan regret / worst-case ratio (unguarded over guarded, paired)",
            regret["per_pair"],
            _max_ratio("regret_no_guard", "regret_guarded"),
        )
    )
    jobs.append(
        (
            "plan regret / p95 difference (guarded minus unguarded, paired)",
            regret["per_pair"],
            _p95_difference("regret_guarded", "regret_no_guard"),
        )
    )

    out: dict[str, Any] = {}
    for label, rows, statistic in jobs:
        summary = cluster_bootstrap(
            rows, statistic, replicates=args.replicates, seed=args.seed
        )
        out[label] = summary
        logger.info(
            "%-58s %12.3f  [%12.3f, %12.3f]  (n=%d rows / %d queries)",
            label,
            summary["point"],
            summary["ci_low"],
            summary["ci_high"],
            summary["n_rows"],
            summary["n_clusters"],
        )

    validity = json.loads(
        (results / "bound_verification/bound_validity.json").read_text()
    )
    upgrade = json.loads(
        (results / "certificate_upgrade/cert_upgrade.json").read_text()
    )
    tally = json.loads((results / "utility_table/intervention_tally.json").read_text())
    # Firing sets are nested in tau (fires iff estimate > tau * bound), so the
    # 1,564 rows across four thresholds are 525 distinct firings at tau = 1.
    distinct_firings = sum(int(t["fired"]) for t in tally["per_table"] if t["tau"] == 1)
    zero_events = {
        "pessimistic bound violations (synthetic)": {
            "events": validity["num_violations"],
            "trials": validity["num_queries_tested"],
            "upper_95": zero_event_upper_bound(validity["num_queries_tested"]),
        },
        "MOLP bound violations (certificate upgrade)": {
            "events": 0,
            "trials": upgrade["n_queries"],
            "upper_95": zero_event_upper_bound(upgrade["n_queries"]),
        },
        "one-draw JOB certificates costing more than the exact count": {
            "events": int(
                sum(1 for r in one_draw["per_template"] if r["ratio_1draw"] > 1.0)
            ),
            "trials": len(one_draw["per_template"]),
            "upper_95": zero_event_upper_bound(len(one_draw["per_template"])),
        },
        "interventions that worsened an estimate": {
            "events": tally["worsened_beyond_tolerance"],
            "trials": distinct_firings,
            "trials_with_multiplicity_across_tau": tally["interventions"],
            "upper_95": zero_event_upper_bound(distinct_firings),
            "conservative_upper_95_query_level": zero_event_upper_bound(
                int(tally["distinct_queries_with_a_firing"])
            ),
            "note": (
                "Both real benchmarks, tau in {1, 2, 4, 8}. Firing sets are nested "
                "in tau, so trials counts distinct (query, proposal, certificate) "
                "firings at tau = 1. Interventions within a query are not "
                "independent, so the conservative bound uses the number of "
                "distinct queries with a firing as the effective sample size. "
                "The paper quotes that one."
            ),
        },
    }
    for label, entry in zero_events.items():
        logger.info(
            "%-58s %d/%d events   upper 95%% %.4f",
            f"zero-event / {label}",
            entry["events"],
            entry["trials"],
            entry["upper_95"],
        )

    payload = {
        "experiment": "query_clustered_bootstrap_ci",
        "note": (
            "Percentile 95% CIs under resampling of whole queries. Rows within a "
            "query share a true cardinality and a bound, so row-level resampling "
            "would understate the width. Intervals cover sampling error only; "
            "for the real-JOB row they do not cover the censoring selection."
        ),
        "replicates": args.replicates,
        "seed": args.seed,
        "statistics": out,
        "zero_event_bounds": zero_events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("-> %s", args.output)


if __name__ == "__main__":
    main()
