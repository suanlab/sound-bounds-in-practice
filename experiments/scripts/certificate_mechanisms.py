"""Mechanisms behind the joint plan certificate (exploratory; ``PREDICTIONS.md``).

Four offline analyses of the LpBound x in-engine-sample bracket, each tested
against a prediction frozen before it was run:

* **H1, shared mass.** By the pair inequality behind the joint bound, a plan is
  certified only if the floor of what every plan shares pays for the upper bound
  of what the plan does alone. We summarise this per query as
  ``rho = L(final) / (L(final) + B(non-final joins of the minimax plan))`` and
  ask how well ``rho`` separates certified from uncertified queries (AUC), with
  a threshold chosen on STATS-CEB and applied unchanged to JOB-light.
* **H2, exchange rate.** Every upper bound is tightened geometrically,
  ``B' = |S| (B/|S|)^a`` (still sound), with ``a`` set so the median
  whole-query looseness falls by a factor ``f``; the certified share at
  r = 0.10 is compared with the share at r = 0.25 under the real bounds.
* **H4, a-posteriori certification.** After PostgreSQL executes its own plan,
  every intermediate size of that plan is known exactly. With those sizes and
  the floor (or no floor) on every other sub-join, the regret bound of the
  executed plan over all left-deep orders is ``C(native) / min_p' C_lo(p')``,
  where ``C_lo`` takes a sub-join at its exact size when the native plan
  computed it and at ``L`` otherwise. This is sound by construction and is
  checked on every query.
* **H6, probabilistic floor.** The floor ``max(L, L-hat / c)``, with ``c`` the
  95th percentile of ``L-hat / |S|`` on seed 0, applied to seeds 1-4 and to
  JOB-light; reports the certified share, floor violations, and the true regret
  of every plan it certifies (no soundness assertion: the floor may be wrong).

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.certificate_mechanisms
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from experiments.pg_injection.inject_plans import _read_jsonl, load_estimates
from experiments.pg_injection.run_forced_orders import join_sets
from experiments.scripts.certificate_context import true_spread
from experiments.scripts.plan_certificate_joint import (
    Vector,
    incidence,
    left_deep_orders,
    query_brackets,
    regret_bounds,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

STATS_SIZES = Path("results/plan_guard/subjoin_sizes_maxdeg_fixed.jsonl")
JL_SIZES = Path("results/plan_certificate/joblight_subjoin_sizes.jsonl")
JL_ESTIMATES = Path(
    "external/LpBound/results/evaluation_time/subquery_estimations/joblight"
)
SAMPLES = {
    "stats_r010": Path("results/pg_injection/subset_counts.jsonl"),
    "stats_r025": Path("results/pg_injection/subset_counts_r025.jsonl"),
    **{
        f"stats_r010_seed{s}": Path(
            f"results/pg_injection/v3/subset_counts_r010_seed{s}.jsonl"
        )
        for s in range(1, 5)
    },
    "joblight_r010": Path("results/pg_injection/joblight/subset_counts.jsonl"),
}
UPPER = "lpbound"
TARGET = 2.0
TIGHTEN = (1.0, 1.5, 2.0, 2.5, 4.0, 8.0, 16.0)


def load_samples(path: Path) -> dict[tuple[int, tuple[str, ...]], dict[str, Any]]:
    """In-engine sample rows keyed by (query, sorted aliases)."""
    return {(r["query_id"], tuple(sorted(r["aliases"]))): r for r in _read_jsonl(path)}


def queries(sizes: Path, samples_path: Path) -> list[dict[str, Any]]:
    """Per query with 3+ relations: orders, incidence, truth, bounds and samples."""
    samples = load_samples(samples_path)
    out = []
    for record in _read_jsonl(sizes):
        if record["n_aliases"] < 3:
            continue
        brackets = query_brackets(record, samples)
        if brackets is None or UPPER not in brackets["uppers"]:
            continue
        if brackets["violated"][UPPER]:
            continue
        index = {frozenset(s["aliases"]): i for i, s in enumerate(record["subjoins"])}
        orders = left_deep_orders(record["aliases"], index)
        rows = [samples[(record["query_id"], tuple(sorted(k)))] for k in index]
        lhat = np.array(
            [math.nan if r["sample_censored"] else float(r["lower_hat"]) for r in rows]
        )
        out.append(
            {
                "query_id": record["query_id"],
                "n_aliases": record["n_aliases"],
                "record": record,
                "index": index,
                "orders": orders,
                "inc": incidence(orders, index, include_final=True),
                "truth": brackets["truth"],
                "upper": brackets["uppers"][UPPER],
                "floor": brackets["lowers"]["sample"],
                "lhat": lhat,
                "final": index[frozenset(record["aliases"])],
                "matters": true_spread(record) > TARGET,
            }
        )
    return out


def certified(q: dict[str, Any], lower: Vector, upper: Vector) -> tuple[bool, float]:
    """Whether the best-bound plan is certified, and its true regret."""
    bound, _ = regret_bounds(q["inc"], lower, upper)
    best = int(np.argmin(bound))
    c_true = q["inc"] @ q["truth"]
    return bool(bound[best] <= TARGET), float(c_true[best] / c_true.min())


def auc(scores: list[float], labels: list[bool]) -> float:
    """Area under the ROC curve (Mann-Whitney, ties counted half)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def rho(q: dict[str, Any]) -> float:
    """Shared-mass statistic of the minimax plan (smallest C_hi)."""
    plan = int(np.argmin(q["inc"] @ q["upper"]))
    own = q["inc"][plan] * q["upper"]
    own[q["final"]] = 0.0
    shared = q["floor"][q["final"]]
    return float(shared / (shared + own.sum()))


def h1_shared_mass(
    stats: list[dict[str, Any]], joblight: list[dict[str, Any]]
) -> dict[str, Any]:
    """AUC of rho for certification; STATS-chosen threshold applied to JOB-light."""
    out: dict[str, Any] = {}
    threshold = None
    for name, qs in (("stats_r010", stats), ("joblight_r010", joblight)):
        labels = [certified(q, q["floor"], q["upper"])[0] for q in qs]
        scores = [rho(q) for q in qs]
        if threshold is None:  # accuracy-maximising threshold, STATS-CEB only
            candidates = sorted(set(scores))
            threshold = max(
                candidates,
                key=lambda t: sum(
                    (s >= t) == y for s, y in zip(scores, labels, strict=True)
                ),
            )
        predicted = [s >= threshold for s in scores]
        by_size: dict[int, dict[str, float]] = {}
        for n in sorted({q["n_aliases"] for q in qs}):
            sub = [i for i, q in enumerate(qs) if q["n_aliases"] == n]
            by_size[n] = {
                "n": len(sub),
                "share_certified": sum(labels[i] for i in sub) / len(sub),
                "median_rho": median(scores[i] for i in sub),
            }
        out[name] = {
            "n": len(qs),
            "n_certified": sum(labels),
            "auc": auc(scores, labels),
            "threshold_from_stats": threshold,
            "accuracy_at_threshold": sum(
                p == y for p, y in zip(predicted, labels, strict=True)
            )
            / len(qs),
            "by_n_aliases": by_size,
            "n_rho_at_least_half": sum(sc >= 0.5 for sc in scores),
            "n_rho_at_least_half_certified": sum(
                sc >= 0.5 and y for sc, y in zip(scores, labels, strict=True)
            ),
            "per_query": [
                {
                    "query_id": q["query_id"],
                    "n_aliases": q["n_aliases"],
                    "rho": score,
                    "certified": label,
                }
                for q, score, label in zip(qs, scores, labels, strict=True)
            ],
        }
    return out


def h2_exchange_rate(
    r010: list[dict[str, Any]], r025: list[dict[str, Any]]
) -> dict[str, Any]:
    """Certified share when the upper bounds are tightened by each factor."""
    looseness = median(
        float(q["upper"][q["final"]] / q["truth"][q["final"]]) for q in r010
    )

    def share(qs: list[dict[str, Any]], factor: float) -> float:
        a = 1.0 if factor == 1.0 else math.log(looseness / factor) / math.log(looseness)
        hits = 0
        for q in qs:
            upper = q["truth"] * (q["upper"] / q["truth"]) ** a
            hits += certified(q, q["floor"], np.maximum(upper, q["truth"]))[0]
        return hits / len(qs)

    curve = {f"{f:g}": share(r010, f) for f in TIGHTEN if f < looseness}
    reference = share(r025, 1.0)
    factors = [float(f) for f in curve]
    shares = list(curve.values())
    exchange = None
    for (f0, s0), (f1, s1) in zip(
        zip(factors, shares, strict=True),
        zip(factors[1:], shares[1:], strict=True),
        strict=False,
    ):
        if s0 <= reference <= s1 and s1 > s0:
            exchange = f0 + (reference - s0) * (f1 - f0) / (s1 - s0)
            break

    def share_scaled(
        qs: list[dict[str, Any]], upper_div: float, floor_mul: float
    ) -> float:
        return sum(
            certified(q, q["floor"] * floor_mul, q["upper"] / upper_div)[0] for q in qs
        ) / len(qs)

    sub_join_shrink = float(
        np.median(
            np.concatenate(
                [
                    (q["upper"] / q["truth"])
                    ** (1 - math.log(looseness / 2.5) / math.log(looseness))
                    for q in r010
                ]
            )
        )
    )
    clipped = sum(
        certified(q, q["floor"], np.maximum(q["upper"] / 2.5, q["truth"]))[0]
        for q in r010
    ) / len(r010)
    below = sum(int((q["upper"] / 2.5 < q["truth"] * (1 - 1e-12)).sum()) for q in r010)
    floors_025 = {q["query_id"]: q["floor"] for q in r025}
    ratio = np.concatenate(
        [
            floors_025[q["query_id"]][
                (q["floor"] > 1) & (floors_025[q["query_id"]] > 1)
            ]
            / q["floor"][(q["floor"] > 1) & (floors_025[q["query_id"]] > 1)]
            for q in r010
            if q["query_id"] in floors_025
        ]
    )
    return {
        "share_r010_upper_divided_2p5_clipped_at_truth": clipped,
        "n_subjoins_upper_divided_2p5_below_truth": below,
        "n_subjoins": sum(len(q["truth"]) for q in r010),
        "median_floor_ratio_r025_over_r010": float(np.median(ratio)),
        "share_r010_upper_divided_2p5": share_scaled(r010, 2.5, 1.0),
        "share_r010_floor_times_2p5": share_scaled(r010, 1.0, 2.5),
        "median_subjoin_shrink_at_2p5": sub_join_shrink,
        "median_whole_query_looseness": looseness,
        "share_r010_by_tightening": curve,
        "share_r025_real_bounds": reference,
        "tightening_matching_r025": exchange,
    }


def native_plans(path: Path) -> dict[int, list[frozenset[str]]]:
    """Join alias sets of PostgreSQL's native plan per query (execution run)."""
    out = {}
    for run in _read_jsonl(path):
        sets: list[frozenset[str]] = []
        join_sets(run["plans"][run["policy_plan"]["native"]], sets)
        out[run["query_id"]] = sets
    return out


def h4_a_posteriori(
    qs: list[dict[str, Any]],
    plans: dict[int, list[frozenset[str]]],
    artifact_truth: dict[tuple[int, tuple[str, ...]], float],
) -> dict[str, Any]:
    """Regret bound of the executed native plan from its exact intermediate sizes.

    The native plan may join through an implied equality, a sub-join absent from
    the left-deep enumeration; its size then comes from the artifact's truth.
    Competitors are the left-deep orders. With ``cout_minus_final`` the final
    result, which every plan shares, is left out of every cost.
    """
    out: dict[str, Any] = {}
    for variant in ("cout", "cout_minus_final"):
        include_final = variant == "cout"
        rows = []
        n_missing = 0
        for q in qs:
            sets = plans.get(q["query_id"])
            sizes = []
            for s in set(sets or []):
                key = (q["query_id"], tuple(sorted(s)))
                if s in q["index"]:
                    sizes.append(float(q["truth"][q["index"][s]]))
                elif key in artifact_truth:
                    sizes.append(max(artifact_truth[key], 1.0))
                else:
                    sizes = []
                    break
            if not sizes:
                n_missing += 1
                continue
            final = frozenset(q["record"]["aliases"])
            c_native = sum(sizes) - (0.0 if include_final else q["truth"][q["final"]])
            inc = incidence(q["orders"], q["index"], include_final=include_final)
            if not inc.any(axis=1).all() or c_native <= 0:
                continue
            observed = np.zeros(len(q["index"]), dtype=bool)
            observed[[q["index"][s] for s in set(sets or []) if s in q["index"]]] = True
            if not include_final:
                observed[q["index"][final]] = False
            c_true = inc @ q["truth"]
            regret = c_native / min(c_native, float(c_true.min()))
            row: dict[str, Any] = {
                "query_id": q["query_id"],
                "matters": q["matters"],
                "true_regret": regret,
                "implied_join": any(s not in q["index"] for s in sets or []),
            }
            for name, floor in (
                ("sample", q["floor"]),
                ("none", np.ones_like(q["truth"])),
            ):
                lower = np.where(observed, q["truth"], floor)
                bound = c_native / min(c_native, float((inc @ lower).min()))
                if regret > bound * (1 + 1e-9):
                    msg = f"a-posteriori certificate unsound on query {q['query_id']}"
                    raise AssertionError(msg)
                row[f"bound_{name}"] = bound
            rows.append(row)

        def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "n": len(subset),
                "share_true_regret_le_2": sum(
                    r["true_regret"] <= TARGET for r in subset
                )
                / len(subset),
                **{
                    f"share_certified_{name}": sum(
                        r[f"bound_{name}"] <= TARGET for r in subset
                    )
                    / len(subset)
                    for name in ("sample", "none")
                },
            }

        out[variant] = {
            "n_skipped_no_truth": n_missing,
            "n_implied_join_plans": sum(r["implied_join"] for r in rows),
            "all": block(rows),
            "choice_matters": block([r for r in rows if r["matters"]]),
            "choice_does_not_matter": block([r for r in rows if not r["matters"]]),
            "per_query": rows,
        }
    return out


def h6_probabilistic_floor(
    calibration: list[dict[str, Any]], tests: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Calibrated L-hat floor: coverage on seed 0, certificates on held-out draws."""
    ratios = np.concatenate(
        [
            q["lhat"][np.isfinite(q["lhat"]) & (q["lhat"] > 0)]
            / q["truth"][np.isfinite(q["lhat"]) & (q["lhat"] > 0)]
            for q in calibration
        ]
    )
    c = float(np.quantile(ratios, 0.95))
    out: dict[str, Any] = {"c": c, "n_calibration_subjoins": len(ratios)}
    for name, qs in tests.items():
        n_sub = violations = sound_hits = prob_hits = regret_ok = 0
        for q in qs:
            lhat = np.where(np.isfinite(q["lhat"]), q["lhat"], 0.0)
            floor = np.maximum(q["floor"], lhat / c)
            n_sub += len(floor)
            violations += int((floor > q["truth"] * (1 + 1e-9)).sum())
            sound_hits += certified(q, q["floor"], q["upper"])[0]
            hit, regret = certified(q, floor, q["upper"])
            prob_hits += hit
            regret_ok += hit and regret <= TARGET
        out[name] = {
            "n_queries": len(qs),
            "violation_rate": violations / n_sub,
            "share_certified_sound": sound_hits / len(qs),
            "share_certified_calibrated": prob_hits / len(qs),
            "share_of_certified_with_true_regret_le_2": regret_ok / prob_hits
            if prob_hits
            else math.nan,
        }
    return out


def h6_leave_one_draw_out(draws: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Calibrate c on each draw in turn; floor violations on the other draws."""

    def c_of(qs: list[dict[str, Any]]) -> float:
        ratios = np.concatenate(
            [
                q["lhat"][np.isfinite(q["lhat"]) & (q["lhat"] > 0)]
                / q["truth"][np.isfinite(q["lhat"]) & (q["lhat"] > 0)]
                for q in qs
            ]
        )
        return float(np.quantile(ratios, 0.95))

    out: dict[str, Any] = {}
    for name, qs in draws.items():
        c = c_of(qs)
        rates = {}
        for other, test in draws.items():
            if other == name:
                continue
            n_sub = violations = 0
            for q in test:
                lhat = np.where(np.isfinite(q["lhat"]), q["lhat"], 0.0)
                floor = np.maximum(q["floor"], lhat / c)
                n_sub += len(floor)
                violations += int((floor > q["truth"] * (1 + 1e-9)).sum())
            rates[other] = violations / n_sub
        out[name] = {"c": c, "violation_rate_on_others": rates}
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-runs",
        type=Path,
        default=Path("results/pg_injection/v3/stats_inject.jsonl"),
    )
    parser.add_argument(
        "--native-runs-joblight",
        type=Path,
        default=Path("results/pg_injection/v3/joblight_inject.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/certificate_mechanisms.json"),
    )
    args = parser.parse_args()

    stats = queries(STATS_SIZES, SAMPLES["stats_r010"])
    joblight = queries(JL_SIZES, SAMPLES["joblight_r010"])
    out: dict[str, Any] = {
        "experiment": "certificate_mechanisms",
        "preregistered": False,
        "predictions": "docs/icde_2027/PREDICTIONS.md",
        "h1_shared_mass": h1_shared_mass(stats, joblight),
    }
    logger.info("H1 %s", json.dumps(out["h1_shared_mass"])[:600])
    out["h2_exchange_rate"] = h2_exchange_rate(
        stats, queries(STATS_SIZES, SAMPLES["stats_r025"])
    )
    logger.info("H2 %s", json.dumps(out["h2_exchange_rate"]))
    out["h4_a_posteriori"] = h4_a_posteriori(
        stats, native_plans(args.native_runs), load_estimates("truecardinality")
    )
    out["h4_a_posteriori_joblight"] = h4_a_posteriori(
        joblight,
        native_plans(args.native_runs_joblight),
        load_estimates("truecardinality", directory=JL_ESTIMATES),
    )
    for variant, entry in [
        *out["h4_a_posteriori"].items(),
        *((f"joblight {k}", v) for k, v in out["h4_a_posteriori_joblight"].items()),
    ]:
        logger.info(
            "H4 %s %s",
            variant,
            json.dumps({k: v for k, v in entry.items() if k != "per_query"}),
        )
    tests = {
        f"stats_r010_seed{s}": queries(STATS_SIZES, SAMPLES[f"stats_r010_seed{s}"])
        for s in range(1, 5)
    }
    tests["joblight_r010"] = joblight
    out["h6_probabilistic_floor"] = h6_probabilistic_floor(stats, tests)
    draws = {"stats_r010": stats} | {
        k: v for k, v in tests.items() if k.startswith("stats_r010_seed")
    }
    out["h6_leave_one_draw_out"] = h6_leave_one_draw_out(draws)
    logger.info("H6 leave-one-draw-out %s", json.dumps(out["h6_leave_one_draw_out"]))
    logger.info("H6 %s", json.dumps(out["h6_probabilistic_floor"]))
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
