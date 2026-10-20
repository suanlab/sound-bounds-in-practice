"""Plan-driven selective sampling ("certify-then-stop"), simulated offline.

Sampling every sub-plan of a query costs a median 2.82x executing it
(``results/pg_injection/v2/sampling_overhead.json``; cross-run timings). This
asks whether sampling
*only the sub-plans that decide the plan* certifies a near-optimal plan for a
fraction of that cost.

Per query (left-deep orders, C_out, the joint certificate of
``plan_certificate_joint``): start with the upper bound B (LpBound by default)
and no lower bound (L = 1). Repeatedly take the plan p* with the smallest joint
regret bound R and the competitor p' that attains it; sample the unsampled
sub-join of p' with the largest upper bound (a sub-join can only lower R through
its lower bound, and p' is where a too-small L hurts p*); replace its L by the
sound in-engine sample count from ``subset_counts.jsonl`` and pay that sample's
measured in-engine time. Stop when R(p*) <= target or nothing is left to sample.
The rule uses no information a deployment would lack: the order of sampling
depends only on B, L and plan structure, never on unseen truth.

Costs are compared with the time PostgreSQL takes to execute its own native
plan (``inject_plans_native.jsonl``). Post-hoc design, offline simulation.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.selective_sampling
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from experiments.pg_injection.inject_plans import _read_jsonl
from experiments.scripts.bootstrap_ci import cluster_bootstrap, percentile
from experiments.scripts.plan_certificate import repair_rounding
from experiments.scripts.plan_certificate_joint import incidence, left_deep_orders

if TYPE_CHECKING:
    from collections.abc import Callable

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

Vector = npt.NDArray[np.float64]
CHUNK = 256


def joint_bounds(inc: Vector, lower: Vector, upper: Vector) -> tuple[Vector, list[int]]:
    """Joint regret bound R(p) for every plan and the competitor attaining it."""
    c_lo, c_hi = inc @ lower, inc @ upper
    n = inc.shape[0]
    bound = np.zeros(n)
    rival: list[int] = [0] * n
    for start in range(0, n, CHUNK):
        block = inc[start : start + CHUNK]
        shared_hi = block @ (inc * upper).T
        shared_lo = block @ (inc * lower).T
        own_hi = c_hi[start : start + CHUNK, None] - shared_hi
        other_lo = c_lo[None, :] - shared_lo
        ratio = np.maximum(
            (own_hi + shared_lo) / (other_lo + shared_lo),
            (own_hi + shared_hi) / (other_lo + shared_hi),
        )
        bound[start : start + CHUNK] = ratio.max(axis=1)
        rival[start : start + CHUNK] = [int(i) for i in ratio.argmax(axis=1)]
    return bound, rival


def simulate_query(
    record: dict[str, Any],
    samples: dict[tuple[int, tuple[str, ...]], dict[str, Any]],
    upper_name: str,
    target: float,
    include_final: bool,
    give_up: bool = True,
) -> dict[str, Any] | None:
    """Selective and full sampling for one query; None if the bound is invalid."""
    subjoins = record["subjoins"]
    index = {frozenset(s["aliases"]): i for i, s in enumerate(subjoins)}
    truth = np.array([max(float(s["truth"]), 1.0) for s in subjoins])
    upper_values = []
    for s, t in zip(subjoins, truth, strict=True):
        value, violated = repair_rounding(float(s["upper"][upper_name]), float(t))
        if violated:
            return None
        upper_values.append(max(value, 1.0))
    upper = np.array(upper_values)
    keys = [(record["query_id"], tuple(sorted(s["aliases"]))) for s in subjoins]
    sampled_lower = np.array(
        [
            1.0
            if samples[k]["sample_censored"]
            else max(float(samples[k]["lower_sound"]), 1.0)
            for k in keys
        ]
    )
    sample_ms = np.array([float(samples[k]["sample_ms"]) for k in keys])
    if (sampled_lower > truth * (1 + 1e-12)).any():
        msg = f"unsound sample on query {record['query_id']}"
        raise AssertionError(msg)

    orders = left_deep_orders(record["aliases"], index)
    inc = incidence(orders, index, include_final=include_final)
    if not inc.any(axis=1).all():
        return None
    true_cost = inc @ truth
    optimum = float(true_cost.min())

    # Full sampling: every sub-join sampled.
    full_bound, _ = joint_bounds(inc, sampled_lower, upper)
    full_best = int(np.argmin(full_bound))

    # Selective sampling.
    lower = np.ones_like(truth)
    taken = np.zeros(len(truth), dtype=bool)
    cost = 0.0
    gave_up = False
    trajectory: list[tuple[float, float]] = []
    while True:
        bound, rival = joint_bounds(inc, lower, upper)
        best = int(np.argmin(bound))
        trajectory.append((cost, float(bound[best])))
        if bound[best] <= target or taken.all():
            break
        if give_up:
            # Best case sampling could reach: every unsampled lower bound at its
            # upper bound (a sample never exceeds the truth, so never exceeds B).
            hopeful, _ = joint_bounds(inc, np.where(taken, lower, upper), upper)
            if hopeful.min() > target:
                gave_up = True
                break
        candidates = [j for j in np.flatnonzero(inc[rival[best]]) if not taken[j]]
        if not candidates:  # the competitor is fully sampled: widen to any sub-join
            candidates = [int(j) for j in np.flatnonzero(~taken)]
        pick = max(candidates, key=lambda j: upper[j])
        taken[pick] = True
        lower[pick] = sampled_lower[pick]
        cost += sample_ms[pick]
    if (true_cost / optimum > bound * (1 + 1e-9)).any():
        msg = f"certificate unsound on query {record['query_id']}"
        raise AssertionError(msg)
    return {
        "query_id": record["query_id"],
        "n_subjoins": len(truth),
        "n_orders": len(orders),
        "selective_bound": float(bound[best]),
        "selective_certified": bool(bound[best] <= target),
        "selective_sampled": int(taken.sum()),
        "selective_ms": cost,
        "selective_true_regret": float(true_cost[best] / optimum),
        "selective_gave_up": gave_up,
        "selective_order": list(orders[best]),
        "full_order": list(orders[full_best]),
        "full_bound": float(full_bound[full_best]),
        "full_certified": bool(full_bound[full_best] <= target),
        "full_ms": float(sample_ms.sum()),
        "full_true_regret": float(true_cost[full_best] / optimum),
        "trajectory": trajectory,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=Path,
        default=Path("results/plan_guard/subjoin_sizes_maxdeg_fixed.jsonl"),
    )
    parser.add_argument(
        "--samples", type=Path, default=Path("results/pg_injection/subset_counts.jsonl")
    )
    parser.add_argument(
        "--native-runs",
        type=Path,
        default=Path("results/pg_injection/inject_plans_native.jsonl"),
    )
    parser.add_argument("--upper", default="lpbound")
    parser.add_argument("--target", type=float, default=2.0)
    parser.add_argument("--exclude-final", action="store_true")
    parser.add_argument(
        "--no-give-up", action="store_true", help="sample until certified or exhausted"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/selective_sampling.json"),
    )
    args = parser.parse_args()

    samples = {
        (r["query_id"], tuple(r["aliases"])): r for r in _read_jsonl(args.samples)
    }
    native_ms: dict[int, float] = {}
    for run in _read_jsonl(args.native_runs):
        digest = run["policy_plan"]["native"]
        if not run["plan_censored"][digest]:
            native_ms[run["query_id"]] = run["plan_ms"][digest]
    rows: list[dict[str, Any]] = []
    for record in _read_jsonl(args.sizes):
        if record["n_aliases"] < 3:
            continue
        row = simulate_query(
            record,
            samples,
            args.upper,
            args.target,
            not args.exclude_final,
            give_up=not args.no_give_up,
        )
        if row is None:
            continue
        exec_ms = native_ms.get(row["query_id"])
        row["native_exec_ms"] = exec_ms
        if exec_ms:
            row["selective_cost_over_exec"] = row["selective_ms"] / exec_ms
            row["full_cost_over_exec"] = row["full_ms"] / exec_ms
        rows.append(row)
        logger.info(
            "q%d: selective %s R=%.2f sampled %d/%d %.0f ms | full %s R=%.2f %.0f ms",
            row["query_id"],
            "cert" if row["selective_certified"] else "----",
            row["selective_bound"],
            row["selective_sampled"],
            row["n_subjoins"],
            row["selective_ms"],
            "cert" if row["full_certified"] else "----",
            row["full_bound"],
            row["full_ms"],
        )

    def share(field: str) -> Callable[[list[dict[str, Any]]], float]:
        return lambda rs: sum(1.0 for r in rs if r[field]) / len(rs)

    def med(field: str) -> Callable[[list[dict[str, Any]]], float]:
        return lambda rs: percentile([r[field] for r in rs], 0.5)

    timed = [r for r in rows if "selective_cost_over_exec" in r]
    if not timed:
        msg = "no query has a timed native plan; check --native-runs"
        raise RuntimeError(msg)
    summary = {
        "n_queries": len(rows),
        "n_timed": len(timed),
        "selective_certified": cluster_bootstrap(rows, share("selective_certified")),
        "full_certified": cluster_bootstrap(rows, share("full_certified")),
        "selective_cost_over_exec": cluster_bootstrap(
            timed, med("selective_cost_over_exec")
        ),
        "full_cost_over_exec": cluster_bootstrap(timed, med("full_cost_over_exec")),
        "selective_share_cost_below_exec": sum(
            r["selective_cost_over_exec"] < 1 for r in timed
        )
        / len(timed),
        "median_fraction_sampled": median(
            r["selective_sampled"] / r["n_subjoins"] for r in rows
        ),
        "max_true_regret_selective": max(r["selective_true_regret"] for r in rows),
        "max_true_regret_full": max(r["full_true_regret"] for r in rows),
        "share_gave_up": sum(r["selective_gave_up"] for r in rows) / len(rows),
        "selective_cost_over_exec_certified_only": percentile(
            [r["selective_cost_over_exec"] for r in timed if r["selective_certified"]],
            0.5,
        ),
    }
    payload = {
        "experiment": "selective_sampling",
        "preregistered": False,
        "upper": args.upper,
        "target": args.target,
        "include_final": not args.exclude_final,
        "give_up_rule": not args.no_give_up,
        "summary": summary,
        "per_query": rows,
    }
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    logger.info(json.dumps(summary, indent=1)[:2500])


if __name__ == "__main__":
    main()
