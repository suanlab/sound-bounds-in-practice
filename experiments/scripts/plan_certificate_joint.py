"""Joint (shared-sub-join-aware) plan certificates from sound brackets.

The registered certificate (Proposition 4(3), ``plan_certificate.py``) bounds
regret by ``G = max_{p in S} C_hi(p) / min_p' C_lo(p')``. It evaluates the
numerator at the upper bound and the denominator at the lower bound
*independently*, so a sub-join that two plans share -- under C_out at least the
final result, which every plan shares -- enters at ``B`` above and at ``L``
below at once. ``G`` is therefore a decoupled relaxation, and with a sampling
floor ``L ~ r|S|`` it can never fall below about ``1/r``.

The tightest regret bound the same box ``[L, B]`` supports is the joint one,
``R(p) = max_{s in box} C(p, s) / min_p' C(p', s)
       = max_p' max_{s in box} C(p, s) / C(p', s)``.
For a fixed pair, write ``C(p, s) = A + S`` and ``C(p', s) = D + S`` with ``A``
the sub-joins only ``p`` uses, ``D`` those only ``p'`` uses and ``S`` the shared
ones. The ratio is maximised at ``A`` on ``B``, ``D`` on ``L``, and -- because
``(a + x) / (d + x)`` is monotone in ``x`` -- at the shared total on either its
lower or its upper extreme. Joint dominance is exact in the same way: ``p`` is
never optimal if some ``p'`` has ``A(L) > D(B)`` (shared terms cancel).

Post-hoc (suggested by an internal review panel), not pre-registered. Every
query asserts soundness: the truth lies in the box, the optimum survives joint
pruning, and every plan's true regret is at most its joint bound.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.plan_certificate_joint
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from experiments.scripts.bootstrap_ci import cluster_bootstrap
from experiments.scripts.plan_certificate import repair_rounding

if TYPE_CHECKING:
    from collections.abc import Callable

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

Vector = npt.NDArray[np.float64]

UPPERS = ("maxdeg", "safebound", "lpbound")
CHUNK = 256
TOL = 1e-9


def left_deep_orders(
    aliases: list[str], index: dict[frozenset[str], int]
) -> list[tuple[str, ...]]:
    """Alias orders whose every prefix of size >= 2 is a connected sub-join."""
    return [
        perm
        for perm in itertools.permutations(aliases)
        if all(frozenset(perm[: i + 1]) in index for i in range(1, len(perm)))
    ]


def incidence(
    orders: list[tuple[str, ...]],
    index: dict[frozenset[str], int],
    *,
    include_final: bool,
) -> Vector:
    """0/1 matrix: order x sub-join, 1 where the sub-join is a costed prefix."""
    n_sub = len(index)
    matrix = np.zeros((len(orders), n_sub))
    for row, order in enumerate(orders):
        last = len(order) if include_final else len(order) - 1
        for i in range(1, last):
            matrix[row, index[frozenset(order[: i + 1])]] = 1.0
    return matrix


def regret_bounds(
    inc: Vector, lower: Vector, upper: Vector
) -> tuple[Vector, npt.NDArray[np.bool_]]:
    """Joint regret bound ``R(p)`` of every plan, and which plans are dominated."""
    c_lo, c_hi = inc @ lower, inc @ upper
    n = inc.shape[0]
    bound = np.zeros(n)
    dominated = np.zeros(n, dtype=bool)
    for start in range(0, n, CHUNK):
        block = inc[start : start + CHUNK]
        shared_hi = block @ (inc * upper).T  # shared sub-joins at B
        shared_lo = block @ (inc * lower).T  # shared sub-joins at L
        own_hi = c_hi[start : start + CHUNK, None] - shared_hi  # p-only at B
        other_lo = c_lo[None, :] - shared_lo  # p'-only at L
        ratio_lo = (own_hi + shared_lo) / (other_lo + shared_lo)
        ratio_hi = (own_hi + shared_hi) / (other_lo + shared_hi)
        bound[start : start + CHUNK] = np.maximum(ratio_lo, ratio_hi).max(axis=1)
        own_lo = c_lo[start : start + CHUNK, None] - shared_lo  # p-only at L
        other_hi = c_hi[None, :] - shared_hi  # p'-only at B
        margin = (own_lo - other_hi) > TOL * float(c_hi.max())
        dominated[start : start + CHUNK] = margin.any(axis=1)
    return bound, dominated


def joint_certificate(
    inc: Vector, lower: Vector, upper: Vector, truth: Vector
) -> dict[str, Any]:
    """Decoupled G, joint regret bounds, joint pruning, and soundness checks."""
    c_lo, c_hi, c_true = inc @ lower, inc @ upper, inc @ truth
    optimum = float(c_true.min())
    regret = c_true / optimum

    decoupled_survivors = c_lo <= c_hi.min()
    decoupled_g = float(c_hi[decoupled_survivors].max() / c_lo.min())

    n = inc.shape[0]
    bound, dominated = regret_bounds(inc, lower, upper)
    if dominated[np.argmin(c_true)] or (regret > bound * (1 + 1e-9)).any():
        msg = "joint certificate unsound on this query"
        raise AssertionError(msg)
    best = int(np.argmin(bound))
    survivors = ~dominated
    return {
        "n_plans": n,
        "G_decoupled": decoupled_g,
        "minimax_bound_decoupled": float(c_hi.min() / c_lo.min()),
        "joint_bound_best": float(bound[best]),
        "joint_bound_minimax_plan": float(bound[int(np.argmin(c_hi))]),
        "joint_bound_survivors": float(bound[survivors].max()),
        "true_regret_best": float(regret[best]),
        "n_survivors_joint": int(survivors.sum()),
        "n_survivors_decoupled": int(decoupled_survivors.sum()),
        "pruning_joint": float(dominated.mean()),
        "pruning_decoupled": float(1.0 - decoupled_survivors.mean()),
    }


def query_brackets(
    record: dict[str, Any],
    pg_samples: dict[tuple[int, tuple[str, ...]], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Truth, upper and lower vectors for one query, or None if not usable.

    With ``pg_samples`` the sampled floor is PostgreSQL's in-engine sample of the
    same sub-join (``subset_counts.jsonl``) instead of the stored Python sample.
    """
    subjoins = record["subjoins"]
    truth = np.array([max(float(s["truth"]), 1.0) for s in subjoins])
    uppers: dict[str, Vector] = {}
    violated: dict[str, bool] = {}
    for name in UPPERS:
        if not all(name in s["upper"] for s in subjoins):
            continue  # e.g. no max-degree statistics for this benchmark
        values, bad = [], False
        for s, t in zip(subjoins, truth, strict=True):
            repaired, real = repair_rounding(float(s["upper"][name]), float(t))
            bad = bad or real
            values.append(max(repaired, 1.0))
        uppers[name], violated[name] = np.array(values), bad
    if pg_samples is not None:
        keys = [(record["query_id"], tuple(sorted(s["aliases"]))) for s in subjoins]
        sample = np.array(
            [
                1.0
                if pg_samples[k]["sample_censored"]
                else max(float(pg_samples[k]["lower_sound"]), 1.0)
                for k in keys
            ]
        )
    else:
        sample = np.array(
            [
                1.0
                if s["sample"]["censored"]
                else max(float(s["sample"]["sound"]), 1.0)
                for s in subjoins
            ]
        )
    if (sample > truth * (1 + 1e-12)).any():
        msg = f"sampling floor above truth on query {record['query_id']}"
        raise AssertionError(msg)
    return {
        "truth": truth,
        "uppers": uppers,
        "violated": violated,
        "lowers": {"one": np.ones_like(truth), "sample": sample},
    }


def summarise(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    """Shares and medians with query-clustered intervals (one row per query)."""

    def share_within(
        field: str, limit: float
    ) -> Callable[[list[dict[str, Any]]], float]:
        return lambda rs: sum(1.0 for r in rs if r[field] <= limit) / len(rs)

    def median_of(field: str) -> Callable[[list[dict[str, Any]]], float]:
        return lambda rs: float(median(r[field] for r in rs))

    out: dict[str, Any] = {"n_queries": len(rows)}
    stats = {
        "share_joint_bound_le_2": share_within("joint_bound_best", 2.0),
        "share_decoupled_G_le_2": share_within("G_decoupled", 2.0),
        "median_joint_bound_best": median_of("joint_bound_best"),
        "median_joint_bound_minimax_plan": median_of("joint_bound_minimax_plan"),
        "median_minimax_bound_decoupled": median_of("minimax_bound_decoupled"),
        "median_G_decoupled": median_of("G_decoupled"),
        "median_true_regret_best": median_of("true_regret_best"),
        "mean_pruning_joint": lambda rs: sum(r["pruning_joint"] for r in rs) / len(rs),
        "mean_pruning_decoupled": lambda rs: (
            sum(r["pruning_decoupled"] for r in rs) / len(rs)
        ),
    }
    for name, statistic in stats.items():
        out[name] = cluster_bootstrap(rows, statistic, replicates=replicates, seed=seed)
    out["count_joint_bound_le_2"] = sum(1 for r in rows if r["joint_bound_best"] <= 2)
    out["max_true_regret_best"] = max(r["true_regret_best"] for r in rows)
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=Path, default=Path("results/plan_guard/subjoin_sizes.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/plan_certificate_joint.json"),
    )
    parser.add_argument(
        "--pg-samples",
        type=Path,
        default=None,
        help="use PostgreSQL's in-engine samples (subset_counts.jsonl) as the floor",
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.sizes.read_text().splitlines() if line]
    pg_samples = None
    if args.pg_samples is not None:
        pg_samples = {}
        for line in args.pg_samples.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pg_samples[(row["query_id"], tuple(row["aliases"]))] = row
    rate = (
        {json.loads(args.pg_samples.read_text().splitlines()[0])["rate"]}
        if args.pg_samples
        else {r["subjoins"][0]["sample"]["rate"] for r in records}
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["n_aliases"] < 3:
            continue
        brackets = query_brackets(record, pg_samples)
        if brackets is None:
            continue
        index = {frozenset(s["aliases"]): i for i, s in enumerate(record["subjoins"])}
        orders = left_deep_orders(record["aliases"], index)
        for include_final in (True, False):
            inc = incidence(orders, index, include_final=include_final)
            if not inc.any(axis=1).all():
                continue  # a 2-alias remainder has no costed prefix without final
            for upper_name in brackets["uppers"]:
                if brackets["violated"][upper_name]:
                    continue
                for lower_name, lower in brackets["lowers"].items():
                    variant = "cout" if include_final else "cout_minus_final"
                    key = f"{upper_name}x{lower_name}|{variant}"
                    result = joint_certificate(
                        inc, lower, brackets["uppers"][upper_name], brackets["truth"]
                    )
                    result["query_id"] = record["query_id"]
                    result["n_aliases"] = record["n_aliases"]
                    rows.setdefault(key, []).append(result)
        logger.info("query %s done (%d orders)", record["query_id"], len(orders))

    payload = {
        "experiment": "plan_certificate_joint",
        "preregistered": False,
        "sample_rate": sorted(rate),
        "source": str(args.sizes),
        "sample_source": str(args.pg_samples) if args.pg_samples else "python sampler",
        "replicates": args.replicates,
        "seed": args.seed,
        "summary": {
            k: summarise(v, args.replicates, args.seed) for k, v in rows.items()
        },
        "per_query": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    for key, entry in payload["summary"].items():
        share = entry["share_joint_bound_le_2"]
        logger.info(
            "%-34s n=%3d  joint<=2 %3d (%.1f%% [%.1f, %.1f])  median joint %.2f  "
            "median G %.3g  pruning joint %.1f%% vs %.1f%%",
            key,
            entry["n_queries"],
            entry["count_joint_bound_le_2"],
            100 * share["point"],
            100 * share["ci_low"],
            100 * share["ci_high"],
            entry["median_joint_bound_best"]["point"],
            entry["median_G_decoupled"]["point"],
            100 * entry["mean_pruning_joint"]["point"],
            100 * entry["mean_pruning_decoupled"]["point"],
        )


if __name__ == "__main__":
    main()
