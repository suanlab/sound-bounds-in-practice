"""Price of certification: the cheapest set of samples that still certifies (H5).

``selective_sampling`` decides online which sub-joins to sample. This asks what
an oracle that knew every sample's value and time in advance would pay: the
minimum total sampling time over sets ``T`` of sub-joins such that, with the
sampled floor on ``T`` and ``L = 1`` elsewhere, some plan has a joint regret
bound of at most 2 (LpBound above, ``C_out`` with the final result, r = 0.10).
Given these measured samples (their values and times), no choice of
sub-joins to sample certifies the query for less; other samplers or floors
might. The joint bound can only fall when a
floor rises, so feasibility is monotone in ``T`` and an exact branch and bound
(include / exclude each useful sub-join, prune on cost and on infeasibility of
the most optimistic completion) finds the optimum. Sub-joins whose sample timed
out or drew no tuple cannot raise the floor and are never sampled.

Costs are compared with the native plan's execution time in each of the three
v3 runs that timed it, and with the online rule's cost. Exploratory; prediction
in ``docs/icde_2027/PREDICTIONS.md``.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.price_of_certification
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import numpy.typing as npt

from experiments.scripts.certificate_mechanisms import (
    SAMPLES,
    STATS_SIZES,
    TARGET,
    load_samples,
    queries,
)
from experiments.scripts.plan_certificate_joint import Vector, regret_bounds
from experiments.scripts.selective_denominators import V3, native_times

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MAX_CANDIDATES = 24


def feasible(q: dict[str, Any], chosen: npt.NDArray[np.bool_]) -> bool:
    """Whether sampling exactly the sub-joins in ``chosen`` certifies the query."""
    lower = np.where(chosen, q["floor"], 1.0)
    bound, _ = regret_bounds(q["inc"], lower, q["upper"])
    return bool(bound.min() <= TARGET)


def oracle_cost(q: dict[str, Any], times: Vector) -> tuple[float, list[int], int]:
    """Minimum sampling time that certifies; (cost, sub-joins sampled, nodes)."""
    useful = [i for i in range(len(times)) if q["floor"][i] > 1.0]
    useful.sort(key=lambda i: -times[i])
    best_cost = [float(times[useful].sum())]
    best_set = [sorted(useful)]  # sampling every useful sub-join certifies
    nodes = [0]

    def search(k: int, chosen: npt.NDArray[np.bool_], cost: float) -> None:
        nodes[0] += 1
        if cost >= best_cost[0]:
            return
        optimistic = chosen.copy()
        optimistic[useful[k:]] = True
        if not feasible(q, optimistic):
            return
        if feasible(q, chosen):
            best_cost[0], best_set[0] = cost, [int(i) for i in np.flatnonzero(chosen)]
            return
        if k == len(useful):
            return
        i = useful[k]
        chosen[i] = True
        search(k + 1, chosen, cost + float(times[i]))
        chosen[i] = False
        search(k + 1, chosen, cost)

    search(0, np.zeros(len(times), dtype=bool), 0.0)
    return best_cost[0], best_set[0], nodes[0]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selective",
        type=Path,
        default=Path("results/plan_certificate/selective_sampling_v3.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/price_of_certification.json"),
    )
    args = parser.parse_args()

    samples = load_samples(SAMPLES["stats_r010"])
    greedy = {
        r["query_id"]: r for r in json.loads(args.selective.read_text())["per_query"]
    }
    native = native_times(V3)
    rows = []
    for q in queries(STATS_SIZES, SAMPLES["stats_r010"]):
        if not feasible(q, np.ones(len(q["truth"]), dtype=bool)):
            continue  # not certified even with every sample
        times = np.array(
            [
                float(samples[(q["query_id"], tuple(sorted(k)))]["sample_ms"])
                for k in q["index"]
            ]
        )
        n_useful = int((q["floor"] > 1.0).sum())
        if n_useful > MAX_CANDIDATES:
            msg = f"query {q['query_id']}: {n_useful} candidates, too many to search"
            raise RuntimeError(msg)
        cost, chosen, nodes = oracle_cost(q, times)
        keys = list(q["index"])
        row: dict[str, Any] = {
            "query_id": q["query_id"],
            "n_aliases": q["n_aliases"],
            "matters": q["matters"],
            "n_subjoins": len(times),
            "n_sampled_oracle": len(chosen),
            "oracle_subjoins": [sorted(keys[i]) for i in chosen],
            "oracle_ms": cost,
            "greedy_ms": greedy[q["query_id"]]["selective_ms"],
            "full_ms": float(times.sum()),
            "search_nodes": nodes,
        }
        for name, ref in native.items():
            if q["query_id"] in ref:
                row[f"oracle_over_{name}"] = cost / ref[q["query_id"]]
        rows.append(row)
        logger.info(
            "q%d: oracle %.0f ms (%d/%d sampled) greedy %.0f ms full %.0f ms",
            q["query_id"],
            cost,
            len(chosen),
            len(times),
            row["greedy_ms"],
            row["full_ms"],
        )

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n": len(subset),
            "median_oracle_over_greedy": median(
                r["oracle_ms"] / r["greedy_ms"] for r in subset if r["greedy_ms"] > 0
            ),
            "min_oracle_over_greedy": min(
                r["oracle_ms"] / r["greedy_ms"] for r in subset if r["greedy_ms"] > 0
            ),
            "share_greedy_optimal": sum(
                r["oracle_ms"] >= r["greedy_ms"] * (1 - 1e-9) for r in subset
            )
            / len(subset),
            "median_sampled_fraction_oracle": median(
                r["n_sampled_oracle"] / r["n_subjoins"] for r in subset
            ),
        }
        for name in native:
            values = [
                r[f"oracle_over_{name}"] for r in subset if f"oracle_over_{name}" in r
            ]
            out[f"{name}_n"] = len(values)
            out[f"{name}_median_oracle_over_exec"] = median(values)
            out[f"{name}_share_oracle_above_exec"] = sum(v > 1 for v in values) / len(
                values
            )
        return out

    summary = {
        "all_certified": block(rows),
        "choice_matters": block([r for r in rows if r["matters"]]),
        "choice_does_not_matter": block([r for r in rows if not r["matters"]]),
    }
    payload = {
        "experiment": "price_of_certification",
        "preregistered": False,
        "predictions": "docs/icde_2027/PREDICTIONS.md",
        "summary": summary,
        "per_query": rows,
    }
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    logger.info("%s", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
