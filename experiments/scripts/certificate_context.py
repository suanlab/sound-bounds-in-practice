"""Context for the joint certificate: where plan choice matters, and an a-priori bound.

Two questions the joint-certificate shares alone do not answer:

* **Does the choice matter?** A query whose worst left-deep order is within 2x
  of the best in true ``C_out`` is "certified" by any plan. We report the share
  of such queries and the certified share among the others.
* **What would a q-error argument certify?** Planning on the geometric midpoint
  ``sqrt(L B)`` of every bracket gives every intermediate result a q-error of at
  most ``q = max_S sqrt(B(S) / L(S))``. For a cost that sums intermediate sizes
  (``C_out``) the chosen plan ``p`` then satisfies
  ``C(p) <= q C_est(p) <= q C_est(p*) <= q^2 C(p*)``: an a-priori regret bound of
  ``q^2`` that needs no plan enumeration, in the spirit of Moerkotte et al.

Brackets: LpBound above, PostgreSQL's in-engine sampled count below (a timed-out
sample counts as ``L = 1``), on the sub-joins that left-deep orders use.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.certificate_context
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from experiments.scripts.bootstrap_ci import cluster_bootstrap
from experiments.scripts.plan_certificate_joint import left_deep_orders

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CASES = {
    "stats_r025": (
        "results/plan_guard/subjoin_sizes_maxdeg_fixed.jsonl",
        "results/pg_injection/subset_counts_r025.jsonl",
        "results/plan_certificate/plan_certificate_joint_pg_r025.json",
    ),
    "stats_r010": (
        "results/plan_guard/subjoin_sizes_maxdeg_fixed.jsonl",
        "results/pg_injection/subset_counts.jsonl",
        "results/plan_certificate/plan_certificate_joint_pg_r010.json",
    ),
    "joblight_r010": (
        "results/plan_certificate/joblight_subjoin_sizes.jsonl",
        "results/pg_injection/joblight/subset_counts.jsonl",
        "results/plan_certificate/plan_certificate_joint_joblight_pg_r010.json",
    ),
}


def true_spread(record: dict[str, Any]) -> float:
    """Worst over best true ``C_out`` of a query's left-deep orders."""
    truth = {
        frozenset(s["aliases"]): max(float(s["truth"]), 1.0) for s in record["subjoins"]
    }
    orders = left_deep_orders(record["aliases"], dict.fromkeys(truth, 0))
    costs = [
        sum(truth[frozenset(o[: i + 1])] for i in range(1, len(o))) for o in orders
    ]
    return max(costs) / min(costs)


def query_context(
    record: dict[str, Any], samples: dict[tuple[int, tuple[str, ...]], dict[str, Any]]
) -> dict[str, float]:
    """True-cost spread of the left-deep orders and the q-squared bound."""
    truth = {
        frozenset(s["aliases"]): max(float(s["truth"]), 1.0) for s in record["subjoins"]
    }
    upper = {
        frozenset(s["aliases"]): max(float(s["upper"]["lpbound"]), 1.0)
        for s in record["subjoins"]
    }
    orders = left_deep_orders(record["aliases"], dict.fromkeys(truth, 0))
    used = {frozenset(o[: i + 1]) for o in orders for i in range(1, len(o))}
    q = 1.0
    for key in used:
        sample = samples[(record["query_id"], tuple(sorted(key)))]
        lower = (
            1.0 if sample["sample_censored"] else max(float(sample["lower_sound"]), 1.0)
        )
        q = max(q, math.sqrt(max(upper[key], lower) / lower))
    return {"spread": true_spread(record), "q_squared": q * q}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/certificate_context.json"),
    )
    args = parser.parse_args()

    out: dict[str, Any] = {}
    for case, (sizes, sample_file, joint_file) in CASES.items():
        samples = {}
        for line in Path(sample_file).read_text().splitlines():
            row = json.loads(line)
            samples[(row["query_id"], tuple(sorted(row["aliases"])))] = row
        records = {
            r["query_id"]: r
            for r in (json.loads(x) for x in Path(sizes).read_text().splitlines() if x)
        }
        joint = json.loads(Path(joint_file).read_text())["per_query"][
            "lpboundxsample|cout"
        ]
        rows = []
        for result in joint:
            context = query_context(records[result["query_id"]], samples)
            rows.append(
                {
                    "query_id": result["query_id"],
                    "certified": result["joint_bound_best"] <= 2.0,
                    "matters": context["spread"] > 2.0,
                    "q_squared": context["q_squared"],
                }
            )
        matters = [r for r in rows if r["matters"]]
        if not matters:
            msg = f"{case}: no query where the choice of plan matters"
            raise RuntimeError(msg)
        out[case] = {
            "n_queries": len(rows),
            "n_choice_matters": len(matters),
            "share_all_orders_within_2x": sum(not r["matters"] for r in rows)
            / len(rows),
            "certified_where_choice_matters": cluster_bootstrap(
                matters,
                lambda rs: sum(r["certified"] for r in rs) / len(rs),
                replicates=args.replicates,
                seed=args.seed,
            ),
            "n_certified": sum(r["certified"] for r in rows),
            "n_certified_where_choice_matters": sum(r["certified"] for r in matters),
            "share_q_squared_le_2": sum(r["q_squared"] <= 2.0 for r in rows)
            / len(rows),
            "median_q_squared": sorted(r["q_squared"] for r in rows)[len(rows) // 2],
        }
        logger.info("%s: %s", case, json.dumps(out[case])[:400])
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
