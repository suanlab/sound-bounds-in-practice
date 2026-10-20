"""Post-hoc analyses of the plan-certificate run --- NOT pre-registered.

Two questions the pre-registered result (``plan_certificate.json``) raises, and
which were not in ``docs/c4_preregistration.md``; they are reported as
exploratory and never as findings.

1. **Censoring sensitivity.** Sub-joins whose sampling lower certificate ran
   past the 30 s budget were floored at $L = 1$, which can only widen the
   bracket. Recomputing the primary statistics on the queries with *no*
   censored sub-join says whether the vacuous result is a budget artefact.

2. **The minimax plan.** The plan minimising the upper-bound cost
   $C(p, B)$ needs no estimator at all. Its regret was a pre-registered
   *output*; comparing it to the engine estimators' picks was not a
   pre-registered hypothesis, so the paired comparison here is post-hoc.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.plan_certificate_posthoc
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from statistics import median
from typing import Any

from experiments.scripts.bootstrap_ci import cluster_bootstrap

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PRIMARY = "lpboundxsample0.25"
ESTIMATORS = ("duckdb", "postgres")


def _rows(per_query: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    rows = []
    for query in per_query:
        record = query["brackets"][key]
        if record["flagged"]:
            continue
        row = {
            "query_id": query["query_id"],
            "censored": sum(query["sample_censored"].values()) > 0,
            "certified": record["certified"],
            "pruning_rate": record["pruning_rate"],
            "G": record["G"],
            "width": record["width"],
            "minimax_regret": record["minimax_regret"],
            "minimax_bound": record["minimax_bound"],
        }
        for name in ESTIMATORS:
            row[f"{name}_regret_free"] = record["estimators"][name]["regret_free"]
        rows.append(row)
    return rows


def _mean(field: str) -> Any:
    return lambda rows: sum(float(r[field]) for r in rows) / len(rows)


def _share_equal(field: str, target: float) -> Any:
    return lambda rows: sum(1.0 for r in rows if r[field] == target) / len(rows)


def _median(field: str) -> Any:
    return lambda rows: float(median(float(r[field]) for r in rows))


def _p95(field: str) -> Any:
    def statistic(rows: list[dict[str, Any]]) -> float:
        ordered = sorted(float(r[field]) for r in rows)
        position = 0.95 * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return statistic


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/plan_certificate/plan_certificate.json"),
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/plan_certificate_posthoc.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    rows = _rows(payload["per_query"], PRIMARY)
    uncensored = [r for r in rows if not r["censored"]]
    boot = dict(replicates=args.replicates, seed=args.seed)

    censoring = {
        "n_queries": len(rows),
        "n_uncensored_queries": len(uncensored),
        "uncensored": {
            "certified_share": cluster_bootstrap(
                uncensored, _mean("certified"), **boot
            ),
            "mean_pruning_rate": cluster_bootstrap(
                uncensored, _mean("pruning_rate"), **boot
            ),
            "median_G": cluster_bootstrap(uncensored, _median("G"), **boot),
            "median_width": cluster_bootstrap(uncensored, _median("width"), **boot),
        },
    }

    minimax: dict[str, Any] = {
        "median_regret": cluster_bootstrap(rows, _median("minimax_regret"), **boot),
        "p95_regret": cluster_bootstrap(rows, _p95("minimax_regret"), **boot),
        "max_regret": max(r["minimax_regret"] for r in rows),
        # What the bracket can *prove* about its own plan, min C_hi / min C_lo.
        "median_bound": cluster_bootstrap(rows, _median("minimax_bound"), **boot),
        "p95_bound": cluster_bootstrap(rows, _p95("minimax_bound"), **boot),
        "optimal_plan_rate": cluster_bootstrap(
            rows, _share_equal("minimax_regret", 1.0), **boot
        ),
    }
    for name in ESTIMATORS:
        field = f"{name}_regret_free"
        paired = [
            {
                "query_id": r["query_id"],
                "log_ratio": math.log(r["minimax_regret"] / r[field]),
                "better": r["minimax_regret"] < r[field],
                "worse": r["minimax_regret"] > r[field],
            }
            for r in rows
        ]
        minimax[f"vs_{name}"] = {
            "estimator_p95_regret": cluster_bootstrap(rows, _p95(field), **boot),
            "estimator_optimal_plan_rate": cluster_bootstrap(
                rows, _share_equal(field, 1.0), **boot
            ),
            "share_minimax_strictly_better": cluster_bootstrap(
                paired, _mean("better"), **boot
            ),
            "share_minimax_strictly_worse": cluster_bootstrap(
                paired, _mean("worse"), **boot
            ),
            "mean_log_regret_ratio": cluster_bootstrap(
                paired, _mean("log_ratio"), **boot
            ),
        }

    out = {
        "experiment": "plan_certificate_posthoc",
        "status": "POST-HOC: not pre-registered; exploratory, not a finding",
        "bracket": PRIMARY,
        "censoring_sensitivity": censoring,
        "minimax_plan": minimax,
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    logger.info(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
