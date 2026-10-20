"""Summarise phase 1 (``subset_counts.jsonl``) into a citable JSON.

Truth agreement and soundness of the in-engine sampled lower bound, the cost of
the sampled count relative to the exact count of the same sub-plan in the same
engine (overall and by how long the exact count takes), and the accuracy of
``L_hat``. Intervals resample whole queries (all sub-plans of a query move
together).

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.summarize_counts
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.pg_injection.inject_plans import _read_jsonl
from experiments.scripts.bootstrap_ci import cluster_bootstrap, percentile

if TYPE_CHECKING:
    from collections.abc import Callable

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BUCKETS = ((0.0, 100.0), (100.0, 1_000.0), (1_000.0, 10_000.0), (10_000.0, 1e12))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts", type=Path, default=Path("results/pg_injection/subset_counts.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/subset_counts_summary.json"),
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = _read_jsonl(args.counts)
    finished = [r for r in rows if not r["exact_censored"]]
    both = [r for r in finished if not r["sample_censored"]]
    for r in both:
        r["cost_ratio"] = r["sample_ms"] / r["exact_ms"]
        truth = max(float(r["truth_artifact"] or 1.0), 1.0)
        hat = max(float(r["lower_hat"]), 1.0)
        r["lhat_qerror"] = max(hat / truth, truth / hat)

    def median_of(field: str) -> Callable[[list[dict[str, Any]]], float]:
        return lambda rs: percentile([r[field] for r in rs], 0.5)

    def p95_of(field: str) -> Callable[[list[dict[str, Any]]], float]:
        return lambda rs: percentile([r[field] for r in rs], 0.95)

    boot = {"replicates": args.replicates, "seed": args.seed}
    summary: dict[str, Any] = {
        "n_subplans": len(rows),
        "n_queries": len({r["query_id"] for r in rows}),
        "exact_censored": sum(r["exact_censored"] for r in rows),
        "sample_censored": sum(r["sample_censored"] for r in rows),
        "truth_agreement": {
            "compared": sum(r["truth_artifact"] is not None for r in finished),
            "mismatches": sum(
                r["truth_artifact"] is not None and r["exact_pg"] != r["truth_artifact"]
                for r in finished
            ),
        },
        "lower_unsound": sum(
            not r["sample_censored"]
            and r["truth_artifact"] is not None
            and r["lower_sound"] > r["truth_artifact"]
            for r in rows
        ),
        "lower_zero": sum(1 for r in both if r["lower_sound"] == 0),
        "cost_ratio": {
            "n": len(both),
            "median": cluster_bootstrap(both, median_of("cost_ratio"), **boot),
            "p25": percentile([r["cost_ratio"] for r in both], 0.25),
            "p75": percentile([r["cost_ratio"] for r in both], 0.75),
            "share_above_one": sum(r["cost_ratio"] > 1 for r in both) / len(both),
        },
        "cost_ratio_by_exact_ms": {},
        "lhat_qerror": {
            "median": cluster_bootstrap(both, median_of("lhat_qerror"), **boot),
            "p95": cluster_bootstrap(both, p95_of("lhat_qerror"), **boot),
            "max": max(r["lhat_qerror"] for r in both),
        },
    }
    for low, high in BUCKETS:
        bucket = [r for r in both if low <= r["exact_ms"] < high]
        if bucket:
            summary["cost_ratio_by_exact_ms"][f"{low:g}-{high:g}"] = {
                "n": len(bucket),
                "median": percentile([r["cost_ratio"] for r in bucket], 0.5),
            }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(json.dumps(summary, indent=1)[:1500])


if __name__ == "__main__":
    main()
