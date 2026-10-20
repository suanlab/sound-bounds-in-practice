"""Total sampling cost per query against the query's own execution time.

For each STATS-CEB query: the sum of the in-engine sampled-count times over all
its sub-plans (``subset_counts.jsonl``) divided by the median execution time of
PostgreSQL's native plan (``inject_plans_native.jsonl``). Queries with a
censored sample or a censored native execution are left out.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.sampling_overhead
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from experiments.pg_injection.inject_plans import _read_jsonl
from experiments.scripts.bootstrap_ci import cluster_bootstrap, percentile

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts", type=Path, default=Path("results/pg_injection/subset_counts.jsonl")
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/pg_injection/inject_plans_native.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/sampling_overhead.json"),
    )
    args = parser.parse_args()

    sample_ms: dict[int, float] = defaultdict(float)
    censored: dict[int, bool] = defaultdict(bool)
    for record in _read_jsonl(args.counts):
        sample_ms[record["query_id"]] += record["sample_ms"]
        censored[record["query_id"]] |= record["sample_censored"]
    rows, capped = [], []
    native_censored = sample_censored = 0
    for record in _read_jsonl(args.runs):
        query_id = record["query_id"]
        if query_id not in sample_ms:
            # Without this a query with no counts would cost 0 to sample.
            msg = f"q{query_id} has no sampled counts in {args.counts}"
            raise RuntimeError(msg)
        digest = record["policy_plan"]["native"]
        if record["plan_censored"][digest]:
            native_censored += 1
            continue
        row = {
            "query_id": query_id,
            "ratio": sample_ms[query_id] / record["plan_ms"][digest],
        }
        # Sensitivity: keep queries whose sample timed out, with the timed-out
        # counts at their cap (a lower bound on the sampling cost).
        capped.append(row)
        if censored[query_id]:
            sample_censored += 1
            continue
        rows.append(row)

    def median_ratio(rs: list[dict[str, float]]) -> float:
        return percentile([r["ratio"] for r in rs], 0.5)

    ratios = [r["ratio"] for r in rows]
    summary = {
        "n_queries": len(rows),
        "n_excluded_native_timeout": native_censored,
        "n_excluded_sample_timeout": sample_censored,
        "median": cluster_bootstrap(rows, median_ratio),
        "share_above_one": sum(x > 1 for x in ratios) / len(ratios),
        "with_sample_timeouts_at_cap": {
            "n_queries": len(capped),
            "median": cluster_bootstrap(capped, median_ratio),
            "share_above_one": sum(r["ratio"] > 1 for r in capped) / len(capped),
        },
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
