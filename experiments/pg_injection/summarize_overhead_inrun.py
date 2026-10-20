"""Summarise the single-run sampling-cost measurement (``sampling_overhead_inrun``).

Per STATS-CEB query, the median time of sampling every sub-plan (r = 0.10)
divided by the median time PostgreSQL takes to execute its own plan, both timed
back to back in one run. Reports the median ratio with a query-clustered
interval and the share of queries where sampling costs more than the query, on
the queries without a timed-out sample or native plan, and with timed-out
samples kept at their cap (a lower bound on their cost).

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.summarize_overhead_inrun
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import _read_jsonl
from experiments.scripts.bootstrap_ci import cluster_bootstrap, percentile

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def median_ratio(rows: list[dict[str, Any]]) -> float:
    """Median over queries of sampling time / execution time."""
    return percentile([r["ratio"] for r in rows], 0.5)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/pg_injection/v3/sampling_overhead_inrun.jsonl"),
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v3/sampling_overhead_inrun_summary.json"),
    )
    args = parser.parse_args()

    records = _read_jsonl(args.runs)
    clean, capped = [], []
    native_censored = sample_censored = 0
    for record in records:
        if record["native_censored"]:
            native_censored += 1
            continue
        row = {
            "query_id": record["query_id"],
            "ratio": record["sample_median_ms"] / record["native_median_ms"],
        }
        capped.append(row)
        if record["sample_censored"]:
            sample_censored += 1
            continue
        clean.append(row)

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n_queries": len(rows),
            "median": cluster_bootstrap(
                rows, median_ratio, replicates=args.replicates, seed=args.seed
            ),
            "share_above_one": sum(r["ratio"] > 1 for r in rows) / len(rows),
        }

    summary = {
        "inputs": {str(args.runs): hashlib.sha256(args.runs.read_bytes()).hexdigest()},
        "n_queries_measured": len(records),
        "n_excluded_native_timeout": native_censored,
        "n_excluded_sample_timeout": sample_censored,
        "n_high_load": sum(r["high_load"] for r in records),
        "clean": block(clean),
        "with_sample_timeouts_at_cap": block(capped),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("%s", json.dumps(summary)[:700])


if __name__ == "__main__":
    main()
