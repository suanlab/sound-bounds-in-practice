"""Summarise the witness-floor measurement (H7; ``witness_floor``).

Per sub-join of the hindsight-optimal certifying sets: the time of witnessing
the sampled floor with ``SELECT COUNT(*) FROM (... LIMIT k)`` over the time of
the sampled count of the same sub-join, both timed back to back in one run.
Reports the median ratio with a query-clustered interval, the share of
sub-joins where the witness is cheaper, censoring, and the summed times.
Exploratory; prediction in ``docs/icde_2027/PREDICTIONS.md`` (H7).

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.summarize_witness_floor
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
    """Median of witness time / sampled-count time."""
    return percentile([r["ratio"] for r in rows], 0.5)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=Path("results/pg_injection/v3/witness_floor.jsonl")
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v3/witness_floor_summary.json"),
    )
    args = parser.parse_args()

    records = _read_jsonl(args.runs)
    rows = [
        {
            "query_id": r["query_id"],
            "ratio": max(r["witness_ms"], 1e-3) / max(r["sample_ms"], 1e-3),
        }
        for r in records
        if not r["witness_censored"] and not r["sample_censored"]
    ]
    summary = {
        "inputs": {str(args.runs): hashlib.sha256(args.runs.read_bytes()).hexdigest()},
        "n_subjoins": len(records),
        "n_queries": len({r["query_id"] for r in records}),
        "n_censored": sum(
            r["witness_censored"] or r["sample_censored"] for r in records
        ),
        "median_witness_over_sample": cluster_bootstrap(
            rows, median_ratio, replicates=args.replicates, seed=args.seed
        ),
        "share_witness_cheaper": sum(r["ratio"] < 1 for r in rows) / len(rows),
        "total_witness_ms": sum(r["witness_ms"] for r in records),
        "total_sample_ms": sum(r["sample_ms"] for r in records),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("%s", json.dumps(summary))


if __name__ == "__main__":
    main()
