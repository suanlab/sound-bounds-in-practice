"""Sampling rate r = 0.10 against r = 0.25: cost, accuracy and certificates.

Accuracy and certificates use every sub-plan: a count is the same under any
host load, although more samples time out under load (they then count as
L = 1). Cost compares the two rates on the same sub-plans, and only on
those whose r = 0.25 timing was taken while the host was quiet (1-min load at
most ``--max-load``): the r = 0.25 run shared the host with heavy jobs for most
of its duration, which leaves its counts valid but its timings unusable. The
r = 0.10 run recorded no load, so only the r = 0.25 side can be filtered.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.sampling_rate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import median
from typing import Any

from experiments.pg_injection.subset_counts import _read_jsonl
from experiments.scripts.bootstrap_ci import cluster_bootstrap

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

R = Path("results")


def _ratio(key: str) -> Any:
    """Median over rows of the sampled / exact time at one rate."""

    def statistic(rows: list[dict[str, Any]]) -> float:
        return float(median(r[key] for r in rows))

    return statistic


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--low", type=Path, default=R / "pg_injection/subset_counts.jsonl"
    )
    parser.add_argument(
        "--high", type=Path, default=R / "pg_injection/subset_counts_r025.jsonl"
    )
    parser.add_argument("--max-load", type=float, default=60.0)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=R / "pg_injection/v2/sampling_rate.json"
    )
    args = parser.parse_args()

    low = {(r["query_id"], tuple(r["aliases"])): r for r in _read_jsonl(args.low)}
    rows: list[dict[str, Any]] = []
    for high in _read_jsonl(args.high):
        base = low.get((high["query_id"], tuple(high["aliases"])))
        if base is None or high["loadavg_1m"] > args.max_load:
            continue
        if any(
            r[f] for r in (base, high) for f in ("sample_censored", "exact_censored")
        ):
            continue
        rows.append(
            {
                "query_id": high["query_id"],
                "exact_ms_low": base["exact_ms"],
                "cost_low": base["sample_ms"] / base["exact_ms"],
                "cost_high": high["sample_ms"] / high["exact_ms"],
            }
        )
    if not rows:
        msg = f"no sub-plan timed at load <= {args.max_load} at both rates"
        raise RuntimeError(msg)
    expensive = [r for r in rows if r["exact_ms_low"] > 1_000]
    out: dict[str, Any] = {
        "max_load": args.max_load,
        "n_subplans": len(rows),
        "n_queries": len({r["query_id"] for r in rows}),
        "max_load_seen": max(
            h["loadavg_1m"]
            for h in _read_jsonl(args.high)
            if h["loadavg_1m"] <= args.max_load
        ),
        "n_high_load_rows_excluded": sum(
            h["loadavg_1m"] > args.max_load for h in _read_jsonl(args.high)
        ),
    }
    for label, subset in (("all", rows), ("exact_over_1s", expensive)):
        out[label] = {
            "n_subplans": len(subset),
            "cost_r010": cluster_bootstrap(
                subset, _ratio("cost_low"), replicates=args.replicates, seed=args.seed
            ),
            "cost_r025": cluster_bootstrap(
                subset, _ratio("cost_high"), replicates=args.replicates, seed=args.seed
            ),
        }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("%s", json.dumps(out)[:800])


if __name__ == "__main__":
    main()
