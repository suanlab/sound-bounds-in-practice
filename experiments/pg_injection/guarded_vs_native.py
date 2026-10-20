"""Guarded plans against PostgreSQL's own native plans.

``analyze_injection`` compares each guard with the source's own injected
estimates. This compares the guarded plans of the published PostgreSQL
estimates directly with PostgreSQL's native, non-injected plan, per benchmark:
the geometric-mean speedup native time / guarded time (< 1: the guarded plans
are still slower than PostgreSQL's own) with query-clustered intervals.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.guarded_vs_native
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import _read_jsonl
from experiments.scripts.bootstrap_ci import cluster_bootstrap

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

POLICIES = ("postgres_raw", "postgres_upper", "postgres_lower", "postgres_bi")


def geo(rows: list[dict[str, Any]]) -> float:
    """Geometric mean of native / policy."""
    return math.exp(sum(math.log(r["native"] / r["policy"]) for r in rows) / len(rows))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("results/pg_injection/v3")
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out: dict[str, Any] = {}
    for benchmark in ("stats", "joblight"):
        records = _read_jsonl(args.runs_dir / f"{benchmark}_inject.jsonl")
        entry: dict[str, Any] = {}
        for policy in POLICIES:
            rows = [
                {
                    "query_id": r["query_id"],
                    "native": max(r["plan_ms"][r["policy_plan"]["native"]], 1e-3),
                    "policy": max(r["plan_ms"][r["policy_plan"][policy]], 1e-3),
                }
                for r in records
                if policy in r["policy_plan"]
            ]
            entry[policy] = cluster_bootstrap(
                rows, geo, replicates=args.replicates, seed=args.seed
            )
            logger.info(
                "%s native->%s %.2fx [%.2f, %.2f]",
                benchmark,
                policy,
                entry[policy]["point"],
                entry[policy]["ci_low"],
                entry[policy]["ci_high"],
            )
        out[benchmark] = entry
    output = args.runs_dir / "guarded_vs_native.json"
    output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
