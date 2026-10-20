"""Our execution results against the LpBound artifact's own end-to-end runs.

The artifact reports, per query, the runtime of PostgreSQL planning on each
estimator's sub-plan estimates (``results/evaluation_time/<benchmark>``). Its
``postgres`` runs plan on that instance's own estimates; ours inject the same
published numbers into another PostgreSQL 13.1 instance, whose own estimates
differ. This script puts the two side by side: geometric-mean speedups with
query-clustered intervals for the same comparisons (PostgreSQL to the true
sizes, to LpBound; the true sizes to LpBound), and how the published
PostgreSQL sub-plan estimates compare with our instance's.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.artifact_replication
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from experiments.pg_injection.inject_plans import (
    WORKLOAD_PATHS,
    _read_jsonl,
    load_estimates,
)
from experiments.scripts.bootstrap_ci import cluster_bootstrap

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_TIMES = Path("external/LpBound/results/evaluation_time")
RUNS = {
    "stats": Path("results/pg_injection/v3/stats_inject.jsonl"),
    "joblight": Path("results/pg_injection/v3/joblight_inject.jsonl"),
}
NATIVE_CSV = {
    # PostgreSQL's own estimates read inside the full query.
    "stats": Path("results/pg_injection/v2/native_incontext_estimations.csv"),
    "joblight": Path(
        "results/pg_injection/v2/native_incontext_estimations_joblight.csv"
    ),
}
# (label, artifact baseline, artifact policy, our baseline, our policy)
COMPARISONS = (
    ("pg_to_truth", "postgres", "truecardinality", "native", "truth"),
    ("pg_to_lpbound", "postgres", "lpbound", "native", "lpbound_est"),
    ("truth_to_lpbound", "truecardinality", "lpbound", "truth", "lpbound_est"),
    ("published_to_truth", None, None, "postgres_raw", "truth"),
)


def artifact_runtimes(benchmark: str, name: str) -> dict[int, float]:
    """Per-query runtime (s) of one artifact estimator."""
    path = ARTIFACT_TIMES / benchmark / f"{name}_evaluation_time.csv"
    with path.open(newline="") as handle:
        return {int(r["QueryID"]): float(r["Runtime"]) for r in csv.DictReader(handle)}


def geo(rows: list[dict[str, Any]]) -> float:
    """Geometric mean of the per-query ratios base / policy."""
    return math.exp(sum(math.log(r["base"] / r["policy"]) for r in rows) / len(rows))


def compare(pairs: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    """Geometric-mean speedup with a query-clustered interval."""
    return cluster_bootstrap(pairs, geo, replicates=replicates, seed=seed)


def estimate_gap(benchmark: str) -> dict[str, Any]:
    """Published PostgreSQL sub-plan estimates against our instance's own."""
    directory = WORKLOAD_PATHS[benchmark][0]
    published = load_estimates("postgres", directory=directory)
    native = load_estimates("native", path=NATIVE_CSV[benchmark])
    truth = load_estimates("truecardinality", directory=directory)
    keys = sorted(set(published) & set(native) & set(truth))

    def qerr(e: float, t: float) -> float:
        e, t = max(e, 1.0), max(t, 1.0)
        return max(e / t, t / e)

    by_size: dict[int, list[float]] = defaultdict(list)
    for key in keys:
        by_size[len(key[1])].append(max(published[key], 1.0) / max(native[key], 1.0))
    return {
        "n_subplans": len(keys),
        "published_over_native_median": median(
            max(published[k], 1.0) / max(native[k], 1.0) for k in keys
        ),
        "published_over_native_median_by_relations": {
            str(n): median(v) for n, v in sorted(by_size.items())
        },
        "qerror_median_published": median(qerr(published[k], truth[k]) for k in keys),
        "qerror_median_native": median(qerr(native[k], truth[k]) for k in keys),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v3/artifact_replication.json"),
    )
    args = parser.parse_args()

    out: dict[str, Any] = {}
    for benchmark, runs in RUNS.items():
        ours: dict[int, dict[str, float]] = {}
        for record in _read_jsonl(runs):
            ours[record["query_id"]] = {
                policy: record["plan_ms"][digest] / 1000.0
                for policy, digest in record["policy_plan"].items()
            }
        entry: dict[str, Any] = {"estimates": estimate_gap(benchmark)}
        for label, a_base, a_policy, o_base, o_policy in COMPARISONS:
            row: dict[str, Any] = {}
            if a_base is not None and a_policy is not None:
                base, policy = (
                    artifact_runtimes(benchmark, a_base),
                    artifact_runtimes(benchmark, a_policy),
                )
                pairs = [
                    {"query_id": q, "base": base[q], "policy": policy[q]}
                    for q in sorted(set(base) & set(policy))
                ]
                row["artifact"] = compare(pairs, args.replicates, args.seed)
            pairs = [
                {"query_id": q, "base": t[o_base], "policy": t[o_policy]}
                for q, t in sorted(ours.items())
            ]
            row["ours"] = compare(pairs, args.replicates, args.seed)
            entry[label] = row
        out[benchmark] = entry
        logger.info("%s: %s", benchmark, json.dumps(entry, default=str)[:600])
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
