"""Summarise the executed tightness-sweep points (``execute_tightness_points``).

For our instance's in-context estimates on STATS-CEB: the geometric-mean speedup
over the raw-estimate plan of planning on the true sizes and of both synthetic
guards at ``k = 2`` and ``k = 8`` (query-clustered 95% intervals), the share of
the true-size speedup each guard recovers (log scale), the shares of queries
faster and slower beyond +-10%, the same without queries that started above a
one-minute load of 30, and, for comparison, the true-``C_out`` shares the
EXPLAIN-only sweep gives on the same queries. Exploratory (H3 confirmation in
time; ``docs/icde_2027/PREDICTIONS.md``).

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.summarize_tightness_points
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.pg_injection.inject_plans import _read_jsonl
from experiments.scripts.bootstrap_ci import cluster_bootstrap

if TYPE_CHECKING:
    from collections.abc import Callable

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

POLICIES = ("truth", "bi_k2", "bi_k8")
BAND = 1.1
QUIET_LOAD = 30.0


def geo(field: str) -> Callable[[list[dict[str, Any]]], float]:
    """Statistic: geometric mean of raw / policy time."""

    def stat(rows: list[dict[str, Any]]) -> float:
        return math.exp(sum(r[field] for r in rows) / len(rows))

    return stat


def recovered(field: str) -> Callable[[list[dict[str, Any]]], float]:
    """Statistic: share of the true-size speedup recovered (log scale)."""

    def stat(rows: list[dict[str, Any]]) -> float:
        total = sum(r["truth"] for r in rows)
        return sum(r[field] for r in rows) / total if total > 0 else math.nan

    return stat


def block(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    """Speedups, recovered shares and faster/slower shares on a set of queries."""
    out: dict[str, Any] = {"n": len(rows)}
    for name in POLICIES:
        out[name] = {
            "geo_speedup": cluster_bootstrap(
                rows, geo(name), replicates=replicates, seed=seed
            ),
            "share_faster": sum(r[name] > math.log(BAND) for r in rows) / len(rows),
            "share_slower": sum(r[name] < -math.log(BAND) for r in rows) / len(rows),
        }
        if name != "truth":
            out[name]["share_of_truth_speedup"] = cluster_bootstrap(
                rows, recovered(name), replicates=replicates, seed=seed
            )
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/pg_injection/v3/tightness_points_stats.jsonl"),
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("results/pg_injection/v3/tightness_sweep_stats.jsonl"),
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v3/tightness_points_summary.json"),
    )
    args = parser.parse_args()

    records = _read_jsonl(args.runs)
    rows = []
    for r in records:
        ms = {n: max(r["plan_ms"][d], 1e-3) for n, d in r["policy_plan"].items()}
        rows.append(
            {
                "query_id": r["query_id"],
                "quiet": r["loadavg_1m_start"] <= QUIET_LOAD,
                **{n: math.log(ms["raw"] / ms[n]) for n in POLICIES},
            }
        )
    sweep = {
        r["query_id"]: r["native"]["cout"]
        for r in _read_jsonl(args.sweep)
        if "native" in r
    }
    cout_rows = [
        {
            "query_id": q,
            **{n: math.log(sweep[q]["raw"] / sweep[q][n]) for n in POLICIES},
        }
        for q in (r["query_id"] for r in rows)
        if q in sweep
    ]
    summary = {
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.runs, args.sweep)
        },
        "n_queries": len(records),
        "n_censored_plans": sum(sum(r["plan_censored"].values()) for r in records),
        "n_high_load": sum(r["high_load"] for r in records),
        "time": block(rows, args.replicates, args.seed),
        "time_quiet_start": block(
            [r for r in rows if r["quiet"]], args.replicates, args.seed
        ),
        "cout_same_queries": {
            name: recovered(name)(cout_rows) for name in ("bi_k2", "bi_k8")
        },
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("%s", json.dumps(summary)[:1500])


if __name__ == "__main__":
    main()
