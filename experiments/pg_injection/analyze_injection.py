"""Summarise the PostgreSQL injection runs (``inject_plans.jsonl``).

Every query contributes exactly one row per policy pair. For each pair it
reports the per-query speedup (baseline time / policy time), its geometric
mean, the share of queries faster and slower beyond a 10% noise band, and the
ratio of workload totals, with bootstrap intervals over queries. Alongside the
wall-clock figures it reports a machine-independent work measure: the true
C_out of the *executed* plan (sum of the exact sizes of its join nodes, from the
artifact's truth), so that conclusions do not rest on timings from a shared host
alone. Censored executions (statement timeout) are kept at the timeout and
counted.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.analyze_injection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from functools import partial
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import (
    WORKLOAD_PATHS,
    _read_jsonl,
    load_estimates,
)
from experiments.scripts.bootstrap_ci import cluster_bootstrap

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

JOINS = {"Hash Join", "Merge Join", "Nested Loop"}
SOURCE_POLICIES = ("upper", "upper_t1", "lower", "bi", "bi_hat")
SHARED_POLICIES = ("truth", "lhat_est", "lpbound_est")


def pairs_for(policies: set[str]) -> list[tuple[str, str]]:
    """Every (baseline, policy) comparison the run's policies support.

    For each estimate source ``X`` with injected values ``X_raw``: its guards
    against ``X_raw``, the shared policies (truth, L-hat, LpBound) against
    ``X_raw``, and ``X_raw`` against PostgreSQL's native plan. Plus the native
    plan against the truth and the truth against the two certificate-based
    plans.
    """
    sources = sorted(p[: -len("_raw")] for p in policies if p.endswith("_raw"))
    pairs: list[tuple[str, str]] = []
    for source in sources:
        raw = f"{source}_raw"
        pairs.append(("native", raw))
        pairs += [(raw, f"{source}_{g}") for g in SOURCE_POLICIES]
        pairs += [(raw, shared) for shared in SHARED_POLICIES]
    pairs += [("native", "truth"), ("truth", "lhat_est"), ("truth", "lpbound_est")]
    return [(b, p) for b, p in pairs if b in policies and p in policies]


Rows = list[dict[str, Any]]
_LHAT = "ms:lhat_est"


def geo_ratio(rows: Rows, *, base: str, policy: str, kind: str) -> float:
    """Geometric mean over queries of base / policy (time or true C_out)."""
    logs = [
        math.log(
            max(float(r[f"{kind}:{base}"]), 1e-9)
            / max(float(r[f"{kind}:{policy}"]), 1e-9)
        )
        for r in rows
    ]
    return math.exp(sum(logs) / len(logs))


def total_ratio(rows: Rows, *, base: str, policy: str) -> float:
    """Workload total time under base over total under policy."""
    return sum(float(r[f"ms:{base}"]) for r in rows) / sum(
        float(r[f"ms:{policy}"]) for r in rows
    )


def share(rows: Rows, *, base: str, policy: str, faster: bool) -> float:
    """Share of queries where policy is >10% faster (or slower) than base."""

    def hit(r: dict[str, Any]) -> bool:
        ratio = float(r[f"ms:{base}"]) / max(float(r[f"ms:{policy}"]), 1e-9)
        return ratio > 1.1 if faster else ratio < 1 / 1.1

    return sum(hit(r) for r in rows) / len(rows)


def true_cout(plan: dict[str, Any], query_id: int, truth: dict[Any, float]) -> float:
    """Sum of exact sizes over the join nodes of an executed plan."""

    def aliases(node: dict[str, Any]) -> set[str]:
        found = {node["Alias"]} if "Alias" in node else set()
        for child in node.get("Plans", []):
            found |= aliases(child)
        return found

    total = 0.0

    def walk(node: dict[str, Any]) -> None:
        nonlocal total
        if node.get("Node Type") in JOINS:
            key = (query_id, tuple(sorted(aliases(node))))
            total += truth.get(key, math.nan)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    return total


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=Path("results/pg_injection/inject_plans.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/injection_summary.json"),
    )
    parser.add_argument("--workload", choices=sorted(WORKLOAD_PATHS), default="stats")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    truth = load_estimates(
        "truecardinality", directory=WORKLOAD_PATHS[args.workload][0]
    )
    records = _read_jsonl(args.runs)
    per_query: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {"query_id": record["query_id"]}
        for policy, digest in record["policy_plan"].items():
            row[f"ms:{policy}"] = record["plan_ms"][digest]
            row[f"cens:{policy}"] = record["plan_censored"][digest]
            row[f"plan:{policy}"] = digest
            row[f"cout:{policy}"] = true_cout(
                record["plans"][digest], record["query_id"], truth
            )
        row["high_load"] = record.get("high_load", False)
        row["truth_ok"] = record["truth_injection_verified"]
        per_query.append(row)

    summary: dict[str, Any] = {
        # The audit re-hashes this file and rejects the summary if it changed.
        "inputs": {str(args.runs): hashlib.sha256(args.runs.read_bytes()).hexdigest()},
        "n_queries": len(per_query),
        "n_high_load": sum(r["high_load"] for r in per_query),
        "truth_injection_failures": [
            r["query_id"] for r in per_query if not r["truth_ok"]
        ],
        "pairs": {},
    }
    policies = {
        key.split(":", 1)[1]
        for row in per_query
        for key in row
        if key.startswith("ms:")
    }
    for base, policy in pairs_for(policies):
        rows = [r for r in per_query if f"ms:{base}" in r and f"ms:{policy}" in r]
        if not rows:
            continue

        stats = {
            "geo_speedup": partial(geo_ratio, base=base, policy=policy, kind="ms"),
            "total_ratio": partial(total_ratio, base=base, policy=policy),
            "share_faster": partial(share, base=base, policy=policy, faster=True),
            "share_slower": partial(share, base=base, policy=policy, faster=False),
        }
        # True C_out is undefined for a plan with a join outside the artifact's
        # sub-plan list; such queries are left out of the C_out ratio, and counted.
        cout_rows = [
            r
            for r in rows
            if not math.isnan(r[f"cout:{base}"]) and not math.isnan(r[f"cout:{policy}"])
        ]
        entry: dict[str, Any] = {
            "n": len(rows),
            "plans_differ": sum(r[f"plan:{base}"] != r[f"plan:{policy}"] for r in rows),
            "censored_base": sum(r[f"cens:{base}"] for r in rows),
            "censored_policy": sum(r[f"cens:{policy}"] for r in rows),
            "n_cout_undefined": len(rows) - len(cout_rows),
        }
        for name, statistic in stats.items():
            entry[name] = cluster_bootstrap(
                rows, statistic, replicates=args.replicates, seed=args.seed
            )
        entry["geo_cout_ratio"] = cluster_bootstrap(
            cout_rows,
            partial(geo_ratio, base=base, policy=policy, kind="cout"),
            replicates=args.replicates,
            seed=args.seed,
        )
        # L-hat policies exist only where the sample covers every sub-plan, so
        # every pair is also reported on that query set (a matched denominator
        # for comparisons with lhat_est) and on the queries it excludes.
        geo = stats["geo_speedup"]
        for label, subset in (
            ("matched_lhat", [r for r in rows if _LHAT in r]),
            ("without_lhat", [r for r in rows if _LHAT not in r]),
        ):
            entry[label] = {
                "n": len(subset),
                "geo_speedup": (
                    cluster_bootstrap(
                        subset, geo, replicates=args.replicates, seed=args.seed
                    )
                    if subset
                    else None
                ),
            }
        summary["pairs"][f"{base}->{policy}"] = entry
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(
        "%d queries (%d under high load)", summary["n_queries"], summary["n_high_load"]
    )
    for name, entry in summary["pairs"].items():
        g, tr = entry["geo_speedup"], entry["total_ratio"]
        logger.info(
            "%-30s n=%3d differ=%3d  geo %.2fx [%.2f, %.2f]  total %.2fx [%.2f, %.2f]  "
            "faster %.0f%% slower %.0f%%  cout %.2fx  cens %d/%d",
            name,
            entry["n"],
            entry["plans_differ"],
            g["point"],
            g["ci_low"],
            g["ci_high"],
            tr["point"],
            tr["ci_low"],
            tr["ci_high"],
            100 * entry["share_faster"]["point"],
            100 * entry["share_slower"]["point"],
            entry["geo_cout_ratio"]["point"],
            entry["censored_base"],
            entry["censored_policy"],
        )


if __name__ == "__main__":
    main()
