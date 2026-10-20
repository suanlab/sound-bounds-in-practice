"""How the selective-sampling cost ratios depend on the native-time reference.

``selective_sampling`` divides the time of the sampled counts (timed in the
sampling run) by the time PostgreSQL takes to execute its own native plan, taken
from the execution run. Two other v3 runs also time the same native plan: the
forced-order run and the in-run sampling-cost run. The three agree on the plan
but not on its time, so this recomputes the median cost ratios of selective and
of full sampling against each reference, on the queries all three timed, and
reports how the references differ (geometric mean of one time over another),
also split by whether the query started in the execution run at a one-minute
load above 30. Post-hoc.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.selective_denominators
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
from experiments.scripts.bootstrap_ci import cluster_bootstrap, percentile

if TYPE_CHECKING:
    from collections.abc import Callable

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

V3 = Path("results/pg_injection/v3")


def median_of(key: str) -> Callable[[list[dict[str, Any]]], float]:
    """Statistic: median of ``key`` over rows."""

    def stat(rows: list[dict[str, Any]]) -> float:
        return percentile([r[key] for r in rows], 0.5)

    return stat


def native_times(v3: Path) -> dict[str, dict[int, float]]:
    """Uncensored native-plan times per query in each of the three v3 runs."""
    execution = {}
    for run in _read_jsonl(v3 / "stats_inject.jsonl"):
        digest = run["policy_plan"]["native"]
        if not run["plan_censored"][digest]:
            execution[run["query_id"]] = float(run["plan_ms"][digest])
    forced = {
        r["query_id"]: float(r["ms"]["native"])
        for r in _read_jsonl(v3 / "forced_orders.jsonl")
        if not r["censored"]["native"]
    }
    inrun = {
        r["query_id"]: float(r["native_median_ms"])
        for r in _read_jsonl(v3 / "sampling_overhead_inrun.jsonl")
        if not r["native_censored"]
    }
    return {"execution": execution, "forced_order": forced, "in_run": inrun}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selective",
        type=Path,
        default=Path("results/plan_certificate/selective_sampling_v3.json"),
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/selective_denominators.json"),
    )
    args = parser.parse_args()

    per_query = {
        r["query_id"]: r for r in json.loads(args.selective.read_text())["per_query"]
    }
    times = native_times(V3)
    common = sorted(set(per_query).intersection(*(set(t) for t in times.values())))
    out: dict[str, Any] = {
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                args.selective,
                V3 / "stats_inject.jsonl",
                V3 / "forced_orders.jsonl",
                V3 / "sampling_overhead_inrun.jsonl",
            )
        },
        "n_common": len(common),
        "by_reference": {},
        "reference_ratio": {},
    }
    for name, native in times.items():
        rows = [
            {
                "query_id": q,
                "selective": per_query[q]["selective_ms"] / native[q],
                "full": per_query[q]["full_ms"] / native[q],
            }
            for q in common
        ]
        entry: dict[str, Any] = {}
        for key in ("selective", "full"):
            entry[key] = cluster_bootstrap(
                rows, median_of(key), replicates=args.replicates, seed=args.seed
            )
            entry[f"{key}_share_below_one"] = sum(r[key] < 1 for r in rows) / len(rows)
        out["by_reference"][name] = entry
        logger.info(
            "%-12s selective %.2fx [%.2f, %.2f]  full %.2fx [%.2f, %.2f]",
            name,
            entry["selective"]["point"],
            entry["selective"]["ci_low"],
            entry["selective"]["ci_high"],
            entry["full"]["point"],
            entry["full"]["ci_low"],
            entry["full"]["ci_high"],
        )
    base = times["execution"]
    burst = set(
        json.loads((V3 / "stats_protocol.json").read_text())["queries_start_above_30"]
    )
    out["reference_ratio_by_load"] = {}
    for name in ("forced_order", "in_run"):
        other = times[name]
        logs = [math.log(base[q] / other[q]) for q in common]
        out["reference_ratio"][f"execution/{name}"] = math.exp(sum(logs) / len(logs))
        for label, subset in (
            ("start_above_30", [q for q in common if q in burst]),
            ("rest", [q for q in common if q not in burst]),
        ):
            sub = [math.log(base[q] / other[q]) for q in subset]
            out["reference_ratio_by_load"][f"execution/{name}|{label}"] = {
                "n": len(sub),
                "geo": math.exp(sum(sub) / len(sub)),
            }
    logger.info("execution-run time over others: %s", out["reference_ratio"])
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
