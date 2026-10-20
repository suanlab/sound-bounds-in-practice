"""Why guarding helps some plans and hurts others: the clamp rate.

The runtime experiment (``plan_runtime``) found the guard to be high-variance at
the plan level: large speedups on a few queries, larger aggregate slowdown. This
script tests the mechanism we suspect from the data.

Plan choice depends on the *ranking* of subplan estimates, not on their absolute
accuracy. The guard replaces an estimate by the bound only where it fires, so it
perturbs that ranking non-uniformly. The prediction is therefore:

  * clamping *few* subplans surgically removes one catastrophic overestimate and
    improves the ranking -> speedup;
  * clamping *many* subplans rewrites much of the ranking -> the chosen order is
    effectively arbitrary -> slowdown.

For every (query, configuration) pair that ``plan_runtime`` timed, we recompute
the fraction of connected subplans the guard clamps and correlate it with the
measured speedup.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.clamp_rate_analysis
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
from pathlib import Path
from statistics import median
from typing import Any

from experiments.scripts.plan_regret import _connected, _sub_query, time_limit
from src.core.ceg import pessimistic_bound
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.utils.job_eval import build_job_query
from src.utils.job_sql_parser import parse_job_sql

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def clamp_rate(
    sql: str,
    query_id: str,
    config: str,
    rows: dict[str, list[dict[str, str]]],
    tau: float,
) -> tuple[float, int, int] | None:
    """Fraction of connected subplans whose estimate the guard replaces."""
    parsed = parse_job_sql(sql, query_id)
    aliases = sorted(parsed.aliases)
    full = build_job_query(parsed, rows)
    estimator = BaselineCEGEstimator()
    total = clamped = 0
    for size in range(2, len(aliases) + 1):
        for subset in itertools.combinations(aliases, size):
            if not _connected(subset, parsed):
                continue
            sub = _sub_query(full, subset)
            if sub is None:
                continue
            bound = max(pessimistic_bound(sub), 1.0)
            estimate = max(estimator.estimate_all_configs(sub)[config].estimate, 1.0)
            total += 1
            clamped += int(estimate > tau * bound)
    if total == 0:
        return None
    return clamped / total, clamped, total


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (ties averaged)."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("external/LpBound")
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("results/plan_regret/plan_runtime.json"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=base / "benchmarks/workloads/stats/statsQueries.sql",
    )
    parser.add_argument("--data-dir", type=Path, default=base / "data/datasets/stats")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--query-timeout-s", type=float, default=90.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/plan_regret/clamp_rate.json")
    )
    args = parser.parse_args()

    timed = json.loads(args.runtime.read_text())["per_pair"]
    queries = [
        line.strip() for line in args.queries.read_text().splitlines() if line.strip()
    ]
    cache: dict[str, list[dict[str, str]]] = {}
    records: list[dict[str, Any]] = []

    for pair in timed:
        index = int(pair["query_id"])
        sql = queries[index - 1]
        parsed = parse_job_sql(sql, str(index))
        try:
            with time_limit(args.query_timeout_s):
                for table in parsed.base_relations:
                    if table not in cache:
                        with (args.data_dir / f"{table}.csv").open(newline="") as fh:
                            cache[table] = list(csv.DictReader(fh))
                result = clamp_rate(
                    sql,
                    str(index),
                    pair["config"],
                    {t: cache[t] for t in parsed.base_relations},
                    args.tau,
                )
        except Exception:
            continue
        if result is None:
            continue
        rate, clamped, total = result
        records.append(
            {
                "query_id": index,
                "config": pair["config"],
                "clamp_rate": rate,
                "clamped": clamped,
                "subplans": total,
                "speedup": pair["speedup_guarded_over_no_guard"],
            }
        )

    rates = [r["clamp_rate"] for r in records]
    speedups = [r["speedup"] for r in records]
    rho = spearman(rates, speedups)
    low = [r for r in records if r["clamp_rate"] <= 0.25]
    high = [r for r in records if r["clamp_rate"] > 0.25]
    payload = {
        "experiment": "clamp_rate_vs_plan_speedup",
        "hypothesis": (
            "Plan choice depends on the ranking of subplan estimates. Clamping few "
            "subplans fixes the worst outlier and improves the ranking; clamping "
            "many rewrites the ranking and the chosen order becomes arbitrary."
        ),
        "tau": args.tau,
        "n_pairs": len(records),
        "spearman_clamp_rate_vs_speedup": rho,
        "low_clamp_rate_le_25pct": {
            "n": len(low),
            "median_speedup": median(r["speedup"] for r in low) if low else None,
            "n_faster": sum(1 for r in low if r["speedup"] > 1.0),
        },
        "high_clamp_rate_gt_25pct": {
            "n": len(high),
            "median_speedup": median(r["speedup"] for r in high) if high else None,
            "n_faster": sum(1 for r in high if r["speedup"] > 1.0),
        },
        "per_pair": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    logger.info("n=%d pairs, Spearman(clamp rate, speedup) = %+.3f", len(records), rho)
    for label, group in (("clamp <=25%", low), ("clamp  >25%", high)):
        if group:
            logger.info(
                "  %s: n=%3d  median speedup %.2fx  faster in %d/%d",
                label,
                len(group),
                median(r["speedup"] for r in group),
                sum(1 for r in group if r["speedup"] > 1.0),
                len(group),
            )
    logger.info("-> %s", args.output)


if __name__ == "__main__":
    main()
