"""Where a sound guard loses its effect: a query-level funnel (STATS-CEB, v3).

For an estimate source and a guard, count the queries that pass each gate on
the way from an estimation error to a faster plan:

1. some join sub-plan is misestimated by more than 10x (an error to repair);
2. such a sub-plan lies outside the bracket on the guard's side (the bound sees
   it: above LpBound for the upper guard, below the in-engine sampled floor at
   r = 0.10 for the floor, either for both);
3. the guard changes the plan PostgreSQL chooses;
4. the changed plan does less work (true ``C_out`` of the executed plan falls
   by more than 10%);
5. the changed plan runs faster beyond the +-10% band.

Exploratory: it organises measurements already in the execution run.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.guard_funnel
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import _read_jsonl, load_estimates
from experiments.pg_injection.run_forced_orders import join_sets

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SOURCES = {
    "native": Path("results/pg_injection/v2/native_incontext_estimations.csv"),
    "postgres": None,
}
GUARDS = {"upper": ("upper",), "lower": ("lower",), "bi": ("upper", "lower")}
LARGE = 10.0
BAND = 1.1


def qerror(estimate: float, truth: float) -> float:
    """q-error with both sides floored at one row."""
    e, t = max(estimate, 1.0), max(truth, 1.0)
    return max(e / t, t / e)


def true_cout(plan: dict[str, Any], query_id: int, truth: dict[Any, float]) -> float:
    """True C_out of a plan: the sum of the true sizes of its join nodes."""
    sets: list[frozenset[str]] = []
    join_sets(plan, sets)
    return sum(max(truth[(query_id, tuple(sorted(s)))], 1.0) for s in sets)


def seen(
    key: tuple[int, tuple[str, ...]],
    sides: tuple[str, ...],
    estimates: dict[tuple[int, tuple[str, ...]], float],
    upper: dict[tuple[int, tuple[str, ...]], float],
    floor: dict[tuple[int, tuple[str, ...]], float],
) -> bool:
    """Whether the bracket sees the estimate's error on one of the guard's sides."""
    above = "upper" in sides and key in upper and estimates[key] > upper[key]
    below = "lower" in sides and key in floor and estimates[key] < floor[key]
    return above or below


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=Path("results/pg_injection/v3/stats_inject.jsonl")
    )
    parser.add_argument(
        "--samples", type=Path, default=Path("results/pg_injection/subset_counts.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v3/guard_funnel.json"),
    )
    args = parser.parse_args()

    truth = load_estimates("truecardinality")
    upper = load_estimates("lpbound")
    floor = {
        (r["query_id"], tuple(sorted(r["aliases"]))): float(r["lower_sound"])
        for r in _read_jsonl(args.samples)
        if not r["sample_censored"]
    }
    records = _read_jsonl(args.runs)
    out: dict[str, Any] = {"n_queries": len(records)}
    for source, path in SOURCES.items():
        estimates = load_estimates(source, path=path)
        for guard, sides in GUARDS.items():
            counts = dict.fromkeys(
                (
                    "error",
                    "seen",
                    "plan_changed",
                    "less_work",
                    "faster",
                    "faster_and_less_work",
                    "slower",
                ),
                0,
            )
            for record in records:
                qid = record["query_id"]
                keys = [k for k in estimates if k[0] == qid and k in truth]
                large = [k for k in keys if qerror(estimates[k], truth[k]) > LARGE]
                if not large:
                    continue
                counts["error"] += 1

                if not any(seen(k, sides, estimates, upper, floor) for k in large):
                    continue
                counts["seen"] += 1
                plan_of = record["policy_plan"]
                base, guarded = plan_of[f"{source}_raw"], plan_of[f"{source}_{guard}"]
                if base == guarded:
                    continue
                counts["plan_changed"] += 1
                c_base = true_cout(record["plans"][base], qid, truth)
                c_guard = true_cout(record["plans"][guarded], qid, truth)
                counts["less_work"] += c_base / c_guard > BAND
                speedup = record["plan_ms"][base] / record["plan_ms"][guarded]
                counts["faster"] += speedup > BAND
                counts["faster_and_less_work"] += (
                    speedup > BAND and c_base / c_guard > BAND
                )
                counts["slower"] += speedup < 1 / BAND
            out[f"{source}|{guard}"] = counts
            logger.info("%-16s %s", f"{source}|{guard}", counts)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
