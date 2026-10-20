"""Where PostgreSQL's own estimates fall relative to the sound bracket.

For every STATS-CEB sub-plan: is our instance's in-context estimate above
LpBound or below the sound sampled floor (r = 0.10)? Counted over all sub-plans
and over those whose estimate is off by more than 10x. A guard can only act on
estimates outside the bracket, and it moves them only to the bound.

Usage::

    python -m experiments.pg_injection.native_bracket
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from experiments.pg_injection.inject_plans import _read_jsonl, load_estimates

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def qerror(estimate: float, truth: float) -> float:
    """q-error with both sides floored at one row."""
    e, t = max(estimate, 1.0), max(truth, 1.0)
    return max(e / t, t / e)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native",
        type=Path,
        default=Path("results/pg_injection/v2/native_incontext_estimations.csv"),
    )
    parser.add_argument(
        "--samples", type=Path, default=Path("results/pg_injection/subset_counts.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v3/native_bracket.json"),
    )
    args = parser.parse_args()

    native = load_estimates("native", path=args.native)
    upper = load_estimates("lpbound")
    truth = load_estimates("truecardinality")
    floor = {
        (r["query_id"], tuple(r["aliases"])): float(r["lower_sound"])
        for r in _read_jsonl(args.samples)
        if not r["sample_censored"]
    }
    keys = [k for k in native if k in upper and k in truth]

    def outside(key: tuple[int, tuple[str, ...]]) -> bool:
        return native[key] > upper[key] or (key in floor and native[key] < floor[key])

    large = [k for k in keys if qerror(native[k], truth[k]) > 10]
    out = {
        "n_subplans": len(keys),
        "above_upper": sum(native[k] > upper[k] for k in keys),
        "below_floor": sum(k in floor and native[k] < floor[k] for k in keys),
        "n_large_errors": len(large),
        "large_errors_outside_bracket": sum(outside(k) for k in large),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("%s", out)


if __name__ == "__main__":
    main()
