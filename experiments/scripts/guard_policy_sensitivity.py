"""Leave-one-estimator-out sensitivity of the guard-policy tail (Table 4).

The worst 5% of unguarded pairs on STATS-CEB are dominated by one estimator, so
the headline p95 q-error reduction is re-computed with each estimator removed
in turn. Post-hoc (added after an internal review), descriptive only.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.guard_policy_sensitivity
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.scripts.guard_policy_ablation import apply_policies, load_pairs

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

POLICIES = (
    "no_guard",
    "upper_only",
    "lower_only",
    "bidirectional",
    "bidirectional_sound",
)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("results/utility_table/real_stats_bidirectional.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/utility_table/guard_policy_leave_one_out.json"),
    )
    parser.add_argument("--tau", type=float, default=2.0)
    args = parser.parse_args()

    pairs = load_pairs(args.table)
    summary, records = apply_policies(pairs, args.tau)
    ranked = sorted(records, key=lambda r: r["qerror_no_guard"], reverse=True)
    tail_size = round(0.05 * len(records))
    tail = Counter(r["proposal"] for r in ranked[:tail_size])

    def p95s(stats: dict[str, dict[str, Any]]) -> dict[str, float]:
        return {p: float(stats[p]["qerror_p95"]) for p in POLICIES}

    result: dict[str, Any] = {
        "experiment": "guard_policy_leave_one_out",
        "preregistered": False,
        "tau": args.tau,
        "n_pairs": len(pairs),
        "worst_5pct_unguarded": {"n": tail_size, "by_proposal": dict(tail)},
        "all": p95s(summary),
        "drop": {},
    }
    for proposal in sorted({p for _, p in pairs}):
        subset = {k: v for k, v in pairs.items() if k[1] != proposal}
        stats, _ = apply_policies(subset, args.tau)
        result["drop"][proposal] = {"n_pairs": len(subset), **p95s(stats)}
        logger.info("drop %-11s %s", proposal, result["drop"][proposal])
    guarded = [entry["bidirectional"] for entry in result["drop"].values()]
    result["bidirectional_p95_range_leave_one_out"] = [min(guarded), max(guarded)]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("tail by proposal %s; wrote %s", dict(tail), args.output)


if __name__ == "__main__":
    main()
