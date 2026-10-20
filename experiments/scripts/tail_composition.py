"""Error-direction composition of the worst-case tail.

Underestimation and overestimation do not occupy the same part of the error
distribution, and that difference explains why the two guard directions produce
such different median/tail profiles (Table~\\ref{tab:policy} in the paper).

For every (query, proposal) pair on real STATS-CEB we record the unguarded
q-error and its direction, then compare the composition of the worst ``k``
percent of pairs against the composition of all pairs. Everything here is a
direct measurement -- no plan-cost surrogate is involved.

Usage::

    python -m experiments.scripts.tail_composition --tail-fraction 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from statistics import median
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_UPPER_CERT = "lpbound"
_LOWER_CERT = "sample_lb"


def load_pairs(path: Path) -> list[tuple[float, int]]:
    """Return [(q_error, direction)] per (query, proposal) pair."""
    merged: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["query_id"], row["proposal"])
            entry = merged.setdefault(
                key,
                {
                    "estimate": float(row["estimate"]),
                    "true": float(row["true_cardinality"]),
                },
            )
            if row["certificate"] == _UPPER_CERT:
                entry["upper"] = float(row["bound"])
            elif row["certificate"] == _LOWER_CERT:
                entry["lhat"] = float(row["estimate"]) / math.exp(
                    float(row["tightness_proxy"])
                )
    out: list[tuple[float, int]] = []
    for entry in merged.values():
        if "upper" not in entry or "lhat" not in entry:
            continue
        y = max(entry["true"], 1.0)
        e = max(entry["estimate"], 1.0)
        out.append((max(e / y, y / e), 1 if e >= y else -1))
    return out


def summarize(pairs: list[tuple[float, int]], fraction: float) -> dict[str, Any]:
    """Compare direction composition of the worst ``fraction`` against all."""
    ordered = sorted(pairs, key=lambda t: -t[0])
    cut = max(1, int(len(ordered) * fraction))
    tail = ordered[:cut]

    def compose(subset: list[tuple[float, int]]) -> dict[str, Any]:
        under = sum(1 for _, d in subset if d == -1)
        return {
            "n": len(subset),
            "under": under,
            "over": len(subset) - under,
            "under_share": under / len(subset),
            "qerror_median": median(q for q, _ in subset),
        }

    return {
        "tail_fraction": fraction,
        "all_pairs": compose(ordered),
        "worst_tail": compose(tail),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("results/utility_table/real_stats_bidirectional.csv"),
    )
    parser.add_argument("--tail-fraction", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/utility_table/tail_composition.json"),
    )
    args = parser.parse_args()

    pairs = load_pairs(args.table)
    summary = summarize(pairs, args.tail_fraction)
    payload = {
        "experiment": "error_direction_composition_of_tail",
        "dataset": "real STATS-CEB (bidirectional utility table)",
        "note": (
            "Direct measurement of unguarded q-errors; no plan-cost surrogate is "
            "used anywhere in this analysis."
        ),
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for label in ("all_pairs", "worst_tail"):
        stats = summary[label]
        logger.info(
            "%-11s n=%4d  under=%4d (%.1f%%)  over=%4d  median q-err=%.0f",
            label,
            stats["n"],
            stats["under"],
            100 * stats["under_share"],
            stats["over"],
            stats["qerror_median"],
        )
    logger.info("-> %s", args.output)


if __name__ == "__main__":
    main()
