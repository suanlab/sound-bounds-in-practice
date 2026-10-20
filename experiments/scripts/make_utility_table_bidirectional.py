"""Merge the sampling lower certificate into the STATS-CEB utility table.

The external table (``make_utility_table_external``) holds *upper*-certificate
rows (LpBound, SafeBound) whose cost is a statistics lookup, so the
pre-registered ``cost`` variable is constant there. This script appends the
*lower*-certificate rows measured by ``make_lower_column_stats`` --- same real
queries, same proposals, but a certificate that must partially execute the join
and therefore has a real, per-query cost.

The result is the first table where all three pre-registered variables
(direction, tightness, cost) vary, which is what makes the C1 test
non-degenerate (see ``results/c1_pilot/c1_pilot.json`` caveats).

Lower-guard semantics, per ``docs/c1_preregistration.md``:
  * fires when ``estimate < L_hat / tau`` (the proposal underestimates), and the
    intervention raises the estimate to ``L_hat``;
  * ``tightness_true = log(y / L)``, ``tightness_proxy = log(e / L_hat)``;
  * ``intervention_help`` iff the post-intervention q-error strictly improves.

Usage::

    python -m experiments.scripts.make_utility_table_bidirectional \
        --output results/utility_table/real_stats_bidirectional.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from math import log
from pathlib import Path
from typing import Any

from experiments.scripts.make_utility_table_external import FIELDS
from src.utils.metrics import q_error

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_lower(path: Path) -> dict[int, dict[str, float]]:
    """Read the measured lower-certificate column keyed by query id."""
    lower: dict[int, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            lower[int(row["query_id"])] = {
                "sound_lower_bound": float(row["sound_lower_bound"]),
                "l_hat": float(row["l_hat"]),
                "cost_ratio": float(row["cost_ratio"]),
                "lower_cert_ms": float(row["lower_cert_ms"]),
                "true_cardinality": float(row["true_cardinality"]),
            }
    return lower


def build_lower_rows(
    upper_rows: list[dict[str, str]],
    lower: dict[int, dict[str, float]],
    tau: float,
) -> list[dict[str, Any]]:
    """One lower-guard row per (query, proposal) that has a measured certificate."""
    seen: set[tuple[int, str]] = set()
    rows: list[dict[str, Any]] = []
    for upper in upper_rows:
        qid = int(upper["query_id"])
        proposal = upper["proposal"]
        if (qid, proposal) in seen or qid not in lower:
            continue
        seen.add((qid, proposal))

        cert = lower[qid]
        y = max(float(upper["true_cardinality"]), 1.0)
        e = max(float(upper["estimate"]), 1.0)
        l_sound = max(cert["sound_lower_bound"], 1.0)
        l_hat = max(cert["l_hat"], 1.0)

        base_q = q_error(e, y)
        fired = e < l_hat / tau
        final = l_hat if fired else e
        rows.append(
            {
                "workload": upper["workload"],
                "query_id": qid,
                "proposal": proposal,
                "certificate": "sample_lb",
                "estimate": e,
                "true_cardinality": y,
                "bound": l_sound,
                "bound_valid": l_sound <= y,
                "direction": 1 if e >= y else -1,
                "tightness_true": log(y / l_sound),
                "tightness_proxy": log(e / l_hat),
                "cert_ms": cert["lower_cert_ms"],
                "cost_ratio": cert["cost_ratio"],
                "guard_fired": fired,
                "intervention_help": fired and q_error(final, y) < base_q,
                "proposal_qerror": base_q,
                "final_qerror": q_error(final, y),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fire/help rates per certificate x direction, plus validity."""
    summary: dict[str, Any] = {}
    for cert in sorted({str(r["certificate"]) for r in rows}):
        subset = [r for r in rows if str(r["certificate"]) == cert]
        valid = sum(1 for r in subset if str(r["bound_valid"]) in ("True", "true"))
        entry: dict[str, Any] = {"bound_validity": valid / len(subset)}
        for regime, sign in (("over", 1), ("under", -1)):
            part = [r for r in subset if int(r["direction"]) == sign]
            fired = [r for r in part if str(r["guard_fired"]) in ("True", "true")]
            helped = [
                r for r in part if str(r["intervention_help"]) in ("True", "true")
            ]
            entry[regime] = {
                "n": len(part),
                "fire_rate": len(fired) / len(part) if part else float("nan"),
                "help_rate": len(helped) / len(part) if part else float("nan"),
            }
        summary[cert] = entry
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upper", type=Path, default=Path("results/utility_table/real_stats.csv")
    )
    parser.add_argument(
        "--lower", type=Path, default=Path("results/utility_table/stats_lower.csv")
    )
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/utility_table/real_stats_bidirectional.csv"),
    )
    args = parser.parse_args()

    with args.upper.open(newline="") as handle:
        upper_rows = list(csv.DictReader(handle))
    lower = load_lower(args.lower)
    lower_rows = build_lower_rows(upper_rows, lower, args.tau)
    combined: list[dict[str, Any]] = [dict(r) for r in upper_rows] + lower_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(combined)

    summary = summarize(combined)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for cert, entry in summary.items():
        logger.info(
            "%-10s validity=%5.1f%%  over: n=%4d fire=%5.1f%% help=%5.1f%%  "
            "under: n=%4d fire=%5.1f%% help=%5.1f%%",
            cert,
            100 * entry["bound_validity"],
            entry["over"]["n"],
            100 * entry["over"]["fire_rate"],
            100 * entry["over"]["help_rate"],
            entry["under"]["n"],
            100 * entry["under"]["fire_rate"],
            100 * entry["under"]["help_rate"],
        )
    logger.info(
        "%d upper + %d lower = %d rows -> %s",
        len(upper_rows),
        len(lower_rows),
        len(combined),
        args.output,
    )


if __name__ == "__main__":
    main()
