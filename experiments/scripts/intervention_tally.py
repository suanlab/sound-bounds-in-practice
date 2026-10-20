"""How many times a sound certificate fired, and whether it ever made things worse.

The safety claim is a count, and a count needs a provenance. This script
recomputes it from the per-query utility tables rather than quoting a number
from an earlier draft: every (query, proposal, certificate) row on both real
benchmarks at every threshold we swept, classified into fired / helped /
worsened.

"Worsened" needs a tolerance, and saying so is not a hedge. At $\\tau=1$ a bound
that equals the true cardinality to within $2.5\\times10^{-10}$ trips its own
gate, and returning it changes the q-error in the fifteenth decimal place. Those
firings are ties, not degradations, and counting them as degradations would be
as misleading as ignoring them. We therefore report both: the number of firings
that increase q-error at all, and the number that increase it by more than a
relative ``--tolerance``.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.intervention_tally
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# (tau, benchmark) -> table. The tau=2 tables are the ones without a suffix.
TABLES: dict[tuple[int, str], str] = {
    (1, "job-light"): "real_joblight_tau1.csv",
    (2, "job-light"): "real_joblight.csv",
    (4, "job-light"): "real_joblight_tau4.csv",
    (8, "job-light"): "real_joblight_tau8.csv",
    (1, "stats-ceb"): "real_stats_tau1.csv",
    (2, "stats-ceb"): "real_stats.csv",
    (4, "stats-ceb"): "real_stats_tau4.csv",
    (8, "stats-ceb"): "real_stats_tau8.csv",
}


def zero_event_upper_bound(trials: int, level: float = 0.95) -> float:
    """Exact one-sided upper bound on a rate after observing zero events."""
    return 1.0 - (1.0 - level) ** (1.0 / trials) if trials > 0 else float("nan")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=Path("results/utility_table"))
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Relative q-error increase below which a firing counts as a tie.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/utility_table/intervention_tally.json"),
    )
    args = parser.parse_args()

    per_table: list[dict[str, Any]] = []
    ties: list[dict[str, Any]] = []
    fired = helped = worse_any = worse_beyond_tol = 0
    queries: set[tuple[str, str]] = set()

    for (tau, benchmark), name in sorted(TABLES.items()):
        path = args.tables / name
        if not path.exists():
            msg = f"missing utility table {path}"
            raise FileNotFoundError(msg)
        rows = list(csv.DictReader(path.open(newline="")))
        table_fired = table_worse = table_tol = 0
        for row in rows:
            if row["guard_fired"] != "True":
                continue
            table_fired += 1
            queries.add((benchmark, row["query_id"]))
            if row["intervention_help"] == "True":
                helped += 1
            before = float(row["proposal_qerror"])
            after = float(row["final_qerror"])
            if after > before:
                table_worse += 1
                if after > before * (1.0 + args.tolerance):
                    table_tol += 1
                else:
                    ties.append(
                        {
                            "benchmark": benchmark,
                            "tau": tau,
                            "query_id": row["query_id"],
                            "proposal": row["proposal"],
                            "certificate": row["certificate"],
                            "qerror_before": before,
                            "qerror_after": after,
                        }
                    )
        fired += table_fired
        worse_any += table_worse
        worse_beyond_tol += table_tol
        per_table.append(
            {
                "benchmark": benchmark,
                "tau": tau,
                "rows": len(rows),
                "fired": table_fired,
                "worsened_any": table_worse,
                "worsened_beyond_tolerance": table_tol,
            }
        )
        logger.info(
            "%-10s tau=%-2d rows=%5d fired=%4d worsened=%d (beyond tolerance: %d)",
            benchmark,
            tau,
            len(rows),
            table_fired,
            table_worse,
            table_tol,
        )

    payload: dict[str, Any] = {
        "experiment": "intervention_tally",
        "note": (
            "Every (query, proposal, certificate) row on both real benchmarks at "
            "every swept threshold. 'Worsened' is reported twice: any increase in "
            "q-error, and an increase beyond a relative tolerance. The difference "
            "is exact-tight ties where the bound equals the truth to within "
            "floating-point error."
        ),
        "tolerance": args.tolerance,
        "interventions": fired,
        "helped": helped,
        "worsened_any_amount": worse_any,
        "worsened_beyond_tolerance": worse_beyond_tol,
        "distinct_queries_with_a_firing": len(queries),
        "degradation_rate_upper_95": zero_event_upper_bound(fired),
        "degradation_rate_upper_95_query_level": zero_event_upper_bound(len(queries)),
        "ties_at_floating_point": ties,
        "per_table": per_table,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    logger.info(
        "\n%d interventions, %d helped, %d worsened beyond tolerance "
        "(%d ties at floating point)\n"
        "  degradation rate upper 95%%: %.4f (per intervention), "
        "%.4f (per query, n=%d)\n-> %s",
        fired,
        helped,
        worse_beyond_tol,
        worse_any - worse_beyond_tol,
        payload["degradation_rate_upper_95"],
        payload["degradation_rate_upper_95_query_level"],
        len(queries),
        args.output,
    )


if __name__ == "__main__":
    main()
