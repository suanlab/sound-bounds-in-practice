"""Utility table on REAL benchmarks from the LpBound artifact (W2 GO/KILL data).

Consumes the per-query estimation CSVs shipped in the SIGMOD'25 LpBound
reproducibility artifact (``external/LpBound/results/accuracy_acyclic/``),
which cover the *real* JOB-light (70 queries) and STATS-CEB (146 queries)
benchmarks with:

  * true cardinalities,
  * real proposal estimators (PostgreSQL, NeuroCard, DeepDB, FLAT,
    FactorJoin, BayesCard, DuckDB, DBX), and
  * two strong sound upper certificates (LpBound, SafeBound).

It emits one row per (query, proposal, certificate) with the *pre-registered*
variables of ``docs/c1_preregistration.md`` (frozen 2026-07-11). Notes kept
honest:

  * ``cost_ratio`` needs the full exact-count time, which the artifact does not
    ship; we emit the certificate's own per-query inference time (``cert_ms``)
    when available and leave ``cost_ratio`` empty rather than fabricate it.
    Statistics-based certificates are precomputed-summary lookups, so their
    cost is negligible relative to join execution in any case.
  * ``bound_valid`` audits the certificate against the true cardinality with
    the 1e-9 relative tolerance used elsewhere in this repo.

Usage::

    python -m experiments.scripts.make_utility_table_external \
        --workload joblight --output results/utility_table/real_joblight.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from math import log
from pathlib import Path
from typing import Any

from src.utils.metrics import q_error

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROPOSALS = [
    "postgres",
    "duckdb",
    "dbx",
    "neurocard",
    "deepdb",
    "flat",
    "factorjoin",
    "bayescard",
]
CERTIFICATES = ["lpbound", "safebound"]
_REL_TOL = 1e-9

FIELDS = [
    "workload",
    "query_id",
    "proposal",
    "certificate",
    "estimate",
    "true_cardinality",
    "bound",
    "bound_valid",
    "direction",
    "tightness_true",
    "tightness_proxy",
    "cert_ms",
    "cost_ratio",
    "guard_fired",
    "intervention_help",
    "proposal_qerror",
    "final_qerror",
]


def _load_column(path: Path) -> dict[int, dict[str, float]]:
    """Read one artifact CSV into {QueryID: {column: value}}."""
    out: dict[int, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            qid = int(row["QueryID"])
            out[qid] = {
                key: float(value)
                for key, value in row.items()
                if key != "QueryID" and value not in ("", "nan")
            }
    return out


def build_rows(results_dir: Path, workload: str, tau: float) -> list[dict[str, Any]]:
    """Join artifact CSVs into pre-registered utility rows."""
    base = results_dir / workload

    def path_of(system: str) -> Path:
        return base / f"{system}_{workload}_full_estimations.csv"

    truth = _load_column(path_of("truecardinality"))
    certs = {name: _load_column(path_of(name)) for name in CERTIFICATES}
    rows: list[dict[str, Any]] = []
    missing_proposals: list[str] = []

    for proposal in PROPOSALS:
        prop_path = path_of(proposal)
        if not prop_path.exists():
            missing_proposals.append(proposal)
            continue
        estimates = _load_column(prop_path)
        for qid, truth_row in truth.items():
            y = max(next(iter(truth_row.values())), 1.0)
            est_row = estimates.get(qid)
            if est_row is None:
                continue
            e = max(next(iter(est_row.values())), 1.0)
            base_q = q_error(e, y)
            direction = 1 if e >= y else -1
            for cert_name in CERTIFICATES:
                cert_row = certs[cert_name].get(qid)
                if cert_row is None:
                    continue
                bound = max(cert_row[f"{cert_name}_Estimate"], 1.0)
                cert_ms = cert_row.get(f"{cert_name}_InferenceTime", float("nan"))
                fired = e > tau * bound
                final = bound if fired else e
                rows.append(
                    {
                        "workload": workload,
                        "query_id": qid,
                        "proposal": proposal,
                        "certificate": cert_name,
                        "estimate": e,
                        "true_cardinality": y,
                        "bound": bound,
                        "bound_valid": y <= bound * (1.0 + _REL_TOL),
                        "direction": direction,
                        "tightness_true": log(bound / y),
                        "tightness_proxy": log(bound / e),
                        "cert_ms": cert_ms,
                        "cost_ratio": "",  # exact-count reference time not shipped
                        "guard_fired": fired,
                        "intervention_help": fired and q_error(final, y) < base_q,
                        "proposal_qerror": base_q,
                        "final_qerror": q_error(final, y),
                    }
                )
    if missing_proposals:
        logger.warning("missing proposal files (skipped): %s", missing_proposals)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fire/help rates by certificate x direction, plus validity audit."""
    summary: dict[str, Any] = {}
    for cert_name in CERTIFICATES:
        cert_rows = [r for r in rows if r["certificate"] == cert_name]
        validity = (
            sum(1 for r in cert_rows if r["bound_valid"]) / len(cert_rows)
            if cert_rows
            else float("nan")
        )
        entry: dict[str, Any] = {"bound_validity": validity}
        for regime, sign in (("over", 1), ("under", -1)):
            subset = [r for r in cert_rows if r["direction"] == sign]
            fired = [r for r in subset if r["guard_fired"]]
            helped = [r for r in subset if r["intervention_help"]]
            entry[regime] = {
                "n": len(subset),
                "fire_rate": len(fired) / len(subset) if subset else float("nan"),
                "help_rate": len(helped) / len(subset) if subset else float("nan"),
            }
        summary[cert_name] = entry
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("external/LpBound/results/accuracy_acyclic"),
    )
    parser.add_argument("--workload", choices=["joblight", "stats"], default="joblight")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/utility_table/real_joblight.csv")
    )
    args = parser.parse_args()

    rows = build_rows(args.results_dir, args.workload, args.tau)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for cert_name, entry in summary.items():
        logger.info(
            "%-10s validity=%5.1f%%  over: n=%3d fire=%5.1f%% help=%5.1f%%  "
            "under: n=%3d fire=%5.1f%% help=%5.1f%%",
            cert_name,
            100 * entry["bound_validity"],
            entry["over"]["n"],
            100 * entry["over"]["fire_rate"],
            100 * entry["over"]["help_rate"],
            entry["under"]["n"],
            100 * entry["under"]["fire_rate"],
            100 * entry["under"]["help_rate"],
        )
    logger.info("Wrote %d rows -> %s (+ %s)", len(rows), args.output, summary_path)


if __name__ == "__main__":
    main()
