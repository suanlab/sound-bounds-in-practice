"""Tightness ablation: does a tighter *sound* upper bound intervene more usefully?

Pre-registered as TOP_PLAN Phase 3.2 ablation #3, testing claim C3. Three sound
upper certificates of increasing tightness are compared on the *same* real
STATS-CEB queries and the *same* proposals:

  * ``max_degree`` -- our own max-degree (MOLP-style) bound, computed here from
    the filtered real relations,
  * ``safebound``  -- SafeBound, from the LpBound artifact,
  * ``lpbound``    -- LpBound (SIGMOD'25), from the same artifact.

For each we report how tight the bound actually is (``B/|Q|``), how often it is
valid, and how often it fires and helps at a fixed threshold. This separates
*tightness* from *soundness*: all three are sound by construction, so any
difference in utility is attributable to tightness alone --- the controlled
comparison the geometric-mean counterexample cannot provide.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.tightness_ablation --limit 40
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import signal
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

from src.core.ceg import pessimistic_bound
from src.utils.job_eval import build_job_query
from src.utils.job_sql_parser import parse_job_sql
from src.utils.metrics import q_error

if TYPE_CHECKING:
    from collections.abc import Iterator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_REL_TOL = 1e-9


class _TimeoutError(Exception):
    """Raised when a per-query computation exceeds its budget."""


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Abort the enclosed block after ``seconds`` (SIGALRM, main thread only)."""

    def _handler(signum: int, frame: object) -> None:
        raise _TimeoutError

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def load_artifact(path: Path, column: str) -> dict[int, float]:
    """Read one artifact CSV column keyed by QueryID."""
    out: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out[int(row["QueryID"])] = float(row[column])
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("external/LpBound")
    parser.add_argument(
        "--queries",
        type=Path,
        default=base / "benchmarks/workloads/stats/statsQueries.sql",
    )
    parser.add_argument("--data-dir", type=Path, default=base / "data/datasets/stats")
    parser.add_argument(
        "--artifact-dir", type=Path, default=base / "results/accuracy_acyclic/stats"
    )
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--query-timeout-s", type=float, default=20.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/utility_table/tightness.json")
    )
    args = parser.parse_args()

    truth = load_artifact(
        args.artifact_dir / "truecardinality_stats_full_estimations.csv",
        "truecardinality_Estimate",
    )
    bounds = {
        "lpbound": load_artifact(
            args.artifact_dir / "lpbound_stats_full_estimations.csv", "lpbound_Estimate"
        ),
        "safebound": load_artifact(
            args.artifact_dir / "safebound_stats_full_estimations.csv",
            "safebound_Estimate",
        ),
    }
    proposals = {
        name: load_artifact(
            args.artifact_dir / f"{name}_stats_full_estimations.csv", f"{name}_Estimate"
        )
        for name in ("postgres", "duckdb", "neurocard", "flat", "factorjoin", "dbx")
    }

    queries = [
        line.strip() for line in args.queries.read_text().splitlines() if line.strip()
    ]
    if args.limit:
        queries = queries[: args.limit]

    cache: dict[str, list[dict[str, str]]] = {}
    per_cert: dict[str, dict[str, list[float]]] = {
        name: {"tightness": [], "valid": [], "fire": [], "help": []}
        for name in ("max_degree", "safebound", "lpbound")
    }
    evaluated = 0
    records: list[dict[str, Any]] = []
    skipped: list[str] = []

    for index, sql in enumerate(queries, start=1):
        parsed = parse_job_sql(sql, str(index))
        if not parsed.fully_parsed or index not in truth:
            skipped.append(f"{index}:unusable")
            continue
        try:
            with time_limit(args.query_timeout_s):
                for table in parsed.base_relations:
                    if table not in cache:
                        with (args.data_dir / f"{table}.csv").open(newline="") as fh:
                            cache[table] = list(csv.DictReader(fh))
                query = build_job_query(
                    parsed, {t: cache[t] for t in parsed.base_relations}
                )
                max_degree = pessimistic_bound(query)
        except (_TimeoutError, KeyError, ValueError) as exc:
            logger.warning("  q%d skipped: %s", index, type(exc).__name__)
            skipped.append(f"{index}:{type(exc).__name__}")
            continue

        y = max(truth[index], 1.0)
        optional: dict[str, float | None] = {"max_degree": max_degree}
        optional.update({k: v.get(index) for k, v in bounds.items()})
        if any(b is None for b in optional.values()):
            skipped.append(f"{index}:missing-bound")
            continue
        cert_bounds: dict[str, float] = {
            k: float(v) for k, v in optional.items() if v is not None
        }
        evaluated += 1

        for name, bound_value in cert_bounds.items():
            bound = max(bound_value, 1.0)
            stats = per_cert[name]
            stats["tightness"].append(bound / y)
            stats["valid"].append(1.0 if y <= bound * (1.0 + _REL_TOL) else 0.0)
            for proposal, estimates in proposals.items():
                if index not in estimates:
                    continue
                e = max(estimates[index], 1.0)
                # Same tolerance as validity: a bound equal to an exact
                # estimate up to rounding must not count as firing.
                fired = e > args.tau * bound * (1.0 + _REL_TOL)
                stats["fire"].append(1.0 if fired else 0.0)
                helped = fired and q_error(bound, y) < q_error(e, y)
                stats["help"].append(1.0 if helped else 0.0)
                # Per-pair record so the rates can carry a query-clustered
                # interval; the aggregate alone cannot.
                records.append(
                    {
                        "query_id": str(index),
                        "certificate": name,
                        "proposal": proposal,
                        "bound_over_true": bound / y,
                        "fired": fired,
                        "helped": helped,
                    }
                )
        logger.info(
            "  q%-3d max_degree/true=%.1f safebound/true=%.1f lpbound/true=%.1f",
            index,
            cert_bounds["max_degree"] / y,
            cert_bounds["safebound"] / y,
            cert_bounds["lpbound"] / y,
        )

    summary: dict[str, Any] = {}
    for name, stats in per_cert.items():
        if not stats["tightness"]:
            continue
        summary[name] = {
            "bound_over_true_median": median(stats["tightness"]),
            "bound_over_true_geomean": math.exp(
                sum(math.log(v) for v in stats["tightness"]) / len(stats["tightness"])
            ),
            "validity": sum(stats["valid"]) / len(stats["valid"]),
            "fire_rate": (
                sum(stats["fire"]) / len(stats["fire"]) if stats["fire"] else 0.0
            ),
            "help_rate": (
                sum(stats["help"]) / len(stats["help"]) if stats["help"] else 0.0
            ),
        }

    payload = {
        "experiment": "tightness_vs_soundness_ablation",
        "preregistered_as": "TOP_PLAN Phase 3.2 ablation #3 (claim C3)",
        "dataset": "real STATS-CEB",
        "tau": args.tau,
        "queries_evaluated": evaluated,
        "queries_skipped": skipped,
        "certificates": summary,
        "per_pair": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    logger.info(
        "\n%-12s %14s %10s %10s %10s",
        "certificate",
        "B/true (med)",
        "valid",
        "fire",
        "help",
    )
    for name in ("max_degree", "safebound", "lpbound"):
        if name not in summary:
            continue
        s = summary[name]
        logger.info(
            "%-12s %14.1f %9.0f%% %9.1f%% %9.1f%%",
            name,
            s["bound_over_true_median"],
            100 * s["validity"],
            100 * s["fire_rate"],
            100 * s["help_rate"],
        )
    logger.info("evaluated %d queries -> %s", evaluated, args.output)


if __name__ == "__main__":
    main()
