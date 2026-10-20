"""Summarise the execution of certified join orders (``certified_plans.jsonl``).

Geometric-mean speedup of the certified order over PostgreSQL's native plan
(execution only, and with the in-engine sampling time added), counts of queries
faster or slower beyond a 10% band, overall and by native execution time, with
bootstrap intervals over queries.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.summarize_certified
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import _read_jsonl
from experiments.scripts.bootstrap_ci import cluster_bootstrap

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def geo(values: list[float]) -> float:
    """Geometric mean."""
    return math.exp(sum(math.log(v) for v in values) / len(values))


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Speedups with intervals and faster/slower counts for a set of queries."""
    return {
        "n": len(rows),
        "geo_exec": cluster_bootstrap(
            rows, lambda rs: geo([r["native_ms"] / r["certified_ms"] for r in rs])
        ),
        "geo_end_to_end": cluster_bootstrap(
            rows,
            lambda rs: geo(
                [r["native_ms"] / (r["certified_ms"] + r["sampling_ms"]) for r in rs]
            ),
        ),
        "slower": sum(r["certified_ms"] > 1.1 * r["native_ms"] for r in rows),
        "faster": sum(r["certified_ms"] < r["native_ms"] / 1.1 for r in rows),
        "censored_native": sum(r["native_censored"] for r in rows),
        "censored_certified": sum(r["certified_censored"] for r in rows),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=Path("results/pg_injection/certified_plans.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/certified_plans_summary.json"),
    )
    args = parser.parse_args()
    rows = _read_jsonl(args.runs)
    summary = {
        "all": summarise(rows),
        "native_below_1s": summarise([r for r in rows if r["native_ms"] < 1000]),
        "native_at_least_1s": summarise([r for r in rows if r["native_ms"] >= 1000]),
        "high_load": sum(r["high_load"] for r in rows),
        "same_plan_as_native": sum(r["same_plan_as_native"] for r in rows),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for key in ("all", "native_below_1s", "native_at_least_1s"):
        entry = summary[key]
        logger.info(
            "%-20s n=%d exec %.2f [%.2f, %.2f] e2e %.2f slower %d faster %d",
            key,
            entry["n"],
            entry["geo_exec"]["point"],
            entry["geo_exec"]["ci_low"],
            entry["geo_exec"]["ci_high"],
            entry["geo_end_to_end"]["point"],
            entry["slower"],
            entry["faster"],
        )


if __name__ == "__main__":
    main()
