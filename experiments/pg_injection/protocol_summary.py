"""Protocol facts of an injection run, for the paper's audit.

Counts what the run did rather than what it found: plans timed, plans censored
at the timeout, server crashes, queries whose re-injected native estimates
reproduce the native plan, and the host load at query start.

Usage::

    python -m experiments.pg_injection.protocol_summary \
        --runs results/pg_injection/v2/stats_inject.jsonl \
        --output results/pg_injection/v2/stats_protocol.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import median

from experiments.pg_injection.inject_plans import _read_jsonl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = _read_jsonl(args.runs)
    summary = {
        "n_queries": len(records),
        "n_plans": sum(len(r["plans"]) for r in records),
        "n_censored_plans": sum(sum(r["plan_censored"].values()) for r in records),
        "n_crashed_plans": sum(len(r["crashed_plans"]) for r in records),
        "n_queries_with_censored_plan": sum(
            any(r["plan_censored"].values()) for r in records
        ),
        "native_reproduced": sum(
            r["policy_plan"]["native"] == r["policy_plan"]["native_raw"]
            for r in records
        ),
        "truth_injection_verified": sum(r["truth_injection_verified"] for r in records),
        "n_high_load": sum(r["high_load"] for r in records),
        "load_start_median": median(r["loadavg_1m_start"] for r in records),
        "load_start_max": max(r["loadavg_1m_start"] for r in records),
        "load_end_max": max(r["loadavg_1m_end"] for r in records),
        "load_start_max_query": max(records, key=lambda r: r["loadavg_1m_start"])[
            "query_id"
        ],
        "n_end_above_60": sum(r["loadavg_1m_end"] > 60 for r in records),
        "n_start_above_30": sum(r["loadavg_1m_start"] > 30 for r in records),
        "queries_start_above_30": [
            r["query_id"] for r in records if r["loadavg_1m_start"] > 30
        ],
        "queries_with_censored_plan": [
            r["query_id"] for r in records if any(r["plan_censored"].values())
        ],
        "uncensored_plans_timed_reps": sorted(
            {
                n
                for r in records
                for d, n in r["plan_reps"].items()
                if not r["plan_censored"][d]
            }
        ),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("%s", json.dumps(summary))


if __name__ == "__main__":
    main()
