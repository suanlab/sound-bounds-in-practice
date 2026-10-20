"""Queries covered by the exact-count agreement check, per benchmark.

Sub-plans of one query share data, so the paper bounds the share of *queries*
with a disagreement between PostgreSQL's exact count and the artifact's truth.
This writes the number of queries with at least one compared sub-plan, after
asserting that none disagrees.

Usage::

    python -m experiments.pg_injection.agreement_queries
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from experiments.pg_injection.subset_counts import _read_jsonl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COUNTS = (
    Path("results/pg_injection/subset_counts.jsonl"),
    Path("results/pg_injection/joblight/subset_counts.jsonl"),
)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v2/agreement_queries.json"),
    )
    args = parser.parse_args()
    out = []
    for path in COUNTS:
        compared = [
            r
            for r in _read_jsonl(path)
            if not r["exact_censored"] and r.get("truth_artifact") is not None
        ]
        bad = [r for r in compared if abs(r["exact_pg"] - r["truth_artifact"]) > 0.5]
        if bad:
            msg = f"{len(bad)} disagreements in {path}"
            raise AssertionError(msg)
        out.append(len({r["query_id"] for r in compared}))
    args.output.write_text(json.dumps(out) + "\n")
    logger.info("queries compared (STATS-CEB, JOB-light): %s", out)


if __name__ == "__main__":
    main()
