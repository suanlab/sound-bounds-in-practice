"""Recompute the max-degree upper bound of stored sub-joins after the NULL fix.

``build_job_query`` used to count SQL NULL as a join-key value, which made the
NULL group the "max degree" of mostly-NULL keys and loosened the max-degree
bound (fixed 2026-10-04; ``docs/AUDIT_REPAIR_LOG.md``). This rewrites only the
``maxdeg`` column of a sub-join sizes file -- truth, LpBound, SafeBound and the
sampling bound are untouched -- keeping the old value as ``maxdeg_null_counted``
and writing new files rather than overwriting.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.recompute_maxdeg \
        --sizes results/plan_guard/subjoin_sizes.json \
        --output results/plan_guard/subjoin_sizes_maxdeg_fixed.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from experiments.scripts.make_lower_column_stats import load_tables
from experiments.scripts.plan_certificate import repair_rounding
from experiments.scripts.plan_regret import _sub_query
from src.core.ceg import pessimistic_bound
from src.utils.job_eval import build_job_query
from src.utils.job_sql_parser import parse_job_sql

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE = Path("external/LpBound")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--queries",
        type=Path,
        default=BASE / "benchmarks/workloads/stats/statsQueries.sql",
    )
    parser.add_argument("--data-dir", type=Path, default=BASE / "data/datasets/stats")
    args = parser.parse_args()

    payload: dict[str, Any] = json.loads(args.sizes.read_text())
    sql_by_id = {
        i: line.strip()
        for i, line in enumerate(args.queries.read_text().splitlines(), start=1)
        if line.strip()
    }
    cache: dict[str, list[dict[str, str]]] = {}
    changed = tighter = violations = 0
    for record in payload["per_query"]:
        parsed = parse_job_sql(sql_by_id[record["query_id"]], str(record["query_id"]))
        rows = load_tables(args.data_dir, set(parsed.base_relations), cache)
        full = build_job_query(parsed, rows)
        for subjoin in record["subjoins"]:
            sub = _sub_query(full, tuple(sorted(subjoin["aliases"])))
            if sub is None:
                msg = f"q{record['query_id']}: {subjoin['aliases']} not a valid query"
                raise RuntimeError(msg)
            old = subjoin["upper"]["maxdeg"]
            new, violated = repair_rounding(
                max(pessimistic_bound(sub), 1.0), float(subjoin["truth"])
            )
            if violated:
                msg = f"q{record['query_id']} {subjoin['aliases']}: maxdeg below truth"
                raise AssertionError(msg)
            subjoin["upper"]["maxdeg_null_counted"] = old
            subjoin["upper"]["maxdeg"] = new
            subjoin["upper_violated"]["maxdeg"] = violated
            changed += abs(new / old - 1) > 1e-9
            tighter += new < old * (1 - 1e-9)
            violations += violated
    payload["note"] = (
        payload.get("note", "")
        + " | maxdeg recomputed 2026-10-04 with NULL keys excluded from degree "
        "statistics (old value kept as maxdeg_null_counted)"
    )
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    jsonl = args.output.with_suffix(".jsonl")
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in payload["per_query"]))
    logger.info(
        "maxdeg changed on %d sub-joins (%d tighter), violations %d -> %s, %s",
        changed,
        tighter,
        violations,
        args.output,
        jsonl,
    )


if __name__ == "__main__":
    main()
