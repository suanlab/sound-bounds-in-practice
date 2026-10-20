"""Does injecting PostgreSQL's own estimates reproduce its native plan?

For each query, compares the plan fingerprint of the un-hinted query with that
of the query carrying ``Rows`` hints for every sub-plan, once with the
stand-alone estimates (``native_estimates``) and once with the in-context ones
(``native_estimates_incontext``). Planning only.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.check_native_reproduction
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import (
    WORKLOAD_PATHS,
    hint_text,
    load_estimates,
    plan_of,
    reconnect,
)
from experiments.pg_injection.subset_counts import count_sql

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SOURCES = {
    "stats": {
        "standalone": "results/pg_injection/native_subquery_estimations.csv",
        "in_context": "results/pg_injection/v2/native_incontext_estimations.csv",
    },
    "joblight": {
        "standalone": "results/pg_injection/joblight/native_subquery_estimations.csv",
        "in_context": (
            "results/pg_injection/v2/native_incontext_estimations_joblight.csv"
        ),
    },
}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v2/native_reproduction_check.json"),
    )
    args = parser.parse_args()
    out: dict[str, Any] = {}
    for workload, sources in SOURCES.items():
        _, _, query_file, dbname = WORKLOAD_PATHS[workload]
        cursor = reconnect(
            f"host=127.0.0.1 port=55432 dbname={dbname} user=postgres password=postgres"
        )
        lines = [raw for raw in query_file.read_text().splitlines() if raw.strip()]
        queries = {
            i + 1: count_sql(line.split("||", 1)[-1].strip().rstrip(";")) + ";"
            for i, line in enumerate(lines)
        }
        native = {q: plan_of(cursor, sql)[0] for q, sql in queries.items()}
        entry: dict[str, Any] = {"n_queries": len(queries)}
        for label, path in sources.items():
            estimates = load_estimates("native", path=Path(path))
            differs = []
            for query_id, sql in queries.items():
                values = {k: v for k, v in estimates.items() if k[0] == query_id}
                if plan_of(cursor, hint_text(values) + sql)[0] != native[query_id]:
                    differs.append(query_id)
            entry[label] = {
                "reproduced": len(queries) - len(differs),
                "differs": differs,
            }
        out[workload] = entry
        logger.info("%s: %s", workload, json.dumps(entry))
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
