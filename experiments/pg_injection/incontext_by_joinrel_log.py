"""PostgreSQL's in-context sub-plan estimates from a stock server (``joinrel_log``).

The PostgreSQL 13.1 harness reads the planner's estimate of every join sub-plan
inside the full query through hooks of the STATS-CEB benchmark patch, which
exists for 13.1 only. The small ``joinrel_log`` extension (this directory's
``joinrel_log/``) wraps ``join_search_hook``: the standard join search runs
unchanged and every join relation it built is reported with its row estimate.
A join relation's size is fixed when the search first builds it, from whichever
split it meets first, so these are exactly the in-context estimates; forcing a
join order instead (``incontext_by_prefix``) builds the relation from another
split and gives different values for 3+ relations. ``EXPLAIN`` of each unhinted
query is enough; nothing is executed.

``--compare`` checks the values against the hook-based CSV of PostgreSQL 13.1
(the extension reports clamped, i.e. rounded, row counts).

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.incontext_by_joinrel_log \
        --workload stats --output results/pg_injection/v3/joinrel_log_stats_pg13.csv \
        --compare results/pg_injection/v2/native_incontext_estimations.csv
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import (
    WORKLOAD_PATHS,
    Key,
    _key,
    load_estimates,
    reconnect,
)
from experiments.pg_injection.subset_counts import count_sql, with_dbname

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PREFIX = "joinrel_log|"


def joinrel_estimates(cursor: Any, sql: str) -> dict[tuple[str, ...], float]:
    """Row estimate of every join relation the planner built for ``sql``."""
    connection = cursor.connection
    connection.notices.clear()
    cursor.execute("EXPLAIN " + sql)
    cursor.fetchall()
    out: dict[tuple[str, ...], float] = {}
    for notice in connection.notices:
        text = notice.strip()
        start = text.find(PREFIX)
        if start < 0:
            continue
        aliases, rows = text[start + len(PREFIX) :].rsplit("|", 1)
        out[tuple(sorted(a.lower() for a in aliases.split()))] = float(rows)
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=(
            "host=127.0.0.1 port=55432 dbname=stats user=postgres password=postgres"
        ),
    )
    parser.add_argument("--workload", choices=sorted(WORKLOAD_PATHS), default="stats")
    parser.add_argument("--queries", default="", help="e.g. 1,2,10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, default=None)
    args = parser.parse_args()

    _, subplan_file, query_file, dbname = WORKLOAD_PATHS[args.workload]
    subplans: dict[int, list[Key]] = {}
    for line in subplan_file.read_text().splitlines()[1:]:
        qid, tables, _ = line.split("|", 2)
        subplans.setdefault(int(qid), []).append(_key(qid, tables))
    lines = [raw for raw in query_file.read_text().splitlines() if raw.strip()]
    queries = {
        i + 1: count_sql(line.split("||", 1)[-1].strip().rstrip(";"))
        for i, line in enumerate(lines)
    }
    wanted = {int(q) for q in args.queries.split(",") if q} or set(subplans)
    cursor = reconnect(with_dbname(args.dsn, dbname))
    # psycopg2 keeps only the last 50 notices by default; a 7-relation query
    # builds far more join relations than that.
    cursor.connection.notices = collections.deque(maxlen=1_000_000)
    cursor.execute("LOAD 'joinrel_log'")
    cursor.execute("SET client_min_messages = notice")
    rows: list[tuple[int, tuple[str, ...], float]] = []
    missing: list[Key] = []
    for query_id in sorted(wanted):
        estimates = joinrel_estimates(cursor, queries[query_id])
        for key in subplans.get(query_id, []):
            if key[1] in estimates:
                rows.append((query_id, key[1], estimates[key[1]]))
            else:
                missing.append(key)
    lines_out = ["QueryID,Tables,native_Estimate"] + [
        f"{q},{' '.join(a.upper() for a in t)},{v:.17g}" for q, t, v in rows
    ]
    args.output.write_text("\n".join(lines_out) + "\n")
    summary: dict[str, Any] = {
        "n_subplans": sum(len(subplans.get(q, [])) for q in wanted),
        "n_estimated": len(rows),
        "n_missing": len(missing),
        "missing_examples": [[q, list(t)] for q, t in missing[:10]],
    }
    if args.compare is not None:
        hooks = load_estimates("native", args.compare, directory=Path("."))
        pairs = [(v, hooks[(q, t)]) for q, t, v in rows if (q, t) in hooks]
        summary["n_compared"] = len(pairs)
        summary["n_equal_after_rounding"] = sum(
            abs(v - max(round(h), 1)) <= 1 for v, h in pairs
        )
        summary["max_rel_diff"] = max(
            abs(v - max(round(h), 1)) / max(round(h), 1) for v, h in pairs
        )
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    logger.info("%s", json.dumps(summary))


if __name__ == "__main__":
    main()
