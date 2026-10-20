"""Phase 1 of the PostgreSQL study: exact and sampled counts of every sub-plan.

For every (query, sub-plan) in the LpBound artifact's STATS-CEB sub-query list
(``benchmarks/workloads/stats/stats_subqueries.csv``; PostgreSQL's notion of a
connected sub-plan, i.e. including joins implied by key equalities), this runs
inside the patched PostgreSQL 13.1 of ``experiments/pg_injection``:

* the exact ``COUNT(*)`` -- a cross-check of the artifact's truth, and the
  engine-level cost an exact answer would take;
* the same count with the largest base relation replaced by
  ``TABLESAMPLE BERNOULLI(100 r) REPEATABLE(seed)`` -- every sampled tuple set is
  a subset of the relation, so the count is a *sound* lower bound ``L`` with
  probability one, and ``L / r`` is the unbiased point ``L_hat``.

Timing is wall-clock around execute+fetch on a local socket, second of two
runs (the first warms the buffer cache; counts slower than 5 s are timed once);
a run that exceeds the statement timeout is recorded as censored. Results
append to a JSONL file so the run can
resume.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.subset_counts
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import psycopg2

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MIN_FREE_BYTES = 10 * 1024**3  # stop before a shared disk fills up


def _enough_disk(path: Path) -> bool:
    """True while the output file system keeps at least 10 GB free."""
    return shutil.disk_usage(path.parent).free >= MIN_FREE_BYTES


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Records of a resume log, ignoring a truncated final line."""
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("ignoring a corrupt line in %s", path)
    return records


ARTIFACT = Path("external/LpBound")
HELDOUT = Path("results/heldout_jobrange")
WORKLOADS: dict[str, dict[str, Any]] = {
    "stats": {
        "subqueries": ARTIFACT / "benchmarks/workloads/stats/stats_subqueries.csv",
        "estimates": ARTIFACT / "results/evaluation_time/subquery_estimations/stats",
        "dbname": "stats",
    },
    "joblight": {
        "subqueries": ARTIFACT
        / "benchmarks/workloads/joblight/joblight_subqueries.csv",
        "estimates": ARTIFACT / "results/evaluation_time/subquery_estimations/joblight",
        "dbname": "imdb",
    },
    # Held-out JOB-range sample (docs/icde_2027/PREREGISTRATION_JOBRANGE.md); its
    # truth file is built from our own exact counts.
    "jobrange_heldout": {
        "subqueries": HELDOUT / "jobrange_heldout_subqueries.csv",
        "estimates": HELDOUT / "estimates",
        "dbname": "imdb",
    },
}
SUBQUERIES = WORKLOADS["stats"]["subqueries"]
TRUTH = WORKLOADS["stats"]["estimates"] / "truecardinality_subquery_estimations.csv"
FROM_ITEM = re.compile(r"(\w+)\s+AS\s+(\w+)", re.IGNORECASE)


def load_subqueries(path: Path) -> list[dict[str, Any]]:
    """Rows of the artifact's pipe-delimited sub-query list."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines()[1:]:
        query_id, tables, sql = line.split("|", 2)
        rows.append(
            {
                "query_id": int(query_id),
                "aliases": sorted(t.lower() for t in tables.split()),
                "sql": sql.strip().rstrip(";"),
            }
        )
    return rows


def load_truth(path: Path) -> dict[tuple[int, tuple[str, ...]], float]:
    """Artifact truth keyed by (query id, sorted lower-case aliases)."""
    truth: dict[tuple[int, tuple[str, ...]], float] = {}
    for line in path.read_text().splitlines()[1:]:
        query_id, tables, value = line.split(",")
        key = (int(query_id), tuple(sorted(t.lower() for t in tables.split())))
        truth[key] = float(value)
    return truth


def count_sql(sql: str, sample: tuple[str, float, int] | None = None) -> str:
    """Turn the artifact's ``SELECT * FROM ...`` into a (sampled) count."""
    body = re.sub(r"^\s*SELECT\s+\*\s+FROM", "SELECT COUNT(*) FROM", sql, flags=re.I)
    if sample is None:
        return body
    alias, rate, seed = sample
    sampled = 0

    def add_sample(match: re.Match[str]) -> str:
        nonlocal sampled
        if match.group(2).lower() != alias:
            return match.group(0)
        sampled += 1
        return (
            f"{match.group(0)} TABLESAMPLE BERNOULLI ({100 * rate:g}) "
            f"REPEATABLE ({seed})"
        )

    from_end = re.search(r"\bWHERE\b", body, flags=re.I)
    cut = from_end.start() if from_end else len(body)
    out = FROM_ITEM.sub(add_sample, body[:cut]) + body[cut:]
    if sampled != 1:
        # Without this check a missing alias would silently yield an exact count.
        msg = f"alias {alias!r} matched {sampled} FROM items in: {sql[:120]}"
        raise ValueError(msg)
    return out


class BackendCrashedError(Exception):
    """The PostgreSQL backend died while counting; the count is censored."""


def with_dbname(dsn: str, dbname: str) -> str:
    """``dsn`` with its database set to ``dbname`` (added if absent)."""
    if re.search(r"\bdbname=\S+", dsn):
        return re.sub(r"\bdbname=\S+", f"dbname={dbname}", dsn)
    return f"{dsn} dbname={dbname}"


def connect(dsn: str, attempts: int = 30) -> tuple[Any, Any]:
    """Connect (retrying while the server recovers from a crash)."""
    for attempt in range(attempts):
        try:
            connection = psycopg2.connect(dsn)
        except psycopg2.OperationalError:
            if attempt == attempts - 1:
                raise
            time.sleep(10)
            continue
        connection.autocommit = True
        return connection, connection.cursor()
    msg = "unreachable"
    raise RuntimeError(msg)


def safe_count(
    cursor: Any, dsn: str, sql: str, timeout_ms: int
) -> tuple[Any, float | None, float, bool]:
    """``timed_count`` that survives a backend crash.

    Returns (cursor to use next, count or None, ms, crashed). A crash is
    censored at the timeout and followed by a reconnect.
    """
    try:
        value, elapsed = timed_count(cursor, sql, timeout_ms)
    except BackendCrashedError:
        logger.warning("backend crashed; censoring and reconnecting")
        _, cursor = connect(dsn)
        return cursor, None, float(timeout_ms), True
    return cursor, value, elapsed, False


def timed_count(cursor: Any, sql: str, timeout_ms: int) -> tuple[float | None, float]:
    """Run ``sql`` twice; return (count or None if censored, ms of the 2nd run).

    A count the first run obtained is kept even if the timed second run hits the
    timeout; only the timing is then censored (reported at the timeout).
    """
    cursor.execute(f"SET statement_timeout = {timeout_ms}")
    value: float | None = None
    elapsed = float(timeout_ms)
    for run in range(2):
        start = time.perf_counter()
        try:
            cursor.execute(sql)
            value = float(cursor.fetchone()[0])
            elapsed = 1000.0 * (time.perf_counter() - start)
        except psycopg2.errors.QueryCanceled:
            cursor.connection.rollback()
            return value, float(timeout_ms)
        except psycopg2.OperationalError as exc:
            # The backend was killed (out of memory) before the timeout fired.
            raise BackendCrashedError(str(exc)) from exc
        if run == 0 and elapsed > 5_000:
            break  # long count: cache warm-up is negligible, time it once
    return value, elapsed


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default="host=127.0.0.1 port=55432 dbname=stats "
        "user=postgres password=postgres",
    )
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="stats")
    parser.add_argument("--rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/subset_counts.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=0, help="debug: first N rows")
    parser.add_argument(
        "--skip-exact",
        action="store_true",
        help="time only the sampled count (for extra sampling seeds)",
    )
    parser.add_argument(
        "--redo-censored",
        type=Path,
        default=None,
        help="measure only the sub-plans whose sample timed out in this earlier file",
    )
    args = parser.parse_args()

    preset = WORKLOADS[args.workload]
    rows = load_subqueries(preset["subqueries"])
    if args.redo_censored is not None:
        redo = {
            (r["query_id"], tuple(r["aliases"]))
            for r in _read_jsonl(args.redo_censored)
            if r["sample_censored"]
        }
        rows = [r for r in rows if (r["query_id"], tuple(r["aliases"])) in redo]
    truth = load_truth(preset["estimates"] / "truecardinality_subquery_estimations.csv")
    if args.workload != "stats":
        args.dsn = with_dbname(args.dsn, preset["dbname"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        for record in _read_jsonl(args.output):
            done.add((record["query_id"], tuple(record["aliases"])))
    logger.info("%d sub-plans, %d already done", len(rows), len(done))

    _, cursor = connect(args.dsn)
    cursor.execute("SELECT relname, reltuples FROM pg_class WHERE relkind = 'r'")
    reltuples = {name.lower(): float(n) for name, n in cursor.fetchall()}

    with args.output.open("a") as sink:
        for i, row in enumerate(rows):
            if args.limit and i >= args.limit:
                break
            key = (row["query_id"], tuple(row["aliases"]))
            if key in done:
                continue
            if not _enough_disk(args.output):
                logger.error("less than 10 GB free on the output disk; stopping")
                break
            tables = {a.lower(): t.lower() for t, a in FROM_ITEM.findall(row["sql"])}
            largest = max(row["aliases"], key=lambda a: reltuples[tables[a]])
            if args.skip_exact:
                exact, exact_ms, exact_crashed = None, 0.0, False
            else:
                cursor, exact, exact_ms, exact_crashed = safe_count(
                    cursor, args.dsn, count_sql(row["sql"]), args.timeout_ms
                )
            cursor, sampled, sample_ms, sample_crashed = safe_count(
                cursor,
                args.dsn,
                count_sql(row["sql"], (largest, args.rate, args.seed)),
                args.timeout_ms,
            )
            record = {
                "query_id": row["query_id"],
                "aliases": row["aliases"],
                "truth_artifact": truth.get(key),
                "exact_pg": exact,
                "exact_ms": exact_ms,
                "exact_censored": exact is None and not args.skip_exact,
                "exact_skipped": args.skip_exact,
                "sampled_alias": largest,
                "rate": args.rate,
                "seed": args.seed,
                "lower_sound": sampled,
                "lower_hat": None if sampled is None else sampled / args.rate,
                "sample_ms": sample_ms,
                "sample_censored": sampled is None,
                "exact_crashed": exact_crashed,
                "sample_crashed": sample_crashed,
                "loadavg_1m": os.getloadavg()[0],
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if i % 50 == 0:
                logger.info(
                    "%d/%d q%d %s exact %.0fms sample %.0fms",
                    i,
                    len(rows),
                    row["query_id"],
                    row["aliases"],
                    exact_ms,
                    sample_ms,
                )


if __name__ == "__main__":
    main()
