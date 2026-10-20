"""End-to-end runtime of the plans chosen from guarded vs unguarded estimates.

Section~\\ref{sec:planregret} scores chosen plans by their true intermediate
cardinalities. This script closes the remaining step: it *executes* those plans
on a real engine and measures wall-clock time.

Two engines are supported, because a runtime claim that rests on one engine is a
claim about that engine. Both are made to honour the join order we place in the
FROM clause, by the mechanism each one documents:

``duckdb``
    a vectorised column store; ``SET disabled_optimizers='join_order'`` stops it
    from reordering.
``sqlite``
    a row store with index-nested-loop joins; the ``CROSS JOIN`` operator is
    defined to fix the loop nesting order. Tables are loaded in memory with
    empty CSV fields converted to SQL ``NULL`` and indexes built on every join
    column, which is the configuration a row store would actually be deployed
    in --- and the one where join order matters most.

The order itself comes from the same textbook left-deep optimizer as the
plan-regret experiment: it minimises the sum of estimated intermediate sizes,
once with raw CEG subplan estimates and once with each subplan estimate passed
through the guard.

Only the queries where the two policies actually choose different orders can
show a runtime difference, so those are reported separately from the (majority)
queries where the guard changes nothing.

Whether an engine really honours the order is a precondition, not an
assumption: :mod:`experiments.scripts.check_order_honoured` verifies it for
both engines before these numbers mean anything.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.plan_runtime --max-aliases 5
    PYTHONHASHSEED=0 python -m experiments.scripts.plan_runtime --engine sqlite \\
        --output results/plan_regret/plan_runtime_sqlite.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

import duckdb

from experiments.scripts.plan_regret import (
    _connected,
    _left_deep_orders,
    _order_cost,
    _sub_query,
    time_limit,
)
from src.core.ceg import pessimistic_bound
from src.utils.job_eval import build_alias_rows, build_job_query
from src.utils.job_sql_parser import parse_job_sql
from src.utils.join_counting import count_join
from src.utils.types import ALL_CEG_CONFIGS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.utils.job_sql_parser import ParsedJobQuery

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_STATS_TABLES = (
    "badges",
    "comments",
    "postHistory",
    "postLinks",
    "posts",
    "tags",
    "users",
    "votes",
)


ENGINES = ("duckdb", "sqlite")


def _column_type(values: list[str]) -> str:
    """Infer a SQLite column type from the non-empty values in a column.

    Timestamps stay ``TEXT``: the STATS predicates compare them to ISO-8601
    literals, which orders correctly as text.
    """
    seen = [v for v in values if v != ""]
    if seen and all(re.fullmatch(r"-?\d+", v) for v in seen):
        return "INTEGER"
    if seen and all(re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", v) for v in seen):
        return "REAL"
    return "TEXT"


def connect_sqlite(data_dir: Path) -> sqlite3.Connection:
    """Load the STATS tables into an in-memory SQLite database.

    Empty CSV fields become SQL ``NULL`` so that the engine applies the same
    NULL semantics our own counting does, and every join column is indexed.
    """
    con = sqlite3.connect(":memory:")
    for table in _STATS_TABLES:
        with (data_dir / f"{table}.csv").open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = [[cell if cell != "" else None for cell in row] for row in reader]
        types = [
            _column_type([str(row[i]) for row in rows if row[i] is not None])
            for i in range(len(header))
        ]
        spec = ", ".join(f'"{c}" {t}' for c, t in zip(header, types, strict=True))
        con.execute(f'CREATE TABLE "{table}" ({spec})')
        placeholders = ", ".join("?" * len(header))
        con.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', rows)
    con.commit()
    return con


def index_join_columns(con: sqlite3.Connection, parsed: ParsedJobQuery) -> None:
    """Ensure an index exists on every column this query joins on."""
    for join in parsed.joins:
        sides = (
            (join.left_relation, join.left_attribute),
            (join.right_relation, join.right_attribute),
        )
        for alias, attribute in sides:
            table = parsed.alias_to_relation[alias]
            con.execute(
                f'CREATE INDEX IF NOT EXISTS "ix_{table}_{attribute}" '
                f'ON "{table}" ("{attribute}")'
            )
    con.commit()


def connect(data_dir: Path, engine: str = "duckdb") -> object:
    """Load the STATS tables and pin the join order to the query."""
    if engine == "sqlite":
        return connect_sqlite(data_dir)
    con = duckdb.connect()
    for table in _STATS_TABLES:
        con.execute(
            f"CREATE TABLE {table} AS "
            f"SELECT * FROM read_csv_auto('{data_dir / f'{table}.csv'}', header=true)"
        )
    con.execute("SET disabled_optimizers='join_order'")
    return con


def render_sql(
    sql: str,
    order: tuple[str, ...],
    parsed: ParsedJobQuery,
    engine: str = "duckdb",
) -> str:
    """Rewrite the query as COUNT(*) with the FROM list in the given order.

    SQLite reorders a comma-separated FROM list freely but is defined to keep
    the loop nesting order of ``CROSS JOIN``, so that is how the order is
    expressed there.
    """
    where = re.split(r"\bWHERE\b", sql, maxsplit=1, flags=re.IGNORECASE)[1]
    where = where.strip().rstrip(";").strip()
    separator = " CROSS JOIN " if engine == "sqlite" else ", "
    from_list = separator.join(f'"{parsed.alias_to_relation[a]}" AS {a}' for a in order)
    return f"SELECT count(*) FROM {from_list} WHERE {where}"


def _execute_one(con: object, query_sql: str) -> int:
    """Run a COUNT(*) query on either engine and return the count."""
    row = con.execute(query_sql).fetchone()  # type: ignore[attr-defined]
    return int(row[0]) if row else 0


@contextmanager
def query_deadline(con: object, seconds: float | None) -> Iterator[None]:
    """Abort a query that runs past ``seconds``.

    A ``SIGALRM``-based timeout cannot interrupt SQLite, which spends the whole
    query inside a C call that never returns to the interpreter for the signal
    handler to run. SQLite's own progress handler is called every N virtual
    machine instructions and aborts the statement when it returns non-zero, so
    that is what is used here. DuckDB needs no such help at these table sizes.
    """
    if seconds is None or not isinstance(con, sqlite3.Connection):
        yield
        return
    deadline = time.perf_counter() + seconds
    con.set_progress_handler(lambda: int(time.perf_counter() > deadline), 10_000)
    try:
        yield
    finally:
        con.set_progress_handler(None, 0)


def time_order(con: object, query_sql: str, repeats: int) -> tuple[float, int]:
    """Return (best wall-clock seconds, result) for a query, after one warm-up."""
    result = _execute_one(con, query_sql)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        _execute_one(con, query_sql)
        best = min(best, time.perf_counter() - start)
    return best, result


def choose_orders(
    parsed: ParsedJobQuery,
    rows_by_relation: dict[str, list[dict[str, str]]],
    tau: float,
) -> dict[str, Any] | None:
    """Pick the left-deep order under no-guard / guarded / true sizes."""
    from src.estimators.baseline_ceg import BaselineCEGEstimator

    alias_rows = build_alias_rows(parsed, rows_by_relation)
    aliases = sorted(parsed.aliases)
    full = build_job_query(parsed, rows_by_relation)
    estimator = BaselineCEGEstimator()

    true_sizes: dict[frozenset[str], float] = {}
    bound_sizes: dict[frozenset[str], float] = {}
    est_sizes: dict[str, dict[frozenset[str], float]] = {
        c.name: {} for c in ALL_CEG_CONFIGS
    }
    for size in range(2, len(aliases) + 1):
        for subset in _subsets(aliases, size, parsed):
            key = frozenset(subset)
            joins = frozenset(
                j
                for j in parsed.joins
                if j.left_relation in subset and j.right_relation in subset
            )
            true_sizes[key] = max(
                count_join({a: alias_rows[a] for a in subset}, key, joins), 1.0
            )
            sub = _sub_query(full, subset)
            if sub is None:
                return None
            bound_sizes[key] = max(pessimistic_bound(sub), 1.0)
            per_config = estimator.estimate_all_configs(sub)
            for config in ALL_CEG_CONFIGS:
                est_sizes[config.name][key] = max(per_config[config.name].estimate, 1.0)

    orders = _left_deep_orders(aliases, parsed)
    if not orders:
        return None
    optimal = min(orders, key=lambda o: _order_cost(o, true_sizes))

    picks: dict[str, dict[str, tuple[str, ...]]] = {}
    for config in ALL_CEG_CONFIGS:
        raw = est_sizes[config.name]
        guarded = {
            k: (bound_sizes[k] if raw[k] > tau * bound_sizes[k] else raw[k])
            for k in raw
        }
        picks[config.name] = {
            "no_guard": min(orders, key=lambda o: _order_cost(o, raw)),
            "guarded": min(orders, key=lambda o: _order_cost(o, guarded)),
        }
    return {"optimal": optimal, "picks": picks}


def _subsets(
    aliases: list[str], size: int, parsed: ParsedJobQuery
) -> list[tuple[str, ...]]:
    """Connected alias subsets of a given size."""
    import itertools

    return [s for s in itertools.combinations(aliases, size) if _connected(s, parsed)]


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
    parser.add_argument("--engine", choices=ENGINES, default="duckdb")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--max-aliases", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--query-timeout-s", type=float, default=90.0)
    parser.add_argument("--exec-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--order-timeout-s",
        type=float,
        default=60.0,
        help=(
            "Abort a single plan that runs longer than this. A SIGALRM timeout "
            "cannot interrupt SQLite, so its progress handler is used instead; "
            "a pair with any timed-out plan is skipped rather than compared "
            "against a censored time."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/plan_regret/plan_runtime.json")
    )
    args = parser.parse_args()

    con = connect(args.data_dir, args.engine)
    queries = [
        line.strip() for line in args.queries.read_text().splitlines() if line.strip()
    ]
    cache: dict[str, list[dict[str, str]]] = {}
    records: list[dict[str, Any]] = []
    skipped: list[str] = []

    for index, sql in enumerate(queries, start=1):
        parsed = parse_job_sql(sql, str(index))
        if not parsed.fully_parsed or not 3 <= len(parsed.aliases) <= args.max_aliases:
            continue
        try:
            with time_limit(args.query_timeout_s):
                for table in parsed.base_relations:
                    if table not in cache:
                        with (args.data_dir / f"{table}.csv").open(newline="") as fh:
                            cache[table] = list(csv.DictReader(fh))
                chosen = choose_orders(
                    parsed, {t: cache[t] for t in parsed.base_relations}, args.tau
                )
        except Exception as exc:
            skipped.append(f"{index}:{type(exc).__name__}")
            continue
        if chosen is None:
            skipped.append(f"{index}:degenerate")
            continue

        differing = {
            name: p
            for name, p in chosen["picks"].items()
            if p["no_guard"] != p["guarded"]
        }
        if not differing:
            continue

        timings: dict[tuple[str, ...], float] = {}
        try:
            with time_limit(args.exec_timeout_s):
                needed = {chosen["optimal"]}
                for p in differing.values():
                    needed.update((p["no_guard"], p["guarded"]))
                if args.engine == "sqlite":
                    index_join_columns(con, parsed)
                for order in needed:
                    with query_deadline(con, args.order_timeout_s):
                        seconds, _ = time_order(
                            con,
                            render_sql(sql, order, parsed, args.engine),
                            args.repeats,
                        )
                    timings[order] = seconds
        except Exception as exc:
            skipped.append(f"{index}:exec-{type(exc).__name__}")
            continue

        for config, p in differing.items():
            records.append(
                {
                    "query_id": index,
                    "n_aliases": len(parsed.aliases),
                    "config": config,
                    "ms_no_guard": timings[p["no_guard"]] * 1000.0,
                    "ms_guarded": timings[p["guarded"]] * 1000.0,
                    "ms_optimal": timings[chosen["optimal"]] * 1000.0,
                    "speedup_guarded_over_no_guard": (
                        timings[p["no_guard"]] / timings[p["guarded"]]
                    ),
                }
            )
        logger.info(
            "  q%-3d aliases=%d differing-configs=%d  no-guard %.1fms  guarded %.1fms",
            index,
            len(parsed.aliases),
            len(differing),
            median(r["ms_no_guard"] for r in records if r["query_id"] == index),
            median(r["ms_guarded"] for r in records if r["query_id"] == index),
        )

    speedups = [r["speedup_guarded_over_no_guard"] for r in records]
    payload = {
        "experiment": f"plan_runtime_{args.engine}",
        "engine": (
            "DuckDB (join-order optimizer disabled; FROM order honoured)"
            if args.engine == "duckdb"
            else "SQLite (CROSS JOIN fixes loop nesting order; join columns indexed)"
        ),
        "dataset": "real STATS-CEB",
        "note": (
            "Wall-clock execution of the plans chosen by guarded vs unguarded "
            "subplan estimates. Only (query, config) pairs where the two policies "
            "choose different orders are timed; all others execute identically."
        ),
        "tau": args.tau,
        "repeats": args.repeats,
        "n_pairs_timed": len(records),
        "n_queries_timed": len({r["query_id"] for r in records}),
        "skipped": skipped,
        "speedup_guarded_over_no_guard": {
            "median": median(speedups) if speedups else float("nan"),
            "min": min(speedups) if speedups else float("nan"),
            "max": max(speedups) if speedups else float("nan"),
            "n_faster": sum(1 for s in speedups if s > 1.0),
            "n_slower": sum(1 for s in speedups if s < 1.0),
        },
        "total_ms_no_guard": sum(r["ms_no_guard"] for r in records),
        "total_ms_guarded": sum(r["ms_guarded"] for r in records),
        "total_ms_optimal": sum(r["ms_optimal"] for r in records),
        "per_pair": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    s = payload["speedup_guarded_over_no_guard"]
    logger.info(
        "\n%d pairs across %d queries where the guard changed the plan\n"
        "  speedup guarded/no-guard: median %.2fx  min %.2fx  max %.2fx"
        "  (faster %d, slower %d)\n"
        "  total wall-clock: no-guard %.0f ms | guarded %.0f ms | "
        "optimal %.0f ms -> %s",
        payload["n_pairs_timed"],
        payload["n_queries_timed"],
        s["median"],
        s["min"],
        s["max"],
        s["n_faster"],
        s["n_slower"],
        payload["total_ms_no_guard"],
        payload["total_ms_guarded"],
        payload["total_ms_optimal"],
        args.output,
    )


if __name__ == "__main__":
    main()
