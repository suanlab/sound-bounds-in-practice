"""Phase 2: inject (guarded) sub-plan estimates into PostgreSQL and time the plans.

For every STATS-CEB query and every policy below, the sub-plan cardinalities
are injected with ``pg_hint_plan`` ``Rows(<aliases> #<n>)`` hints (one per
sub-plan in the LpBound artifact's list), PostgreSQL 13.1 plans the query, and
the plan is fingerprinted from ``EXPLAIN (FORMAT JSON)`` with estimates and
costs stripped. Each *distinct* plan of a query is executed -- policies that
pick the same plan share its time -- so every query contributes one comparison
per policy pair, never several copies. The plans of a query run interleaved
(one round over all plans per repetition) after waiting for the 1-minute load
average to fall below ``--max-load`` (the host is shared); the load at start and
end is recorded, and the plan JSON is kept so that a machine-independent work
measure (true intermediate sizes) can be computed afterwards.

Policies (tau = 2 unless named; all bounds as published in the artifact, the
sampling floor from ``subset_counts.py``):

* ``native``: no hints (PostgreSQL's own estimates on this instance).
* ``truth``: exact sub-plan cardinalities.
* ``lpbound_est``, ``lhat_est``: plan directly on LpBound / on ``L_hat``.
* for each estimator E in {postgres, duckdb} (the artifact's estimates):
  ``E_raw``, ``E_upper`` (clamp to LpBound when E > tau*B), ``E_upper_t1``
  (tau = 1), ``E_lower`` (floor to the sound L when E < L/tau), ``E_bi``
  (both), ``E_bi_hat`` (LpBound above, unbiased L_hat below; not sound).

Post-hoc (added after an internal review panel); results append to a JSONL file
so the run can resume.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.inject_plans
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import psycopg2

# The disk guard and the JSONL reader live in subset_counts; they are
# re-exported here (``as`` marks an explicit re-export) for the other scripts.
from experiments.pg_injection.subset_counts import MIN_FREE_BYTES as MIN_FREE_BYTES
from experiments.pg_injection.subset_counts import _enough_disk as _enough_disk
from experiments.pg_injection.subset_counts import _read_jsonl as _read_jsonl
from experiments.pg_injection.subset_counts import count_sql, with_dbname

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ARTIFACT = Path("external/LpBound")
ESTIMATES = ARTIFACT / "results/evaluation_time/subquery_estimations/stats"
SUBQUERIES = ARTIFACT / "benchmarks/workloads/stats/stats_subqueries.csv"
QUERIES = Path(
    "external/End-to-End-CardEst-Benchmark/workloads/stats_CEB/stats_CEB.sql"
)
WORKLOAD_PATHS: dict[str, tuple[Path, Path, Path, str]] = {
    # (estimates dir, sub-plan list, query file, database)
    "stats": (ESTIMATES, SUBQUERIES, QUERIES, "stats"),
    "joblight": (
        ARTIFACT / "results/evaluation_time/subquery_estimations/joblight",
        ARTIFACT / "benchmarks/workloads/joblight/joblight_subqueries.csv",
        ARTIFACT / "benchmarks/workloads/joblight/joblightQueries.sql",
        "imdb",
    ),
    "jobrange_heldout": (
        Path("results/heldout_jobrange/estimates"),
        Path("results/heldout_jobrange/jobrange_heldout_subqueries.csv"),
        ARTIFACT / "benchmarks/workloads/jobrange/jobrangeQueries.sql",
        "imdb",
    ),
}
ESTIMATORS = ("postgres", "duckdb")
Key = tuple[int, tuple[str, ...]]


def _key(query_id: str, tables: str) -> Key:
    return int(query_id), tuple(sorted(t.lower() for t in tables.split()))


def load_estimates(
    name: str, path: Path | None = None, directory: Path = ESTIMATES
) -> dict[Key, float]:
    """One estimate file (artifact format) keyed by (query id, sorted aliases)."""
    out: dict[Key, float] = {}
    path = path or directory / f"{name}_subquery_estimations.csv"
    for line in path.read_text().splitlines()[1:]:
        parts = line.split(",")
        out[_key(parts[0], parts[1])] = float(parts[2])
    return out


def load_lower(path: Path) -> dict[Key, tuple[float | None, float | None]]:
    """Sound floor L and unbiased L_hat per sub-plan from phase 1."""
    out: dict[Key, tuple[float | None, float | None]] = {}
    for record in _read_jsonl(path):
        out[(record["query_id"], tuple(record["aliases"]))] = (
            record["lower_sound"],
            record["lower_hat"],
        )
    return out


def policy_values(
    keys: list[Key],
    est: dict[str, dict[Key, float]],
    lower: dict[Key, tuple[float | None, float | None]],
    tau: float,
    estimators: tuple[str, ...] = ESTIMATORS,
) -> dict[str, dict[Key, float] | None]:
    """Sub-plan cardinalities to inject, per policy (None = no hints)."""
    policies: dict[str, dict[Key, float] | None] = {
        "native": None,
        "truth": {k: est["truecardinality"][k] for k in keys},
        "lpbound_est": {k: est["lpbound"][k] for k in keys},
    }
    lhat: dict[Key, float] = {}
    for k in keys:
        point = lower.get(k, (None, None))[1]
        if point is not None:
            lhat[k] = float(point)
    if len(lhat) == len(keys):
        policies["lhat_est"] = lhat
    for name in estimators:
        if not all(k in est[name] for k in keys):
            continue  # this estimator does not cover every sub-plan of the query
        raw = {k: est[name][k] for k in keys}
        upper = {k: max(est["lpbound"][k], 1.0) for k in keys}

        def clamp_up(value: float, bound: float, t: float) -> float:
            return bound if value > t * bound else value

        def floor_up(value: float, key: Key, hat: bool) -> float:
            sound, point = lower.get(key, (None, None))
            floor = point if hat else sound
            if floor is None:
                return value  # censored sample: no floor
            return floor if value < floor / tau else value

        policies[f"{name}_raw"] = raw
        policies[f"{name}_upper"] = {k: clamp_up(raw[k], upper[k], tau) for k in keys}
        if tau != 1.0:
            policies[f"{name}_upper_t1"] = {
                k: clamp_up(raw[k], upper[k], 1.0) for k in keys
            }
        policies[f"{name}_lower"] = {k: floor_up(raw[k], k, False) for k in keys}
        policies[f"{name}_bi"] = {
            k: floor_up(clamp_up(raw[k], upper[k], tau), k, False) for k in keys
        }
        policies[f"{name}_bi_hat"] = {
            k: floor_up(clamp_up(raw[k], upper[k], tau), k, True) for k in keys
        }
    return policies


def hint_text(values: dict[Key, float] | None) -> str:
    """A ``pg_hint_plan`` comment setting every sub-plan's row estimate."""
    if not values:
        return ""
    if not all(math.isfinite(v) for v in values.values()):
        msg = "a sub-plan value to inject is not finite"
        raise ValueError(msg)
    rows = " ".join(
        f"Rows({' '.join(aliases)} #{max(round(v), 1)})"
        for (_, aliases), v in sorted(values.items())
    )
    return f"/*+ {rows} */ "


def _strip(node: Any) -> Any:
    """Plan tree without estimates or costs, for fingerprinting."""
    if isinstance(node, dict):
        return {
            k: _strip(v)
            for k, v in node.items()
            if k not in {"Startup Cost", "Total Cost", "Plan Rows", "Plan Width"}
        }
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def plan_of(cursor: Any, hinted_sql: str) -> tuple[str, Any]:
    """Fingerprint and JSON plan of a hinted statement."""
    sql_start = hinted_sql.index("SELECT")
    cursor.execute(
        hinted_sql[:sql_start] + "EXPLAIN (FORMAT JSON) " + hinted_sql[sql_start:]
    )
    plan = cursor.fetchone()[0][0]["Plan"]
    digest = hashlib.sha256(
        json.dumps(_strip(plan), sort_keys=True).encode()
    ).hexdigest()
    return digest[:16], plan


def wait_for_quiet(max_load: float, max_wait_s: float) -> tuple[float, bool]:
    """Wait (up to ``max_wait_s``) until the 1-min load average drops below
    ``max_load``. Returns the load when measuring starts and whether it is high."""
    waited = 0.0
    while os.getloadavg()[0] > max_load and waited < max_wait_s:
        time.sleep(30)
        waited += 30
    load = os.getloadavg()[0]
    return load, load > max_load


class PlanCrashedError(RuntimeError):
    """The server died while executing one plan (e.g. out of memory)."""

    def __init__(self, digest: str) -> None:
        super().__init__(f"server terminated while executing plan {digest}")
        self.digest = digest


def run_interleaved(
    cursor: Any,
    sqls: dict[str, str],
    timeout_ms: int,
    reps: int,
    always_reps: bool = False,
) -> dict[str, dict[str, Any]]:
    """Execute every plan once per round, ``reps`` rounds, so all plans of a query
    see the same machine conditions; median wall-clock ms per plan.

    A plan that times out is censored and not re-run; a plan slower than 10 s is
    timed in the first round only.
    """
    cursor.execute(f"SET statement_timeout = {timeout_ms}")
    times: dict[str, list[float]] = {d: [] for d in sqls}
    censored: dict[str, bool] = dict.fromkeys(sqls, False)
    for rep in range(reps):
        for digest, sql in sqls.items():
            if censored[digest] or (
                not always_reps and rep > 0 and times[digest][0] > 10_000
            ):
                continue
            start = time.perf_counter()
            try:
                cursor.execute(sql)
                cursor.fetchone()
            except psycopg2.errors.QueryCanceled:
                cursor.connection.rollback()
                censored[digest] = True
                continue
            except psycopg2.OperationalError as error:
                raise PlanCrashedError(digest) from error
            times[digest].append(1000.0 * (time.perf_counter() - start))
    return {
        d: {
            "ms": float(timeout_ms) if censored[d] else statistics.median(times[d]),
            "censored": censored[d],
            "reps": len(times[d]) + int(censored[d]),
        }
        for d in sqls
    }


def truth_injected(plan: Any, values: dict[Key, float], query_id: int) -> bool:
    """Every join node's estimate equals the injected truth for its aliases."""
    ok = True

    def aliases_of(node: Any) -> set[str]:
        found = {node["Alias"]} if "Alias" in node else set()
        for child in node.get("Plans", []):
            found |= aliases_of(child)
        return found

    def walk(node: Any) -> None:
        nonlocal ok
        if node.get("Node Type") in {"Hash Join", "Merge Join", "Nested Loop"}:
            key = (query_id, tuple(sorted(aliases_of(node))))
            if (
                key in values
                and abs(node["Plan Rows"] - max(round(values[key]), 1)) > 1
            ):
                ok = False
        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    return ok


def reconnect(dsn: str, attempts: int = 30) -> Any:
    """A fresh hinted cursor, retrying while the server recovers from a crash."""
    for attempt in range(attempts):
        try:
            connection = psycopg2.connect(dsn)
        except psycopg2.OperationalError:
            if attempt == attempts - 1:
                raise
            time.sleep(10)
            continue
        connection.autocommit = True
        cursor = connection.cursor()
        cursor.execute("LOAD 'pg_hint_plan'")
        cursor.execute("SET pg_hint_plan.enable_hint = on")
        return cursor
    msg = "unreachable"
    raise RuntimeError(msg)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=(
            "host=127.0.0.1 port=55432 dbname=stats user=postgres password=postgres"
        ),
    )
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--timeout-ms", type=int, default=300_000)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--always-reps",
        action="store_true",
        help="time every plan --reps times (default: plans over 10 s once)",
    )
    parser.add_argument(
        "--max-load", type=float, default=60.0, help="wait while 1-min load is above"
    )
    parser.add_argument("--max-wait-s", type=float, default=1800.0)
    parser.add_argument(
        "--lower", type=Path, default=Path("results/pg_injection/subset_counts.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/pg_injection/inject_plans.jsonl")
    )
    parser.add_argument("--queries", type=str, default="", help="e.g. 1,2,10")
    parser.add_argument("--workload", choices=sorted(WORKLOAD_PATHS), default="stats")
    parser.add_argument(
        "--estimators",
        type=str,
        default=",".join(ESTIMATORS),
        help="comma-separated; 'native' reads --native-csv",
    )
    parser.add_argument(
        "--native-csv",
        type=Path,
        default=Path("results/pg_injection/native_subquery_estimations.csv"),
    )
    args = parser.parse_args()

    estimates_dir, subplan_file, query_file, dbname = WORKLOAD_PATHS[args.workload]
    if args.workload != "stats":
        args.dsn = with_dbname(args.dsn, dbname)
    estimators = tuple(e for e in args.estimators.split(",") if e)
    est = {
        n: load_estimates(n, directory=estimates_dir)
        for n in ("lpbound", "truecardinality")
    }
    for name in estimators:
        est[name] = load_estimates(
            name,
            args.native_csv if name == "native" else None,
            directory=estimates_dir,
        )
    lower = load_lower(args.lower)
    subplans: dict[int, list[Key]] = {}
    for line in subplan_file.read_text().splitlines()[1:]:
        qid, tables, _ = line.split("|", 2)
        subplans.setdefault(int(qid), []).append(_key(qid, tables))
    queries = {
        i + 1: count_sql(line.split("||", 1)[-1].strip().rstrip(";")) + ";"
        for i, line in enumerate(
            raw for raw in query_file.read_text().splitlines() if raw.strip()
        )
    }
    wanted = {int(q) for q in args.queries.split(",") if q} or set(queries)
    done = set()
    if args.output.exists():
        done = {record["query_id"] for record in _read_jsonl(args.output)}

    cursor = reconnect(args.dsn)

    with args.output.open("a") as sink:
        for query_id in sorted(wanted - done):
            if not _enough_disk(args.output):
                logger.error("less than 10 GB free on the output disk; stopping")
                break
            keys = subplans.get(query_id, [])
            policies = policy_values(keys, est, lower, args.tau, estimators)
            plans: dict[str, dict[str, Any]] = {}
            chosen: dict[str, str] = {}
            injected_ok = True
            for name, values in policies.items():
                sql = hint_text(values) + queries[query_id]
                digest, plan = plan_of(cursor, sql)
                chosen[name] = digest
                if name == "truth" and values:
                    injected_ok = truth_injected(plan, values, query_id)
                plans.setdefault(digest, {"sql": sql, "plan": plan})
            load_start, high_load = wait_for_quiet(args.max_load, args.max_wait_s)
            pending = {d: p["sql"] for d, p in plans.items()}
            crashed: list[str] = []
            timings: dict[str, dict[str, Any]] = {}
            while pending:
                try:
                    timings.update(
                        run_interleaved(
                            cursor,
                            pending,
                            args.timeout_ms,
                            args.reps,
                            always_reps=args.always_reps,
                        )
                    )
                    pending = {}
                except PlanCrashedError as crash:
                    # A plan that kills the backend (out of memory) is a failed
                    # plan: record it as censored, reconnect, measure the rest.
                    logger.warning("q%d: %s; reconnecting", query_id, crash)
                    crashed.append(crash.digest)
                    timings[crash.digest] = {
                        "ms": float(args.timeout_ms),
                        "censored": True,
                        "reps": 1,
                    }
                    pending = {d: s for d, s in pending.items() if d != crash.digest}
                    cursor = reconnect(args.dsn)
            record = {
                "query_id": query_id,
                "n_subplans": len(keys),
                "truth_injection_verified": injected_ok,
                "policy_plan": chosen,
                "plan_ms": {d: t["ms"] for d, t in timings.items()},
                "plan_censored": {d: t["censored"] for d, t in timings.items()},
                "plan_reps": {d: t["reps"] for d, t in timings.items()},
                "plans": {d: p["plan"] for d, p in plans.items()},
                "loadavg_1m_start": load_start,
                "loadavg_1m_end": os.getloadavg()[0],
                "high_load": high_load,
                "crashed_plans": crashed,
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            logger.info(
                "q%d: %d policies -> %d distinct plans; truth injected %s; %s",
                query_id,
                len(policies),
                len(plans),
                injected_ok,
                {n: round(timings[chosen[n]]["ms"]) for n in ("native", "truth")},
            )


if __name__ == "__main__":
    main()
