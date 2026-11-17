"""S3: one independent time block of the small runtime effects and the sampling cost.

Pre-registered in ``docs/vldb_2027/PREDICTIONS_S3.md``. The query sets are
fixed in ``docs/vldb_2027/S3_DESIGN.json``; the two files are hashed together in
``docs/vldb_2027/PREDICTIONS_S3.sha256``, and this runner refuses to start if
the hashes do not match.

The protocol differs from the v3 execution runs on the points raised by the
review (``REVIEW_VLDB.md`` R3-M2, R4-M1, R5-M3):

* queries run in a seeded random order per block;
* every distinct plan of a query gets one untimed warm-up, then ``--reps`` timed
  rounds. Each round runs the plans in a fresh seeded random permutation (v3
  always ran native first), and every plan is timed in every round, slow ones
  included;
* every repetition is stored: its time, its position in the round, a timestamp,
  and the one-minute load before and after;
* for every query with a certifiable order, the certify-then-stop chain runs
  live in the same block: it draws the in-engine samples the rule picks
  (``TABLESAMPLE BERNOULLI(10) REPEATABLE(0)``, each live count checked
  against the stored one), times the certificate computation, then forces the
  certified order and executes it among the other plans.

Policies per query: ``native`` (no hints, forced-order form), ``native_raw``
(our instance's in-context estimates injected), ``native_upper``,
``native_lower`` and ``native_bi`` (sound guards, tau = 1, as in v3), ``truth``,
``bi_k2`` and ``bi_k8`` (synthetic brackets), and ``certified``.

Usage (one block; the driver ``run_s3_blocks.sh`` restarts the server between
blocks)::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.run_blocks_s3 \
        --block 1 --output /stage/s3/block_1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2

from experiments.pg_injection.forced_order import UNFORCE, forced_join_sql
from experiments.pg_injection.inject_plans import (
    WORKLOAD_PATHS,
    Key,
    _enough_disk,
    _key,
    _read_jsonl,
    hint_text,
    load_estimates,
    load_lower,
    plan_of,
    policy_values,
    reconnect,
    wait_for_quiet,
)
from experiments.pg_injection.run_forced_orders import honours
from experiments.pg_injection.subset_counts import WORKLOADS, count_sql, load_subqueries
from experiments.pg_injection.tightness_sweep import policies as tight_policies
from experiments.scripts.plan_certificate import repair_rounding
from experiments.scripts.plan_certificate_joint import incidence, left_deep_orders
from experiments.scripts.selective_sampling import joint_bounds

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DOCS = Path("docs/vldb_2027")
DESIGN = DOCS / "S3_DESIGN.json"
PREDICTIONS = DOCS / "PREDICTIONS_S3.md"
HASHES = DOCS / "PREDICTIONS_S3.sha256"
SIZES = Path("results/plan_guard/subjoin_sizes_maxdeg_fixed.jsonl")
SAMPLES = Path("results/pg_injection/subset_counts.jsonl")
NATIVE_CSV = Path("results/pg_injection/v2/native_incontext_estimations.csv")
RATE, SAMPLE_SEED, TARGET = 0.10, 0, 2.0
TEMP_FILE_LIMIT = "20GB"
GUARDS = ("native_raw", "native_upper", "native_lower", "native_bi", "truth")


def check_registration() -> None:
    """Refuse to run unless the registration and design match their hashes."""
    for line in HASHES.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split()
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            msg = f"{name} changed after registration"
            raise SystemExit(msg)


def cursor_for(dsn: str, timeout_ms: int) -> Any:
    """A hinted cursor with the statement timeout and the temp-file cap set."""
    cursor = reconnect(dsn)
    cursor.execute(f"SET temp_file_limit = '{TEMP_FILE_LIMIT}'")
    cursor.execute(f"SET statement_timeout = {timeout_ms}")
    return cursor


def live_chain(
    cursor: Any,
    record: dict[str, Any],
    subsql: dict[Key, str],
    stored: dict[Key, dict[str, Any]],
    sample_timeout_ms: int,
) -> dict[str, Any]:
    """Run the certify-then-stop rule live: sample, certify, return the order.

    Mirrors ``selective_sampling.simulate_query`` (target 2, LpBound upper,
    final result included, give-up rule on) with live samples. A sample is drawn
    twice: the first run is the deployment cost, the second the warm time v3
    reported.
    """
    t_setup = time.perf_counter()
    subjoins = record["subjoins"]
    index = {frozenset(s["aliases"]): i for i, s in enumerate(subjoins)}
    upper = np.array(
        [
            max(
                repair_rounding(float(s["upper"]["lpbound"]), float(s["truth"]))[0], 1.0
            )
            for s in subjoins
        ]
    )
    keys = [(record["query_id"], tuple(sorted(s["aliases"]))) for s in subjoins]
    orders = left_deep_orders(record["aliases"], index)
    inc = incidence(orders, index, include_final=True)
    cert_ms = 1000.0 * (time.perf_counter() - t_setup)
    lower = np.ones(len(subjoins))
    taken = np.zeros(len(subjoins), dtype=bool)
    draws: list[dict[str, Any]] = []
    gave_up = False
    cursor.execute(f"SET statement_timeout = {sample_timeout_ms}")
    while True:
        t0 = time.perf_counter()
        bound, rival = joint_bounds(inc, lower, upper)
        best = int(np.argmin(bound))
        stop = bound[best] <= TARGET or taken.all()
        pick = -1
        if not stop:
            hopeful, _ = joint_bounds(inc, np.where(taken, lower, upper), upper)
            if hopeful.min() > TARGET:
                gave_up, stop = True, True
            else:
                cands = [j for j in np.flatnonzero(inc[rival[best]]) if not taken[j]]
                if not cands:
                    cands = [int(j) for j in np.flatnonzero(~taken)]
                pick = int(max(cands, key=lambda j: upper[j]))
        cert_ms += 1000.0 * (time.perf_counter() - t0)
        if stop:
            break
        key = keys[pick]
        sql = count_sql(subsql[key], (stored[key]["sampled_alias"], RATE, SAMPLE_SEED))
        runs: list[float] = []
        value: float | None = None
        for _ in range(2):
            start = time.perf_counter()
            try:
                cursor.execute(sql)
                value = float(cursor.fetchone()[0])
                runs.append(1000.0 * (time.perf_counter() - start))
            except psycopg2.errors.QueryCanceled:
                cursor.connection.rollback()
                runs.append(float(sample_timeout_ms))
                break
        censored = value is None
        taken[pick] = True
        lower[pick] = 1.0 if censored else max(value or 1.0, 1.0)
        expected = (
            None if stored[key]["sample_censored"] else stored[key]["lower_sound"]
        )
        draws.append(
            {
                "aliases": list(key[1]),
                "count": value,
                "stored_count": expected,
                "matches_stored": (value == expected) if not censored else None,
                "first_ms": runs[0],
                "second_ms": runs[1] if len(runs) > 1 else None,
                "censored": censored,
            }
        )
    return {
        "certified": bool(bound[best] <= TARGET),
        "gave_up": gave_up,
        "bound": float(bound[best]),
        "order": list(orders[best]),
        "draws": draws,
        "sampling_first_ms": sum(d["first_ms"] for d in draws),
        "sampling_second_ms": sum(
            d["second_ms"] if d["second_ms"] is not None else d["first_ms"]
            for d in draws
        ),
        "certificate_ms": cert_ms,
    }


def execute_block_query(
    cursor: Any,
    dsn: str,
    sqls: dict[str, str],
    reps: int,
    timeout_ms: int,
    rng: random.Random,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Warm-up, then ``reps`` rounds in a fresh random order; every repetition kept."""
    out: dict[str, dict[str, Any]] = {
        d: {"warmup_ms": None, "reps": [], "censored": False, "crashed": False}
        for d in sqls
    }

    def run(digest: str) -> float | None:
        nonlocal cursor
        cursor.execute(UNFORCE)  # forced SQL sets its own limit; reset for the others
        start = time.perf_counter()
        try:
            cursor.execute(sqls[digest])
            cursor.fetchone()
        except psycopg2.errors.QueryCanceled:
            cursor.connection.rollback()
            out[digest]["censored"] = True
            return None
        except psycopg2.Error as error:  # out of memory, temp-file cap, crash
            logger.warning("plan %s failed: %s", digest, error)
            out[digest]["censored"] = out[digest]["crashed"] = True
            cursor = cursor_for(dsn, timeout_ms)
            return None
        return 1000.0 * (time.perf_counter() - start)

    for digest in rng.sample(sorted(sqls), len(sqls)):
        out[digest]["warmup_ms"] = run(digest)
    for rnd in range(reps):
        order = rng.sample(sorted(sqls), len(sqls))
        for pos, digest in enumerate(order):
            if out[digest]["censored"]:
                continue
            load0 = os.getloadavg()[0]
            ms = run(digest)
            out[digest]["reps"].append(
                {
                    "round": rnd,
                    "position": pos,
                    "n_plans": len(order),
                    "ms": ms,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "load_before": load0,
                    "load_after": os.getloadavg()[0],
                }
            )
    for rec in out.values():
        times = [r["ms"] for r in rec["reps"] if r["ms"] is not None]
        rec["median_ms"] = (
            float(timeout_ms)
            if rec["censored"] or not times
            else float(np.median(times))
        )
    return cursor, out


def main() -> None:
    """CLI entry point: one block."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default="host=127.0.0.1 port=55432 dbname=stats "
        "user=postgres password=postgres",
    )
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=300_000)
    parser.add_argument("--sample-timeout-ms", type=int, default=120_000)
    parser.add_argument("--max-load", type=float, default=60.0)
    parser.add_argument("--max-wait-s", type=float, default=1800.0)
    parser.add_argument(
        "--queries", default="", help="smoke test only: queries outside the sets"
    )
    args = parser.parse_args()

    design = json.loads(DESIGN.read_text())
    if args.queries:
        wanted = [int(q) for q in args.queries.split(",")]
        registered = set(design["set_a"]) | set(design["set_b"])
        if registered & set(wanted):
            msg = "smoke tests may not touch registered queries"
            raise SystemExit(msg)
        sets = {q: "smoke" for q in wanted}
    else:
        check_registration()
        sets = {q: "A" for q in design["set_a"]} | {q: "B" for q in design["set_b"]}
    seed = int(design["seed"])

    estimates_dir, subplan_file, query_file, _ = WORKLOAD_PATHS["stats"]
    est = {
        n: load_estimates(n, directory=estimates_dir)
        for n in ("lpbound", "truecardinality")
    }
    est["native"] = load_estimates("native", NATIVE_CSV, directory=estimates_dir)
    lower = load_lower(SAMPLES)
    stored = {(r["query_id"], tuple(r["aliases"])): r for r in _read_jsonl(SAMPLES)}
    subsql = {
        (r["query_id"], tuple(r["aliases"])): r["sql"]
        for r in load_subqueries(WORKLOADS["stats"]["subqueries"])
    }
    sizes = {r["query_id"]: r for r in _read_jsonl(SIZES)}
    subplans: dict[int, list[Key]] = {}
    for line in subplan_file.read_text().splitlines()[1:]:
        qid, tables, _ = line.split("|", 2)
        subplans.setdefault(int(qid), []).append(_key(qid, tables))
    lines = [x for x in query_file.read_text().splitlines() if x.strip()]
    raw_sql = {
        i + 1: x.split("||", 1)[-1].strip().rstrip(";") for i, x in enumerate(lines)
    }

    done = set()
    if args.output.exists():
        done = {r["query_id"] for r in _read_jsonl(args.output) if "query_id" in r}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cursor = cursor_for(args.dsn, args.timeout_ms)
    cursor.execute("SELECT pg_postmaster_start_time()::text, version()")
    started, version = cursor.fetchone()
    order = sorted(sets)
    random.Random(f"{seed}-block-{args.block}").shuffle(order)
    with args.output.open("a") as sink:
        if not done:
            sink.write(
                json.dumps(
                    {
                        "block_header": args.block,
                        "start": datetime.now(timezone.utc).isoformat(),
                        "postmaster_start": started,
                        "version": version,
                        "query_order": order,
                        "reps": args.reps,
                        "timeout_ms": args.timeout_ms,
                        "temp_file_limit": TEMP_FILE_LIMIT,
                        "smoke": bool(args.queries),
                    }
                )
                + "\n"
            )
        for position, query_id in enumerate(order):
            if query_id in done:
                continue
            if not _enough_disk(args.output):
                logger.error("less than 10 GB free on the output disk; stopping")
                break
            keys = subplans[query_id]
            pol = policy_values(keys, est, lower, 1.0, ("native",))
            values: dict[str, dict[Key, float] | None] = {
                "native_raw": pol["native_raw"],
                "native_upper": pol["native_upper"],
                "native_lower": pol["native_lower"],
                "native_bi": pol["native_bi"],
                "truth": pol["truth"],
            }
            tight = tight_policies(
                keys,
                {k: est["native"][k] for k in keys},
                {k: est["truecardinality"][k] for k in keys},
                est["lpbound"],
            )
            values["bi_k2"], values["bi_k8"] = tight["bi_k2"], tight["bi_k8"]
            counted = count_sql(raw_sql[query_id]) + ";"
            sqls: dict[str, str] = {}
            chosen: dict[str, str] = {}
            plans: dict[str, Any] = {}
            cursor.execute(UNFORCE)
            for name, vals in values.items():
                sql = hint_text(vals) + counted
                digest, plan = plan_of(cursor, sql)
                chosen[name] = digest
                sqls.setdefault(digest, sql)
                plans.setdefault(digest, plan)
            native_sql = UNFORCE + raw_sql[query_id] + ";"
            digest, plan = plan_of(cursor, native_sql)
            chosen["native"] = digest
            sqls.setdefault(digest, native_sql)
            plans.setdefault(digest, plan)

            load_start, high_load = wait_for_quiet(args.max_load, args.max_wait_s)
            chain = None
            record = sizes.get(query_id)
            if record is not None and record["n_aliases"] >= 3:
                chain = live_chain(
                    cursor, record, subsql, stored, args.sample_timeout_ms
                )
                cursor.execute(f"SET statement_timeout = {args.timeout_ms}")
                if chain["certified"]:
                    forced = forced_join_sql(raw_sql[query_id], chain["order"])
                    t0 = time.perf_counter()
                    digest, plan = plan_of(cursor, forced)
                    chain["plan_ms"] = 1000.0 * (time.perf_counter() - t0)
                    cursor.execute(UNFORCE)
                    if not honours(plan, chain["order"]):
                        msg = f"q{query_id}: forced order {chain['order']} not honoured"
                        raise RuntimeError(msg)
                    chosen["certified"] = digest
                    sqls.setdefault(digest, forced)
                    plans.setdefault(digest, plan)
            rng = random.Random(f"{seed}-block-{args.block}-q{query_id}")
            cursor, timings = execute_block_query(
                cursor, args.dsn, sqls, args.reps, args.timeout_ms, rng
            )
            out = {
                "block": args.block,
                "query_id": query_id,
                "set": sets[query_id],
                "position_in_block": position,
                "policy_plan": chosen,
                "timings": timings,
                "chain": chain,
                "plans": plans,
                "loadavg_1m_start": load_start,
                "loadavg_1m_end": os.getloadavg()[0],
                "high_load": high_load,
                "finished": datetime.now(timezone.utc).isoformat(),
            }
            sink.write(json.dumps(out) + "\n")
            sink.flush()
            logger.info(
                "block %d q%d (%s): %s%s",
                args.block,
                query_id,
                sets[query_id],
                {p: round(timings[d]["median_ms"]) for p, d in chosen.items()},
                ""
                if chain is None
                else f" | chain {chain['certified']} "
                f"{round(chain['sampling_first_ms'])} ms",
            )
    logger.info("block %d done", args.block)


if __name__ == "__main__":
    main()
