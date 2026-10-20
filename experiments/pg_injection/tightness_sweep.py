"""Dose-response of a sound guard to the tightness of its bound (H3; EXPLAIN only).

A guard on an estimate source ``e`` replaces every join sub-plan's value by
``min(e, B)`` (upper), ``max(e, L)`` (floor) or both. Here the bounds are
synthetic and sound by construction, ``B_k = k |S|`` and ``L_k = |S| / k``, so
tightness ``k`` is the only variable. For every query, source and ``k`` the
policy's values are injected as in ``inject_plans``, PostgreSQL plans the query
(``EXPLAIN``, nothing is executed), and the chosen plan is scored by its true
``C_out``: the sum of the true sizes of its join nodes. Planning on the true
sizes gives the reference gain; the real LpBound upper guard is planned too, as
the point a real bound occupies.

Exploratory on STATS-CEB and JOB-light; prediction in
``docs/icde_2027/PREDICTIONS.md``.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.tightness_sweep \
        --workload stats \
        --native-csv results/pg_injection/v2/native_incontext_estimations.csv \
        --output results/pg_injection/v3/tightness_sweep_stats.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import (
    WORKLOAD_PATHS,
    Key,
    _key,
    _read_jsonl,
    hint_text,
    load_estimates,
    plan_of,
    reconnect,
)
from experiments.pg_injection.run_forced_orders import join_sets
from experiments.pg_injection.subset_counts import count_sql, with_dbname

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TIGHTNESS = (1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
SOURCES = ("native", "postgres")


def policies(
    keys: list[Key],
    raw: dict[Key, float],
    truth: dict[Key, float],
    real: dict[Key, float],
) -> dict[str, dict[Key, float]]:
    """Injected values per policy: raw, truth, real upper guard, and B_k / L_k."""
    out = {"raw": raw, "truth": truth}
    if all(k in real for k in keys):  # no LpBound sub-plan bounds for JOB-range
        out["lpbound_upper"] = {k: min(raw[k], max(real[k], 1.0)) for k in keys}
    for t in TIGHTNESS:
        upper = {k: min(raw[k], t * truth[k]) for k in keys}
        out[f"upper_k{t:g}"] = upper
        out[f"lower_k{t:g}"] = {k: max(raw[k], truth[k] / t) for k in keys}
        out[f"bi_k{t:g}"] = {k: max(upper[k], truth[k] / t) for k in keys}
    return out


def true_cout(
    plan: dict[str, Any], query_id: int, truth: dict[Key, float]
) -> float | None:
    """True C_out of a plan, or None if a join node's size is unknown."""
    sets: list[frozenset[str]] = []
    join_sets(plan, sets)
    keys = [(query_id, tuple(sorted(s))) for s in sets]
    if any(k not in truth for k in keys):
        return None
    return sum(max(truth[k], 1.0) for k in keys)


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Geometric-mean C_out gain over raw, plan-change share, share of truth gain."""
    out: dict[str, Any] = {}
    for source in SOURCES:
        rows = [r[source] for r in records if source in r]
        if not rows:
            continue
        names = [
            n
            for n in rows[0]["cout"]
            if n != "raw" and all(n in r["cout"] for r in rows)
        ]

        def geo(name: str, rows: list[dict[str, Any]] = rows) -> float:
            logs = [math.log(r["cout"]["raw"] / r["cout"][name]) for r in rows]
            return math.exp(sum(logs) / len(logs))

        truth_gain = geo("truth")
        entry: dict[str, Any] = {"n": len(rows), "truth_gain": truth_gain}
        for name in names:
            gain = geo(name)
            entry[name] = {
                "gain": gain,
                "share_of_truth_gain_log": math.log(gain) / math.log(truth_gain)
                if truth_gain > 1
                else math.nan,
                "share_plan_changed": sum(
                    r["plan"][name] != r["plan"]["raw"] for r in rows
                )
                / len(rows),
            }
        out[source] = entry
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
    parser.add_argument("--native-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    estimates_dir, subplan_file, query_file, dbname = WORKLOAD_PATHS[args.workload]
    if args.workload != "stats":
        args.dsn = with_dbname(args.dsn, dbname)
    truth = load_estimates("truecardinality", directory=estimates_dir)
    real: dict[Key, float] = {}
    if (estimates_dir / "lpbound_subquery_estimations.csv").exists():
        real = load_estimates("lpbound", directory=estimates_dir)
    sources = {
        "native": load_estimates("native", args.native_csv, directory=estimates_dir)
    }
    if (estimates_dir / "postgres_subquery_estimations.csv").exists():
        sources["postgres"] = load_estimates("postgres", directory=estimates_dir)
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
    done = set()
    if args.output.exists():
        done = {r["query_id"] for r in _read_jsonl(args.output)}
    cursor = reconnect(args.dsn)
    with args.output.open("a") as sink:
        for query_id in sorted(set(queries) - done):
            keys = subplans.get(query_id, [])
            record: dict[str, Any] = {"query_id": query_id}
            for source, est in sources.items():
                if not keys or not all(k in est and k in truth for k in keys):
                    continue
                raw = {k: est[k] for k in keys}
                plan_digest: dict[str, str] = {}
                cout: dict[str, float] = {}
                ok = True
                for name, values in policies(
                    keys, raw, {k: truth[k] for k in keys}, real
                ).items():
                    digest, plan = plan_of(
                        cursor, hint_text(values) + queries[query_id]
                    )
                    value = true_cout(plan, query_id, truth)
                    if value is None:
                        ok = False
                        break
                    plan_digest[name], cout[name] = digest, value
                if ok:
                    record[source] = {"plan": plan_digest, "cout": cout}
                else:
                    record.setdefault("skipped_unknown_join_size", []).append(source)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            logger.info("query %d done", query_id)
    records = [r for r in _read_jsonl(args.output) if any(s in r for s in SOURCES)]
    skipped = [
        s
        for r in _read_jsonl(args.output)
        for s in r.get("skipped_unknown_join_size", [])
    ]
    summary = {
        "n_skipped_unknown_join_size": {s: skipped.count(s) for s in SOURCES},
        **summarise(records),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    logger.info("%s", json.dumps(summary)[:1500])


if __name__ == "__main__":
    main()
