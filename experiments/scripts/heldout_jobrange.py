"""Held-out JOB-range confirmation (``docs/icde_2027/PREREGISTRATION_JOBRANGE.md``).

Three steps, run after ``subset_counts`` and ``native_estimates_incontext`` on
the ``jobrange_heldout`` workload:

* ``native-plans``: PostgreSQL's own plan of every sampled query (no hints);
* ``truth``: the artifact-format truth file from our exact counts (sub-plans
  whose count timed out or crashed are left out, which excludes their query);
* ``analyze``: the pre-registered tests C1-C7, after ``tightness_sweep``.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.heldout_jobrange native-plans
    PYTHONHASHSEED=0 python -m experiments.scripts.heldout_jobrange truth
    PYTHONHASHSEED=0 python -m experiments.scripts.heldout_jobrange analyze
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.pg_injection.inject_plans import (
    WORKLOAD_PATHS,
    _read_jsonl,
    load_estimates,
    plan_of,
    reconnect,
)
from experiments.pg_injection.run_forced_orders import join_sets
from experiments.pg_injection.subset_counts import (
    FROM_ITEM,
    count_sql,
    load_subqueries,
    with_dbname,
)
from experiments.pg_injection.tightness_sweep import summarise
from experiments.scripts.plan_certificate_joint import incidence, left_deep_orders

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HELDOUT = Path("results/heldout_jobrange")
SAMPLE = HELDOUT / "sample.json"
COUNTS = HELDOUT / "subset_counts.jsonl"
NATIVE_PLANS = HELDOUT / "native_plans.jsonl"
TRUTH = HELDOUT / "estimates/truecardinality_subquery_estimations.csv"
NATIVE_CSV = HELDOUT / "native_incontext_estimations.csv"
SWEEP = HELDOUT / "tightness_sweep.jsonl"
RESULT = HELDOUT / "confirmatory_results.json"
TARGET = 2.0


def sample_ids() -> list[int]:
    """The pre-registered query ids."""
    ids: list[int] = json.loads(SAMPLE.read_text())["query_ids"]
    return ids


def native_plans(dsn: str) -> None:
    """Record PostgreSQL's unhinted plan (join alias sets) for every query."""
    _, _, query_file, dbname = WORKLOAD_PATHS["jobrange_heldout"]
    lines = [raw for raw in query_file.read_text().splitlines() if raw.strip()]
    cursor = reconnect(with_dbname(dsn, dbname))
    with NATIVE_PLANS.open("w") as sink:
        for qid in sample_ids():
            sql = count_sql(lines[qid - 1].strip().rstrip(";")) + ";"
            digest, plan = plan_of(cursor, sql)
            sets: list[frozenset[str]] = []
            join_sets(plan, sets)
            sink.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "digest": digest,
                        "join_sets": [sorted(s) for s in sets],
                    }
                )
                + "\n"
            )
    logger.info("native plans of %d queries written", len(sample_ids()))


def build_truth() -> None:
    """Artifact-format truth file from our exact counts."""
    rows = [r for r in _read_jsonl(COUNTS) if r["exact_pg"] is not None]
    lines = ["QueryID,Tables,truecardinality_Estimate"] + [
        ",".join(
            (
                str(r["query_id"]),
                " ".join(a.upper() for a in r["aliases"]),
                f"{r['exact_pg']:.17g}",
            )
        )
        for r in rows
    ]
    TRUTH.write_text("\n".join(lines) + "\n")
    logger.info("truth for %d sub-plans written", len(rows))


def registered_subplans() -> dict[int, set[frozenset[str]]]:
    """The registered sub-plan list (``jobrange_heldout_subqueries.csv``) per query."""
    out: dict[int, set[frozenset[str]]] = {}
    for row in load_subqueries(WORKLOAD_PATHS["jobrange_heldout"][1]):
        out.setdefault(row["query_id"], set()).add(frozenset(row["aliases"]))
    return out


def included(
    counts: list[dict[str, Any]],
) -> tuple[dict[int, dict[frozenset[str], dict[str, Any]]], dict[str, int]]:
    """Queries that pass the registered exclusions, with their sub-plan counts.

    Excluded (counted once, first failing rule): the full query is not in the
    registered list (no final result to analyse); a registered sub-plan without
    an exact count (missing row, timeout or crash); a missing in-context
    estimate; a missing native plan; a native plan joining a set outside the
    registered list.
    """
    rows: dict[int, dict[frozenset[str], dict[str, Any]]] = {}
    for r in counts:
        rows.setdefault(r["query_id"], {})[frozenset(r["aliases"])] = r
    estimates = load_estimates("native", NATIVE_CSV, directory=TRUTH.parent)
    plans = {r["query_id"]: r for r in _read_jsonl(NATIVE_PLANS)}
    listed = registered_subplans()
    _, _, query_file, _ = WORKLOAD_PATHS["jobrange_heldout"]
    lines = [raw for raw in query_file.read_text().splitlines() if raw.strip()]
    excluded = dict.fromkeys(
        (
            "final_not_listed",
            "missing_count",
            "missing_estimate",
            "missing_plan",
            "plan_outside_list",
        ),
        0,
    )
    keep: dict[int, dict[frozenset[str], dict[str, Any]]] = {}
    for qid in sample_ids():
        expected = listed.get(qid, set())
        got = rows.get(qid, {})
        final = frozenset(a.lower() for _, a in FROM_ITEM.findall(lines[qid - 1]))
        if final not in expected:
            excluded["final_not_listed"] += 1
            continue
        if not expected or any(
            s not in got or got[s]["exact_pg"] is None for s in expected
        ):
            excluded["missing_count"] += 1
            continue
        if any((qid, tuple(sorted(s))) not in estimates for s in expected):
            excluded["missing_estimate"] += 1
            continue
        if qid not in plans:
            excluded["missing_plan"] += 1
            continue
        if any(frozenset(s) not in expected for s in plans[qid]["join_sets"]):
            excluded["plan_outside_list"] += 1
            continue
        keep[qid] = {s: got[s] for s in expected}
    return keep, excluded


def a_posteriori(counts: list[dict[str, Any]]) -> dict[str, Any]:
    """C1-C5: a-posteriori certificates of the native plan on 3+-relation queries."""
    keep, excluded = included(counts)
    plans = {r["query_id"]: r for r in _read_jsonl(NATIVE_PLANS)}
    excluded = {**excluded, "fewer_than_3": 0}
    cells: dict[str, list[bool]] = {
        f"{v}|{f}": [] for v in ("cout", "cout_minus_final") for f in ("sample", "none")
    }
    violations = 0
    n = 0
    for qid, subplans in keep.items():
        full: frozenset[str] = max(list(subplans), key=len)
        aliases = sorted(full)
        if len(aliases) < 3:
            excluded["fewer_than_3"] += 1
            continue
        sets = {frozenset(s) for s in plans[qid]["join_sets"]}
        index = {s: i for i, s in enumerate(subplans)}
        truth = np.array([max(float(subplans[s]["exact_pg"]), 1.0) for s in index])
        floor = np.array(
            [
                1.0
                if subplans[s]["sample_censored"]
                else max(float(subplans[s]["lower_sound"]), 1.0)
                for s in index
            ]
        )
        orders = left_deep_orders(aliases, index)
        final = index[frozenset(aliases)]
        n += 1
        for variant in ("cout", "cout_minus_final"):
            include = variant == "cout"
            inc = incidence(orders, index, include_final=include)
            observed = np.zeros(len(index), dtype=bool)
            observed[[index[s] for s in sets]] = True
            if not include:
                observed[final] = False
            c_native = float(sum(truth[index[s]] for s in sets)) - (
                0.0 if include else float(truth[final])
            )
            c_true = inc @ truth
            regret = c_native / min(c_native, float(c_true.min()))
            for name, lower in (("sample", floor), ("none", np.ones_like(truth))):
                low = np.where(observed, truth, lower)
                bound = c_native / min(c_native, float((inc @ low).min()))
                violations += regret > bound * (1 + 1e-9)
                cells[f"{variant}|{name}"].append(bound <= TARGET)
    shares = {k: sum(v) / len(v) for k, v in cells.items()}
    return {
        "n_queries": n,
        "excluded": excluded,
        "shares": shares,
        "violations": violations,
    }


def analyze() -> None:
    """Evaluate the pre-registered decision rules."""
    counts = _read_jsonl(COUNTS)
    post = a_posteriori(counts)
    s = post["shares"]
    keep, _ = included(counts)
    records = [r for r in _read_jsonl(SWEEP) if r["query_id"] in keep and "native" in r]
    n_sweep_unknown = sum(
        1
        for r in _read_jsonl(SWEEP)
        if r["query_id"] in keep and "native" in r.get("skipped_unknown_join_size", [])
    )
    sweep = summarise(records)["native"]
    changed = {
        k: sweep[f"bi_k{k}"]["share_plan_changed"]
        for k in ("1", "1.5", "2", "4", "8", "16", "32", "64")
    }
    steps = list(changed.values())
    truth_gain = sweep["truth_gain"]

    def verdict(confirmed: bool, falsified: bool) -> str:
        return (
            "confirmed" if confirmed else "falsified" if falsified else "inconclusive"
        )

    tests = {
        "C1": verdict(s["cout|none"] >= 0.70, s["cout|none"] < 0.50),
        "C2": verdict(s["cout|sample"] >= 0.80, s["cout|sample"] < 0.60),
        "C3": verdict(
            s["cout_minus_final|none"] <= 0.05, s["cout_minus_final|none"] > 0.10
        ),
        "C4": verdict(
            0.10 <= s["cout_minus_final|sample"] <= 0.70,
            s["cout_minus_final|sample"] < 0.05 or s["cout_minus_final|sample"] > 0.80,
        ),
        "C5": verdict(post["violations"] == 0, post["violations"] > 0),
    }
    c6 = changed["2"] >= 2 * changed["16"] and all(
        b - a <= 0.05 for a, b in itertools.pairwise(steps)
    )
    tests["C6"] = verdict(c6, not c6)
    if truth_gain < 1.05:
        tests["C7"] = "not testable"
    else:
        half = [sweep[f"bi_k{k}"]["share_of_truth_gain_log"] for k in ("2", "16")]
        c7 = half[0] >= 0.5 and half[1] < 0.5
        tests["C7"] = verdict(c7, not c7)
    payload = {
        "preregistration": "docs/icde_2027/PREREGISTRATION_JOBRANGE.md",
        "a_posteriori": post,
        "n_sweep_queries": len(records),
        "n_sweep_excluded_unknown_guarded_join_size": n_sweep_unknown,
        "plan_changed_by_k": changed,
        "truth_gain": truth_gain,
        "share_of_truth_gain_log": {
            k: sweep[f"bi_k{k}"]["share_of_truth_gain_log"] for k in changed
        },
        "tests": tests,
    }
    RESULT.write_text(json.dumps(payload, indent=2, default=lambda x: None) + "\n")
    logger.info("%s", json.dumps(payload, default=lambda x: None))
    if any(math.isnan(v) for v in s.values()):
        msg = "an a-posteriori share is undefined"
        raise RuntimeError(msg)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("native-plans", "truth", "analyze"))
    parser.add_argument(
        "--dsn",
        default=(
            "host=127.0.0.1 port=55432 dbname=stats user=postgres password=postgres"
        ),
    )
    args = parser.parse_args()
    if args.step == "native-plans":
        native_plans(args.dsn)
    elif args.step == "truth":
        build_truth()
    else:
        analyze()


if __name__ == "__main__":
    main()
