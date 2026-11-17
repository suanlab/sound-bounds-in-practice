"""Checks answering the simulated PVLDB review, from stored results.

Nothing is re-measured. Five questions:

* **Censoring.** A plan that hits the 300 s timeout is kept at 300 s. For a
  speedup ``T_base / T_policy`` the capped ratio is a lower bound on the true
  ratio when only the baseline is censored, an upper bound when only the policy
  is, exact when both pick the same plan, and of unknown direction when two
  *different* plans are both censored. For every comparison of every executed
  run the paper reports (injection, tightness points, forced orders), this
  counts each case.
* **Headroom.** Planning on the true join sizes is a measured reference, not a
  bound: per source, the queries on which a guard beats the truth's plan by
  more than 10%, and those on which the truth's plan is >10% slower than the
  source's own.
* **Agreement coverage.** Queries whose every sub-plan has an exact count
  compared with the artifact's truth, and the exact one-sided 95% bound on the
  share of such queries with a disagreement (none found).
* **Positivity.** The certificates floor sizes at 1 (``max(., 1)``) and give a
  timed-out sample ``L = 1``; that is sound only when every sub-join is
  non-empty. This checks every true size and exact count the certificates use.
* **Held-out exclusions.** The range of each certificate share (C1-C4) if the
  9 queries excluded for a missing count had been certified or not.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.review_checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

from experiments.pg_injection.analyze_injection import pairs_for
from experiments.pg_injection.inject_plans import _read_jsonl, _strip

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RUNS = {
    "pg13_stats": Path("results/pg_injection/v3/stats_inject.jsonl"),
    "pg13_joblight": Path("results/pg_injection/v3/joblight_inject.jsonl"),
    "pg16_stats": Path("results/pg_injection/pg16_jitoff/stats_inject.jsonl"),
}
# Every other executed run the paper reports a speedup from.
OTHER_RUNS = {
    "pg16_stats_jit_on": Path("results/pg_injection/pg16/stats_inject.jsonl"),
    "pg13_points": Path("results/pg_injection/v3/tightness_points_stats.jsonl"),
    "pg16_points": Path(
        "results/pg_injection/pg16_jitoff/tightness_points_stats.jsonl"
    ),
    "pg16_points_jit_on": Path(
        "results/pg_injection/pg16/tightness_points_stats.jsonl"
    ),
}
FORCED = Path("results/pg_injection/v3/forced_orders.jsonl")
FORCED_PAIRS = (
    ("native", "cout_opt"),
    ("native", "certified"),
    ("native", "native_lo"),
    ("native_lo", "cout_opt"),
)
GUARDS = ("upper", "lower", "bi")
BAND = 1.1
COUNTS = {
    "stats": Path("results/pg_injection/subset_counts.jsonl"),
    "joblight": Path("results/pg_injection/joblight/subset_counts.jsonl"),
}
SIZES = (
    Path("results/plan_guard/subjoin_sizes_maxdeg_fixed.jsonl"),
    Path("results/plan_certificate/joblight_subjoin_sizes.jsonl"),
)
EXTRA_COUNTS = (
    Path("results/pg_injection/subset_counts_r025.jsonl"),
    Path("results/heldout_jobrange/subset_counts.jsonl"),
)
HELDOUT = Path("results/heldout_jobrange/confirmatory_results.json")


def censoring_case(record: dict[str, Any], base: str, policy: str) -> str:
    """How the timeout cap affects the speedup of ``policy`` over ``base``."""
    b, p = record["policy_plan"][base], record["policy_plan"][policy]
    cb, cp = record["plan_censored"][b], record["plan_censored"][p]
    if not cb and not cp:
        return "none"
    if cb and cp:
        return "both_same_plan" if b == p else "both_distinct_plans"
    return "base_only" if cb else "policy_only"


def reported_pairs(policies: set[str]) -> list[tuple[str, str]]:
    """Every comparison of an injection or tightness run the paper can report.

    The analysis pairs of ``analyze_injection`` (guards, truth, the sample and
    LpBound against each source's own estimates; sources against the native
    plan), every policy against the native plan and against our own injected
    estimates (guarded against native), and, for the tightness runs, every
    policy against ``raw``.
    """
    pairs = set(pairs_for(policies))
    for base in ("native", "native_raw", "raw"):
        if base in policies:
            pairs |= {(base, p) for p in policies if p != base}
    return sorted(pairs)


def censoring(
    records: list[dict[str, Any]], pairs: list[tuple[str, str]]
) -> dict[str, Any]:
    """Censoring cases of the given comparisons, with a sensitivity mean."""
    out: dict[str, Any] = {}
    for base, policy in pairs:
        cases: dict[str, Any] = {}
        logs: list[float] = []
        kept: list[float] = []
        for r in records:
            if base not in r["policy_plan"] or policy not in r["policy_plan"]:
                continue  # e.g. planning on L-hat needs every sample to finish
            case = censoring_case(r, base, policy)
            if case != "none":
                cases.setdefault(case, []).append(int(r["query_id"]))
            t_base = float(r["plan_ms"][r["policy_plan"][base]])
            t_pol = float(r["plan_ms"][r["policy_plan"][policy]])
            logs.append(math.log(max(t_base, 1e-9) / max(t_pol, 1e-9)))
            if case != "both_distinct_plans":
                kept.append(logs[-1])
        # Sensitivity: leave out the queries whose ratio has unknown direction.
        cases["geo_all"] = math.exp(sum(logs) / len(logs))
        cases["geo_without_distinct_both"] = math.exp(sum(kept) / len(kept))
        out[f"{base}->{policy}"] = cases
    return out


def forced_as_injection(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Forced-order runs in the injection format, plans fingerprinted as there.

    Two roles whose plans share a fingerprint keep separate times here (each
    order was executed on its own), so each role gets its own key.
    """
    out = []
    for r in records:
        policy_plan, plan_ms, plan_censored = {}, {}, {}
        for role, ms in r["ms"].items():
            digest = hashlib.sha256(
                json.dumps(_strip(r["plans"][role]), sort_keys=True).encode()
            ).hexdigest()[:16]
            key = f"{digest}:{role}"
            policy_plan[role] = key
            plan_ms[key] = ms
            plan_censored[key] = r["censored"][role]
        out.append(
            {
                "query_id": r["query_id"],
                "policy_plan": policy_plan,
                "plan_ms": plan_ms,
                "plan_censored": plan_censored,
                "digest": {k: v.split(":")[0] for k, v in policy_plan.items()},
            }
        )
    return out


def forced_censoring(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Censoring cases of the forced-order comparisons, by plan fingerprint."""
    out: dict[str, Any] = {}
    for base, policy in FORCED_PAIRS:
        cases: dict[str, Any] = {}
        logs: list[float] = []
        kept: list[float] = []
        for r in records:
            if base not in r["policy_plan"] or policy not in r["policy_plan"]:
                continue
            cb = r["plan_censored"][r["policy_plan"][base]]
            cp = r["plan_censored"][r["policy_plan"][policy]]
            t_base = float(r["plan_ms"][r["policy_plan"][base]])
            t_pol = float(r["plan_ms"][r["policy_plan"][policy]])
            logs.append(math.log(max(t_base, 1e-9) / max(t_pol, 1e-9)))
            case = "none"
            if cb and cp:
                same = r["digest"][base] == r["digest"][policy]
                case = "both_same_plan" if same else "both_distinct_plans"
            elif cb or cp:
                case = "base_only" if cb else "policy_only"
            if case != "none":
                cases.setdefault(case, []).append(int(r["query_id"]))
            if case != "both_distinct_plans":
                kept.append(logs[-1])
        cases["geo_all"] = math.exp(sum(logs) / len(logs))
        cases["geo_without_distinct_both"] = math.exp(sum(kept) / len(kept))
        out[f"{base}->{policy}"] = cases
    return out


def censoring_summary(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Over every comparison: the policy-only cases, the largest count of
    distinct censored pairs, and the largest sensitivity shift."""
    policy_only = [
        [run, pair, q]
        for run, pairs in runs.items()
        for pair, cases in pairs.items()
        for q in cases.get("policy_only", [])
    ]
    shifts = [
        abs(c["geo_without_distinct_both"] / c["geo_all"] - 1)
        for pairs in runs.values()
        for c in pairs.values()
        if "geo_all" in c
    ]
    return {
        "n_comparisons": sum(len(p) for p in runs.values()),
        "policy_only": policy_only,
        "policy_only_queries": sorted({(run, q) for run, _, q in policy_only}),
        "max_distinct_both": max(
            len(c.get("both_distinct_plans", []))
            for pairs in runs.values()
            for c in pairs.values()
        ),
        "max_sensitivity_shift": max(shifts),
    }


def headroom_exceedance(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per source and guard, queries where the guard beats the truth's plan."""
    policies = set(records[0]["policy_plan"])
    sources = sorted(p[: -len("_raw")] for p in policies if p.endswith("_raw"))
    out: dict[str, Any] = {}

    def ms(r: dict[str, Any], policy: str) -> float:
        return max(float(r["plan_ms"][r["policy_plan"][policy]]), 1e-9)

    for source in sources:
        raw = f"{source}_raw"
        entry: dict[str, Any] = {
            "truth_slower_than_raw": sum(
                ms(r, raw) / ms(r, "truth") < 1 / BAND for r in records
            )
        }
        for guard in GUARDS:
            policy = f"{source}_{guard}"
            if policy in policies:
                entry[f"{guard}_beats_truth"] = sum(
                    ms(r, "truth") / ms(r, policy) > BAND for r in records
                )
        out[source] = entry
    return out


def coverage(path: Path) -> dict[str, Any]:
    """Queries whose every sub-plan was compared, and the zero-event bound."""
    by_query: dict[int, list[bool]] = {}
    for r in _read_jsonl(path):
        ok = not r["exact_censored"] and r.get("truth_artifact") is not None
        if ok and abs(r["exact_pg"] - r["truth_artifact"]) > 0.5:
            msg = f"disagreement in {path}: {r['query_id']} {r['aliases']}"
            raise AssertionError(msg)
        by_query.setdefault(int(r["query_id"]), []).append(ok)
    complete = sum(all(v) for v in by_query.values())
    return {
        "n_queries": len(by_query),
        "n_any_compared": sum(any(v) for v in by_query.values()),
        "n_all_compared": complete,
        "n_subplans": sum(len(v) for v in by_query.values()),
        "n_subplans_compared": sum(sum(v) for v in by_query.values()),
        "upper95_share_complete_queries": 1 - 0.05 ** (1 / complete),
    }


def positivity() -> dict[str, Any]:
    """Smallest true size and exact count the certificates use."""
    out: dict[str, Any] = {}
    for path in SIZES:
        truths = [
            float(s["truth"])
            for line in path.read_text().splitlines()
            if line
            for s in json.loads(line)["subjoins"]
        ]
        out[str(path)] = {"n": len(truths), "min_truth": min(truths)}
    for path in (*COUNTS.values(), *EXTRA_COUNTS):
        exact = [
            float(r["exact_pg"]) for r in _read_jsonl(path) if not r["exact_censored"]
        ]
        out[str(path)] = {"n_exact": len(exact), "min_exact": min(exact)}
    return out


def heldout_ranges() -> dict[str, Any]:
    """C1-C4 shares if the queries excluded for a missing count were included."""
    data = json.loads(HELDOUT.read_text())["a_posteriori"]
    n = int(data["n_queries"])
    missing = int(data["excluded"]["missing_count"])
    out: dict[str, Any] = {"n_observed": n, "n_missing_count": missing}
    for cell, share in data["shares"].items():
        hits = round(share * n)
        out[cell] = {
            "observed": hits / n,
            "if_none_certified": hits / (n + missing),
            "if_all_certified": (hits + missing) / (n + missing),
        }
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/review/review_checks.json")
    )
    args = parser.parse_args()
    inputs = [
        *RUNS.values(),
        *OTHER_RUNS.values(),
        FORCED,
        *COUNTS.values(),
        *SIZES,
        *EXTRA_COUNTS,
        HELDOUT,
    ]
    out: dict[str, Any] = {
        "experiment": "review_checks",
        "preregistered": False,
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "censoring": {},
        "headroom_exceedance": {},
    }
    for name, path in {**RUNS, **OTHER_RUNS}.items():
        records = _read_jsonl(path)
        pairs = reported_pairs(set(records[0]["policy_plan"]))
        out["censoring"][name] = censoring(records, pairs)
        if name in RUNS:
            out["headroom_exceedance"][name] = headroom_exceedance(records)
    out["censoring_forced_orders"] = forced_censoring(
        forced_as_injection(_read_jsonl(FORCED))
    )
    out["censoring_summary"] = censoring_summary(
        {**out["censoring"], "forced_orders": out["censoring_forced_orders"]}
    )
    out["agreement_coverage"] = {k: coverage(p) for k, p in COUNTS.items()}
    out["positivity"] = positivity()
    out["heldout_exclusion_ranges"] = heldout_ranges()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("censoring: %s", json.dumps(out["censoring_summary"]))
    logger.info("forced orders: %s", json.dumps(out["censoring_forced_orders"]))
    logger.info("headroom: %s", json.dumps(out["headroom_exceedance"]))
    logger.info("coverage: %s", json.dumps(out["agreement_coverage"]))
    logger.info("positivity: %s", json.dumps(out["positivity"]))
    logger.info("heldout: %s", json.dumps(out["heldout_exclusion_ranges"]))


if __name__ == "__main__":
    main()
