"""Check every headline number of the ICDE 2027 draft against its result file.

Same contract as ``audit_paper_numbers`` (render the number from the JSON exactly
as the paper writes it and require that string in the LaTeX source); every check
is anchored to surrounding words so that an unrelated occurrence of the same
digits elsewhere cannot satisfy it.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.audit_icde_numbers
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

from experiments.scripts.audit_common import Check, _dig
from experiments.scripts.render_heldout_section import render as render_heldout
from experiments.scripts.render_icde_tables import (
    DIRECTION_SOURCES,
    EXECUTION_SOURCES,
    TABLE_SOURCES,
    certificate_rows,
    direction_rows,
    execution_rows,
    pg16_rows,
    replication_rows,
    sampling_rate_rows,
    tightness_rows,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

R = Path("results")
BASE = R / "utility_table/cardinality_baselines_pg.json"
TIGHT = R / "utility_table/tightness_all_maxdegfix_t1.json"
COUNTS = R / "pg_injection/subset_counts_summary.json"
JL_COUNTS = R / "pg_injection/joblight/subset_counts_summary.json"
SELECTIVE = R / "plan_certificate/selective_sampling_v3.json"
PG25 = R / "plan_certificate/plan_certificate_joint_pg_r025.json"
PG10 = R / "plan_certificate/plan_certificate_joint_pg_r010.json"
REG = R / "plan_certificate/plan_certificate.json"
COMM = R / "plan_certificate/plan_certificate_commutation.json"
RATE = R / "pg_injection/v2/sampling_rate.json"
JLCERT = R / "plan_certificate/plan_certificate_joint_joblight_pg_r010.json"
CTX = R / "plan_certificate/certificate_context.json"
MAXDEG = R / "plan_guard/subjoin_sizes_maxdeg_fixed.json"
PG25_N = "summary::lpboundxsample|cout::n_queries"
SB25_N = "summary::safeboundxsample|cout::n_queries"
JL_N = "summary::lpboundxsample|cout::n_queries"
MATCHED_N = "pairs::postgres_raw->truth::matched_lhat::n"
GVN = R / "pg_injection/v3/guarded_vs_native.json"
INRUN = R / "pg_injection/v3/sampling_overhead_inrun_summary.json"
FORCED = R / "pg_injection/v3/forced_orders_summary.json"
SELDEN = R / "plan_certificate/selective_denominators.json"
PG_UPPER = "pairs::postgres_raw->postgres_upper::geo_speedup::point"
MECH = R / "plan_certificate/certificate_mechanisms.json"
JL_RHO_ACCURACY = "h1_shared_mass::joblight_r010::accuracy_at_threshold"
MINT = R / "plan_certificate/mechanism_intervals.json"
POC = R / "plan_certificate/price_of_certification.json"
POC_GREEDY_MATTERS = "summary::choice_matters::share_greedy_optimal"
FUNNEL = R / "pg_injection/v3/guard_funnel.json"
POINTS = R / "pg_injection/v3/tightness_points_summary.json"
POINTS_K2 = "time::bi_k2::share_of_truth_speedup::point"
HELDOUT = R / "heldout_jobrange/confirmatory_results.json"
WITNESS = R / "pg_injection/v3/witness_floor_summary.json"
HINT = R / "heldout_jobrange/heldout_intervals.json"
PG16CMP = R / "pg_injection/pg16/pg16_comparison.json"
PG16PROT = R / "pg_injection/pg16/stats_protocol.json"
PG16JO = R / "pg_injection/pg16_jitoff/pg16_comparison.json"
PG16JOPROT = R / "pg_injection/pg16_jitoff/stats_protocol.json"
JRL_STATS = R / "pg_injection/v3/joinrel_log_stats_pg13.summary.json"
JRL_JL = R / "pg_injection/v3/joinrel_log_joblight_pg13.summary.json"
PREFIX = R / "pg_injection/v3/prefix_estimates_stats_pg13.summary.json"
SEEDS = R / "pg_injection/v3/seed_robustness.json"
BRACKET = R / "pg_injection/v3/native_bracket.json"
TSENS = R / "pg_injection/v3/stats_timing_sensitivity.json"
JL_NOFINAL = "summary::lpboundxsample|cout_minus_final::share_joint_bound_le_2::point"
REPRO = R / "pg_injection/v2/native_reproduction_check.json"
COUNTS_JSONL_QUERIES = R / "pg_injection/v2/agreement_queries.json"
BASE25 = R / "utility_table/cardinality_baselines_pg_r025.json"
PG25_SHARE = "summary::lpboundxsample|cout::share_joint_bound_le_2::point"
JLB = R / "utility_table/cardinality_baselines_pg_joblight.json"
S2 = R / "pg_injection/v3/stats_summary.json"
J2 = R / "pg_injection/v3/joblight_summary.json"
SP = R / "pg_injection/v3/stats_protocol.json"
JP = R / "pg_injection/v3/joblight_protocol.json"
REP = R / "pg_injection/v3/artifact_replication.json"


def thousands(value: float) -> str:
    """``109{,}640`` style integer."""
    return f"{round(value):,}".replace(",", "{,}")


def pct(value: float) -> str:
    """Share as a one-decimal percentage."""
    return f"{100 * value:.1f}"


NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def normalise(text: str) -> str:
    """LaTeX source without comments, with every run of whitespace one space.

    Comments must not satisfy a check, and reflowing the text must not break one.
    """
    lines = (re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())
    return re.sub(r"\s+", " ", " ".join(lines))


def superseded_entries(path: Path = R / "SUPERSEDED.md") -> list[str]:
    """Result paths that ``results/SUPERSEDED.md`` lists as superseded or withdrawn.

    Table rows list superseded files in their first cell (rows marked "not
    superseded" or "still read" are skipped); bullet entries list them before
    the first em dash.
    """
    entries: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("| `"):
            cells = line.split("|")
            if "not superseded" in line or "still read" in line:
                continue
            entries += re.findall(r"`([^`]+)`", cells[1])
        elif line.startswith("- `"):
            entries += re.findall(r"`([^`]+)`", line.split(" — ", 1)[0])
    return [e.removeprefix("results/") for e in entries]


def is_superseded(source: Path, entries: list[str]) -> str | None:
    """The SUPERSEDED.md entry that covers ``source``, if any."""
    relative = source.as_posix().removeprefix("results/")
    for entry in entries:
        if entry.endswith("/") and relative.startswith(entry):
            return entry
        if "/" in entry and fnmatch.fnmatch(relative, entry):
            return entry
        if "/" not in entry and fnmatch.fnmatch(source.name, entry):
            return entry
    return None


def native_guard_speedups(pairs: dict[str, Any]) -> list[float]:
    """Speedups of the three sound guards on our instance's own estimates."""
    return [
        pairs[f"native_raw->native_{g}"]["geo_speedup"]["point"]
        for g in ("upper", "lower", "bi")
    ]


def directions_kept(_: object) -> bool:
    """Whether every reported comparison keeps its direction under the
    alternative tie bands and timeout caps (both benchmarks)."""
    for path in (TSENS, R / "pg_injection/v3/joblight_timing_sensitivity.json"):
        data = json.loads(path.read_text())
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            geos = [entry[f"geo_cap_{c}s"] for c in (300, 120, 60)]
            geos.append(entry["geo_quiet_start"])
            if len({g > 1.0 for g in geos if abs(g - 1.0) > 0.02}) > 1:
                return False
    return True


PARITY_BAND = 1.10
NO_FINAL = "cout_minus_final"


def pct_iv1(stat: dict[str, float]) -> str:
    """A share with its interval, in percent to one decimal."""
    return (
        f"{100 * stat['point']:.1f}\\% "
        f"[{100 * stat['ci_low']:.1f}, {100 * stat['ci_high']:.1f}]"
    )


def pct_iv(stat: dict[str, float]) -> str:
    """A share with its interval, in percent, as the paper writes it."""
    return (
        f"{round(100 * stat['point'])}\\% "
        f"[{round(100 * stat['ci_low'])}, {round(100 * stat['ci_high'])}]"
    )


def real_bracket_share() -> int:
    """Share of our estimates' headroom the real sound guards recover (log)."""
    pairs = json.loads(S2.read_text())["pairs"]
    gain = pairs["native_raw->native_bi"]["geo_speedup"]["point"]
    headroom = pairs["native_raw->truth"]["geo_speedup"]["point"]
    return round(100 * math.log(gain) / math.log(headroom))


def log_share(block: dict[str, Any]) -> int:
    """Share of the headroom that both sound guards recover (log scale), in %."""
    return round(
        100 * math.log(block["both"]["point"]) / math.log(block["headroom"]["point"])
    )


def oracle_above_range() -> str:
    """Range over the three native-time references of the share of choice-matters
    certified queries whose oracle sampling cost exceeds execution."""
    block = json.loads(POC.read_text())["summary"]["choice_matters"]
    shares = [
        round(100 * block[f"{name}_share_oracle_above_exec"])
        for name in ("execution", "forced_order", "in_run")
    ]
    return f"{min(shares)}--{max(shares)}\\%"


def h4_pct(
    data: dict[str, Any], bench: str, variant: str, field: str, digits: int = 1
) -> str:
    """An a-posteriori share from ``certificate_mechanisms``, as a percentage."""
    key = "h4_a_posteriori" if bench == "stats" else "h4_a_posteriori_joblight"
    return f"{100 * data[key][variant]['all'][field]:.{digits}f}"


NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def balance_reversals() -> tuple[int, float]:
    """Comparisons whose faster-minus-slower balance changes sign across the
    5/10/20% bands, and the band around parity that all of them lie within."""
    flipped = []
    for path in (TSENS, R / "pg_injection/v3/joblight_timing_sensitivity.json"):
        for entry in json.loads(path.read_text()).values():
            if not isinstance(entry, dict):
                continue
            balance = [
                round(
                    entry["n"]
                    * (
                        entry[f"band_{b}"]["share_faster"]
                        - entry[f"band_{b}"]["share_slower"]
                    )
                )
                for b in (5, 10, 20)
            ]
            if len({(x > 0) - (x < 0) for x in balance}) > 1:
                g = entry["geo_cap_300s"]
                flipped.append(max(g, 1 / g))
    if max(flipped) > PARITY_BAND:
        msg = f"a balance reversal at {max(flipped):.3f}x lies beyond the stated band"
        raise ValueError(msg)
    return len(flipped), PARITY_BAND


def iv(stat: dict[str, float]) -> str:
    """A speedup with its interval, as the paper writes it."""
    return f"${stat['point']:.2f}\\times$ [{stat['ci_low']:.2f}, {stat['ci_high']:.2f}]"


def value(source: Path, path: str) -> float:
    """One number from another result file, for checks that compare two."""
    return float(_dig(json.loads(source.read_text()), path))


def median_ratio(pairs: list[dict[str, Any]]) -> float:
    """Median over queries of the max-degree / LpBound tightness ratio."""
    by_query: dict[str, dict[str, float]] = {}
    for row in pairs:
        by_query.setdefault(row["query_id"], {})[row["certificate"]] = row[
            "bound_over_true"
        ]
    ratios = sorted(v["max_degree"] / v["lpbound"] for v in by_query.values())
    middle = len(ratios) // 2
    if len(ratios) % 2:
        return ratios[middle]
    return (ratios[middle - 1] + ratios[middle]) / 2


def tight(field: str) -> float:
    """A value of the tightness summary, for checks that render ratios."""
    return float(_dig(json.loads(TIGHT.read_text()), f"certificates.{field}"))


CHECKS: tuple[Check, ...] = (
    # --- setup / ground truth
    Check(
        "truth agreement compared",
        COUNTS,
        "truth_agreement.compared",
        lambda v: f"({thousands(v)} of {thousands(v)} STATS-CEB",
    ),
    Check(
        "truth agreement mismatches are zero",
        COUNTS,
        "truth_agreement",
        lambda v: (
            f"of {thousands(v['compared'])} STATS-CEB"
            if v["mismatches"] == 0
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "JOB-light truth agreement",
        JL_COUNTS,
        "truth_agreement.compared",
        lambda v: f"{round(v)} of {round(v)} on JOB-light",
    ),
    Check(
        "query-level agreement bound",
        COUNTS_JSONL_QUERIES,
        "",
        lambda v: (
            f"{100 * (1 - 0.05 ** (1 / v[0])):.1f}\\% on the share of queries with "
            f"one, {100 * (1 - 0.05 ** (1 / v[1])):.1f}\\% on JOB-light)"
        ),
        transform=lambda v: v,
    ),
    Check(
        "native reproduction stats",
        REPRO,
        "stats",
        lambda v: (
            f"native plan on {v['in_context']['reproduced']} of "
            f"{v['n_queries']} STATS-CEB queries (the other "
            f"{NUMBER_WORDS[v['n_queries'] - v['in_context']['reproduced']]}"
        ),
        transform=lambda v: v,
    ),
    Check(
        "native reproduction jl",
        REPRO,
        "joblight",
        lambda v: (
            f"on all {v['in_context']['reproduced']} JOB-light queries, and we"
            if v["in_context"]["reproduced"] == v["n_queries"]
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "lesson standalone vs in-context",
        REPRO,
        "stats",
        lambda v: (
            f"reproduced its plan on only {v['standalone']['reproduced']} of "
            f"{v['n_queries']} queries, against "
            f"{v['in_context']['reproduced']} with the "
            "in-context\nvalues"
        ),
        transform=lambda v: v,
    ),
    Check(
        "max-degree loosened by NULL keys",
        MAXDEG,
        "per_query",
        lambda v: f"loosened it up to ${round(v)}\\times$",
        transform=lambda v: max(
            s["upper"]["maxdeg_null_counted"] / s["upper"]["maxdeg"]
            for q in v
            for s in q["subjoins"]
            if s["upper"]["maxdeg"] > 0
        ),
    ),
    Check(
        "query sets",
        BASE,
        "n_queries",
        lambda v: (
            f"the cardinality level the {round(v)} STATS-CEB queries whose whole-query "
            f"sample finished; certificates the {value(PG25, PG25_N):.0f} with 3--7 "
            f"relations ({value(PG25, SB25_N):.0f} for SafeBound) and "
            f"{value(JLCERT, JL_N):.0f} on JOB-light; planning on $\\hat L$ the "
            f"{value(S2, MATCHED_N):.0f} whose every sample finished; "
            "the sampling cost "
            f"the {value(INRUN, 'clean.n_queries'):.0f} without a timed-out "
            "sample or native "
            "plan; selective sampling the "
            f"{value(SELECTIVE, 'summary.n_timed'):.0f} "
            "with a timed native plan; the oracle's sampling price the "
            f"{value(POC, 'summary::all_certified::n'):.0f} queries certified at "
            "$r{=}0.10$; certifying after execution "
            f"{value(MECH, 'h4_a_posteriori::cout::all::n'):.0f} STATS-CEB and "
            f"{value(MECH, 'h4_a_posteriori_joblight::cout::all::n'):.0f} JOB-light "
            "queries; and the guard funnel all "
            f"{value(FUNNEL, 'n_queries'):.0f}"
        ),
    ),
    # --- overview table
    Check(
        "overview after execution",
        MECH,
        "",
        lambda v: (
            "After execution & query (3--7 rel.) & "
            f"{v['h4_a_posteriori']['cout']['all']['n']} & share certified $\\le2$ "
            "\\\\ Execution & query & 146"
        ),
        transform=lambda v: v,
    ),
    Check(
        "overview jl after execution",
        MECH,
        "",
        lambda v: (
            "After execution & query (3--5 rel.) & "
            f"{v['h4_a_posteriori_joblight']['cout']['all']['n']} & share certified "
            "$\\le2$ \\\\ Execution & query & 70"
        ),
        transform=lambda v: v,
    ),
    Check(
        "overview estimate pairs",
        BASE,
        "n_pairs",
        lambda v: f"Estimate & query $\\times$ estimator & {round(v)} &",
    ),
    Check(
        "overview tightness pairs",
        TIGHT,
        "per_pair",
        lambda v: f"Tightness & query $\\times$ estimator & {round(v)} &",
        transform=lambda v: len([r for r in v if r["certificate"] == "lpbound"]),
    ),
    Check(
        "overview sample cost",
        COUNTS,
        "cost_ratio.n",
        lambda v: f"Sample cost & sub-plan & {thousands(v)} &",
    ),
    Check(
        "overview jl estimate pairs",
        JLB,
        "n_pairs",
        lambda v: f"Estimate & query $\\times$ estimator & {round(v)} &",
    ),
    Check(
        "overview jl sample cost",
        JL_COUNTS,
        "cost_ratio.n",
        lambda v: f"Sample cost & sub-plan & {round(v)} &",
    ),
    # --- cardinality: pooled table
    Check("pairs", BASE, "n_pairs", lambda v: f"Rows are the {round(v)}"),
    Check(
        "queries", BASE, "n_queries", lambda v: f"pairs of\nthe {round(v)} STATS-CEB"
    ),
    Check("none p95", BASE, "pooled.none.p95.point", lambda v: f"& {thousands(v)} ["),
    Check("none p95 lo", BASE, "pooled.none.p95.ci_low", lambda v: f"[{thousands(v)},"),
    Check(
        "none median",
        BASE,
        "pooled.none.median.point",
        lambda v: f"No guard            & {v:.1f}",
    ),
    Check(
        "upper p95", BASE, "pooled.upper_t1.p95.point", lambda v: f"& {thousands(v)} [2"
    ),
    Check(
        "upper median",
        BASE,
        "pooled.upper_t1.median.point",
        lambda v: f"(LpBound) & {v:.2f}",
    ),
    Check(
        "both p95",
        BASE,
        "pooled.bi_sound_t1.p95.point",
        lambda v: f"& {round(v)} [96, 648]",
    ),
    Check(
        "both p95 hi",
        BASE,
        "pooled.bi_sound_t1.p95.ci_high",
        lambda v: f"[96, {round(v)}]",
    ),
    Check(
        "both p95 in text",
        BASE,
        "pooled.bi_sound_t1.p95.point",
        lambda v: f"against ${round(v)}$ for the best guard",
    ),
    Check(
        "lpbound est p95",
        BASE,
        "pooled.lpbound_as_estimate.p95.point",
        lambda v: f"& {thousands(v)} [958",
    ),
    Check(
        "sound L p95",
        BASE,
        "pooled.l_sound_as_estimate.p95.point",
        lambda v: f"a tail of ${v:.1f}$",
    ),
    Check(
        "lhat p95",
        BASE,
        "pooled.l_hat_as_estimate.p95.point",
        lambda v: f"$q$-error of\n${v:.2f}$",
    ),
    Check(
        "lhat p95 ci",
        BASE,
        "pooled.l_hat_as_estimate.p95.ci_high",
        lambda v: f"1.72 [1.37, {v:.2f}]",
    ),
    # --- cardinality: per estimator
    Check(
        "neurocard upper",
        BASE,
        "by_proposal.neurocard.p95_upper_t1",
        lambda v: f"to ${thousands(v)}$ and the floor",
    ),
    Check(
        "duckdb none",
        BASE,
        "by_proposal.duckdb.p95_none",
        lambda v: f"from ${thousands(v)}$ to",
    ),
    Check(
        "duckdb both",
        BASE,
        "by_proposal.duckdb.p95_bi_sound_t1",
        lambda v: f"to ${v:.1f}$. Pooled",
    ),
    Check(
        "xbound contrast",
        BASE,
        "by_proposal",
        lambda v: (
            f"holds here for DuckDB and the commercial\nsystem "
            f"({round(100 * v['duckdb']['under_share'])}\\% and "
            f"{round(100 * v['dbx']['under_share'])}\\% of queries), but NeuroCard and "
            "FactorJoin mostly\noverestimate on STATS-CEB"
            if v["neurocard"]["under_share"] < 0.5
            and v["factorjoin"]["under_share"] < 0.5
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "postgres both",
        BASE,
        "by_proposal.postgres.p95_bi_sound_t1",
        lambda v: f"& 329 & {round(v)} \\\\",
    ),
    # --- tightness
    Check(
        "maxdeg B/true",
        TIGHT,
        "certificates.max_degree.bound_over_true_median",
        lambda v: f"median ${thousands(v)}\\times$ above",
    ),
    Check(
        "safebound B/true",
        TIGHT,
        "certificates.safebound.bound_over_true_median",
        lambda v: f"${v:.1f}\\times$ and",
    ),
    Check(
        "tightness gate",
        TIGHT,
        "tau",
        lambda v: f"gated at the bound itself\n($\\tau{{=}}{round(v)}$)",
    ),
    Check(
        "lpbound B/true",
        TIGHT,
        "certificates.lpbound.bound_over_true_median",
        lambda v: f"and ${v:.1f}\\times$; per query",
    ),
    Check(
        "per-query tightness ratio",
        TIGHT,
        "per_pair",
        lambda v: f"median {round(v)} times tighter than max-degree",
        transform=lambda v: median_ratio(v),
    ),
    Check(
        "pooled fire ratio",
        TIGHT,
        "certificates",
        lambda v: (
            f"fires 2.7\ntimes as often ({100 * v['lpbound']['fire_rate']:.1f}\\% of "
            "(query, estimator) pairs against "
            f"{100 * v['max_degree']['fire_rate']:.1f}\\%)"
            if f"{v['lpbound']['fire_rate'] / v['max_degree']['fire_rate']:.1f}"
            == "2.7"
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    # --- in-engine cost
    Check(
        "cost median",
        COUNTS,
        "cost_ratio.median.point",
        lambda v: f"${v:.2f}\\times$ [0.31",
    ),
    Check(
        "cost median ci",
        COUNTS,
        "cost_ratio.median.ci_high",
        lambda v: f"[0.31, {v:.2f}]",
    ),
    Check("cost n", COUNTS, "cost_ratio.n", lambda v: f"over {thousands(v)} sub-plans"),
    Check(
        "cost <100ms",
        COUNTS,
        "cost_ratio_by_exact_ms::0-100::median",
        lambda v: f"${v:.2f}$ when the exact",
    ),
    Check(
        "cost >10s",
        COUNTS,
        "cost_ratio_by_exact_ms::10000-1e+12::median",
        lambda v: f"${v:.2f}$ beyond",
    ),
    # --- sampling rate (matched quiet-host sub-plans for cost)
    Check(
        "rate matched n",
        RATE,
        "",
        lambda v: f"same {v['n_subplans']} sub-plans of {v['n_queries']} queries",
        transform=lambda v: v,
    ),
    Check(
        "rate quiet host",
        RATE,
        "max_load_seen",
        lambda v: "only where the load stayed below 60" if v <= 60 else "MISMATCH",
    ),
    Check(
        "rate cost",
        RATE,
        "all",
        lambda v: (
            f"median ${v['cost_r025']['point']:.2f}\\times$ the exact\n"
            f"count instead of ${v['cost_r010']['point']:.2f}\\times$"
        ),
        transform=lambda v: v,
    ),
    Check(
        "rate cost expensive",
        RATE,
        "exact_over_1s",
        lambda v: (
            f"${v['cost_r025']['point']:.2f}\\times$ instead of "
            f"${v['cost_r010']['point']:.2f}\\times$: there the cost is the rate"
        ),
        transform=lambda v: v,
    ),
    Check(
        "rate L median",
        BASE25,
        "pooled.l_sound_as_estimate.median.point",
        lambda v: (
            f"(median $q$-error {v:.2f} against\n"
            f"{value(BASE, 'pooled.l_sound_as_estimate.median.point'):.2f})"
        ),
    ),
    Check(
        "rate lhat p95",
        BASE25,
        "pooled.l_hat_as_estimate.p95.point",
        lambda v: (
            f"(95th percentile {v:.2f} against "
            f"{value(BASE, 'pooled.l_hat_as_estimate.p95.point'):.2f})"
        ),
    ),
    Check(
        "rate certified",
        PG10,
        "summary::lpboundxsample|cout::share_joint_bound_le_2::point",
        lambda v: (
            f"rises from {100 * v:.1f}\\% to {100 * value(PG25, PG25_SHARE):.1f}\\%"
        ),
    ),
    Check(
        "seed robustness",
        SEEDS,
        "range",
        lambda v: (
            f"draws, $\\hat L$'s 95th-percentile $q$-error ranges from "
            f"{v['lhat_p95']['min']:.2f} to {v['lhat_p95']['max']:.2f}, "
            "the best guard's "
            f"from {round(v['both_guards_p95']['min'])} to "
            f"{round(v['both_guards_p95']['max'])}, the sound count's median from "
            f"{v['l_sound_median']['min']:.2f} to "
            f"{v['l_sound_median']['max']:.2f}, and "
            "the share of queries with a certified plan from "
            f"{100 * v['certified_share']['min']:.1f}\\% to "
            f"{100 * v['certified_share']['max']:.1f}\\%"
        ),
        transform=lambda v: v,
    ),
    # --- plan certificates (PostgreSQL's in-engine samples; table rows generated)
    Check(
        "lpbound joint share",
        PG25,
        "summary::lpboundxsample|cout::share_joint_bound_le_2::point",
        lambda v: f"plan within $2\\times$ of optimal on {100 * v:.1f}\\% of queries",
    ),
    Check(
        "lpbound joint median",
        PG25,
        "summary::lpboundxsample|cout::median_joint_bound_best::point",
        lambda v: f"(median certified bound\n${v:.2f}$)",
    ),
    Check(
        "lpbound pruning joint",
        PG25,
        "summary::lpboundxsample|cout::mean_pruning_joint::point",
        lambda v: "prunes a third of all orders" if 0.3 <= v < 0.37 else "MISMATCH",
    ),
    Check(
        "lpbound pruning decoupled",
        PG25,
        "summary::lpboundxsample|cout::mean_pruning_decoupled::point",
        lambda v: f"against ${100 * v:.1f}\\%$ under $G$",
    ),
    Check(
        "certified max true regret",
        PG25,
        "per_query::lpboundxsample|cout",
        lambda v: f"true regret of at most ${v:.2f}$",
        transform=lambda v: max(
            r["true_regret_best"] for r in v if r["joint_bound_best"] <= 2
        ),
    ),
    Check(
        "lpbound r010 share",
        PG10,
        "summary::lpboundxsample|cout::share_joint_bound_le_2::point",
        lambda v: f"the share\nfalls to {100 * v:.1f}\\%",
    ),
    Check(
        "lpbound no-final share",
        PG25,
        "summary::lpboundxsample|cout_minus_final::share_joint_bound_le_2::point",
        lambda v: f"halves the certified share ({100 * v:.1f}\\%)",
    ),
    # JOB-light certificates (in-engine samples, r = 0.10)
    Check(
        "jl cert n",
        JLCERT,
        "summary::lpboundxsample|cout::n_queries",
        lambda v: f"JOB-light ({round(v)} queries with 3--5 relations",
    ),
    Check(
        "jl cert lpbound",
        JLCERT,
        "summary::lpboundxsample|cout::share_joint_bound_le_2",
        lambda v: (
            f"{100 * v['point']:.1f}\\% [{100 * v['ci_low']:.1f}, "
            f"{100 * v['ci_high']:.1f}] of queries"
        ),
        transform=lambda v: v,
    ),
    Check(
        "jl cert safebound",
        JLCERT,
        "summary::safeboundxsample|cout::share_joint_bound_le_2::point",
        lambda v: f"SafeBound for\n{100 * v:.1f}\\%",
    ),
    Check(
        "jl cert optimal",
        JLCERT,
        "per_query::lpboundxsample|cout",
        lambda v: "every certified plan is in fact optimal" if v == 1 else "MISMATCH",
        transform=lambda v: max(
            r["true_regret_best"] for r in v if r["joint_bound_best"] <= 2
        ),
    ),
    Check(
        "jl cert no lower",
        JLCERT,
        "summary",
        lambda v: (
            "no upper bound certifies any\nquery without a lower bound"
            if v == 0
            else "MISMATCH"
        ),
        transform=lambda v: sum(
            e["share_joint_bound_le_2"]["point"] for k, e in v.items() if "xone|" in k
        ),
    ),
    Check(
        "jl cert no final",
        JLCERT,
        "summary::lpboundxsample|cout_minus_final::share_joint_bound_le_2::point",
        lambda v: f"only {100 * v:.1f}\\%\nremain",
    ),
    # where plan choice matters, and the a-priori q^2 bound
    Check(
        "intro range",
        CTX,
        "",
        lambda v: f"for {v[0]}--{v[1]}\\% of the\nqueries where the choice matters",
        transform=lambda v: (
            round(100 * v["stats_r025"]["certified_where_choice_matters"]["point"]),
            round(100 * v["joblight_r010"]["certified_where_choice_matters"]["point"]),
        ),
    ),
    Check(
        "trivial share",
        CTX,
        "stats_r025.share_all_orders_within_2x",
        lambda v: f"including the\n{100 * v:.1f}\\% on which every left-deep order",
    ),
    Check(
        "choice matters body",
        CTX,
        "stats_r025",
        lambda v: (
            f"On the {v['n_choice_matters']} queries where the choice "
            "matters (a classification that uses the true sizes and serves this "
            "analysis only), the certificate covers "
            f"{100 * v['certified_where_choice_matters']['point']:.1f}\\% "
            f"[{100 * v['certified_where_choice_matters']['ci_low']:.1f}, "
            f"{100 * v['certified_where_choice_matters']['ci_high']:.1f}]"
        ),
        transform=lambda v: v,
    ),
    Check(
        "choice matters r010",
        CTX,
        "stats_r010.certified_where_choice_matters.point",
        lambda v: f"and {100 * v:.1f}\\% at\n$r{{=}}0.10$",
    ),
    Check(
        "q squared median",
        CTX,
        "stats_r025",
        lambda v: (
            f"exceeds 2 on every query (median {round(v['median_q_squared'])})"
            if v["share_q_squared_le_2"] == 0
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "sign test",
        COMM,
        "brackets::lpboundxsample0.25::certified_class_share::count",
        lambda v: (
            f"rests on {round(v)}\ndiscordant queries (exact sign test "
            f"$p={2 * 0.5 ** round(v):.4f}$)"
        ),
    ),
    Check(
        "body jl choice matters",
        CTX,
        "joblight_r010",
        lambda v: (
            f"(on the {v['n_choice_matters']} queries where the choice matters, "
            f"LpBound certifies\n"
            f"{100 * v['certified_where_choice_matters']['point']:.1f}\\% "
            f"[{100 * v['certified_where_choice_matters']['ci_low']:.1f}, "
            f"{100 * v['certified_where_choice_matters']['ci_high']:.1f}])"
        ),
        transform=lambda v: v,
    ),
    Check(
        "abstract under-share range",
        BASE,
        "by_proposal",
        lambda v: f"underestimated queries range from {v[0]}\\% to {v[1]}\\%, an upper",
        transform=lambda v: (
            round(100 * min(e["under_share"] for e in v.values())),
            round(100 * max(e["under_share"] for e in v.values())),
        ),
    ),
    Check(
        "abstract choice matters",
        CTX,
        "",
        lambda v: (
            f"on {v[0]}\\% of STATS-CEB and {v[1]}\\% of JOB-light queries "
            "where plan choice matters (25\\% and 10\\% samples), none without a lower "
            "bound"
        ),
        transform=lambda v: (
            round(100 * v["stats_r025"]["certified_where_choice_matters"]["point"]),
            round(100 * v["joblight_r010"]["certified_where_choice_matters"]["point"]),
        ),
    ),
    Check(
        "abstract heldout",
        HELDOUT,
        "tests",
        lambda v: (
            "a pre-registered test confirms "
            f"{NUMBER[sum(r == 'confirmed' for k, r in v.items() if k != 'C5')]} of "
            f"{NUMBER[sum(r != 'not testable' for k, r in v.items() if k != 'C5')]} "
            "testable predictions, on after-execution certificates and plan changes"
        ),
        transform=lambda v: v,
    ),
    Check(
        "abstract guarded vs native",
        GVN,
        "",
        lambda v: (
            "guarded plans do not beat PostgreSQL's own "
            f"(${v['stats']['postgres_upper']['point']:.2f}$--"
            f"${v['stats']['postgres_bi']['point']:.2f}\\times$ and "
            f"${v['joblight']['postgres_lower']['point']:.2f}$--"
            f"${v['joblight']['postgres_bi']['point']:.2f}\\times$)"
            if all(
                v["stats"][p]["ci_low"] < 1 for p in ("postgres_upper", "postgres_bi")
            )
            and all(
                v["joblight"][p]["ci_high"] < 1
                for p in ("postgres_lower", "postgres_bi")
            )
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "body guarded vs native stats",
        GVN,
        "stats",
        lambda v: (
            "the guarded plans only reach "
            f"${v['postgres_upper']['point']:.2f}\\times$ "
            f"[{v['postgres_upper']['ci_low']:.2f}, "
            f"{v['postgres_upper']['ci_high']:.2f}]"
            f" of their speed (${v['postgres_bi']['point']:.2f}\\times$ "
            f"[{v['postgres_bi']['ci_low']:.2f}, "
            f"{v['postgres_bi']['ci_high']:.2f}] with "
            "both guards)"
        ),
        transform=lambda v: v,
    ),
    Check(
        "body guarded vs native jl",
        GVN,
        "joblight.postgres_lower",
        lambda v: (
            f"the floored plans reach ${v['point']:.2f}\\times$ "
            f"[{v['ci_low']:.2f}, {v['ci_high']:.2f}]"
        ),
        transform=lambda v: v,
    ),
    # certified plans executed with forced join orders (v3)
    Check(
        "forced orders counts",
        FORCED,
        "",
        lambda v: (
            f"({v['n_certified']} certified STATS-CEB queries at "
            "$r{=}0.10$, three "
            "interleaved runs). As controls we forced the left-deep order that "
            "is optimal "
            f"in true $C_{{out}}$ on all {v['n_queries']} queries and, on the "
            f"{v['n_native_left_deep']} whose native plan is left-deep"
        ),
        transform=lambda v: v,
    ),
    Check(
        "forced orders speedups",
        FORCED,
        "",
        lambda v: (
            f"runs {iv(v['comparisons']['native->native_lo']['geo_speedup'])} as "
            "fast as the native plan, because on "
            f"{v['n_forcing_changes_physical_plan']} of the "
            f"{v['n_native_left_deep']} queries PostgreSQL then chooses other join "
            "methods or sides "
            f"(${v['native->native_lo|forcing_changed']['geo']:.2f}\\times$ there, "
            f"${v['native->native_lo|forcing_unchanged']['geo']:.2f}\\times$ on the "
            "rest). Yet the $C_{out}$-optimal order runs "
            f"{iv(v['comparisons']['native->cout_opt']['geo_speedup'])} as fast as "
            "the native plan "
            f"({iv(v['comparisons']['native_lo->cout_opt']['geo_speedup'])} against "
            "the forced native order), and the certified orders "
            f"{iv(v['comparisons']['native->certified']['geo_speedup'])} (faster on "
            f"{round(100 * v['comparisons']['native->certified']['share_faster'])}"
            "\\% of queries, slower on "
            f"{round(100 * v['comparisons']['native->certified']['share_slower'])}"
            f"\\%). On {v['n_certified_is_cout_opt']} of the {v['n_certified']} "
            "certified queries the certified order is the $C_{out}$-optimal order "
            "itself"
        ),
        transform=lambda v: v,
    ),
    Check(
        "forced orders by choice",
        FORCED,
        "certified_by_choice",
        lambda v: (
            f"Where the choice matters ({v['matters']['n']} queries; a post-hoc "
            "split), the certified "
            f"plans run {iv(v['matters']['geo_speedup'])} as fast as the native "
            "plan; where every order is within $2\\times$ of the best "
            f"({v['does_not_matter']['n']}), "
            f"{iv(v['does_not_matter']['geo_speedup'])}"
        ),
        transform=lambda v: v,
    ),
    Check(
        "forced orders with sampling by choice",
        FORCED,
        "certified_by_choice::matters::geo_speedup_with_sampling",
        lambda v: f"and {iv(v)} where the choice matters (sample times",
        transform=lambda v: v,
    ),
    Check(
        "forced orders with sampling",
        FORCED,
        "comparisons::native->certified::geo_speedup_with_sampling",
        lambda v: f"plans run {iv(v)} as fast as the native plan",
        transform=lambda v: v,
    ),
    Check(
        "abstract certified plans beat where choice matters",
        FORCED,
        "certified_by_choice::matters",
        lambda v: (
            "on STATS-CEB, certified plans beat PostgreSQL's own where the choice "
            f"matters (${v['geo_speedup']['point']:.2f}\\times$) until the sample "
            "is paid for"
            if v["geo_speedup"]["ci_low"] > 1
            and v["geo_speedup_with_sampling"]["ci_high"] < 1
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "conclusion certified plans",
        FORCED,
        "",
        lambda v: (
            "on STATS-CEB the plans they certify run as fast as PostgreSQL's own, "
            "and faster where the choice matters"
            if v["comparisons"]["native->certified"]["geo_speedup"]["ci_low"]
            <= 1
            <= v["comparisons"]["native->certified"]["geo_speedup"]["ci_high"]
            and v["certified_by_choice"]["matters"]["geo_speedup"]["ci_low"] > 1
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "recommendation headroom",
        S2,
        "pairs::native_raw->truth::geo_speedup::point",
        lambda v: (
            f"it was ${v:.2f}\\times$ on STATS-CEB and "
            f"${value(J2, 'pairs::native_raw->truth::geo_speedup::point'):.2f}\\times$ "
            "on JOB-light, of which the sound guards recovered little"
        ),
    ),
    # registered study (our own Python sampler) and its commutation re-score
    Check(
        "registered null",
        REG,
        "summary",
        lambda v: (
            f"0 of {value(REG, 'n_queries_eligible'):.0f} queries on either rule"
            if v == 0
            else "MISMATCH"
        ),
        transform=lambda v: sum(
            v[b][rule]["point"] > 0
            for b in v
            if b.startswith(("maxdeg", "safebound", "lpbound"))
            for rule in ("certified_share", "useful_share")
        ),
    ),
    Check(
        "registered min G r",
        COMM,
        "brackets",
        lambda v: f"smallest $G\\cdot r$ over all queries is ${v:.2f}$",
        transform=lambda v: min(
            b["min_G"] * float(k.split("sample")[1])
            for k, b in v.items()
            if "sample" in k
        ),
    ),
    Check(
        "commutation count",
        COMM,
        "brackets::lpboundxsample0.25::certified_class_share::count",
        lambda v: f"certifies {round(v)} of 140 queries",
    ),
    Check(
        "commutation share",
        COMM,
        "brackets::lpboundxsample0.25::certified_class_share",
        lambda v: (
            f"${100 * v['point']:.1f}\\%$ [{100 * v['ci_low']:.1f}, "
            f"{100 * v['ci_high']:.1f}], still below"
        ),
        transform=lambda v: v,
    ),
    Check(
        "commutation paired difference",
        COMM,
        "paired_certified_class_share_primary_minus_one",
        lambda v: (
            f"a difference of ${100 * v['point']:.1f}$ points "
            f"[{100 * v['ci_low']:.1f}, {100 * v['ci_high']:.1f}]"
        ),
        transform=lambda v: v,
    ),
    Check(
        "injection neutral",
        S2,
        "pairs::native->native_raw",
        lambda v: (
            f"changes the plan of\n{v['plans_differ']} of {v['n']} STATS-CEB queries "
            f"and none on JOB-light, with no change in speed\n"
            f"(${v['geo_speedup']['point']:.2f}\\times$ "
            f"[{v['geo_speedup']['ci_low']:.2f}, "
            f"{v['geo_speedup']['ci_high']:.2f}])"
            if value(J2, "pairs::native->native_raw::plans_differ") == 0
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "native guards",
        S2,
        "pairs",
        lambda v: (
            f"upper: {v['native_raw->native_upper']['plans_differ']} of 146 plans "
            "change, "
            f"${v['native_raw->native_upper']['geo_speedup']['point']:.2f}\\times$; "
            "floor "
            f"${v['native_raw->native_lower']['geo_speedup']['point']:.2f}\\times$"
        ),
        transform=lambda v: v,
    ),
    Check(
        "lpbound vs own estimates",
        S2,
        "pairs::native_raw->lpbound_est::geo_speedup::point",
        lambda v: f"only ${v:.2f}\\times$ as fast as our own optimizer's estimates",
    ),
    Check(
        "abstract native guards range",
        S2,
        "pairs",
        lambda v: (
            f"guards change little (${v[0]:.2f}$--${v[1]:.2f}\\times$): the bracket "
            "sees their errors but is too wide; in a synthetic experiment, one within "
            "$2\\times$ of the truth recovers "
            f"{round(100 * value(POINTS, POINTS_K2))}\\% of the headroom"
        ),
        transform=lambda v: (
            min(
                native_guard_speedups(v)
                + native_guard_speedups(json.loads(J2.read_text())["pairs"])
            ),
            max(
                native_guard_speedups(v)
                + native_guard_speedups(json.loads(J2.read_text())["pairs"])
            ),
        ),
    ),
    Check(
        "sampling overhead in run",
        INRUN,
        "clean",
        lambda v: (
            f"a query takes a median ${v['median']['point']:.2f}\\times$ "
            f"[{v['median']['ci_low']:.2f}, {v['median']['ci_high']:.2f}] the time of "
            "executing the query\nwith our optimizer's plan, and longer than the "
            "query on "
            f"{round(100 * v['share_above_one'])}\\% of {v['n_queries']} queries"
        ),
        transform=lambda v: v,
    ),
    Check(
        "overhead exclusions",
        INRUN,
        "",
        lambda v: (
            f"These {v['clean']['n_queries']} exclude "
            f"{v['n_excluded_native_timeout'] + v['n_excluded_sample_timeout']} "
            "queries whose native plan "
            f"({v['n_excluded_native_timeout']}) or one of whose samples "
            f"({v['n_excluded_sample_timeout']}) timed out"
        ),
        transform=lambda v: v,
    ),
    Check(
        "overhead capped sensitivity",
        INRUN,
        "with_sample_timeouts_at_cap",
        lambda v: (
            f"raises the median to ${v['median']['point']:.2f}\\times$ "
            f"({round(100 * v['share_above_one'])}\\% of {v['n_queries']} queries)"
        ),
        transform=lambda v: v,
    ),
    Check(
        "sensitivity directions",
        TSENS,
        "",
        lambda v: (
            f"or without those {value(SP, 'n_start_above_30'):.0f} queries (the "
            "upper guard's "
            f"${value(S2, PG_UPPER):.2f}"
            "\\times$ then becomes "
            f"${value(TSENS, 'postgres_raw->postgres_upper::geo_quiet_start'):.2f}"
            "\\times$), every geometric mean below that lies more than 2\\% from "
            "parity keeps its direction; a 5\\% or 20\\% band changes the sign of "
            "the balance of faster and slower queries only on "
            f"{NUMBER[v[0]]} "
            f"comparisons within ${v[1]:.2f}\\times$ of parity"
        ),
        transform=lambda v: balance_reversals() if directions_kept(v) else "MISMATCH",
    ),
    Check(
        "execution load burst",
        SP,
        "queries_start_above_30",
        lambda v: (
            f"the load gate flagged no query, though {len(v)} STATS-CEB queries, in "
            f"one stretch of {v[-1] - v[0] + 1}, started at a load above 30)"
            if v == sorted(v) and value(SP, "n_high_load") == 0
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "native errors outside bracket",
        BRACKET,
        "",
        lambda v: (
            f"of the {v['n_large_errors']} sub-plans it misestimates by more than "
            f"$10\\times$, {v['large_errors_outside_bracket']} lie outside the bracket"
        ),
        transform=lambda v: v,
    ),
    Check(
        "native lhat floor",
        S2,
        "pairs::native_raw->native_bi_hat::geo_speedup::point",
        lambda v: f"flooring with $\\hat L$ instead recovers more (${v:.2f}\\times$)",
    ),
    Check(
        "limitations load and censoring",
        SP,
        "",
        lambda v: (
            "(on PostgreSQL 13.1 the highest start load was "
            f"{v['load_start_max']:.1f}; {NUMBER[v['n_end_above_60']]} STATS-CEB "
            f"queries ended above 60, at most {v['load_end_max']:.1f}), our "
            "intervals are over queries, not repeated runs, and plans cut off at "
            f"300\\,s (on 13.1, {v['n_censored_plans']} on STATS-CEB; "
            f"{value(JP, 'n_censored_plans'):.0f} on JOB-light, all of one query "
            "on which every plan times out; on 16, "
            f"{value(PG16JOPROT, 'n_censored_plans'):.0f} with JIT off)"
            if len(json.loads(JP.read_text())["queries_with_censored_plan"]) == 1
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    # --- mechanism analyses (post-hoc; docs/icde_2027/PREDICTIONS_RESULTS.md)
    Check(
        "scale invariance",
        MECH,
        "h2_exchange_rate",
        lambda v: (
            "The $r{=}0.25$ sample raises the floors a median "
            f"${v['median_floor_ratio_r025_over_r010']:.2f}\\times$ over $r{{=}}0.10$ "
            "and certifies the same share of queries, "
            f"{100 * v['share_r025_real_bounds']:.1f}\\%, as dividing every LpBound "
            "value by 2.5 at $r{=}0.10$. That cut, however, is not sound: it falls "
            "below the truth on "
            f"{v['n_subjoins_upper_divided_2p5_below_truth']} of "
            f"{v['n_subjoins'] // 1000}{{,}}{v['n_subjoins'] % 1000:03d} "
            "sub-joins. Even clipped at the truth, the tightest a bound of that form "
            "can be, it certifies only "
            f"{100 * v['share_r010_upper_divided_2p5_clipped_at_truth']:.1f}\\%, and a "
            "sound tightening that shrinks the whole-query bound $2.5\\times$ "
            "geometrically certifies only "
            f"{100 * v['share_r010_by_tightening']['2.5']:.1f}\\%"
            if v["share_r010_upper_divided_2p5"]
            == v["share_r010_floor_times_2p5"]
            == v["share_r025_real_bounds"]
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "rho auc",
        MINT,
        "h1",
        lambda v: (
            "proposition, $\\rho\\ge1/2$ certifies $p_{hi}$ within $2\\times$ ("
            f"{value(MECH, 'h1_shared_mass::stats_r010::n_rho_at_least_half'):.0f} "
            "STATS-CEB and "
            f"{value(MECH, 'h1_shared_mass::joblight_r010::n_rho_at_least_half'):.0f} "
            "JOB-light queries, all certified, as the proposition guarantees). As a "
            "score, $\\rho$ separates "
            "certified from uncertified queries with an AUC of "
            f"{v['stats_r010']['auc_rho']['point']:.2f} "
            f"[{v['stats_r010']['auc_rho']['ci_low']:.2f}, "
            f"{v['stats_r010']['auc_rho']['ci_high']:.2f}] on STATS-CEB and "
            f"{v['joblight_r010']['auc_rho']['point']:.2f} "
            f"[{v['joblight_r010']['auc_rho']['ci_low']:.2f}, "
            f"{v['joblight_r010']['auc_rho']['ci_high']:.2f}] on JOB-light (the "
            "threshold chosen on STATS-CEB classifies "
            f"{round(100 * value(MECH, JL_RHO_ACCURACY))}\\% of JOB-light queries "
            "correctly), where the number of relations alone gives "
            f"{v['stats_r010']['auc_n_relations']:.2f} and "
            f"{v['joblight_r010']['auc_n_relations']:.2f}"
            if all(
                value(MECH, f"h1_shared_mass::{b}::n_rho_at_least_half")
                == value(MECH, f"h1_shared_mass::{b}::n_rho_at_least_half_certified")
                for b in ("stats_r010", "joblight_r010")
            )
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "certified share by size",
        MECH,
        "h1_shared_mass::stats_r010::by_n_aliases",
        lambda v: (
            f"from {round(100 * v['3']['share_certified'])}\\% of 3-relation queries "
            f"to none of the {NUMBER[v['7']['n']]} with 7 relations"
            if v["7"]["share_certified"] == 0
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "oracle end to end",
        MINT,
        "h5",
        lambda v: (
            "selective sampling already pays this price on "
            f"{round(100 * value(POC, 'summary::all_certified::share_greedy_optimal'))}"
            f"\\% of the {v['all']['n']} certified queries ("
            f"{round(100 * value(POC, POC_GREEDY_MATTERS))}"
            f"\\% of the {v['choice_matters']['n']} where the choice matters). Charged "
            "the oracle's price instead of the online rule's, the certified plans "
            f"still run {iv(v['all']['oracle_geo'])} as fast as the native plan over "
            f"the {v['all']['n']} queries and {iv(v['choice_matters']['oracle_geo'])} "
            "where the choice matters, faster on "
            f"{round(100 * v['choice_matters']['oracle_share_faster'])}\\% of those. "
            "Our frozen prediction that even this price would exceed the query's "
            "execution time on half of those queries failed ("
            f"{oracle_above_range()})"
        ),
        transform=lambda v: v,
    ),
    Check(
        "a posteriori stats",
        MINT,
        "h4::stats|cout",
        lambda v: (
            f"$2\\times$ on {100 * v['bound_sample']['point']:.1f}\\% "
            f"[{100 * v['bound_sample']['ci_low']:.1f}, "
            f"{100 * v['bound_sample']['ci_high']:.1f}] of "
            f"{value(MECH, 'h4_a_posteriori::cout::all::n'):.0f} STATS-CEB queries "
            "(three lack sub-plan truth; "
            f"{100 * v['true_regret']['point']:.1f}\\% are in fact within "
            "$2\\times$), but that needs a sample of every sub-join the plan did not "
            "compute; with no sample at all, the exact sizes alone certify "
            f"{100 * v['bound_none']['point']:.1f}\\% "
            f"[{100 * v['bound_none']['ci_low']:.1f}, "
            f"{100 * v['bound_none']['ci_high']:.1f}]"
            if value(MECH, "h4_a_posteriori::cout::n_skipped_no_truth") == 3
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "a posteriori joblight and no final",
        MECH,
        "",
        lambda v: (
            "JOB-light the shares are "
            f"{h4_pct(v, 'jl', 'cout', 'share_certified_sample', 0)}\\% and "
            f"{h4_pct(v, 'jl', 'cout', 'share_certified_none')}\\% of "
            f"{v['h4_a_posteriori_joblight']['cout']['all']['n']}. These "
            "certificates rest on the final result, as before execution "
            "(Section~\\ref{sec:joint}): without it, the certificate covers "
            f"{h4_pct(v, 'stats', NO_FINAL, 'share_certified_sample')}\\% of "
            f"STATS-CEB and {h4_pct(v, 'jl', NO_FINAL, 'share_certified_sample')}"
            "\\% of JOB-light plans with the sample and almost none without it ("
            f"{h4_pct(v, 'stats', NO_FINAL, 'share_certified_none', 0)}\\% and "
            f"{h4_pct(v, 'jl', NO_FINAL, 'share_certified_none')}\\%), although "
            f"{h4_pct(v, 'stats', NO_FINAL, 'share_true_regret_le_2')}\\% and "
            f"{h4_pct(v, 'jl', NO_FINAL, 'share_true_regret_le_2')}\\% are in fact "
            "within $2\\times$ in that cost"
        ),
        transform=lambda v: v,
    ),
    Check(
        "guard funnel",
        FUNNEL,
        "",
        lambda v: (
            f"of the {v['native|bi']['error']} queries with a sub-plan our optimizer "
            "misestimates by more than $10\\times$, the bracket sees such an error on "
            f"{v['native|bi']['seen']}, but the plan changes on only "
            f"{v['native|bi']['plan_changed']} (both guards together, unlike the "
            f"{value(S2, 'pairs::native_raw->native_upper::plans_differ'):.0f} "
            f"above), and of the {v['native|bi']['faster']} that run faster only "
            f"{v['native|bi']['faster_and_less_work']} do more than 10\\% less true "
            "work. For the published PostgreSQL estimates the same gates pass "
            f"{v['postgres|bi']['error']}, {v['postgres|bi']['seen']} and "
            f"{v['postgres|bi']['plan_changed']} queries; "
            f"{v['postgres|bi']['faster']} run faster, "
            f"{v['postgres|bi']['faster_and_less_work']} of them with less work"
        ),
        transform=lambda v: v,
    ),
    Check(
        "tightness points executed",
        POINTS,
        "",
        lambda v: (
            f"guards choose on our optimizer's estimates (post-hoc; {v['n_queries']} "
            "queries, protocol as above, with temporary files capped at 2\\,GB, "
            "which no plan reached). In this run planning on the true sizes is "
            f"{iv(v['time']['truth']['geo_speedup'])} faster; with $k=2$ both guards "
            f"recover {pct_iv(v['time']['bi_k2']['share_of_truth_speedup'])} of that "
            f"(${v['time']['bi_k2']['geo_speedup']['point']:.2f}\\times$), with $k=8$ "
            f"{pct_iv(v['time']['bi_k8']['share_of_truth_speedup'])} "
            f"(${v['time']['bi_k8']['geo_speedup']['point']:.2f}\\times$), and the "
            f"real bracket recovers {real_bracket_share()}\\%"
        ),
        transform=lambda v: v,
    ),
    Check(
        "intro bracket width",
        POINTS,
        "time::bi_k2::share_of_truth_speedup::point",
        lambda v: (
            "(in a synthetic experiment, a bracket within $2\\times$ of the truth "
            f"recovers {round(100 * v)}\\% of the headroom)"
        ),
    ),
    Check(
        "recommendation bracket width",
        POINTS,
        "time",
        lambda v: (
            "a bracket within $2\\times$ of the truth recovered "
            f"{round(100 * v['bi_k2']['share_of_truth_speedup']['point'])}\\% of the "
            "headroom and one within $8\\times$ "
            f"{round(100 * v['bi_k8']['share_of_truth_speedup']['point'])}\\%, "
            f"against {real_bracket_share()}\\% for today's"
        ),
        transform=lambda v: v,
    ),
    # --- pre-registered held-out test (text rendered by a script hashed beforehand)
    Check(
        "heldout paragraph",
        HELDOUT,
        "",
        lambda v: render_heldout(v),
        transform=lambda v: v,
    ),
    Check(
        "heldout c3 note",
        HELDOUT,
        "a_posteriori::shares::cout_minus_final|none",
        lambda v: (
            "four were confirmed and C3 was inconclusive: on JOB-range "
            f"{100 * v:.1f}\\% of plans are certified without the final result even "
            "without a sample"
        ),
    ),
    Check(
        "overview heldout rows",
        HELDOUT,
        "",
        lambda v: (
            "After execution & query "
            f"({value(HINT, 'n_relations_min'):.0f}--"
            f"{value(HINT, 'n_relations_max'):.0f} rel.) & "
            f"{v['a_posteriori']['n_queries']} & share certified $\\le2$ \\\\ "
            f"Tightness sweep & query & {v['n_sweep_queries']} & share of plans changed"
        ),
        transform=lambda v: v,
    ),
    Check(
        "witness floor",
        WITNESS,
        "",
        lambda v: (
            "has them, costs a median "
            f"{iv(v['median_witness_over_sample'])} of the sampled count of the same "
            f"sub-join ({v['n_subjoins']} sub-joins, timed back to back)"
            if v["n_censored"] == 0
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "heldout intervals",
        HINT,
        "",
        lambda v: (
            "C1--C4 are "
            + ", ".join(
                pct_iv1(v[c])
                for c in ("cout|none", "cout|sample", "cout_minus_final|none")
            )
            + f" and {pct_iv1(v['cout_minus_final|sample'])}. The test is narrow: "
            "JOB-range uses the same IMDb tables as JOB-light, the "
            f"{v['n_queries']} certificate queries have "
            f"{v['n_relations_min']}--{v['n_relations_max']} relations (the sweep "
            f"also includes {value(HELDOUT, 'n_sweep_queries') - v['n_queries']:.0f} "
            "with two)"
            if v["n_queries"] == value(HELDOUT, "a_posteriori::n_queries")
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "implied join plans",
        MECH,
        "h4_a_posteriori::cout::n_implied_join_plans",
        lambda v: f"joins through an implied equality ({v:.0f} queries)",
    ),
    Check(
        "real bracket main run headroom",
        S2,
        "pairs::native_raw->truth::geo_speedup::point",
        lambda v: (
            "(from the main run, at a lower host load, where the true sizes gave "
            f"${v:.2f}\\times$)"
        ),
    ),
    # --- PostgreSQL 16 (post-hoc replication; docs/vldb_2027/PREDICTIONS_PG16.md)
    Check(
        "pg16 estimate extraction",
        JRL_STATS,
        "",
        lambda v: (
            "on 13.1 it reproduces the patch's values on all "
            f"{v['n_compared'] // 1000}{{,}}{v['n_compared'] % 1000:03d} STATS-CEB and "
            f"{value(JRL_JL, 'n_compared'):.0f} JOB-light sub-plans"
            if v["n_equal_after_rounding"] == v["n_compared"] == v["n_subplans"]
            and value(JRL_JL, "n_equal_after_rounding") == value(JRL_JL, "n_compared")
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "pg16 prefix negative",
        PREFIX,
        "",
        lambda v: (
            "rounded input sizes give another estimate, and "
            f"{(v['n_compared'] - v['n_equal_after_rounding']) // 1000}{{,}}"
            f"{(v['n_compared'] - v['n_equal_after_rounding']) % 1000:03d} of the "
            f"{v['n_compared'] // 1000}{{,}}{v['n_compared'] % 1000:03d} values differ "
            "by more than one row"
        ),
        transform=lambda v: v,
    ),
    Check(
        "pg16 reproduced and predictions",
        PG16JO,
        "",
        lambda v: (
            (
                "Injecting PostgreSQL 16's own estimates reproduces its native plan on "
                f"all {v['native_reproduced']} queries"
                if v["native_reproduced"] == v["n_queries"]
                and value(PG16CMP, "native_reproduced") == value(PG16CMP, "n_queries")
                else "MISMATCH"
            )
            + ". Table~\\ref{tab:pg16} sets the two versions side by side. Our first "
            "PostgreSQL 16 run had JIT on"
        ),
        transform=lambda v: v,
    ),
    Check(
        "pg16 predictions held",
        PG16CMP,
        "verdicts",
        lambda v: (
            "no aggregate computed; supplement), and "
            f"{NUMBER[sum(x == 'holds' for x in v.values())]} held"
        ),
        transform=lambda v: v,
    ),
    Check(
        "pg16 guards text",
        PG16JO,
        "",
        lambda v: (
            f"estimates, {iv(v['pg16']['both'])} of a "
            f"${v['pg16']['headroom']['point']:.2f}\\times$ headroom ("
            f"{log_share(v['pg16'])}\\%, against {log_share(v['pg13'])}\\% on "
            "13.1), above the $1.12\\times$ we predicted, and an $8\\times$ synthetic "
            f"bracket recovers {pct_iv(v['pg16']['bi_k8_share'])} of the headroom "
            f"({round(100 * value(PG16CMP, 'pg16::bi_k8_share::point'))}\\% with JIT "
            "on, where we predicted less than half; on the JIT-off run that "
            "prediction holds, which we do not count)"
            if v["pg16"]["bi_k8_share"]["point"] < 0.5
            and value(PG16CMP, "pg16::both::point") > 1.12
            and v["pg16"]["both"]["point"] > 1.12
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "pg16 load and cap",
        PG16PROT,
        "",
        lambda v: (
            "In each run one plan reached the 8\\,GB cap on temporary files"
            if v["n_crashed_plans"] == 1 and value(PG16JOPROT, "n_crashed_plans") == 1
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "abstract pg16",
        PG16JO,
        "pg16",
        lambda v: (
            "the guards do somewhat more for its own estimates "
            f"(${v['both']['point']:.2f}\\times$ of a "
            f"${v['headroom']['point']:.2f}\\times$ headroom)"
        ),
        transform=lambda v: v,
    ),
    Check(
        "intro pg16 guards",
        PG16JO,
        "",
        lambda v: (
            f"(${v['pg13']['both']['point']:.2f}\\times$ on 13.1; somewhat more, "
            f"${v['pg16']['both']['point']:.2f}\\times$, on 16)"
        ),
        transform=lambda v: v,
    ),
    # --- replication against the artifact's own end-to-end runs
    Check(
        "rep published over native stats",
        REP,
        "stats.estimates.published_over_native_median",
        lambda v: f"median ${v:.1f}\\times$ our instance's",
    ),
    Check(
        "rep published over native stats 2-rel",
        REP,
        "stats.estimates.published_over_native_median_by_relations.2",
        lambda v: f"(${round(v)}\\times$ on two-relation",
    ),
    Check(
        "rep published over native jl",
        REP,
        "joblight.estimates.published_over_native_median",
        lambda v: f"JOB-light ${v:.2f}\\times$, and",
    ),
    Check(
        "rep qerror stats",
        REP,
        "stats.estimates",
        lambda v: (
            f"$q$-error is {v['qerror_median_published']:.1f} against\n"
            f"{v['qerror_median_native']:.2f} for ours"
        ),
        transform=lambda v: v,
    ),
    Check(
        "rep qerror jl",
        REP,
        "joblight.estimates",
        lambda v: (
            f"({v['qerror_median_published']:.1f} against "
            f"{v['qerror_median_native']:.2f} on JOB-light)"
        ),
        transform=lambda v: v,
    ),
    Check(
        "rep headroom both",
        REP,
        "",
        lambda v: (
            f"(${v['stats']['pg_to_truth']['artifact']['point']:.2f}\\times$ against "
            f"${v['stats']['pg_to_truth']['ours']['point']:.2f}\\times$ on STATS-CEB, "
            f"${v['joblight']['pg_to_truth']['artifact']['point']:.2f}\\times$ "
            "against\n"
            f"${v['joblight']['pg_to_truth']['ours']['point']:.2f}\\times$ "
            "on JOB-light)"
        ),
        transform=lambda v: v,
    ),
    Check(
        "rep stats lpbound",
        REP,
        "stats",
        lambda v: (
            f"(${v['truth_to_lpbound']['artifact']['point']:.2f}\\times$ "
            f"[{v['truth_to_lpbound']['artifact']['ci_low']:.2f}, "
            f"{v['truth_to_lpbound']['artifact']['ci_high']:.2f}]), where we find\n"
            f"${v['truth_to_lpbound']['ours']['point']:.2f}\\times$, and "
            f"${v['pg_to_lpbound']['artifact']['point']:.2f}\\times$ faster than "
            f"PostgreSQL, where we find\n${v['pg_to_lpbound']['ours']['point']:.2f}"
            "\\times$."
        ),
        transform=lambda v: v,
    ),
    Check(
        "abstract jl floor",
        J2,
        "pairs::postgres_raw->postgres_lower::geo_speedup::point",
        lambda v: f"JOB-light\n(${v:.2f}\\times$)",
    ),
    Check(
        "jl duckdb upper",
        J2,
        "pairs::duckdb_raw->duckdb_upper::geo_speedup",
        lambda v: (
            f"(${v['point']:.2f}\\times$ [{v['ci_low']:.2f}, "
            f"{v['ci_high']:.2f}]) as from the floor"
        ),
        transform=lambda v: v,
    ),
    Check(
        "jl duckdb floor",
        J2,
        "pairs::duckdb_raw->duckdb_lower::geo_speedup",
        lambda v: (
            f"floor (${v['point']:.2f}\\times$ [{v['ci_low']:.2f}, "
            f"{v['ci_high']:.2f}]);"
        ),
        transform=lambda v: v,
    ),
    Check(
        "jl duckdb headroom",
        J2,
        "pairs::duckdb_raw->truth::geo_speedup::point",
        lambda v: f"with a headroom of ${v:.2f}\\times$, its errors",
    ),
    Check(
        "jl duckdb under",
        JLB,
        "by_proposal.duckdb.under_share",
        lambda v: f"underestimates on {round(100 * v)}\\% of\nJOB-light",
    ),
    # --- abstract and introduction (anchored to their sentences)
    Check(
        "abstract lhat p95",
        BASE,
        "pooled.l_hat_as_estimate.p95.point",
        lambda v: f"$q$-error {v:.2f} against",
    ),
    Check(
        "abstract best guard p95",
        BASE,
        "pooled.bi_sound_t1.p95.point",
        lambda v: f"against {round(v)})",
    ),
    # --- selective (certify-then-stop) sampling
    Check(
        "selective certified",
        SELECTIVE,
        "summary.selective_certified.point",
        lambda v: f"sub-join certifies,\n{100 * v:.1f}\\% [",
    ),
    Check(
        "selective certified CI low",
        SELECTIVE,
        "summary.selective_certified.ci_low",
        lambda v: f"[{100 * v:.1f}, ",
    ),
    Check(
        "selective certified CI high",
        SELECTIVE,
        "summary.selective_certified.ci_high",
        lambda v: f", {100 * v:.1f}] of {value(SELECTIVE, 'summary.n_queries'):.0f}",
    ),
    Check(
        "selective fraction sampled",
        SELECTIVE,
        "summary.median_fraction_sampled",
        lambda v: f"a median {round(100 * v)}\\% of the sub-joins",
    ),
    Check(
        "selective gave up",
        SELECTIVE,
        "summary.share_gave_up",
        lambda v: f"gives up on {round(100 * v)}\\% of queries",
    ),
    Check(
        "selective timed n",
        SELECTIVE,
        "summary.n_timed",
        lambda v: f"in the execution run ({round(v)} queries), is a median",
    ),
    Check(
        "selective denominators",
        SELDEN,
        "",
        lambda v: (
            "the execution run timed PostgreSQL's native plans "
            f"${v['reference_ratio']['execution/forced_order']:.2f}\\times$ and "
            f"${v['reference_ratio']['execution/in_run']:.2f}\\times$ slower "
            "(geometric mean) than the two other runs that executed them; with "
            "their times, selective sampling costs "
            f"${v['by_reference']['forced_order']['selective']['point']:.2f}\\times$ "
            f"and ${v['by_reference']['in_run']['selective']['point']:.2f}\\times$, "
            "and sampling all sub-joins "
            f"${v['by_reference']['forced_order']['full']['point']:.2f}\\times$ and "
            f"${v['by_reference']['in_run']['full']['point']:.2f}\\times$"
            if all(e["selective"]["point"] < 1 for e in v["by_reference"].values())
            and v["n_common"] == value(SELECTIVE, "summary.n_timed")
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    Check(
        "selective denominators by load",
        SELDEN,
        "reference_ratio_by_load",
        lambda v: (
            "Most of the gap comes from the "
            f"{v['execution/in_run|start_above_30']['n']} "
            "queries that started under high load in the execution run "
            f"(${v['execution/forced_order|start_above_30']['geo']:.2f}\\times$ and "
            f"${v['execution/in_run|start_above_30']['geo']:.2f}\\times$ there, "
            f"${v['execution/forced_order|rest']['geo']:.2f}\\times$ and "
            f"${v['execution/in_run|rest']['geo']:.2f}\\times$ on the rest)"
        ),
        transform=lambda v: v,
    ),
    Check(
        "limitations cross-run gap",
        SELDEN,
        "reference_ratio",
        lambda v: (
            f"which time the same native plans ${min(v.values()):.2f}$--"
            f"${max(v.values()):.2f}\\times$ apart on average"
        ),
        transform=lambda v: v,
    ),
    Check(
        "selective cost",
        SELECTIVE,
        "summary.selective_cost_over_exec.point",
        lambda v: f"is a median ${v:.2f}\\times$ [",
    ),
    Check(
        "full cost",
        SELECTIVE,
        "summary.full_cost_over_exec.point",
        lambda v: f"against ${v:.2f}\\times$\n[",
    ),
    Check(
        "selective cost certified only",
        SELECTIVE,
        "summary.selective_cost_over_exec_certified_only",
        lambda v: f"(${v:.2f}\\times$ on the certified",
    ),
    # --- certified plans executed in PostgreSQL (Leading hints)
    Check(
        "intro under share neurocard",
        BASE,
        "by_proposal",
        lambda v: (
            f"NeuroCard underestimates {round(100 * v['neurocard']['under_share'])}"
            f"\\% of queries and DuckDB {round(100 * v['duckdb']['under_share'])}\\%"
        ),
        transform=lambda v: v,
    ),
    # --- JOB-light cardinality replication
    Check("jl pairs", JLB, "n_pairs", lambda v: f"({round(v)} pairs, 70 queries)"),
    Check(
        "jl min under",
        JLB,
        "by_proposal.factorjoin.under_share",
        lambda v: f"runs from {round(100 * v)}\\% (FactorJoin)",
    ),
    Check(
        "jl max under",
        JLB,
        "by_proposal.postgres.under_share",
        lambda v: f"to\n{round(100 * v)}\\% (PostgreSQL)",
    ),
    Check(
        "jl factorjoin none",
        JLB,
        "by_proposal.factorjoin.p95_none",
        lambda v: f"(${thousands(v)}\\to",
    ),
    Check(
        "jl factorjoin upper",
        JLB,
        "by_proposal.factorjoin.p95_upper_t1",
        lambda v: f"\\to{round(v)}$) and",
    ),
    Check(
        "jl postgres none",
        JLB,
        "by_proposal.postgres.p95_none",
        lambda v: f"(${round(v)}\\to",
    ),
    Check(
        "jl postgres both",
        JLB,
        "by_proposal.postgres.p95_bi_sound_t1",
        lambda v: f"\\to{v:.1f}$). The direction",
    ),
    Check(
        "jl neurocard under",
        JLB,
        "by_proposal.neurocard.under_share",
        lambda v: f"underestimates on {round(100 * v)}\\% of JOB-light",
    ),
    Check(
        "jl lhat p95",
        JLB,
        "pooled.l_hat_as_estimate.p95.point",
        lambda v: f"reaches ${v:.2f}$ [",
    ),
    Check(
        "jl both p95",
        JLB,
        "pooled.bi_sound_t1.p95.point",
        lambda v: f"against ${v:.1f}$ for the best guard), but",
    ),
    Check(
        "lpbound as estimate sentence",
        BASE,
        "pooled",
        lambda v: (
            "has a median $q$-error of "
            f"{v['lpbound_as_estimate']['median']['point']:.1f}"
            f" against\n{v['upper_t1']['median']['point']:.2f} for guarding with it, "
            "and a tail ten times that of both guards together\n("
            f"{thousands(v['lpbound_as_estimate']['p95']['point'])} against "
            f"{round(v['bi_sound_t1']['p95']['point'])}), though below that of the "
            f"upper guard alone ({thousands(v['upper_t1']['p95']['point'])})"
            if round(
                v["lpbound_as_estimate"]["p95"]["point"]
                / v["bi_sound_t1"]["p95"]["point"]
            )
            == 10
            else "MISMATCH"
        ),
        transform=lambda v: v,
    ),
    # --- execution in PostgreSQL (v2: one run per benchmark, tau = 1)
    Check(
        "load median stats",
        SP,
        "load_start_median",
        lambda v: f"at query start {v:.1f} on STATS-CEB",
    ),
    Check(
        "load median jl",
        JP,
        "load_start_median",
        lambda v: f"and {v:.1f} on JOB-light; the load gate",
    ),
    Check(
        "no high load",
        SP,
        "n_high_load",
        lambda v: "gate flagged no\nquery" if v == 0 else "MISMATCH",
    ),
    Check(
        "jl no high load",
        JP,
        "n_high_load",
        lambda v: "gate flagged no\nquery" if v == 0 else "MISMATCH",
    ),
    Check(
        "three reps",
        SP,
        "uncensored_plans_timed_reps",
        lambda v: "Every plan is timed three times" if v == 3 else "MISMATCH",
        transform=lambda v: v[0] if len(v) == 1 else -1,
    ),
    Check(
        "censored stats",
        SP,
        "n_censored_plans",
        lambda v: f"timeout ({round(v)} plans on STATS-CEB",
    ),
    Check(
        "censored jl", JP, "n_censored_plans", lambda v: f"and {round(v)} on JOB-light)"
    ),
    Check(
        "native reproduced",
        SP,
        "native_reproduced",
        lambda v: f"reproduces its native plan on {round(v)} of 146 STATS-CEB",
    ),
    Check(
        "headroom published",
        S2,
        "pairs::postgres_raw->truth::geo_speedup::point",
        lambda v: f"${v:.2f}\\times$ for the PostgreSQL estimates published",
    ),
    Check(
        "headroom duckdb",
        S2,
        "pairs::duckdb_raw->truth::geo_speedup::point",
        lambda v: f"${v:.2f}\\times$ for DuckDB's",
    ),
    Check(
        "headroom native",
        S2,
        "pairs::native_raw->truth::geo_speedup::point",
        lambda v: f"${v:.2f}\\times$ for our instance's own",
    ),
    Check(
        "published vs native",
        S2,
        "pairs::native->postgres_raw::geo_speedup",
        lambda v: (
            f"runs ${v['point']:.2f}\\times$ [{v['ci_low']:.2f}, {v['ci_high']:.2f}]"
            " as fast"
        ),
        transform=lambda v: v,
    ),
    Check(
        "upper published",
        S2,
        "pairs::postgres_raw->postgres_upper::geo_speedup::point",
        lambda v: f"recovers ${v:.2f}\\times$ of their",
    ),
    Check(
        "upper published faster",
        S2,
        "pairs::postgres_raw->postgres_upper::share_faster::point",
        lambda v: f"({round(100 * v)}\\% of\nqueries faster",
    ),
    Check(
        "upper published slower",
        S2,
        "pairs::postgres_raw->postgres_upper::share_slower::point",
        lambda v: f"faster, {round(100 * v)}\\% slower)",
    ),
    Check(
        "upper duckdb",
        S2,
        "pairs::duckdb_raw->duckdb_upper::geo_speedup::point",
        lambda v: f"(${v:.2f}\\times$), which\nunderestimates",
    ),
    Check(
        "floor duckdb",
        S2,
        "pairs::duckdb_raw->duckdb_lower::geo_speedup::point",
        lambda v: f"(${v:.2f}\\times$ for DuckDB)",
    ),
    Check(
        "native upper plans",
        S2,
        "pairs::native_raw->native_upper::plans_differ",
        lambda v: f"upper: {round(v)} of 146 plans change",
    ),
    Check(
        "cout upper published",
        S2,
        "pairs::postgres_raw->postgres_upper::geo_cout_ratio::point",
        lambda v: f"${v:.2f}\\times$ under the upper guard",
    ),
    Check(
        "cout upper duckdb",
        S2,
        "pairs::duckdb_raw->duckdb_upper::geo_cout_ratio::point",
        lambda v: f"unchanged (${v:.2f}\\times$) for DuckDB's",
    ),
    Check(
        "lhat excluded n",
        S2,
        "pairs::postgres_raw->truth::without_lhat::n",
        lambda v: f"sub-plan of {round(v)}\nqueries",
    ),
    Check(
        "lhat matched published",
        S2,
        "pairs::postgres_raw->truth::matched_lhat::geo_speedup::point",
        lambda v: f"of ${v:.2f}\\times$, $",
    ),
    Check(
        "lhat published",
        S2,
        "pairs::postgres_raw->lhat_est::geo_speedup::point",
        lambda v: f"source: ${v:.2f}\\times$ of",
    ),
    Check(
        "lhat matched duckdb",
        S2,
        "pairs::duckdb_raw->truth::matched_lhat::geo_speedup::point",
        lambda v: f"of\n${v:.2f}\\times$ and",
    ),
    Check(
        "lhat matched native",
        S2,
        "pairs::native_raw->truth::matched_lhat::geo_speedup::point",
        lambda v: f"of ${v:.2f}\\times$, and against",
    ),
    Check(
        "lhat vs truth",
        S2,
        "pairs::truth->lhat_est::geo_speedup::point",
        lambda v: f"indistinguishable (${v:.2f}\\times$)",
    ),
    Check(
        "excluded headroom",
        S2,
        "pairs::postgres_raw->truth::without_lhat::geo_speedup",
        lambda v: (
            f"${v['point']:.2f}\\times$ [{v['ci_low']:.2f}, {v['ci_high']:.2f}] for"
        ),
        transform=lambda v: v,
    ),
    Check(
        "lpbound vs truth",
        S2,
        "pairs::truth->lpbound_est::geo_speedup",
        lambda v: (
            f"${v['point']:.2f}\\times$ [{v['ci_low']:.2f}, {v['ci_high']:.2f}]\nas"
            " fast as planning on the true sizes"
        ),
        transform=lambda v: v,
    ),
    Check(
        "lpbound vs truth slower",
        S2,
        "pairs::truth->lpbound_est::share_slower::point",
        lambda v: f"({round(100 * v)}\\% of queries slower)",
    ),
    Check(
        "abstract upper published",
        S2,
        "pairs::postgres_raw->postgres_upper::geo_speedup::point",
        lambda v: f"LpBound ${v:.2f}\\times$ on STATS-CEB",
    ),
    # JOB-light
    Check(
        "jl floor published",
        J2,
        "pairs::postgres_raw->postgres_lower::geo_speedup",
        lambda v: (
            f"recovers ${v['point']:.2f}\\times$ [{v['ci_low']:.2f}, "
            f"{v['ci_high']:.2f}] of a"
        ),
        transform=lambda v: v,
    ),
    Check(
        "jl headroom published",
        J2,
        "pairs::postgres_raw->truth::geo_speedup::point",
        lambda v: f"${v:.2f}\\times$ headroom, the upper",
    ),
    Check(
        "jl upper published",
        J2,
        "pairs::postgres_raw->postgres_upper::geo_speedup::point",
        lambda v: f"guard only ${v:.2f}\\times$",
    ),
    Check(
        "jl factorjoin upper",
        J2,
        "pairs::factorjoin_raw->factorjoin_upper::geo_speedup::point",
        lambda v: f"recovers ${v:.2f}\\times$ of",
    ),
    Check(
        "jl factorjoin headroom",
        J2,
        "pairs::factorjoin_raw->truth::geo_speedup::point",
        lambda v: f"of ${v:.2f}\\times$ and the floor changes no plan",
    ),
    Check(
        "jl factorjoin floor no change",
        J2,
        "pairs::factorjoin_raw->factorjoin_lower::plans_differ",
        lambda v: "the floor changes no plan" if v == 0 else "MISMATCH",
    ),
    Check(
        "jl bayescard headroom",
        J2,
        "pairs::bayescard_raw->truth::geo_speedup::point",
        lambda v: f"leaves no headroom (${v:.2f}\\times$)",
    ),
    Check(
        "jl bayescard sound guards",
        J2,
        "pairs",
        lambda v: "no sound guard changes any of its plans" if v == 0 else "MISMATCH",
        transform=lambda v: sum(
            v[f"bayescard_raw->bayescard_{g}"]["plans_differ"]
            for g in ("upper", "lower", "bi")
        ),
    ),
    Check(
        "jl lhat vs truth",
        J2,
        "pairs::truth->lhat_est::geo_speedup::point",
        lambda v: f"(${v:.2f}\\times$ against them)",
    ),
    Check(
        "jl lpbound vs truth",
        J2,
        "pairs::truth->lpbound_est::geo_speedup",
        lambda v: (
            f"(${v['point']:.2f}\\times$ [{v['ci_low']:.2f}, {v['ci_high']:.2f}])"
        ),
        transform=lambda v: v,
    ),
)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path("docs/vldb_2027/paper.tex"))
    args = parser.parse_args()
    tex = normalise(args.paper.read_text())
    cache: dict[Path, Any] = {}
    failures: list[str] = []
    entries = superseded_entries()
    sources = {check.source for check in CHECKS} | TABLE_SOURCES
    for source in sorted(sources):
        entry = is_superseded(source, entries)
        if entry is not None:
            logger.error("source %s is listed as superseded (%s)", source, entry)
            failures.append(f"superseded source {source}")
        payload = json.loads(source.read_text())
        cache[source] = payload
        inputs = payload.get("inputs") if isinstance(payload, dict) else None
        for input_path, digest in (inputs or {}).items():
            current = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
            if current != digest:
                logger.error(
                    "%s is stale: %s changed since it was made", source, input_path
                )
                failures.append(f"stale source {source}")
    for check in CHECKS:
        rendered = normalise(
            check.render(check.transform(_dig(cache[check.source], check.path)))
        )
        found = rendered in tex
        logger.info("%s %-34s %r", "ok  " if found else "MISS", check.label, rendered)
        if not found:
            failures.append(check.label)
    # Generated table rows must appear verbatim.
    rows = (
        [
            (f"{benchmark} row {i}", row)
            for benchmark, path in DIRECTION_SOURCES.items()
            for i, row in enumerate(direction_rows(path), start=1)
        ]
        + [
            (f"{benchmark} execution row {i}", row)
            for benchmark, (path, sources) in EXECUTION_SOURCES.items()
            for i, row in enumerate(execution_rows(path, sources), start=1)
        ]
        + [
            (f"replication row {i}", row)
            for i, row in enumerate(replication_rows(), start=1)
        ]
        + [
            (f"certificate row {i}", row)
            for i, row in enumerate(certificate_rows(), start=1)
        ]
        + [
            (f"sampling-rate row {i}", row)
            for i, row in enumerate(sampling_rate_rows(), start=1)
        ]
        + [
            (f"tightness row {i}", row)
            for i, row in enumerate(tightness_rows(), start=1)
        ]
        + [(f"pg16 row {i}", row) for i, row in enumerate(pg16_rows(), start=1)]
    )
    for label, row in rows:
        found = normalise(row) in tex
        logger.info("%s %-34s %r", "ok  " if found else "MISS", label, row)
        if not found:
            failures.append(label)
    total = len(CHECKS) + len(rows)
    if failures:
        logger.error("%d of %d absent: %s", len(failures), total, failures)
        sys.exit(1)
    logger.info("all %d ICDE headline numbers present in %s", total, args.paper)


if __name__ == "__main__":
    main()
