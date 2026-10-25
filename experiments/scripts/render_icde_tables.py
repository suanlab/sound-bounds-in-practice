"""Render the ICDE paper's per-estimator table rows from the result JSONs.

The rows are generated, not typed: ``audit_icde_numbers`` renders them again
and requires each one verbatim in the LaTeX source, so a table cannot drift
from the measurements it reports.

Usage::

    python -m experiments.scripts.render_icde_tables
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

R = Path("results/utility_table")
DIRECTION_SOURCES = {
    "STATS-CEB": R / "cardinality_baselines_pg.json",
    "JOB-light": R / "cardinality_baselines_pg_joblight.json",
}
NAMES = {
    "neurocard": "NeuroCard",
    "factorjoin": "FactorJoin",
    "postgres": "PostgreSQL",
    "flat": "FLAT",
    "dbx": "Commercial",
    "duckdb": "DuckDB",
    "bayescard": "BayesCard",
    "deepdb": "DeepDB",
}


def qerr(value: float) -> str:
    """Format a q-error the way the paper's tables do."""
    if value >= 1e6:
        exponent = math.floor(math.log10(value))
        mantissa = round(value / 10**exponent, 1)
        if mantissa >= 10:  # e.g. 9.96e6 rounds to 10.0: renormalise
            mantissa, exponent = mantissa / 10, exponent + 1
        return f"${mantissa:.1f}{{\\times}}10^{{{exponent}}}$"
    if value >= 1000:
        return f"{round(value):,}".replace(",", "{,}")
    if value >= 100:
        return f"{round(value)}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def direction_rows(path: Path) -> list[str]:
    """One row per estimator, ordered by its share of underestimates."""
    by_proposal = json.loads(path.read_text())["by_proposal"]
    ordered = sorted(by_proposal.items(), key=lambda kv: kv[1]["under_share"])
    return [
        f"{NAMES[name]} & {round(100 * v['under_share'])}\\% & "
        f"{qerr(v['p95_none'])} & {qerr(v['p95_upper_t1'])} & "
        f"{qerr(v['p95_bi_sound_t1'])} \\\\"
        for name, v in ordered
    ]


V2 = Path("results/pg_injection/v2")
V3 = Path("results/pg_injection/v3")
EXECUTION_SOURCES: dict[str, tuple[Path, tuple[str, ...]]] = {
    "STATS-CEB": (V3 / "stats_summary.json", ("postgres", "duckdb", "native")),
    "JOB-light": (
        V3 / "joblight_summary.json",
        ("postgres", "duckdb", "factorjoin", "bayescard", "native"),
    ),
}
# (label, policy; "{s}" is the source) for rows over every query.
EXECUTION_ROWS = (
    ("True join sizes", "truth"),
    ("Upper guard", "{s}_upper"),
    ("Sound floor", "{s}_lower"),
    ("Both guards", "{s}_bi"),
    ("Upper $+$ $\\hat L$ floor", "{s}_bi_hat"),
    ("Plan on LpBound", "lpbound_est"),
)


def interval(stat: dict[str, float]) -> str:
    """A point estimate with its bootstrap interval."""
    return f"{stat['point']:.2f} [{stat['ci_low']:.2f}, {stat['ci_high']:.2f}]"


def execution_rows(path: Path, sources: tuple[str, ...]) -> list[str]:
    """Geometric-mean speedup over each source's own injected estimates.

    Rows over every query come first. Planning on L-hat exists only where the
    sample covers every sub-plan, so it is reported next to the headroom on
    exactly those queries (a matched denominator) when that is a subset.
    """
    pairs = json.loads(path.read_text())["pairs"]
    rows = []
    for label, policy in EXECUTION_ROWS:
        cells = [
            interval(pairs[f"{s}_raw->{policy.format(s=s)}"]["geo_speedup"])
            for s in sources
        ]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    truth = [pairs[f"{s}_raw->truth"] for s in sources]
    matched = [t["matched_lhat"] for t in truth]
    if any(m["n"] != t["n"] for m, t in zip(matched, truth, strict=True)):
        n = matched[0]["n"]
        rows.append(
            f"\\multicolumn{{{len(sources) + 1}}}{{@{{}}l}}"
            f"{{\\emph{{{n} queries with a complete sample}}}} \\\\"
        )
        rows.append(
            "True join sizes & "
            + " & ".join(interval(m["geo_speedup"]) for m in matched)
            + " \\\\"
        )
    rows.append(
        "Plan on $\\hat L$ & "
        + " & ".join(
            interval(pairs[f"{s}_raw->lhat_est"]["geo_speedup"]) for s in sources
        )
        + " \\\\"
    )
    return rows


REPLICATION = V3 / "artifact_replication.json"
REPLICATION_ROWS = (
    ("PostgreSQL $\\to$ true sizes", "pg_to_truth"),
    ("PostgreSQL $\\to$ LpBound", "pg_to_lpbound"),
    ("True sizes $\\to$ LpBound", "truth_to_lpbound"),
)


def replication_rows(path: Path = REPLICATION) -> list[str]:
    """Artifact's own end-to-end speedups next to ours, both benchmarks."""
    data = json.loads(path.read_text())
    rows = []
    for label, key in REPLICATION_ROWS:
        cells = [
            interval(data[benchmark][key][side])
            for benchmark in ("stats", "joblight")
            for side in ("artifact", "ours")
        ]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return rows


CERT = Path("results/plan_certificate")
CERT_PRIMARY = CERT / "plan_certificate_joint_pg_r025.json"
CERT_LOW_RATE = CERT / "plan_certificate_joint_pg_r010.json"
CERT_UPPERS = (
    ("LpBound", "lpbound"),
    ("SafeBound", "safebound"),
    ("max-deg.", "maxdeg"),
)


def _point(value: object) -> float:
    """A summary value: a bare number or a bootstrap record's point."""
    if isinstance(value, dict):
        return float(value["point"])
    if isinstance(value, int | float):
        return float(value)
    msg = f"not a number: {value!r}"
    raise TypeError(msg)


def percent_interval(stat: dict[str, float]) -> str:
    """A share with its interval, in percent."""
    return (
        f"{100 * stat['point']:.1f}\\% "
        f"[{100 * stat['ci_low']:.1f}, {100 * stat['ci_high']:.1f}]"
    )


def certificate_rows(
    primary: Path = CERT_PRIMARY, low_rate: Path = CERT_LOW_RATE
) -> list[str]:
    """Joint-certificate shares from PostgreSQL's in-engine samples."""
    main_summary = json.loads(primary.read_text())["summary"]
    low_summary = json.loads(low_rate.read_text())["summary"]
    rows = []
    for label, key in CERT_UPPERS:
        cout = main_summary[f"{key}xsample|cout"]
        minus = main_summary[f"{key}xsample|cout_minus_final"]
        low = low_summary[f"{key}xsample|cout"]
        rows.append(
            f"{label} & sample & {percent_interval(cout['share_joint_bound_le_2'])} & "
            f"{_point(cout['median_joint_bound_best']):.2f} & "
            f"{100 * _point(minus['share_joint_bound_le_2']):.1f}\\% & "
            f"{100 * _point(low['share_joint_bound_le_2']):.1f}\\% \\\\"
        )
    floor = min(
        _point(main_summary[f"{key}xone|cout"]["median_joint_bound_best"])
        for _, key in CERT_UPPERS
    )
    exponent = math.floor(math.log10(floor))
    rows.append(
        "any & $L\\equiv1$ & 0\\% & "
        f"${{\\ge}}{floor / 10**exponent:.1f}{{\\times}}10^{{{exponent}}}$ & "
        "0\\% & 0\\% \\\\"
        if all(
            _point(summary[f"{key}xone|{cost}"]["share_joint_bound_le_2"]) == 0
            for summary in (main_summary, low_summary)
            for _, key in CERT_UPPERS
            for cost in ("cout", "cout_minus_final")
            if f"{key}xone|{cost}" in summary
        )
        and any(f"{key}xone|cout" in main_summary for _, key in CERT_UPPERS)
        else "MISMATCH: an upper bound alone certified a query"
    )
    return rows


RATE_COST = V2 / "sampling_rate.json"
RATE_CARD = {
    "0.10": R / "cardinality_baselines_pg.json",
    "0.25": R / "cardinality_baselines_pg_r025.json",
}
RATE_CERT = {"0.10": CERT_LOW_RATE, "0.25": CERT_PRIMARY}


def sampling_rate_rows() -> list[str]:
    """One row per sampling rate: cost, accuracy of L and L-hat, certificates."""
    cost = json.loads(RATE_COST.read_text())
    rows = []
    for rate, key in (("0.10", "cost_r010"), ("0.25", "cost_r025")):
        pooled = json.loads(RATE_CARD[rate].read_text())["pooled"]
        cert = json.loads(RATE_CERT[rate].read_text())["summary"]
        share = _point(cert["lpboundxsample|cout"]["share_joint_bound_le_2"])
        rows.append(
            f"${rate}$ & {cost['all'][key]['point']:.2f} & "
            f"{cost['exact_over_1s'][key]['point']:.2f} & "
            f"{pooled['l_sound_as_estimate']['median']['point']:.2f} & "
            f"{pooled['l_hat_as_estimate']['p95']['point']:.2f} & "
            f"{100 * share:.1f}\\% \\\\"
        )
    return rows


TIGHTNESS = R / "tightness_all_maxdegfix_t1.json"


def tightness_firing(path: Path = TIGHTNESS) -> dict[str, dict[str, int]]:
    """Firing count of each upper bound, per estimator (gate at the bound)."""
    counts: dict[str, dict[str, int]] = {}
    for row in json.loads(path.read_text())["per_pair"]:
        per = counts.setdefault(row["proposal"], {})
        per[row["certificate"]] = per.get(row["certificate"], 0) + int(row["fired"])
    return counts


def tightness_rows(path: Path = TIGHTNESS) -> list[str]:
    """One row per estimator, in the order of the direction table."""
    counts = tightness_firing(path)
    by_proposal = json.loads(DIRECTION_SOURCES["STATS-CEB"].read_text())["by_proposal"]
    ordered = sorted(counts, key=lambda name: by_proposal[name]["under_share"])
    return [
        f"{NAMES[name]} & {counts[name]['max_degree']} & "
        f"{counts[name]['safebound']} & {counts[name]['lpbound']} \\\\"
        for name in ordered
    ]


# Every result file a generated table reads (checked by the audit).
PG16 = Path("results/pg_injection/pg16/pg16_comparison.json")
PG16_ROWS = (
    ("Our estimates $\\to$ true sizes", "headroom", "speedup"),
    ("Upper guard on our estimates", "upper", "speedup"),
    ("Floor on our estimates", "floor", "speedup"),
    ("Both guards on our estimates", "both", "speedup"),
    ("Both, synthetic $k=2$ (share)", "bi_k2_share", "share"),
    ("Both, synthetic $k=8$ (share)", "bi_k8_share", "share"),
    ("Published PG: upper guard", "published_upper", "speedup"),
    ("DuckDB: floor", "duckdb_floor", "speedup"),
    ("True sizes $\\to$ LpBound", "lpbound_vs_truth", "speedup"),
)


def pg16_rows(path: Path = PG16) -> list[str]:
    """PostgreSQL 13.1 against 16: point for 13.1, interval for 16."""
    data = json.loads(path.read_text())
    rows = []
    for label, key, kind in PG16_ROWS:
        old, new = data["pg13"][key], data["pg16"][key]
        if kind == "share":
            cells = (
                f"{round(100 * old['point'])}\\%",
                f"{round(100 * new['point'])}\\% "
                f"[{round(100 * new['ci_low'])}, {round(100 * new['ci_high'])}]",
            )
        else:
            cells = (f"{old['point']:.2f}", interval(new))
        rows.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
    return rows


TABLE_SOURCES: set[Path] = {
    *DIRECTION_SOURCES.values(),
    *(path for path, _ in EXECUTION_SOURCES.values()),
    REPLICATION,
    CERT_PRIMARY,
    CERT_LOW_RATE,
    RATE_COST,
    *RATE_CARD.values(),
    TIGHTNESS,
    PG16,
}


def main() -> None:
    """CLI entry point: log every rendered row."""
    for benchmark, path in DIRECTION_SOURCES.items():
        logger.info("%% %s", benchmark)
        for row in direction_rows(path):
            logger.info("%s", row)
    for benchmark, (path, sources) in EXECUTION_SOURCES.items():
        logger.info("%% execution, %s: %s", benchmark, ", ".join(sources))
        for row in execution_rows(path, sources):
            logger.info("%s", row)
    logger.info("%s", "% plan certificates (in-engine samples)")
    for row in certificate_rows():
        logger.info("%s", row)
    logger.info("%s", "% PostgreSQL 13.1 against 16")
    for row in pg16_rows():
        logger.info("%s", row)
    logger.info("%s", "% tightness: firing per estimator")
    for row in tightness_rows():
        logger.info("%s", row)
    logger.info("%s", "% sampling rate")
    for row in sampling_rate_rows():
        logger.info("%s", row)
    logger.info("%s", "% replication against the artifact's own runs")
    for row in replication_rows():
        logger.info("%s", row)


if __name__ == "__main__":
    main()
