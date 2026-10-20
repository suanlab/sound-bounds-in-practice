"""Sensitivity of the execution results to the tie band and the timeout.

The paper counts a query as faster or slower when the speedup leaves a +-10%
band, and keeps plans that hit the 300 s timeout at 300 s. This recomputes, for
the comparisons the paper reports, the faster/slower shares with a 5% and a 20%
band, and the geometric-mean speedup with every plan time re-capped at 120 s
and 60 s (a lower cap can only be imposed after the fact; a higher one would
need new runs), and once more without the queries that started while the
one-minute load was above 30. Besides each source's guards and planning
policies, it covers the guarded plans of the published PostgreSQL estimates
against the native plan and planning on LpBound against the true sizes.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.timing_sensitivity \
        --runs results/pg_injection/v3/stats_inject.jsonl \
        --output results/pg_injection/v3/stats_timing_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import _read_jsonl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BANDS = (0.05, 0.10, 0.20)
CAPS_MS = (300_000.0, 120_000.0, 60_000.0)
QUIET_LOAD = 30.0  # queries that started above this load are left out once


def comparisons(policies: set[str]) -> list[tuple[str, str]]:
    """Guard and planning comparisons per estimate source, as in the paper."""
    out = []
    for source in sorted(p[:-4] for p in policies if p.endswith("_raw")):
        raw = f"{source}_raw"
        for policy in ("truth", f"{source}_upper", f"{source}_lower", "lpbound_est"):
            if policy in policies:
                out.append((raw, policy))
    # Against PostgreSQL's native plan, and planning on LpBound against the truth.
    for policy in ("postgres_upper", "postgres_lower", "postgres_bi"):
        if policy in policies:
            out.append(("native", policy))
    if "lpbound_est" in policies:
        out.append(("truth", "lpbound_est"))
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = _read_jsonl(args.runs)
    times = [{p: r["plan_ms"][d] for p, d in r["policy_plan"].items()} for r in records]
    policies = set().union(*(t.keys() for t in times))
    out: dict[str, Any] = {"runs": str(args.runs), "n_queries": len(records)}
    for base, policy in comparisons(policies):
        pairs = [(t[base], t[policy]) for t in times if base in t and policy in t]
        entry: dict[str, Any] = {"n": len(pairs)}
        for band in BANDS:
            entry[f"band_{round(100 * band)}"] = {
                "share_faster": sum(b / v > 1 + band for b, v in pairs) / len(pairs),
                "share_slower": sum(b / v < 1 / (1 + band) for b, v in pairs)
                / len(pairs),
            }
        for cap in CAPS_MS:
            logs = [
                math.log(max(min(b, cap), 1e-3) / max(min(v, cap), 1e-3))
                for b, v in pairs
            ]
            entry[f"geo_cap_{round(cap / 1000)}s"] = math.exp(sum(logs) / len(logs))
        quiet = [
            math.log(max(t[base], 1e-3) / max(t[policy], 1e-3))
            for t, r in zip(times, records, strict=True)
            if base in t and policy in t and r["loadavg_1m_start"] <= QUIET_LOAD
        ]
        entry["n_quiet_start"] = len(quiet)
        entry["geo_quiet_start"] = math.exp(sum(quiet) / len(quiet))
        out[f"{base}->{policy}"] = entry
        logger.info(
            "%-28s band5 %.2f/%.2f band20 %.2f/%.2f  geo 300s %.2f 120s %.2f 60s %.2f",
            f"{base}->{policy}",
            entry["band_5"]["share_faster"],
            entry["band_5"]["share_slower"],
            entry["band_20"]["share_faster"],
            entry["band_20"]["share_slower"],
            entry["geo_cap_300s"],
            entry["geo_cap_120s"],
            entry["geo_cap_60s"],
        )
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
