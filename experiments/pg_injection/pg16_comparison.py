"""PostgreSQL 16 against PostgreSQL 13.1: the execution results side by side.

Reads the summaries ``analyze_injection`` and ``summarize_tightness_points``
write for both versions (13.1: ``results/pg_injection/v3/``; 16:
``results/pg_injection/pg16/``) plus the raw PostgreSQL 16 run, and evaluates
the predictions frozen in ``docs/vldb_2027/PREDICTIONS_PG16.md`` (R1-R6)
exactly as written there.

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.pg16_comparison
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import _read_jsonl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

V3 = Path("results/pg_injection/v3")
PG16 = Path("results/pg_injection/pg16")
PAIRS = {
    "headroom": "native_raw->truth",
    "upper": "native_raw->native_upper",
    "floor": "native_raw->native_lower",
    "both": "native_raw->native_bi",
    "published_upper": "postgres_raw->postgres_upper",
    "lpbound_vs_truth": "truth->lpbound_est",
    "published_headroom": "postgres_raw->truth",
    "duckdb_floor": "duckdb_raw->duckdb_lower",
}


def numbers(summary: Path, points: Path) -> dict[str, Any]:
    """Point estimates and intervals of the compared quantities for one version."""
    pairs = json.loads(summary.read_text())["pairs"]
    out: dict[str, Any] = {
        name: pairs[key]["geo_speedup"] for name, key in PAIRS.items() if key in pairs
    }
    time = json.loads(points.read_text())["time"]
    for k in ("bi_k2", "bi_k8"):
        out[f"{k}_share"] = time[k]["share_of_truth_speedup"]
    out["points_headroom"] = time["truth"]["geo_speedup"]
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PG16 / "pg16_comparison.json")
    args = parser.parse_args()
    old = numbers(V3 / "stats_summary.json", V3 / "tightness_points_summary.json")
    new = numbers(PG16 / "stats_summary.json", PG16 / "tightness_points_summary.json")
    runs = _read_jsonl(PG16 / "stats_inject.jsonl")
    reproduced = sum(
        r["policy_plan"]["native"] == r["policy_plan"]["native_raw"] for r in runs
    )

    def point(name: str) -> float:
        return float(new[name]["point"])

    verdicts = {
        "R1": reproduced >= 138,
        "R2": 1.15 <= point("headroom") <= 1.60,
        "R3": max(point("upper"), point("floor"), point("both")) <= 1.12,
        "R4": point("published_upper") >= 2.0,
        "R5": point("lpbound_vs_truth") < 0.95,
        "R6": point("bi_k2_share") >= 0.5 > point("bi_k8_share"),
    }
    payload = {
        "predictions": "docs/vldb_2027/PREDICTIONS_PG16.md",
        "n_queries": len(runs),
        "native_reproduced": reproduced,
        "pg13": old,
        "pg16": new,
        "verdicts": {k: "holds" if v else "fails" for k, v in verdicts.items()},
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for name in new:
        logger.info(
            "%-20s 13.1 %.2f  16 %.2f [%.2f, %.2f]",
            name,
            old[name]["point"],
            new[name]["point"],
            new[name]["ci_low"],
            new[name]["ci_high"],
        )
    logger.info("reproduced %d/%d; %s", reproduced, len(runs), payload["verdicts"])


if __name__ == "__main__":
    main()
