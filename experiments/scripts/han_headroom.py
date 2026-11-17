"""Headroom of the true join sizes on total time, set beside Han et al.'s TrueCard.

Han et al. (PVLDB 15(4), Table 3) report STATS-CEB's total time falling from
11.34 h (PostgreSQL) to 5.69 h (TrueCard). Our execution run measures the
headroom as a per-query geometric mean. This script recomputes it on the
workload's total (capped) time from the same per-query medians. It also counts
the queries whose native and true-size plans both hit the 300 s cap, and their
share of the total. Post-hoc; read-only on the execution runs.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.han_headroom
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RUNS = {
    "stats_ceb": Path("results/pg_injection/v3/stats_inject.jsonl"),
    "joblight": Path("results/pg_injection/v3/joblight_inject.jsonl"),
}
OUT = Path("results/pg_injection/v3/han_headroom.json")
HAN_POSTGRESQL_H = 11.34
HAN_TRUECARD_H = 5.69
HAN_SOURCE = "Han et al., PVLDB 15(4):752-765, Table 3 (external)"


def summarise(path: Path) -> dict[str, object]:
    """Return total-time and geometric-mean headroom of true sizes over native plans.

    Args:
        path: An execution run (JSONL, one query per line).

    Returns:
        Totals in hours, both headroom measures, and the queries whose native and
        true-size plans were both capped.
    """
    nat_ms = tru_ms = capped_ms = 0.0
    logs: list[float] = []
    both_capped: list[int] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        plans = rec["policy_plan"]
        native = float(rec["plan_ms"][plans["native"]])
        truth = float(rec["plan_ms"][plans["truth"]])
        nat_ms += native
        tru_ms += truth
        logs.append(math.log(native / truth))
        censored = rec["plan_censored"]
        if censored[plans["native"]] and censored[plans["truth"]]:
            both_capped.append(int(rec["query_id"]))
            capped_ms += native
    return {
        "n_queries": len(logs),
        "native_total_h": nat_ms / 3.6e6,
        "truth_total_h": tru_ms / 3.6e6,
        "headroom_total_time": nat_ms / tru_ms,
        "headroom_geomean": math.exp(sum(logs) / len(logs)),
        "queries_both_plans_capped": both_capped,
        "n_both_plans_capped": len(both_capped),
        "both_capped_share_of_native_total": capped_ms / nat_ms,
    }


def main() -> None:
    """Write ``han_headroom.json`` with input digests for the audit."""
    out: dict[str, object] = {
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in RUNS.values()
        },
        "post_hoc": True,
        **{name: summarise(p) for name, p in RUNS.items()},
        "han_stats_ceb": {
            "postgresql_h": HAN_POSTGRESQL_H,
            "truecard_h": HAN_TRUECARD_H,
            "source": HAN_SOURCE,
            "headroom_total_time": HAN_POSTGRESQL_H / HAN_TRUECARD_H,
        },
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    logger.info("wrote %s", OUT)
    logger.info(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
