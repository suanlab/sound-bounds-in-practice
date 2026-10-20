"""How much the sample-based results depend on the one Bernoulli draw.

Every in-engine sample uses ``TABLESAMPLE BERNOULLI (10) REPEATABLE (seed)``; the
paper's results use seed 0, a single draw shared by all queries. Seeds 1-4 were
drawn afterwards (sampled counts only). For each seed this collects the
statistics the paper reports from the sample at r = 0.10 -- the 95th-percentile
q-error of L-hat and of both guards as the whole-query estimate, the median
q-error of the sound count, and the share of STATS-CEB queries with a plan
certified within 2x (LpBound, joint, C_out) -- and their range over the five.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.seed_robustness
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CARD = {
    0: Path("results/utility_table/cardinality_baselines_pg.json"),
    **{
        s: Path(f"results/utility_table/cardinality_baselines_pg_seed{s}.json")
        for s in range(1, 5)
    },
}
CERT = {
    0: Path("results/plan_certificate/plan_certificate_joint_pg_r010.json"),
    **{
        s: Path(f"results/plan_certificate/plan_certificate_joint_pg_r010_seed{s}.json")
        for s in range(1, 5)
    },
}


def per_seed(seed: int) -> dict[str, float]:
    """The sample-based statistics of one draw."""
    pooled = json.loads(CARD[seed].read_text())["pooled"]
    summary = json.loads(CERT[seed].read_text())["summary"]
    share = summary["lpboundxsample|cout"]["share_joint_bound_le_2"]
    return {
        "lhat_p95": pooled["l_hat_as_estimate"]["p95"]["point"],
        "lhat_median": pooled["l_hat_as_estimate"]["median"]["point"],
        "l_sound_median": pooled["l_sound_as_estimate"]["median"]["point"],
        "both_guards_p95": pooled["bi_sound_t1"]["p95"]["point"],
        "certified_share": share["point"] if isinstance(share, dict) else share,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v3/seed_robustness.json"),
    )
    args = parser.parse_args()

    seeds = {seed: per_seed(seed) for seed in sorted(CARD)}
    keys = next(iter(seeds.values())).keys()
    out: dict[str, Any] = {
        "seeds": seeds,
        "range": {
            k: {
                "min": min(v[k] for v in seeds.values()),
                "max": max(v[k] for v in seeds.values()),
            }
            for k in keys
        },
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for seed, stats in seeds.items():
        logger.info("seed %d: %s", seed, {k: round(v, 3) for k, v in stats.items()})


if __name__ == "__main__":
    main()
