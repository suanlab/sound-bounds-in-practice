"""Re-score the C4 plan certificates with commuting first joins merged.

The pre-registered rule (``docs/c4_preregistration.md`` item 11) calls a query
certified when exactly one left-deep order survives dominance pruning. Under
the C_out cost model that rule cannot be met: an order ``(a, b, c, ...)`` and
its twin ``(b, a, c, ...)`` have the same prefixes of size >= 2, hence the same
cost under *every* bracket, so pruning removes or keeps both and at least two
orders always survive. A twin class has exactly two members (the prefix sets of
size >= 2 fix every position after the second), so survivors come in pairs, and
two survivors mean one cost-equivalence class: the optimal *plan* is identified
up to the commutation that C_out cannot see.

This script is post-hoc. It corrects a defect of the registered rule, which the
paper reports as such, and re-applies the frozen decision thresholds unchanged.
It reads only the committed ``plan_certificate.json``.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.plan_certificate_commutation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from experiments.scripts.bootstrap_ci import cluster_bootstrap, zero_event_upper_bound
from experiments.scripts.plan_certificate import (
    LOWERS,
    UPPERS,
    _bracket_rows,
    _mean,
    _paired_share_difference,
    _share,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PRIMARY = "lpboundxsample0.25"
BASELINE = "lpboundxone"


def add_class_certification(per_query: list[dict[str, Any]]) -> None:
    """Mark ``certified_class`` on every unflagged record, checking the pairing.

    Raises:
        ValueError: if a survivor or plan count is odd, which would contradict
            the twin argument in the module docstring.
    """
    for query in per_query:
        for key, record in query["brackets"].items():
            if record["flagged"]:
                continue
            if record["n_plans"] % 2 or record["n_survivors"] % 2:
                msg = f"odd count for query {query['query_id']} bracket {key}"
                raise ValueError(msg)
            if record["certified"]:
                msg = f"registered rule met on {query['query_id']} {key}"
                raise ValueError(msg)
            record["certified_class"] = record["n_survivors"] == 2


def summarise(
    per_query: list[dict[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    """Class-level certified share per bracket, the paired test and decisions."""
    brackets: dict[str, Any] = {}
    for upper_name in UPPERS:
        for lower_name in LOWERS:
            key = f"{upper_name}x{lower_name}"
            rows = _bracket_rows(per_query, key)
            share = cluster_bootstrap(
                rows, _share("certified_class"), replicates=replicates, seed=seed
            )
            hits = sum(1 for r in rows if r["certified_class"])
            share["count"] = hits
            if hits == 0:
                share["zero_event_upper_95"] = zero_event_upper_bound(len(rows))
            brackets[key] = {
                "n_queries": len(rows),
                "certified_class_share": share,
                "min_survivors": min(r["n_survivors"] for r in rows),
                "useful_count": sum(1 for r in rows if r["useful"]),
                "min_G": min(float(r["G"]) for r in rows),
            }
    diff_rows = _paired_share_difference(
        per_query, PRIMARY, BASELINE, "certified_class"
    )
    paired = cluster_bootstrap(
        diff_rows, _mean("diff"), replicates=replicates, seed=seed
    )
    primary = brackets[PRIMARY]["certified_class_share"]
    decision = {
        "primary_bracket": PRIMARY,
        "certificates_non_trivial": primary["ci_low"] > 0.10,
        "lower_certificate_matters": paired["ci_low"] > 0.0 or paired["ci_high"] < 0.0,
    }
    return {
        "brackets": brackets,
        "paired_certified_class_share_primary_minus_one": paired,
        "decision_with_frozen_thresholds": decision,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/plan_certificate/plan_certificate.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/plan_certificate_commutation.json"),
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    per_query: list[dict[str, Any]] = payload["per_query"]
    add_class_certification(per_query)
    result = {
        "experiment": "plan_certificate_commutation",
        "preregistered": False,
        "purpose": (
            "post-hoc correction of the registered |S|=1 rule, which C_out makes "
            "unattainable because the first two relations of a left-deep order "
            "commute; frozen thresholds re-applied unchanged"
        ),
        "source": str(args.input),
        "replicates": args.replicates,
        "seed": args.seed,
        **summarise(per_query, args.replicates, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for key, entry in result["brackets"].items():
        share = entry["certified_class_share"]
        logger.info(
            "%-22s class-certified %3d/%3d  %.1f%% [%.1f, %.1f]  min G %.2f",
            key,
            share["count"],
            entry["n_queries"],
            100 * share["point"],
            100 * share["ci_low"],
            100 * share["ci_high"],
            entry["min_G"],
        )
    paired = result["paired_certified_class_share_primary_minus_one"]
    logger.info(
        "paired primary - L=1: %.1f pp [%.1f, %.1f]",
        100 * paired["point"],
        100 * paired["ci_low"],
        100 * paired["ci_high"],
    )
    logger.info("decision: %s", result["decision_with_frozen_thresholds"])
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
