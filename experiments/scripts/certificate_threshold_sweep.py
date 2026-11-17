"""Sensitivity of the joint-certificate shares to the regret threshold ``t``.

The paper certifies a plan when its joint regret bound is at most ``t = 2`` and
reports the certified share on the queries where the choice of plan matters
(worst over best true ``C_out`` of the left-deep orders above 2;
``certificate_context``). A reviewer may ask whether ``t = 2`` is arbitrary.
This recomputes, for ``t`` in 1.5, 2, 3 and 4, the certified share

* among the queries whose true spread exceeds ``t`` (where certifying within
  ``t`` is not trivial), and
* among the fixed set of the paper (spread above 2),

from the stored per-query joint bounds (LpBound above, PostgreSQL's in-engine
sample below, ``C_out``); nothing is re-measured. At ``t = 2`` it must reproduce
``certificate_context`` exactly, which it asserts. Post-hoc, prepared for a
revision; not in the submitted paper.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.certificate_threshold_sweep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from experiments.scripts.bootstrap_ci import cluster_bootstrap
from experiments.scripts.certificate_context import CASES, true_spread

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

THRESHOLDS = (1.5, 2.0, 3.0, 4.0)
KEY = "lpboundxsample|cout"


def certified_share(
    rows: list[dict[str, Any]], t: float, replicates: int, seed: int
) -> dict[str, Any]:
    """Share of ``rows`` with a joint bound of at most ``t``, with its interval."""
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "n_certified": sum(r["bound"] <= t for r in rows),
        "share": cluster_bootstrap(
            rows,
            lambda rs: sum(r["bound"] <= t for r in rs) / len(rs),
            replicates=replicates,
            seed=seed,
        ),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--context",
        type=Path,
        default=Path("results/plan_certificate/certificate_context.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/certificate_threshold_sweep.json"),
    )
    args = parser.parse_args()
    context = json.loads(args.context.read_text())

    out: dict[str, Any] = {
        "experiment": "certificate_threshold_sweep",
        "preregistered": False,
        "bracket": KEY,
        "thresholds": list(THRESHOLDS),
        "inputs": {},
        "cases": {},
    }
    for case, (sizes, _, joint_file) in CASES.items():
        for path in (sizes, joint_file):
            out["inputs"][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        records = {
            r["query_id"]: r
            for r in (json.loads(x) for x in Path(sizes).read_text().splitlines() if x)
        }
        joint = json.loads(Path(joint_file).read_text())["per_query"][KEY]
        rows = [
            {
                "query_id": r["query_id"],
                "bound": float(r["joint_bound_best"]),
                "spread": true_spread(records[r["query_id"]]),
            }
            for r in joint
        ]
        paper_set = [r for r in rows if r["spread"] > 2.0]
        entry: dict[str, Any] = {"n_queries": len(rows), "by_threshold": {}}
        for t in THRESHOLDS:
            entry["by_threshold"][f"{t:g}"] = {
                "all_queries": certified_share(rows, t, args.replicates, args.seed),
                "spread_above_t": certified_share(
                    [r for r in rows if r["spread"] > t], t, args.replicates, args.seed
                ),
                "spread_above_2": certified_share(
                    paper_set, t, args.replicates, args.seed
                ),
            }
        at2 = entry["by_threshold"]["2"]["spread_above_2"]
        ref = context[case]
        if (
            at2["n"] != ref["n_choice_matters"]
            or at2["n_certified"] != ref["n_certified_where_choice_matters"]
        ):
            msg = f"{case}: t=2 does not reproduce certificate_context"
            raise AssertionError(msg)
        out["cases"][case] = entry
        for t in THRESHOLDS:
            b = entry["by_threshold"][f"{t:g}"]
            above_t, above_2 = b["spread_above_t"], b["spread_above_2"]
            logger.info(
                "%-14s t=%-3g spread>t: %3d/%3d (%5.1f%%)  spread>2: %3d/%3d (%5.1f%%)",
                case,
                t,
                above_t.get("n_certified", 0),
                above_t["n"],
                100 * above_t["share"]["point"] if above_t["n"] else float("nan"),
                above_2["n_certified"],
                above_2["n"],
                100 * above_2["share"]["point"],
            )
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
