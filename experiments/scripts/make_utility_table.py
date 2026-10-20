"""Emit the per-query certificate-utility table (C1 pilot / W2 GO-KILL data).

Implements the *pre-registered* variable definitions of
``docs/c1_preregistration.md`` (frozen 2026-07-11) verbatim:

  * direction            sign(log e - log y)
  * tightness (true)     log(B/y) upper, log(y/L) lower
  * tightness (proxy)    log(B/e) upper, log(e/L_hat) lower   [deployment-time]
  * relative cost        certificate time / full exact-count time
  * intervention_help    guard fired AND post-intervention q-error < proposal q-error

One output row per (query, proposal, guard-direction). Proposals are the nine
CEG configurations plus the AVI/independence baseline, so the table spans both
over- and under-estimating proposals. Guards: upper (e > tau*B -> B), lower
(e < L_hat/tau -> L_hat).

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.make_utility_table \
        --data-dir data/job_light --output results/utility_table/job_light.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from math import log
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from experiments.scripts.run_real_data import (
    _load_relation_rows,
    build_job_light_queries,
)
from src.core.ceg import pessimistic_bound
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.estimators.independence_estimator import IndependenceEstimator
from src.estimators.lower_certificate import aggregate_lower_certificate
from src.utils.data_loader import load_dataset_catalog
from src.utils.join_counting import count_join
from src.utils.metrics import q_error
from src.utils.types import ALL_CEG_CONFIGS

if TYPE_CHECKING:
    from src.core.query import Query

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FIELDS = [
    "query_id",
    "proposal",
    "guard",
    "estimate",
    "true_cardinality",
    "direction",
    "tightness_true",
    "tightness_proxy",
    "cost_ratio",
    "guard_fired",
    "intervention_help",
    "proposal_qerror",
    "final_qerror",
]


def _proposals(query: Query) -> dict[str, float]:
    """The nine CEG configs plus AVI, as named proposal estimates."""
    estimator = BaselineCEGEstimator()
    all_estimates = estimator.estimate_all_configs(query)
    proposals = {
        config.name: all_estimates[config.name].estimate for config in ALL_CEG_CONFIGS
    }
    proposals["avi"] = IndependenceEstimator().estimate(query)
    return proposals


def evaluate_query(
    query_id: str,
    query: Query,
    true_card: float,
    rows: dict[str, list[dict[str, str]]],
    tau: float,
    rate: float,
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Emit the utility rows for one query (all proposals x both guards)."""
    y = max(true_card, 1.0)

    t0 = perf_counter()
    count_join(rows, query.relation_names, query.joins)  # full-count reference cost
    full_s = max(perf_counter() - t0, 1e-9)

    upper = pessimistic_bound(query)
    t1 = perf_counter()
    cert = aggregate_lower_certificate(
        rows, query.relation_names, query.joins, rate, seeds
    )
    lower_cost_ratio = (perf_counter() - t1) / full_s
    lower_sound = max(cert.sound_lower_bound, 1.0)
    l_hat = max(cert.point_estimate, 1.0)

    records: list[dict[str, Any]] = []
    for name, estimate in _proposals(query).items():
        e = max(estimate, 1.0)
        base_q = q_error(e, y)
        direction = 1 if e >= y else -1

        # --- upper guard: e > tau*B -> fall back to B ---
        fired_u = e > tau * upper
        final_u = upper if fired_u else e
        records.append(
            {
                "query_id": query_id,
                "proposal": name,
                "guard": "upper",
                "estimate": e,
                "true_cardinality": y,
                "direction": direction,
                "tightness_true": log(upper / y),
                "tightness_proxy": log(upper / e),
                "cost_ratio": 0.0,  # bound is precomputed from statistics
                "guard_fired": fired_u,
                "intervention_help": fired_u and q_error(final_u, y) < base_q,
                "proposal_qerror": base_q,
                "final_qerror": q_error(final_u, y),
            }
        )

        # --- lower guard: e < L_hat/tau -> raise to L_hat ---
        fired_l = e < l_hat / tau
        final_l = l_hat if fired_l else e
        records.append(
            {
                "query_id": query_id,
                "proposal": name,
                "guard": "lower",
                "estimate": e,
                "true_cardinality": y,
                "direction": direction,
                "tightness_true": log(y / lower_sound),
                "tightness_proxy": log(e / l_hat) if l_hat > 0 else float("inf"),
                "cost_ratio": lower_cost_ratio,
                "guard_fired": fired_l,
                "intervention_help": fired_l and q_error(final_l, y) < base_q,
                "proposal_qerror": base_q,
                "final_qerror": q_error(final_l, y),
            }
        )
    return records


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/job_light"))
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/utility_table/job_light.csv")
    )
    args = parser.parse_args()

    catalog = load_dataset_catalog(args.data_dir, args.data_dir.name)
    query_pairs = build_job_light_queries(catalog, args.data_dir)
    if args.limit_queries:
        query_pairs = query_pairs[: args.limit_queries]
    logger.info("Loaded %d queries from %s", len(query_pairs), args.data_dir.name)

    rows_out: list[dict[str, Any]] = []
    for index, (query, true_card) in enumerate(query_pairs):
        rows = _load_relation_rows(args.data_dir, query.relation_names)
        rows_out.extend(
            evaluate_query(
                f"q{index}", query, true_card, rows, args.tau, args.rate, args.seeds
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    # Honest summary: guard fire/help rates by direction regime.
    summary: dict[str, dict[str, float]] = {}
    for guard in ("upper", "lower"):
        for regime, sign in (("over", 1), ("under", -1)):
            subset = [
                r for r in rows_out if r["guard"] == guard and r["direction"] == sign
            ]
            fired = [r for r in subset if r["guard_fired"]]
            helped = [r for r in subset if r["intervention_help"]]
            summary[f"{guard}/{regime}"] = {
                "n": len(subset),
                "fire_rate": len(fired) / len(subset) if subset else float("nan"),
                "help_rate": len(helped) / len(subset) if subset else float("nan"),
            }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for key, stats in summary.items():
        logger.info(
            "%-12s n=%4d  fire=%5.1f%%  help=%5.1f%%",
            key,
            stats["n"],
            100 * stats["fire_rate"],
            100 * stats["help_rate"],
        )
    logger.info("Wrote %d rows -> %s (+ %s)", len(rows_out), args.output, summary_path)


if __name__ == "__main__":
    main()
