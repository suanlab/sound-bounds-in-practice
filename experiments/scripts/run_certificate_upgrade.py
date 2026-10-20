"""Certificate-upgrade experiment: MOLP vs. geometric-mean pessimistic bound.

Quantifies the "certificate-upgradeable" claim: replacing the MOLP certificate
with the tighter geometric-mean degree bound improves fallback estimate quality
on cycle queries without changing the selector or the interface.

For each cycle query we compute the true cardinality, the MOLP bound, and the
geometric-mean bound, then report the fallback q-error (q-error of the bound
treated as the returned estimate) for each, plus the bound tightness ratio
geo/MOLP. Outputs JSON consumed by the paper's certificate-upgrade table.

Usage:
    python -m experiments.scripts.run_certificate_upgrade \
        --output results/certificate_upgrade/cert_upgrade.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.core.ceg import pessimistic_bound, tighter_pessimistic_bound
from src.training.workload import generate_synthetic_workload
from src.utils.metrics import q_error, summarize_q_errors

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--relation-counts",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6],
        help="Cycle query sizes to evaluate.",
    )
    parser.add_argument(
        "--queries-per-size", type=int, default=20, help="Queries per size."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 1042, 2042, 3042, 4042]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/certificate_upgrade/cert_upgrade.json"),
    )
    args = parser.parse_args()

    molp_qerrors: list[float] = []
    geo_qerrors: list[float] = []
    tightness_ratios: list[float] = []  # geo_bound / molp_bound (<=1 means tighter)
    geo_strictly_tighter = 0
    molp_violations = 0  # MOLP bound < true cardinality (should be 0; MOLP is sound)
    geo_violations = 0  # geo bound < true cardinality (geo is NOT a sound certificate)
    total = 0

    per_size: dict[int, dict[str, float]] = {}

    for relation_count in args.relation_counts:
        size_molp: list[float] = []
        size_geo: list[float] = []
        size_ratio: list[float] = []
        for seed in args.seeds:
            workload = generate_synthetic_workload(
                family="cycle",
                query_count=args.queries_per_size,
                relation_count=relation_count,
                seed=seed,
            )
            for wq in workload:
                true_card = wq.true_cardinality
                molp = pessimistic_bound(wq.query)
                geo = tighter_pessimistic_bound(wq.query)
                molp_qe = q_error(molp, true_card)
                geo_qe = q_error(geo, true_card)
                molp_qerrors.append(molp_qe)
                geo_qerrors.append(geo_qe)
                size_molp.append(molp_qe)
                size_geo.append(geo_qe)
                ratio = geo / molp if molp > 0 else 1.0
                tightness_ratios.append(ratio)
                size_ratio.append(ratio)
                if geo < molp:
                    geo_strictly_tighter += 1
                if molp < true_card:
                    molp_violations += 1
                if geo < true_card:
                    geo_violations += 1
                total += 1
        per_size[relation_count] = {
            "n": float(len(size_molp)),
            "molp_qerr_mean": summarize_q_errors(size_molp)["mean"],
            "molp_qerr_median": _median(size_molp),
            "geo_qerr_mean": summarize_q_errors(size_geo)["mean"],
            "geo_qerr_median": _median(size_geo),
            "mean_tightness_ratio": sum(size_ratio) / len(size_ratio),
        }

    molp_mean = summarize_q_errors(molp_qerrors)["mean"]
    geo_mean = summarize_q_errors(geo_qerrors)["mean"]
    molp_median = _median(molp_qerrors)
    geo_median = _median(geo_qerrors)
    mean_ratio = (
        sum(tightness_ratios) / len(tightness_ratios) if tightness_ratios else 1.0
    )

    results = {
        "experiment": "certificate_upgrade_molp_vs_geomean",
        "family": "cycle",
        "relation_counts": args.relation_counts,
        "queries_per_size": args.queries_per_size,
        "seeds": args.seeds,
        "n_queries": total,
        "aggregated": {
            "molp_fallback_qerr_mean": molp_mean,
            "molp_fallback_qerr_median": molp_median,
            "geo_fallback_qerr_mean": geo_mean,
            "geo_fallback_qerr_median": geo_median,
            "qerr_mean_improvement_factor": (
                molp_mean / geo_mean if geo_mean > 0 else float("inf")
            ),
            "qerr_median_improvement_factor": (
                molp_median / geo_median if geo_median > 0 else float("inf")
            ),
            "mean_bound_tightness_ratio_geo_over_molp": mean_ratio,
            "bound_tightness_factor": (
                1.0 / mean_ratio if mean_ratio > 0 else float("inf")
            ),
            "geo_strictly_tighter_fraction": geo_strictly_tighter / total
            if total
            else 0.0,
            "molp_bound_violation_rate": molp_violations / total if total else 0.0,
            "geo_bound_violation_rate": geo_violations / total if total else 0.0,
        },
        "per_size": {str(k): v for k, v in per_size.items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    agg = results["aggregated"]
    logger.info("Certificate upgrade (MOLP vs geometric-mean), cycle queries")
    logger.info("  N = %d queries", total)
    logger.info(
        "  MOLP fallback q-error:  mean %.1f  median %.1f",
        agg["molp_fallback_qerr_mean"],
        agg["molp_fallback_qerr_median"],
    )
    logger.info(
        "  Geo  fallback q-error:  mean %.1f  median %.1f",
        agg["geo_fallback_qerr_mean"],
        agg["geo_fallback_qerr_median"],
    )
    logger.info(
        "  Improvement: mean %.1fx  median %.1fx",
        agg["qerr_mean_improvement_factor"],
        agg["qerr_median_improvement_factor"],
    )
    logger.info(
        "  Bound tightness: geo %.1fx tighter (ratio %.3f), strictly tighter %.0f%%",
        agg["bound_tightness_factor"],
        agg["mean_bound_tightness_ratio_geo_over_molp"],
        100 * agg["geo_strictly_tighter_fraction"],
    )
    logger.info(
        "  VALIDITY: MOLP violates true %.1f%%, geo violates true %.1f%% "
        "(geo is NOT a sound certificate)",
        100 * agg["molp_bound_violation_rate"],
        100 * agg["geo_bound_violation_rate"],
    )
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
