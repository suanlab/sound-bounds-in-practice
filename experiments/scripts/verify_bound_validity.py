from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean, median, stdev

from src.core.ceg import pessimistic_bound
from src.training.workload import WorkloadQuery, generate_synthetic_workload

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def verify_bound_validity(
    queries_per_family: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    all_queries: list[WorkloadQuery] = []
    # Deterministic per-family seed offset (avoid non-reproducible hash()).
    family_offset = {"path": 0, "star": 1, "cycle": 2}
    for family in ("path", "star", "cycle"):
        for rel_count in range(2, 8):
            all_queries.extend(
                generate_synthetic_workload(
                    family=family,
                    query_count=queries_per_family,
                    relation_count=rel_count,
                    seed=seed + rel_count * 100 + family_offset[family] * 10000,
                )
            )

    violations: list[tuple[int, float, float, float]] = []
    ratios: list[float] = []
    # Per-family tallies: family -> [tested, violations].
    family_tested: dict[str, int] = {}
    family_violations: dict[str, int] = {}

    for i, wq in enumerate(all_queries):
        bound = pessimistic_bound(wq.query)
        true_card = wq.true_cardinality

        if bound <= 0 or true_card <= 0:
            continue

        ratio = true_card / bound
        ratios.append(ratio)
        family_tested[wq.family] = family_tested.get(wq.family, 0) + 1

        if ratio > 1.0:
            violations.append((i, ratio, true_card, bound))
            family_violations[wq.family] = family_violations.get(wq.family, 0) + 1

    violation_rate = len(violations) / len(ratios) if ratios else 0.0

    per_family = {
        family: {
            "tested": family_tested[family],
            "violations": family_violations.get(family, 0),
            "valid_rate": 1.0
            - family_violations.get(family, 0) / family_tested[family],
        }
        for family in sorted(family_tested)
    }

    stats: dict[str, object] = {
        "total_queries": len(all_queries),
        "num_queries_tested": len(ratios),
        "num_violations": len(violations),
        "violation_rate": violation_rate,
        "valid_rate": 1.0 - violation_rate,
        "per_family": per_family,
        "mean_ratio": mean(ratios) if ratios else 0.0,
        "median_ratio": median(ratios) if ratios else 0.0,
        "max_ratio": max(ratios) if ratios else 0.0,
        "stdev_ratio": stdev(ratios) if len(ratios) > 1 else 0.0,
        "violations": [
            {
                "query_index": v[0],
                "ratio": round(v[1], 4),
                "true_cardinality": round(v[2], 2),
                "bound": round(v[3], 2),
            }
            for v in violations
        ],
    }

    logger.info(f"Queries tested: {stats['num_queries_tested']}")
    logger.info(
        f"Violations: {stats['num_violations']} ({stats['violation_rate']:.2%})"
    )
    logger.info(
        f"Ratio true/bound: mean={stats['mean_ratio']:.4f}, "
        f"median={stats['median_ratio']:.4f}, max={stats['max_ratio']:.4f}"
    )

    if violations:
        logger.warning(f"VIOLATIONS DETECTED ({len(violations)}):")
        for v in violations[:20]:
            logger.warning(
                f"  Query {v[0]}: ratio={v[1]:.4f}, true={v[2]:.1f}, bound={v[3]:.1f}"
            )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify CEG_M bound validity: pessimistic_bound >= true_cardinality"
    )
    parser.add_argument("--queries-per-family", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="results/bound_verification/bound_validity.json"
    )
    args = parser.parse_args()

    stats = verify_bound_validity(
        queries_per_family=args.queries_per_family,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
