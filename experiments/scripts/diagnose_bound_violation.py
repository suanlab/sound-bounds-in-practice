"""Diagnose pessimistic upper-bound violations on a dataset fixture.

The bracket experiment (``run_bidirectional_bracket``) found the max-degree
upper bound ``B(Q)`` violated on a fraction of job_light queries. The catalog
statistics are computed directly from the same CSV data the true cardinality is
counted on (``data_loader.py``: ``max_frequencies = max(value_counter.values())``),
so a violation is *not* a statistics-vs-data mismatch. This script tests the
alternative hypothesis -- that the bound **formula** is unsound on some query
shapes (typically cyclic) -- by, for every query, comparing:

  * ``true``       : exact cardinality (count-based hash join),
  * ``B(Q)``       : the max-degree pessimistic bound,
  * ``cartesian``  : product of relation cardinalities (an *always*-sound bound).

If violators satisfy ``B(Q) < true <= cartesian``, the statistics are
self-consistent and the max-degree *formula* is the culprit; the split by
cyclicity localises where.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.diagnose_bound_violation \
        --data-dir data/job_light --output results/bound_diagnosis/job_light.json
"""

from __future__ import annotations

import argparse
import json
import logging
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.scripts.run_real_data import build_job_light_queries
from src.core.ceg import pessimistic_bound
from src.utils.data_loader import load_dataset_catalog

if TYPE_CHECKING:
    from src.core.query import Query

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _is_cyclic(query: Query) -> bool:
    """A connected query is cyclic when it has at least as many joins as relations."""
    return len(query.joins) >= len(query.relation_names)


def _cartesian(query: Query) -> float:
    """Product of relation cardinalities -- an always-sound upper bound."""
    return float(prod(stats.relation.cardinality for stats in query.relations.values()))


def diagnose(query_pairs: list[tuple[Query, float]]) -> dict[str, Any]:
    """Classify every query and summarise the upper-bound violations."""
    records: list[dict[str, Any]] = []
    for query, true_card in query_pairs:
        bound = pessimistic_bound(query)
        cartesian = _cartesian(query)
        records.append(
            {
                "n_relations": len(query.relation_names),
                "n_joins": len(query.joins),
                "cyclic": _is_cyclic(query),
                "true_cardinality": true_card,
                "pessimistic_bound": bound,
                "cartesian_bound": cartesian,
                "violation": true_card > bound,
                "true_over_bound": true_card / bound if bound > 0 else float("inf"),
                "bound_below_cartesian": bound <= cartesian,
            }
        )

    violators = [r for r in records if r["violation"]]

    def _rate(subset: list[dict[str, Any]]) -> float:
        v = [r for r in subset if r["violation"]]
        return len(v) / len(subset) if subset else float("nan")

    cyclic = [r for r in records if r["cyclic"]]
    acyclic = [r for r in records if not r["cyclic"]]
    return {
        "n_queries": len(records),
        "n_violations": len(violators),
        "violation_rate": len(violators) / len(records) if records else float("nan"),
        "violation_rate_cyclic": _rate(cyclic),
        "violation_rate_acyclic": _rate(acyclic),
        "all_violators_cyclic": all(r["cyclic"] for r in violators),
        "all_violators_below_cartesian": all(
            r["bound_below_cartesian"] for r in violators
        ),
        "max_true_over_bound": max(
            (r["true_over_bound"] for r in violators), default=0.0
        ),
        "violators": sorted(violators, key=lambda r: -r["true_over_bound"])[:20],
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/job_light"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/bound_diagnosis/job_light.json")
    )
    args = parser.parse_args()

    catalog = load_dataset_catalog(args.data_dir, args.data_dir.name)
    query_pairs = build_job_light_queries(catalog, args.data_dir)
    summary = diagnose(query_pairs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(
        "%s: %d/%d violations (%.1f%%)  cyclic %.1f%% vs acyclic %.1f%%\n"
        "  all violators cyclic: %s | all below Cartesian (stats consistent): %s\n"
        "  max true/bound: %.2f -> %s",
        args.data_dir.name,
        summary["n_violations"],
        summary["n_queries"],
        100 * summary["violation_rate"],
        100 * summary["violation_rate_cyclic"],
        100 * summary["violation_rate_acyclic"],
        summary["all_violators_cyclic"],
        summary["all_violators_below_cartesian"],
        summary["max_true_over_bound"],
        args.output,
    )


if __name__ == "__main__":
    main()
