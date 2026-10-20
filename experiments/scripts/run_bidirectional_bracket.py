"""Validate the two-sided Safe-CEG bracket [L(Q), B(Q)] (RQ3).

For each query in a dataset fixture this measures the *guarantee* directly:
  * the sound lower bound  L(Q) = S          (sample-based, aggregated over seeds),
  * the pessimistic upper  B(Q)              (max-degree bound),
  * the true cardinality   |Q|               (exact count-based hash join),
and reports lower-side validity ``L <= |Q|``, upper-side validity ``|Q| <= B``,
joint bracket validity, and the realised bracket *width* ``B / L`` (how tight
the two-sided containment is). Together with the q-error rescue measured by
``run_lower_certificate``, this answers RQ3 (the bidirectional guarantee).

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.run_bidirectional_bracket \
        --data-dir data/job_light --output results/bracket/bracket.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

from experiments.scripts.run_real_data import (
    _load_relation_rows,
    build_job_light_queries,
)
from src.core.ceg import pessimistic_bound
from src.estimators.lower_certificate import aggregate_lower_certificate
from src.utils.data_loader import load_dataset_catalog

if TYPE_CHECKING:
    from src.core.query import Query

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Relative tolerance for the upper-bound validity check: the max-degree bound is
# evaluated as exp(sum of log weights) and can land ~1e-14 below an exactly-tight
# integer bound. 1e-9 absorbs that FP noise while catching any real violation
# (which would exceed the bound by orders of magnitude, not machine epsilon).
_REL_TOL = 1e-9


def _rate_summary(
    query_pairs: list[tuple[Query, float]],
    data_dir: Path,
    rate: float,
    seeds: list[int],
) -> dict[str, float]:
    """Compute bracket metrics for one sample rate over all queries."""
    lower_valid = upper_valid = bracket_valid = 0
    widths: list[float] = []
    total = 0
    for query, true_card in query_pairs:
        rows = _load_relation_rows(data_dir, query.relation_names)
        cert = aggregate_lower_certificate(
            rows, query.relation_names, query.joins, rate, seeds
        )
        lower = cert.sound_lower_bound
        upper = pessimistic_bound(query)
        # Relative tolerance absorbs log-space exp() rounding on tight bounds
        # (the bound is computed as exp(sum of logs), which can fall ~1e-14 below
        # an exactly-tight integer bound -- a numerical artifact, not a violation).
        upper_tol = upper * (1.0 + _REL_TOL)
        total += 1
        if lower <= true_card:
            lower_valid += 1
        if true_card <= upper_tol:
            upper_valid += 1
        if lower <= true_card <= upper_tol:
            bracket_valid += 1
        widths.append(upper / lower if lower > 0 else float("inf"))
    finite = [w for w in widths if w != float("inf")]
    return {
        "lower_validity": lower_valid / total,
        "upper_validity": upper_valid / total,
        "bracket_validity": bracket_valid / total,
        "bracket_width_median": median(finite) if finite else float("inf"),
        "bracket_width_mean": sum(finite) / len(finite) if finite else float("inf"),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/job_light"))
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--output", type=Path, default=Path("results/bracket/bracket.json")
    )
    args = parser.parse_args()

    catalog = load_dataset_catalog(args.data_dir, args.data_dir.name)
    query_pairs = build_job_light_queries(catalog, args.data_dir)
    logger.info("Loaded %d queries from %s", len(query_pairs), args.data_dir.name)

    per_rate: dict[str, dict[str, float]] = {}
    for rate in args.rates:
        summary = _rate_summary(query_pairs, args.data_dir, rate, args.seeds)
        per_rate[str(rate)] = summary
        logger.info(
            "rate=%.2f  bracket valid %.1f%% (lower %.1f%%, upper %.1f%%)  "
            "width median %.1f",
            rate,
            100 * summary["bracket_validity"],
            100 * summary["lower_validity"],
            100 * summary["upper_validity"],
            summary["bracket_width_median"],
        )

    results = {
        "experiment": "bidirectional_bracket",
        "dataset": args.data_dir.name,
        "n_queries": len(query_pairs),
        "rates": args.rates,
        "seeds": args.seeds,
        "per_rate": per_rate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
