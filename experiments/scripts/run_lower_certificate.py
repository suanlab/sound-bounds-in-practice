"""Sample-based lower-bound certificate for the Safe-CEG interface.

The upper-bound (pessimistic) certificate guards against overestimation but
leaves *underestimation* unguarded. On skewed real data (JOB-medium) the
optimistic CEG configurations systematically underestimate, so the upper-only
interface cannot help (Section 5.5). This experiment prototypes the missing
lower certificate.

Key idea: a join over a uniform *sub-sample* of one relation (joined with the
full others) is a subset of the true join, so its size is a sound lower bound
on the true cardinality; dividing by the sampling rate gives an unbiased
point estimate L_hat that serves as a lower floor. Flooring an optimistic
estimate with L_hat rescues the underestimation regime, completing the
bidirectional guarantee.

Usage:
    python -m experiments.scripts.run_lower_certificate \
        --data-dir data/job_medium \
        --output results/lower_certificate/lower_cert.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from experiments.scripts.run_real_data import (
    _load_relation_rows,
    build_job_light_queries,
)
from src.core.ceg import pessimistic_bound
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.estimators.lower_certificate import sample_lower_certificate
from src.utils.data_loader import load_dataset_catalog
from src.utils.metrics import q_error

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return 0.0 if n == 0 else ordered[n // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/job_medium"))
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lower_certificate/lower_cert.json"),
    )
    args = parser.parse_args()

    catalog = load_dataset_catalog(args.data_dir, args.data_dir.name)
    query_pairs = build_job_light_queries(catalog, args.data_dir)
    estimator = BaselineCEGEstimator()
    logger.info("Loaded %d queries from %s", len(query_pairs), args.data_dir.name)

    per_rate: dict[str, dict[str, float]] = {}
    for rate in args.rates:
        oracle_qe: list[float] = []
        floored_qe: list[float] = []
        lhat_qe: list[float] = []
        sound_valid = 0
        total = 0
        for query, true_card in query_pairs:
            rows = _load_relation_rows(args.data_dir, query.relation_names)
            all_est = estimator.estimate_all_configs(query)
            oracle_est = min(
                (e.estimate for e in all_est.values()),
                key=lambda v: q_error(v, true_card),
            )
            # Average the (seeded) lower certificate over seeds.
            ests: list[float] = []
            for seed in args.seeds:
                rng = random.Random(seed)
                cert = sample_lower_certificate(
                    rows, query.relation_names, query.joins, rate, rng
                )
                ests.append(cert.point_estimate)
                if cert.sound_lower_bound <= true_card:
                    sound_valid += 1
                total += 1
            l_hat = sum(ests) / len(ests)
            # Lower floor: never report below the certificate.
            floored = max(oracle_est, l_hat)
            oracle_qe.append(q_error(oracle_est, true_card))
            floored_qe.append(q_error(floored, true_card))
            lhat_qe.append(q_error(l_hat, true_card))
        per_rate[str(rate)] = {
            "oracle_qerr_mean": sum(oracle_qe) / len(oracle_qe),
            "oracle_qerr_median": _median(oracle_qe),
            "floored_qerr_mean": sum(floored_qe) / len(floored_qe),
            "floored_qerr_median": _median(floored_qe),
            "lhat_qerr_mean": sum(lhat_qe) / len(lhat_qe),
            "lhat_qerr_median": _median(lhat_qe),
            "sound_lb_validity": sound_valid / total if total else 0.0,
        }
        logger.info(
            "rate=%.2f  oracle median q-err %.1f -> floored %.1f  "
            "(L_hat median q-err %.2f, sound-LB valid %.1f%%)",
            rate,
            per_rate[str(rate)]["oracle_qerr_median"],
            per_rate[str(rate)]["floored_qerr_median"],
            per_rate[str(rate)]["lhat_qerr_median"],
            100 * per_rate[str(rate)]["sound_lb_validity"],
        )

    # Pessimistic-only reference (upper certificate as estimator).
    pes_qe = [q_error(pessimistic_bound(q), t) for q, t in query_pairs]
    results = {
        "experiment": "sample_based_lower_certificate",
        "dataset": args.data_dir.name,
        "n_queries": len(query_pairs),
        "rates": args.rates,
        "seeds": args.seeds,
        "pessimistic_qerr_mean": sum(pes_qe) / len(pes_qe),
        "pessimistic_qerr_median": _median(pes_qe),
        "per_rate": per_rate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
