"""C1 pilot: pre-registered utility-predictability test (W4 gate, run early).

Implements ``docs/c1_preregistration.md`` (frozen 2026-07-11) exactly:

  * features: direction, tightness *proxy*, relative cost (3 variables, plus
    intercept; no interactions),
  * model: logistic regression (deterministic full-batch gradient descent,
    zero init, standardized features),
  * evaluation: leave-one-dataset-out (LODO) across the utility tables,
    AUROC on the held-out dataset; pooled AUROC over all held-out predictions,
  * gates: pooled >= 0.80 AND per-dataset >= 0.70 (supported);
    < 0.70 (rejected -> failure-atlas PIVOT).

Honesty notes emitted with the result:
  * ``cost_ratio`` is empty for statistics-based certificates (LpBound /
    SafeBound) in the external tables; per pre-registration these bounds are
    precomputed-summary lookups, so cost is imputed as 0.0 and the variable is
    non-discriminative until sampling certificates join the table.
  * We also report intervention *precision* (help rate among fired) --- if it
    is 1.0, the prediction problem degenerates to predicting *fire*, which is
    a deterministic threshold on the tightness proxy; that finding matters as
    much as the AUROC.

Usage::

    python -m experiments.scripts.c1_pilot \
        --tables results/utility_table/real_joblight.csv \
                 results/utility_table/real_stats.csv \
        --output results/c1_pilot/c1_pilot.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GATE_POOLED = 0.80
GATE_PER_DATASET = 0.70


def load_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one utility CSV into (features[n,3], labels[n]) per pre-registration."""
    features: list[list[float]] = []
    labels: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            proxy = float(row["tightness_proxy"])
            if not np.isfinite(proxy):
                continue
            cost_raw = row.get("cost_ratio", "")
            cost = float(cost_raw) if cost_raw not in ("", "nan") else 0.0
            features.append([float(row["direction"]), proxy, cost])
            labels.append(1.0 if row["intervention_help"] == "True" else 0.0)
    return np.asarray(features), np.asarray(labels)


def fit_logistic(
    x: np.ndarray, y: np.ndarray, lr: float = 0.1, iters: int = 2000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic full-batch GD logistic fit; returns (weights, mean, std)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0.0] = 1.0  # constant feature (e.g. cost): keep, zero effect
    xs = (x - mean) / std
    design = np.hstack([np.ones((xs.shape[0], 1)), xs])
    weights = np.zeros(design.shape[1])
    for _ in range(iters):
        preds = 1.0 / (1.0 + np.exp(-design @ weights))
        weights -= lr * (design.T @ (preds - y)) / len(y)
    return weights, mean, std


def predict(
    x: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Predicted P(help) under the fitted standardization."""
    xs = (x - mean) / std
    design = np.hstack([np.ones((xs.shape[0], 1)), xs])
    return 1.0 / (1.0 + np.exp(-design @ weights))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney); NaN when one class is absent."""
    pos = scores[labels == 1.0]
    neg = scores[labels == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    rank_sum = ranks[: len(pos)].sum()
    u = rank_sum - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def precision_among_fired(path: Path) -> tuple[int, int]:
    """(#helped, #fired) --- degeneracy diagnostic."""
    fired = helped = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["guard_fired"] == "True":
                fired += 1
                if row["intervention_help"] == "True":
                    helped += 1
    return helped, fired


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables",
        type=Path,
        nargs="+",
        default=[
            Path("results/utility_table/real_joblight.csv"),
            Path("results/utility_table/real_stats.csv"),
        ],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/c1_pilot/c1_pilot.json")
    )
    args = parser.parse_args()

    datasets = {path.stem: load_table(path) for path in args.tables}
    for name, (_x, y) in datasets.items():
        logger.info(
            "%s: n=%d  help-rate=%.3f", name, len(y), float(y.mean()) if len(y) else 0.0
        )

    per_dataset: dict[str, float] = {}
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    for held_out in datasets:
        train_x = np.vstack([x for n, (x, _) in datasets.items() if n != held_out])
        train_y = np.concatenate([y for n, (_, y) in datasets.items() if n != held_out])
        test_x, test_y = datasets[held_out]
        weights, mean, std = fit_logistic(train_x, train_y)
        scores = predict(test_x, weights, mean, std)
        per_dataset[held_out] = auroc(test_y, scores)
        pooled_labels.append(test_y)
        pooled_scores.append(scores)
        logger.info("LODO held-out=%s  AUROC=%.4f", held_out, per_dataset[held_out])

    pooled = auroc(np.concatenate(pooled_labels), np.concatenate(pooled_scores))
    finite = [v for v in per_dataset.values() if np.isfinite(v)]
    supported = pooled >= GATE_POOLED and all(v >= GATE_PER_DATASET for v in finite)
    rejected = any(v < GATE_PER_DATASET for v in finite)

    degeneracy = {}
    for path in args.tables:
        helped, fired = precision_among_fired(path)
        degeneracy[path.stem] = {
            "fired": fired,
            "helped": helped,
            "precision": helped / fired if fired else float("nan"),
        }

    result: dict[str, Any] = {
        "preregistration": "docs/c1_preregistration.md (frozen 2026-07-11)",
        "model": "logistic(direction, tightness_proxy, cost); LODO",
        "per_dataset_auroc": per_dataset,
        "pooled_auroc": pooled,
        "gates": {"pooled": GATE_POOLED, "per_dataset": GATE_PER_DATASET},
        "verdict": (
            "C1 SUPPORTED"
            if supported
            else ("C1 REJECTED -> failure-atlas PIVOT" if rejected else "C1 WEAKENED")
        ),
        "intervention_precision_among_fired": degeneracy,
        "caveats": [
            "cost_ratio imputed 0.0 for statistics-based certificates "
            "(non-discriminative until sampling certificates join)",
            "if precision==1.0 everywhere, predicting help reduces to "
            "predicting fire (deterministic in tightness_proxy at fixed tau)",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info(
        "pooled AUROC=%.4f -> %s (gates: pooled>=%.2f, per-dataset>=%.2f) -> %s",
        pooled,
        result["verdict"],
        GATE_POOLED,
        GATE_PER_DATASET,
        args.output,
    )


if __name__ == "__main__":
    main()
