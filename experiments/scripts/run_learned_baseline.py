"""Direct learned-regression baseline for cardinality estimation.

A fair external-style baseline that is *not* a CEG config selector: a small MLP
regresses log-cardinality directly from fixed-size query/statistics features
(in the spirit of MSCN-style supervised regressors). This isolates the value of
the Safe-CEG interface from "just train a regressor on the same statistics".

Reports test q-error (mean/median) for the learned regressor against the
pessimistic certificate and the CEG oracle on the same synthetic workload.

Usage:
    python -m experiments.scripts.run_learned_baseline \
        --output results/learned_baseline/learned_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
from math import log1p
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from src.core.ceg import pessimistic_bound
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.training.workload import generate_synthetic_workload
from src.utils.metrics import q_error

if TYPE_CHECKING:
    from src.core.query import Query

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def query_feature_vector(query: Query) -> list[float]:
    """Fixed-size feature vector from query statistics (input to the regressor)."""
    relations = sorted(query.relation_names)
    log_cards: list[float] = []
    log_distincts: list[float] = []
    log_maxfreqs: list[float] = []
    for rel in relations:
        stats = query.relation_statistics(rel)
        log_cards.append(log1p(float(stats.relation.cardinality)))
        for attr in stats.relation.attributes:
            log_distincts.append(log1p(float(stats.distinct_count(attr))))
            log_maxfreqs.append(log1p(float(stats.max_frequency(attr))))

    def agg(xs: list[float]) -> list[float]:
        if not xs:
            return [0.0, 0.0, 0.0, 0.0]
        return [sum(xs), sum(xs) / len(xs), min(xs), max(xs)]

    n_rel = float(len(relations))
    n_join = float(len(query.joins))
    cyclicity = 1.0 if n_join >= n_rel else 0.0
    return [
        n_rel,
        n_join,
        cyclicity,
        *agg(log_cards),
        *agg(log_distincts),
        *agg(log_maxfreqs),
    ]


class CardinalityRegressor(nn.Module):
    """Small MLP predicting log-cardinality from query features."""

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out.squeeze(-1)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return 0.0 if not ordered else ordered[len(ordered) // 2]


def _build_workload(seed: int) -> list[tuple[Query, float]]:
    pairs: list[tuple[Query, float]] = []
    for family in ("path", "star", "cycle"):
        for rel_count in (3, 4, 5):
            wl = generate_synthetic_workload(
                family=family,
                query_count=40,
                relation_count=rel_count,
                seed=seed + rel_count,
            )
            pairs.extend((wq.query, wq.true_cardinality) for wq in wl)
    return pairs


def run_single_seed(seed: int, epochs: int) -> dict[str, float]:
    torch.manual_seed(seed)
    pairs = _build_workload(seed)
    feats = torch.tensor(
        [query_feature_vector(q) for q, _ in pairs], dtype=torch.float32
    )
    targets = torch.tensor([log1p(t) for _, t in pairs], dtype=torch.float32)

    # Split FIRST, then fit normalization on the TRAIN partition only -- fitting
    # on all rows would leak the test distribution into preprocessing.
    n = len(pairs)
    split = int(n * 0.7)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    train_idx, test_idx = perm[:split], perm[split:]

    mean = feats[train_idx].mean(dim=0, keepdim=True)
    std = feats[train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
    feats = (feats - mean) / std

    model = CardinalityRegressor(feats.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(feats[train_idx])
        loss = loss_fn(pred, targets[train_idx])
        loss.backward()
        opt.step()

    model.eval()
    estimator = BaselineCEGEstimator()
    learned_qe: list[float] = []
    pes_qe: list[float] = []
    oracle_qe: list[float] = []
    with torch.no_grad():
        preds = model(feats[test_idx])
    for j, idx in enumerate(test_idx.tolist()):
        query, true_card = pairs[idx]
        learned_est = float(torch.expm1(preds[j]).clamp_min(1.0).item())
        learned_qe.append(q_error(learned_est, true_card))
        pes_qe.append(q_error(pessimistic_bound(query), true_card))
        all_est = estimator.estimate_all_configs(query)
        oracle = min(
            (e.estimate for e in all_est.values()), key=lambda v: q_error(v, true_card)
        )
        oracle_qe.append(q_error(oracle, true_card))

    return {
        "learned_mean": sum(learned_qe) / len(learned_qe),
        "learned_median": _median(learned_qe),
        "pessimistic_mean": sum(pes_qe) / len(pes_qe),
        "oracle_mean": sum(oracle_qe) / len(oracle_qe),
        "oracle_median": _median(oracle_qe),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 1042, 2042, 3042, 4042]
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/learned_baseline/learned_baseline.json"),
    )
    args = parser.parse_args()

    per_seed = [run_single_seed(s, args.epochs) for s in args.seeds]

    def avg(key: str) -> float:
        return sum(r[key] for r in per_seed) / len(per_seed)

    agg = {k: avg(k) for k in per_seed[0]}
    results = {
        "experiment": "learned_regression_baseline",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "aggregated": agg,
        "per_seed": per_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    logger.info("Learned-regression baseline (direct MLP), synthetic mixed workload")
    logger.info(
        "  Learned regressor q-error: mean %.1f  median %.2f",
        agg["learned_mean"],
        agg["learned_median"],
    )
    logger.info("  Pessimistic q-error mean: %.1f", agg["pessimistic_mean"])
    logger.info(
        "  CEG oracle q-error: mean %.2f  median %.2f",
        agg["oracle_mean"],
        agg["oracle_median"],
    )
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
