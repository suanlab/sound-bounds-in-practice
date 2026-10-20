"""Minimal training loop for Safe-CEG configuration prediction.

Run as a module to train a selector on a synthetic workload and write a
checkpoint plus a training log::

    python -m src.training.train --config experiments/configs/train_default.yaml

Determinism note: ``set_seeds`` seeds Python, NumPy, and Torch. For bit-exact
reproducibility also export ``PYTHONHASHSEED=0`` (set/frozenset iteration order
affects floating-point accumulation in the bound computation).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import yaml
from torch import optim

from src.models.features import extract_features
from src.training.loss import expected_risk_loss

if TYPE_CHECKING:
    from src.models.gnn import SafeCEGGNN
    from src.training.dataset import SafeCEGDataset
    from src.training.workload import WorkloadQuery

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy (if present), and Torch RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for the minimal Safe-CEG training loop."""

    epochs: int = 50
    learning_rate: float = 1e-3
    safety_lambda: float = 0.5
    tau: float = 2.0
    ce_lambda: float = 0.5


def train_model(
    model: SafeCEGGNN,
    dataset: SafeCEGDataset,
    config: TrainingConfig,
) -> list[float]:
    """Train the model on an in-memory Safe-CEG dataset."""
    if len(dataset) == 0:
        raise ValueError("Dataset must contain at least one example")
    if config.epochs <= 0:
        raise ValueError("Epoch count must be positive")
    if config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []

    for _ in range(config.epochs):
        epoch_loss = 0.0
        for index in range(len(dataset)):
            example = dataset[index]
            features = extract_features(example.query)
            output = model(features)
            oracle_label = torch.tensor([example.best_config_index], dtype=torch.long)
            loss = expected_risk_loss(
                logits=output.logits,
                config_q_errors=example.q_error_tensor().unsqueeze(0),
                config_estimates=example.estimate_tensor().unsqueeze(0),
                pessimistic_bounds=torch.tensor(
                    [example.pessimistic_bound], dtype=torch.float32
                ),
                safety_lambda=config.safety_lambda,
                tau=config.tau,
                oracle_labels=oracle_label,
                ce_lambda=config.ce_lambda,
            )
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()
            epoch_loss += float(loss.item())

        losses.append(epoch_loss / len(dataset))

    return losses


def _build_workload_from_config(
    workload_cfg: dict[str, object], seed: int
) -> list[WorkloadQuery]:
    """Build a synthetic workload from a config block."""
    from src.training.workload import generate_synthetic_workload

    families = cast(
        "list[str]", workload_cfg.get("families", ["path", "star", "cycle"])
    )
    query_count = int(cast("int", workload_cfg.get("query_count", 40)))
    relation_counts = cast("list[int]", workload_cfg.get("relation_counts", [3, 4, 5]))
    workload: list[WorkloadQuery] = []
    for offset, family in enumerate(families):
        for relation_count in relation_counts:
            workload.extend(
                generate_synthetic_workload(
                    family=str(family),
                    query_count=query_count,
                    relation_count=int(relation_count),
                    seed=seed + offset * 1000 + int(relation_count),
                )
            )
    return workload


def main() -> None:
    """Train a Safe-CEG selector from a YAML config and save a checkpoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Train a Safe-CEG selector.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    from src.models.gnn import SafeCEGGNN
    from src.training.checkpoint import save_checkpoint
    from src.training.dataset import build_dataset_from_workload

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seeds(seed)

    workload = _build_workload_from_config(cfg.get("workload", {}), seed)
    dataset = build_dataset_from_workload(workload)
    logger.info("Training on %d synthetic queries (seed=%d)", len(dataset), seed)

    model_cfg = cfg.get("model", {})
    features = extract_features(dataset[0].query)
    model = SafeCEGGNN(
        node_feature_dim=features.node_features.shape[1],
        global_feature_dim=features.global_features.shape[1],
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 3)),
    )

    train_cfg = cfg.get("training", {})
    config = TrainingConfig(
        epochs=int(train_cfg.get("epochs", 30)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        safety_lambda=float(train_cfg.get("safety_lambda", 0.5)),
        tau=float(train_cfg.get("safety_tau", 2.0)),
        ce_lambda=float(train_cfg.get("ce_lambda", 0.5)),
    )
    losses = train_model(model, dataset, config)

    out_cfg = cfg.get("output", {})
    checkpoint_path = Path(out_cfg.get("checkpoint", "results/checkpoints/safe_ceg.pt"))
    save_checkpoint(model, checkpoint_path)
    log_path = Path(out_cfg.get("log", "results/logs/train_log.json"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "n_train": len(dataset),
                "epochs": config.epochs,
                "final_loss": losses[-1],
                "losses": losses,
                "node_feature_dim": features.node_features.shape[1],
                "global_feature_dim": features.global_features.shape[1],
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(
        "Saved checkpoint -> %s  (final loss %.4f); log -> %s",
        checkpoint_path,
        losses[-1],
        log_path,
    )


if __name__ == "__main__":
    main()
