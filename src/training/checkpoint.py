"""Model checkpoint save and load utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.models.gnn import SafeCEGGNN

if TYPE_CHECKING:
    from pathlib import Path


def save_checkpoint(model: SafeCEGGNN, path: Path) -> None:
    """Save model weights to a PyTorch checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(
    path: Path,
    node_feature_dim: int,
    global_feature_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    num_configs: int = 9,
) -> SafeCEGGNN:
    """Load a SafeCEGGNN model from a saved checkpoint."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    model = SafeCEGGNN(
        node_feature_dim=node_feature_dim,
        global_feature_dim=global_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_configs=num_configs,
    )
    state_dict = torch.load(path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
