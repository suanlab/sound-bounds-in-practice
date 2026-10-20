"""Lightweight message-passing network for Safe-CEG configuration prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from src.models.features import FeatureBundle


@dataclass(frozen=True)
class GNNOutput:
    """Model outputs needed for configuration prediction and safety scoring."""

    logits: torch.Tensor
    confidence: torch.Tensor
    graph_embedding: torch.Tensor


class SafeCEGGNN(nn.Module):
    """A small message-passing model over CEG nodes and edges."""

    def __init__(
        self,
        node_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_configs: int = 9,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("Number of layers must be positive")

        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.self_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.message_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + global_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_configs)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: FeatureBundle) -> GNNOutput:
        """Encode one CEG instance and predict the best configuration."""
        hidden = torch.relu(self.node_encoder(features.node_features))
        source_index = features.edge_index[0]
        target_index = features.edge_index[1]
        edge_gate = self.edge_gate(features.edge_features)

        for self_layer, message_layer in zip(
            self.self_layers, self.message_layers, strict=True
        ):
            aggregated = torch.zeros_like(hidden)
            messages = message_layer(hidden[source_index]) * edge_gate
            aggregated.index_add_(0, target_index, messages)
            hidden = torch.relu(self_layer(hidden) + aggregated)

        graph_embedding = hidden.mean(dim=0, keepdim=True)
        fused_embedding = self.readout(
            torch.cat((graph_embedding, features.global_features), dim=1)
        )
        logits = self.classifier(fused_embedding)
        confidence = torch.sigmoid(self.confidence_head(fused_embedding)).squeeze(1)
        return GNNOutput(
            logits=logits,
            confidence=confidence,
            graph_embedding=fused_embedding,
        )
