"""Prediction helpers that map model outputs to Safe-CEG decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from src.core.ceg import build_optimistic_ceg
from src.models.features import extract_features
from src.utils.types import ALL_CEG_CONFIGS, MLPrediction

if TYPE_CHECKING:
    from src.core.query import Query
    from src.models.gnn import SafeCEGGNN


@dataclass(frozen=True)
class PredictorOutput:
    """The decoded ML prediction plus raw logits for analysis."""

    prediction: MLPrediction
    logits: torch.Tensor


class CEGPredictor:
    """Decode GNN outputs into Safe-CEG configuration predictions."""

    def __init__(self, model: SafeCEGGNN) -> None:
        self.model = model

    def predict(self, query: Query) -> PredictorOutput:
        """Predict the best optimistic CEG configuration for one query."""
        features = extract_features(query)
        model_output = self.model(features)
        config_index = int(torch.argmax(model_output.logits, dim=1).item())
        config = ALL_CEG_CONFIGS[config_index]
        optimistic_ceg = build_optimistic_ceg(query)
        # NOTE: per-edge alpha is a non-functional PLACEHOLDER (constant 0.5).
        # The continuous learned-alpha (Alpha-CEG) interpolation is a prototype
        # only -- the model does not predict per-edge alpha. Alpha-CEG is NOT a
        # claimed contribution in the canonical paper; do not present it as one.
        per_edge_alpha = {
            (
                cast("frozenset[str]", source_node),
                cast("frozenset[str]", target_node),
            ): 0.5
            for source_node, target_node in optimistic_ceg.graph.edges
        }
        prediction = MLPrediction(
            path_length=config.path_length,
            aggregator=config.aggregator,
            per_edge_alpha=per_edge_alpha,
            confidence=float(model_output.confidence.item()),
        )
        return PredictorOutput(prediction=prediction, logits=model_output.logits)
