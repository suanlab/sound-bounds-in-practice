"""Unit tests for alpha-CEG estimator and checkpoint utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.core.query import JoinPredicate, Query, RelationStatistics
from src.estimators.alpha_ceg import alpha_ceg_estimate
from src.models.features import extract_features
from src.models.gnn import SafeCEGGNN
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.utils.types import Relation

if TYPE_CHECKING:
    from pathlib import Path


def _build_query() -> Query:
    relations = {
        "A": RelationStatistics(
            relation=Relation(
                name="A", attributes=frozenset(("a", "b")), cardinality=500
            ),
            attribute_distinct_counts={"a": 500, "b": 50},
            attribute_max_frequencies={"a": 1, "b": 15},
        ),
        "B": RelationStatistics(
            relation=Relation(
                name="B", attributes=frozenset(("b", "c")), cardinality=300
            ),
            attribute_distinct_counts={"b": 40, "c": 100},
            attribute_max_frequencies={"b": 10, "c": 20},
        ),
    }
    joins = frozenset({JoinPredicate("A", "b", "B", "b")})
    return Query(relations=relations, joins=joins)


def test_alpha_ceg_interpolates_between_optimistic_and_pessimistic() -> None:
    query = _build_query()

    pure_optimistic = alpha_ceg_estimate(query, per_edge_alpha={})
    all_zero = alpha_ceg_estimate(
        query,
        per_edge_alpha={
            (frozenset(k1), frozenset(k2)): 0.0
            for k1, k2 in [
                (set(), {"A"}),
                (set(), {"B"}),
                ({"A"}, {"A", "B"}),
                ({"B"}, {"A", "B"}),
            ]
        },
    )
    all_one = alpha_ceg_estimate(
        query,
        per_edge_alpha={
            (frozenset(k1), frozenset(k2)): 1.0
            for k1, k2 in [
                (set(), {"A"}),
                (set(), {"B"}),
                ({"A"}, {"A", "B"}),
                ({"B"}, {"A", "B"}),
            ]
        },
    )

    assert pure_optimistic > 0.0
    assert all_zero > 0.0
    assert all_one > 0.0


def test_checkpoint_save_and_load_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    query = _build_query()
    features = extract_features(query)
    model = SafeCEGGNN(
        node_feature_dim=features.node_features.shape[1],
        global_feature_dim=features.global_features.shape[1],
    )

    checkpoint_path = tmp_path / "test_model.pt"
    save_checkpoint(model, checkpoint_path)

    loaded_model = load_checkpoint(
        checkpoint_path,
        node_feature_dim=features.node_features.shape[1],
        global_feature_dim=features.global_features.shape[1],
    )

    original_output = model(features)
    loaded_output = loaded_model(features)
    assert torch.allclose(original_output.logits, loaded_output.logits, atol=1e-6)
