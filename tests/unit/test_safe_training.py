"""Unit tests for graph I/O, safety fallback, and training scaffolding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.core.query import JoinPredicate, Query, RelationStatistics
from src.estimators.safe_estimator import SafeCEGEstimator
from src.models.features import extract_features
from src.models.gnn import SafeCEGGNN
from src.models.predictor import CEGPredictor
from src.training.dataset import SafeCEGDataset, build_training_example
from src.training.evaluate import evaluate_model
from src.training.train import TrainingConfig, train_model
from src.utils.graph_io import dump_query_spec, load_query_spec
from src.utils.types import (
    AggregatorChoice,
    MLPrediction,
    PathLengthChoice,
    Relation,
)

if TYPE_CHECKING:
    from pathlib import Path


def _build_query() -> Query:
    relations = {
        "A": RelationStatistics(
            relation=Relation(
                name="A",
                attributes=frozenset(("a", "b")),
                cardinality=1_000,
            ),
            attribute_distinct_counts={"a": 1_000, "b": 100},
            attribute_max_frequencies={"a": 1, "b": 20},
        ),
        "B": RelationStatistics(
            relation=Relation(
                name="B",
                attributes=frozenset(("b", "c")),
                cardinality=800,
            ),
            attribute_distinct_counts={"b": 80, "c": 200},
            attribute_max_frequencies={"b": 25, "c": 30},
        ),
        "C": RelationStatistics(
            relation=Relation(
                name="C",
                attributes=frozenset(("c", "d")),
                cardinality=600,
            ),
            attribute_distinct_counts={"c": 120, "d": 600},
            attribute_max_frequencies={"c": 40, "d": 1},
        ),
    }
    joins = frozenset(
        {
            JoinPredicate("A", "b", "B", "b"),
            JoinPredicate("B", "c", "C", "c"),
        }
    )
    return Query(relations=relations, joins=joins)


def test_query_spec_round_trip_yaml(tmp_path: Path) -> None:
    query = _build_query()
    path = tmp_path / "query.yaml"

    dump_query_spec(query, path)
    loaded_query = load_query_spec(path)

    assert loaded_query == query


def test_safe_estimator_falls_back_when_prediction_is_unsafe() -> None:
    query = _build_query()
    prediction = MLPrediction(
        path_length=PathLengthChoice.MIN_HOP,
        aggregator=AggregatorChoice.MAX,
        per_edge_alpha={},
        confidence=0.1,
    )

    result = SafeCEGEstimator(tau=0.001).estimate(query, prediction)

    assert result.used_fallback is True
    assert result.estimate == result.pessimistic_bound
    assert result.config_used is None


def test_feature_extraction_and_predictor_smoke() -> None:
    torch.manual_seed(0)
    query = _build_query()
    features = extract_features(query)
    model = SafeCEGGNN(
        node_feature_dim=features.node_features.shape[1],
        global_feature_dim=features.global_features.shape[1],
    )
    predictor = CEGPredictor(model)

    output = predictor.predict(query)

    assert output.logits.shape == (1, 9)
    assert 0.0 <= output.prediction.confidence <= 1.0


def test_build_training_example_populates_all_config_outcomes() -> None:
    query = _build_query()

    example = build_training_example(query, true_cardinality=24_000.0)

    assert len(example.config_estimates) == 9
    assert len(example.config_q_errors) == 9
    assert example.best_config_index >= 0
    assert example.pessimistic_bound > 0.0


def test_training_loop_runs_on_minimal_dataset() -> None:
    torch.manual_seed(0)
    query = _build_query()
    features = extract_features(query)
    model = SafeCEGGNN(
        node_feature_dim=features.node_features.shape[1],
        global_feature_dim=features.global_features.shape[1],
    )
    dataset = SafeCEGDataset([build_training_example(query, true_cardinality=24_000.0)])

    losses = train_model(model, dataset, TrainingConfig(epochs=2, learning_rate=1e-3))

    assert len(losses) == 2
    assert all(loss > 0.0 for loss in losses)


def test_evaluate_model_returns_summary_metrics() -> None:
    torch.manual_seed(0)
    query = _build_query()
    features = extract_features(query)
    model = SafeCEGGNN(
        node_feature_dim=features.node_features.shape[1],
        global_feature_dim=features.global_features.shape[1],
    )
    dataset = SafeCEGDataset([build_training_example(query, true_cardinality=24_000.0)])

    summary = evaluate_model(model, dataset, tau=2.0)

    assert "mean" in summary
    assert "fallback_rate" in summary
    assert "config_accuracy" in summary
    assert "expected_q_error" in summary
    assert "expected_safety_penalty" in summary
    assert "safety_violation_rate" in summary
    assert "ml_safety_violation_rate" in summary
    assert summary["safety_violation_rate"] == 0.0
