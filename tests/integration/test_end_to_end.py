"""Integration test: raw CSV data through full Safe-CEG pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from src.core.ceg import pessimistic_bound
from src.core.query import JoinPredicate, Query
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.models.features import extract_features
from src.models.gnn import SafeCEGGNN
from src.training.dataset import SafeCEGDataset, build_training_example
from src.training.evaluate import evaluate_model
from src.training.train import TrainingConfig, train_model
from src.utils.data_loader import catalog_to_query_statistics, load_dataset_catalog

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_dataset"


def test_load_dataset_catalog_reads_csv_fixtures() -> None:
    catalog = load_dataset_catalog(FIXTURES_DIR, "sample")

    assert "actors" in catalog.relation_names()
    assert "movies" in catalog.relation_names()
    assert "ratings" in catalog.relation_names()

    actors_stats = catalog.get_statistics("actors")
    assert actors_stats.relation.cardinality == 8
    assert actors_stats.distinct_count("actor_id") == 8
    assert actors_stats.distinct_count("movie_id") == 3


def test_full_pipeline_from_csv_to_evaluation() -> None:
    torch.manual_seed(0)
    catalog = load_dataset_catalog(FIXTURES_DIR, "sample")
    relation_names = frozenset(("actors", "movies"))
    stats = catalog_to_query_statistics(catalog, relation_names)

    query = Query(
        relations=stats,
        joins=frozenset(
            {
                JoinPredicate(
                    left_relation="actors",
                    left_attribute="movie_id",
                    right_relation="movies",
                    right_attribute="movie_id",
                )
            }
        ),
    )

    estimator = BaselineCEGEstimator()
    all_estimates = estimator.estimate_all_configs(query)
    assert len(all_estimates) == 9

    bound = pessimistic_bound(query)
    assert bound > 0.0
    assert bound <= 8.0 * 3.0

    true_cardinality = 8.0
    example = build_training_example(query, true_cardinality, estimator)
    assert len(example.config_estimates) == 9

    features = extract_features(query)
    model = SafeCEGGNN(
        node_feature_dim=features.node_features.shape[1],
        global_feature_dim=features.global_features.shape[1],
    )

    dataset = SafeCEGDataset([example])
    losses = train_model(model, dataset, TrainingConfig(epochs=3, learning_rate=1e-3))
    assert len(losses) == 3

    summary = evaluate_model(model, dataset, tau=2.0)
    assert "mean" in summary
    assert "expected_q_error" in summary
    assert summary["safety_violation_rate"] == 0.0
    assert summary["ml_safety_violation_rate"] >= 0.0
