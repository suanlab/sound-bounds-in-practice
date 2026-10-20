"""Unit tests for core CEG construction and baseline estimation."""

from __future__ import annotations

import pytest

from src.core.ceg import build_optimistic_ceg, build_pessimistic_ceg, pessimistic_bound
from src.core.path import aggregate_estimates, estimate_paths, select_path_estimates
from src.core.query import JoinPredicate, Query, RelationStatistics
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.estimators.independence_estimator import IndependenceEstimator
from src.training.workload import generate_synthetic_workload
from src.utils.metrics import q_error, safety_violation_rate, summarize_q_errors
from src.utils.types import (
    ALL_CEG_CONFIGS,
    AggregatorChoice,
    PathLengthChoice,
    Relation,
)


def _build_path_query() -> Query:
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


def test_query_connected_subsets_exclude_disconnected_pairs() -> None:
    query = _build_path_query()

    connected_subsets = query.connected_subsets()

    assert frozenset(("A", "C")) not in connected_subsets
    assert frozenset(("A", "B")) in connected_subsets
    assert frozenset(("B", "C")) in connected_subsets
    assert frozenset(("A", "B", "C")) in connected_subsets


def test_optimistic_ceg_contains_multiple_path_lengths() -> None:
    query = _build_path_query()
    optimistic_ceg = build_optimistic_ceg(query)

    path_estimates = estimate_paths(optimistic_ceg)
    hop_counts = {path_estimate.hop_count for path_estimate in path_estimates}

    assert hop_counts == {2, 3}


def test_every_config_equals_independence_on_a_path_query() -> None:
    """Regression (2026-10-04): multi-relation edges used to drop internal joins.

    With independence selectivities every path multiplies each relation and each
    join exactly once, so all nine configurations must agree with AVI. Before the
    fix, min-hop paths dropped the B-C join and returned up to 200x the truth.
    """
    query = _build_path_query()
    estimator = BaselineCEGEstimator()
    avi = IndependenceEstimator().estimate(query)

    for config in ALL_CEG_CONFIGS:
        assert estimator.estimate(query, config).estimate == pytest.approx(avi)


def test_every_config_equals_independence_on_cyclic_queries() -> None:
    for item in generate_synthetic_workload(
        "cycle", query_count=3, relation_count=4, seed=3
    ):
        query = item.query
        avi = IndependenceEstimator().estimate(query)
        for config in ALL_CEG_CONFIGS:
            estimate = BaselineCEGEstimator().estimate(query, config).estimate
            assert estimate == pytest.approx(avi, rel=1e-9)


def test_optimistic_estimates_never_drop_joins_above_the_bound() -> None:
    """The reviewer's case: |R| = 1000, d = 100 path; AVI = bound = 1e5, not 1e7."""
    relations = {
        name: RelationStatistics(
            relation=Relation(
                name=name, attributes=frozenset(attrs), cardinality=1_000
            ),
            attribute_distinct_counts=dict.fromkeys(attrs, 100),
            attribute_max_frequencies=dict.fromkeys(attrs, 10),
        )
        for name, attrs in (("A", ("x",)), ("B", ("x", "y")), ("C", ("y",)))
    }
    query = Query(
        relations=relations,
        joins=frozenset(
            {JoinPredicate("A", "x", "B", "x"), JoinPredicate("B", "y", "C", "y")}
        ),
    )
    paths = estimate_paths(build_optimistic_ceg(query))
    assert max(path.estimate for path in paths) == pytest.approx(1e5)


def test_path_selection_and_aggregation_helpers_are_consistent() -> None:
    query = _build_path_query()
    optimistic_ceg = build_optimistic_ceg(query)
    path_estimates = estimate_paths(optimistic_ceg)

    max_hop_paths = select_path_estimates(path_estimates, PathLengthChoice.MAX_HOP)
    all_hop_paths = select_path_estimates(path_estimates, PathLengthChoice.ALL_HOPS)

    assert len(max_hop_paths) < len(all_hop_paths)
    assert aggregate_estimates(
        max_hop_paths, AggregatorChoice.MIN
    ) <= aggregate_estimates(max_hop_paths, AggregatorChoice.MAX)


def test_pessimistic_ceg_builds_and_produces_positive_bound() -> None:
    query = _build_path_query()

    pessimistic_ceg = build_pessimistic_ceg(query)
    bound = pessimistic_bound(query)

    assert pessimistic_ceg.root == frozenset()
    assert pessimistic_ceg.terminal == frozenset(("A", "B", "C"))
    assert bound > 0.0


def test_pessimistic_bound_matches_single_relation_cardinality() -> None:
    relation_statistics = RelationStatistics(
        relation=Relation(
            name="R",
            attributes=frozenset(("x", "y")),
            cardinality=42,
        ),
        attribute_distinct_counts={"x": 42, "y": 21},
        attribute_max_frequencies={"x": 1, "y": 2},
    )
    query = Query(
        relations={"R": relation_statistics},
        joins=frozenset(),
    )

    assert pessimistic_bound(query) == pytest.approx(42.0)


def test_pessimistic_bound_is_tighter_than_product_when_joins_have_low_frequency() -> (
    None
):
    query = _build_path_query()
    relation_product = 1.0
    for relation_name in query.relation_names:
        relation_product *= query.relation_statistics(
            relation_name
        ).relation.cardinality

    bound = pessimistic_bound(query)
    assert bound <= relation_product
    assert bound > 0.0


def test_metrics_cover_common_summary_cases() -> None:
    errors = [q_error(100.0, 80.0), q_error(50.0, 100.0), q_error(120.0, 100.0)]
    summary = summarize_q_errors(errors)
    violation_rate = safety_violation_rate([110.0, 70.0], [100.0, 100.0], tau=1.0)

    assert summary["mean"] >= summary["median"]
    assert summary["p99"] >= summary["p95"]
    assert violation_rate == 0.5
