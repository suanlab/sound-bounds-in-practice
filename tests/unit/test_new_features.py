"""Tests for features added during review revision."""

from __future__ import annotations

from src.core.ceg import pessimistic_bound, tighter_pessimistic_bound
from src.core.query import JoinPredicate, Query, RelationStatistics
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.estimators.independence_estimator import IndependenceEstimator
from src.training.workload import generate_synthetic_workload
from src.utils.metrics import aggregate_multi_seed, format_mean_std
from src.utils.types import (
    AggregatorChoice,
    CEGConfig,
    FallbackDirection,
    PathLengthChoice,
    Relation,
)


def _simple_query() -> Query:
    attrs_a = frozenset(("a", "b"))
    attrs_b = frozenset(("b", "c"))
    relations = {
        "A": RelationStatistics(
            relation=Relation(name="A", attributes=attrs_a, cardinality=100),
            attribute_distinct_counts={"a": 100, "b": 50},
            attribute_max_frequencies={"a": 1, "b": 5},
        ),
        "B": RelationStatistics(
            relation=Relation(name="B", attributes=attrs_b, cardinality=80),
            attribute_distinct_counts={"b": 40, "c": 80},
            attribute_max_frequencies={"b": 3, "c": 1},
        ),
    }
    joins = frozenset({JoinPredicate("A", "b", "B", "b")})
    return Query(relations=relations, joins=joins)


def test_independence_estimator_produces_positive_estimate() -> None:
    query = _simple_query()
    ind = IndependenceEstimator()
    est = ind.estimate(query)
    assert est > 0.0
    # AVI: |A|*|B| / max(distinct_b_A, distinct_b_B) = 100*80/50 = 160
    assert est == 160.0


def test_ceg_caching_produces_same_results_as_individual() -> None:
    """estimate_all_configs (cached) should match individual estimates."""
    query = _simple_query()
    estimator = BaselineCEGEstimator()
    all_configs = estimator.estimate_all_configs(query)
    for config_name, cached_est in all_configs.items():
        config = CEGConfig(cached_est.config.path_length, cached_est.config.aggregator)
        individual_est = estimator.estimate(query, config)
        diff = abs(cached_est.estimate - individual_est.estimate)
        assert diff < 1e-10, f"{config_name}: diff={diff}"


def test_aggregate_multi_seed_computes_mean_and_std() -> None:
    results = [
        {"mean": 10.0, "fallback_rate": 0.5},
        {"mean": 12.0, "fallback_rate": 0.6},
        {"mean": 11.0, "fallback_rate": 0.4},
    ]
    agg = aggregate_multi_seed(results, ["mean", "fallback_rate"])
    assert "mean" in agg
    assert abs(agg["mean"]["mean"] - 11.0) < 1e-10
    assert agg["mean"]["std"] > 0.0
    assert agg["mean"]["n_seeds"] == 3.0


def test_format_mean_std_omits_small_std() -> None:
    assert format_mean_std(1.23, 0.001) == "1.23"
    assert "±" in format_mean_std(1.23, 0.05)


def test_pessimistic_bound_is_sound_on_executed_joins() -> None:
    """The max-degree (MOLP) bound must upper-bound *executed-join* labels.

    Labels are now computed by actually executing the join over generated data
    (estimator-independent), so the sound max-degree bound must never undercount
    the true cardinality. (The old generator inflated labels above the bound;
    that was a label artifact, not real bound invalidity.)
    """
    for family in ("path", "star", "cycle"):
        wl = generate_synthetic_workload(family, 60, 4, seed=42)
        violations = sum(
            1 for wq in wl if wq.true_cardinality > pessimistic_bound(wq.query)
        )
        assert violations == 0, f"{family}: bound undercounts true ({violations} q)"


def test_safe_estimator_fallback_direction_is_upper() -> None:
    """When ML estimate exceeds threshold, direction should be UPPER."""
    from src.estimators.safe_estimator import SafeCEGEstimator
    from src.utils.types import MLPrediction

    query = _simple_query()
    estimator = SafeCEGEstimator(tau=0.01)  # very tight tau to force fallback
    prediction = MLPrediction(
        path_length=PathLengthChoice.MAX_HOP,
        aggregator=AggregatorChoice.MAX,
        per_edge_alpha={},
        confidence=0.5,
    )
    result = estimator.estimate(query, prediction)
    assert result.used_fallback
    assert result.fallback_direction == FallbackDirection.UPPER


def test_tighter_bound_is_at_most_molp() -> None:
    """Tighter bound should always be <= MOLP bound."""
    for family in ("path", "star", "cycle"):
        wl = generate_synthetic_workload(family, 10, 4, seed=99)
        for wq in wl:
            molp = pessimistic_bound(wq.query)
            tight = tighter_pessimistic_bound(wq.query)
            assert tight <= molp + 1e-9, f"{family}: tight={tight} > molp={molp}"


def test_tighter_bound_improves_on_cycle() -> None:
    """On cycle queries, tighter bound should be meaningfully smaller."""
    wl = generate_synthetic_workload("cycle", 10, 4, seed=99)
    ratios = []
    for wq in wl:
        molp = pessimistic_bound(wq.query)
        tight = tighter_pessimistic_bound(wq.query)
        if molp > 0:
            ratios.append(tight / molp)
    avg_ratio = sum(ratios) / len(ratios)
    # Geometric-mean bound should be at least 5x tighter on cycles
    assert avg_ratio < 0.2, f"Expected <0.2, got {avg_ratio:.4f}"
