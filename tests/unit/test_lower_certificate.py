"""Tests for the sample-based sound lower-bound certificate."""

from __future__ import annotations

import random

from src.core.query import JoinPredicate, Query, RelationStatistics
from src.estimators.lower_certificate import (
    LowerCertificate,
    aggregate_lower_certificate,
    sample_lower_certificate,
)
from src.estimators.safe_estimator import SafeCEGEstimator
from src.utils.types import (
    AggregatorChoice,
    FallbackDirection,
    MLPrediction,
    PathLengthChoice,
    Relation,
)

_RELATIONS = frozenset({"A", "B"})
_JOINS = frozenset({JoinPredicate("A", "k", "B", "k")})


def _ab_rows() -> dict[str, list[dict[str, str]]]:
    """A (100 rows, 10 per key) JOIN B (20 rows, 2 per key) -> true join 200."""
    a_rows = [{"k": str(i % 10)} for i in range(100)]
    b_rows = [{"k": str(i % 10)} for i in range(20)]
    return {"A": a_rows, "B": b_rows}


def test_sound_lower_bound_never_exceeds_true() -> None:
    rows = _ab_rows()
    true_card = 200.0
    for rate in (0.1, 0.25, 0.5, 0.75):
        for seed in range(20):
            cert = sample_lower_certificate(
                rows, _RELATIONS, _JOINS, rate, random.Random(seed)
            )
            assert cert.sound_lower_bound <= true_card
            assert cert.point_estimate >= cert.sound_lower_bound


def test_full_rate_recovers_true_exactly() -> None:
    rows = _ab_rows()
    cert = sample_lower_certificate(rows, _RELATIONS, _JOINS, 1.0, random.Random(0))
    assert cert.sound_lower_bound == 200.0
    assert cert.point_estimate == 200.0
    assert cert.sample_rate == 1.0


def test_point_estimate_is_approximately_unbiased() -> None:
    rows = _ab_rows()
    estimates = [
        sample_lower_certificate(
            rows, _RELATIONS, _JOINS, 0.5, random.Random(seed)
        ).point_estimate
        for seed in range(60)
    ]
    mean_estimate = sum(estimates) / len(estimates)
    assert 160.0 <= mean_estimate <= 240.0  # unbiased around true=200


def test_aggregate_reports_tightest_sound_bound() -> None:
    rows = _ab_rows()
    cert = aggregate_lower_certificate(rows, _RELATIONS, _JOINS, 0.5, [1, 2, 3, 4, 5])
    assert cert.sound_lower_bound <= 200.0
    # max of several sound bounds is still sound but tighter than the smallest
    singles = [
        sample_lower_certificate(
            rows, _RELATIONS, _JOINS, 0.5, random.Random(seed)
        ).sound_lower_bound
        for seed in (1, 2, 3, 4, 5)
    ]
    assert cert.sound_lower_bound == max(singles)


def test_invalid_rate_raises() -> None:
    rows = _ab_rows()
    for bad in (0.0, -0.1, 1.5):
        try:
            sample_lower_certificate(rows, _RELATIONS, _JOINS, bad, random.Random(0))
        except ValueError:
            continue
        raise AssertionError(f"rate {bad} should have raised")


def _tiny_query() -> Query:
    relations = {
        "A": RelationStatistics(
            relation=Relation(name="A", attributes=frozenset(("k",)), cardinality=10),
            attribute_distinct_counts={"k": 10},
            attribute_max_frequencies={"k": 1},
        ),
        "B": RelationStatistics(
            relation=Relation(name="B", attributes=frozenset(("k",)), cardinality=10),
            attribute_distinct_counts={"k": 10},
            attribute_max_frequencies={"k": 1},
        ),
    }
    return Query(relations=relations, joins=_JOINS)


def test_safe_estimator_uses_sound_lower_floor() -> None:
    query = _tiny_query()
    prediction = MLPrediction(
        path_length=PathLengthChoice.MIN_HOP,
        aggregator=AggregatorChoice.MIN,
        per_edge_alpha={},
        confidence=0.5,
    )
    cert = LowerCertificate(
        sound_lower_bound=5.0e8, point_estimate=1.0e9, sample_rate=0.5
    )
    result = SafeCEGEstimator(tau=2.0).estimate(
        query, prediction, lower_certificate=cert
    )

    # the optimistic estimate is tiny, so it falls below the sound floor
    assert result.fallback_direction is FallbackDirection.LOWER
    assert result.estimate == 1.0e9
    assert result.optimistic_lower_bound == 1.0e9
    assert result.metadata["sound_lower_bound"] == 5.0e8
    assert result.metadata["bracket"] == (5.0e8, result.pessimistic_bound)


def test_safe_estimator_without_certificate_is_unchanged() -> None:
    query = _tiny_query()
    prediction = MLPrediction(
        path_length=PathLengthChoice.MIN_HOP,
        aggregator=AggregatorChoice.MIN,
        per_edge_alpha={},
        confidence=0.5,
    )
    result = SafeCEGEstimator(tau=2.0).estimate(query, prediction)
    assert result.metadata["sound_lower_bound"] is None
    assert result.metadata["bracket"] is None
