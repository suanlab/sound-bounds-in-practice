"""Unit tests for synthetic workload generation and dataset building."""

from __future__ import annotations

from src.training.dataset import build_dataset_from_workload
from src.training.workload import generate_synthetic_workload


def test_generate_synthetic_workload_is_deterministic() -> None:
    workload_a = generate_synthetic_workload(
        family="path",
        query_count=3,
        relation_count=4,
        seed=7,
    )
    workload_b = generate_synthetic_workload(
        family="path",
        query_count=3,
        relation_count=4,
        seed=7,
    )

    assert workload_a == workload_b


def test_generate_cycle_workload_creates_cyclic_queries() -> None:
    workload = generate_synthetic_workload(
        family="cycle",
        query_count=2,
        relation_count=4,
        seed=11,
    )

    assert all(len(workload_query.query.joins) == 4 for workload_query in workload)
    assert all(workload_query.true_cardinality > 0.0 for workload_query in workload)


def test_build_dataset_from_workload_populates_examples() -> None:
    workload = generate_synthetic_workload(
        family="star",
        query_count=4,
        relation_count=4,
        seed=3,
    )

    dataset = build_dataset_from_workload(workload)

    assert len(dataset) == 4
    assert all(
        len(dataset[index].config_estimates) == 9 for index in range(len(dataset))
    )
    assert all(dataset[index].pessimistic_bound > 0.0 for index in range(len(dataset)))
