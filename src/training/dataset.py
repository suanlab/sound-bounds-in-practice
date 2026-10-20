"""Dataset utilities for Safe-CEG training examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from src.core.ceg import pessimistic_bound
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.utils.metrics import q_error
from src.utils.types import ALL_CEG_CONFIGS

if TYPE_CHECKING:
    from src.core.query import Query
    from src.training.workload import WorkloadQuery
    from src.utils.types import CEGConfig


@dataclass(frozen=True)
class TrainingExample:
    """One supervised training example for configuration prediction."""

    query: Query
    true_cardinality: float
    best_config: CEGConfig
    config_estimates: tuple[float, ...]
    config_q_errors: tuple[float, ...]
    pessimistic_bound: float

    @property
    def best_config_index(self) -> int:
        """Return the index of the oracle-best config in ALL_CEG_CONFIGS."""
        return ALL_CEG_CONFIGS.index(self.best_config)

    def estimate_tensor(self) -> torch.Tensor:
        """Return per-config estimates as a float tensor."""
        return torch.tensor(self.config_estimates, dtype=torch.float32)

    def q_error_tensor(self) -> torch.Tensor:
        """Return per-config q-errors as a float tensor."""
        return torch.tensor(self.config_q_errors, dtype=torch.float32)


class SafeCEGDataset(Dataset[TrainingExample]):
    """A simple in-memory dataset for Safe-CEG experiments."""

    def __init__(self, examples: list[TrainingExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        """Return the number of available training examples."""
        return len(self._examples)

    def __getitem__(self, index: int) -> TrainingExample:
        """Return one training example by index."""
        return self._examples[index]


def build_training_example(
    query: Query,
    true_cardinality: float,
    estimator: BaselineCEGEstimator | None = None,
) -> TrainingExample:
    """Construct a training example with outcomes for all 9 configs."""
    if true_cardinality <= 0:
        raise ValueError("True cardinality must be positive")

    baseline_estimator = estimator or BaselineCEGEstimator()
    estimates_by_name = baseline_estimator.estimate_all_configs(query)
    config_estimates = tuple(
        estimates_by_name[config.name].estimate for config in ALL_CEG_CONFIGS
    )
    config_q_errors = tuple(
        q_error(estimate, true_cardinality) for estimate in config_estimates
    )
    best_config = min(
        ALL_CEG_CONFIGS,
        key=lambda config: q_error(
            estimates_by_name[config.name].estimate, true_cardinality
        ),
    )
    return TrainingExample(
        query=query,
        true_cardinality=true_cardinality,
        best_config=best_config,
        config_estimates=config_estimates,
        config_q_errors=config_q_errors,
        pessimistic_bound=pessimistic_bound(query),
    )


def build_dataset_from_workload(
    workload: list[WorkloadQuery],
    estimator: BaselineCEGEstimator | None = None,
) -> SafeCEGDataset:
    """Convert synthetic workload queries into a training dataset."""
    baseline_estimator = estimator or BaselineCEGEstimator()
    examples = [
        build_training_example(
            workload_query.query,
            workload_query.true_cardinality,
            estimator=baseline_estimator,
        )
        for workload_query in workload
    ]
    return SafeCEGDataset(examples)
