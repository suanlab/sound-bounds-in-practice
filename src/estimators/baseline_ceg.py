"""Baseline optimistic CEG estimators over the 3x3 configuration space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.query import Query

from src.core.ceg import build_optimistic_ceg
from src.core.path import aggregate_estimates, estimate_paths, select_path_estimates
from src.utils.metrics import q_error
from src.utils.types import ALL_CEG_CONFIGS, CEGConfig


@dataclass(frozen=True)
class BaselineEstimate:
    """A baseline CEG estimate and the path candidates used to compute it."""

    estimate: float
    path_count: int
    hop_count: int
    config: CEGConfig


class BaselineCEGEstimator:
    """Evaluate all 9 optimistic CEG configurations for a query."""

    def estimate(self, query: Query, config: CEGConfig) -> BaselineEstimate:
        """Estimate a query cardinality for one optimistic CEG configuration."""
        optimistic_ceg = build_optimistic_ceg(query)
        candidate_paths = estimate_paths(optimistic_ceg)
        selected_paths = select_path_estimates(candidate_paths, config.path_length)
        agg_estimate = aggregate_estimates(selected_paths, config.aggregator)
        hop_count = selected_paths[0].hop_count
        return BaselineEstimate(
            estimate=agg_estimate,
            path_count=len(selected_paths),
            hop_count=hop_count,
            config=config,
        )

    def estimate_all_configs(self, query: Query) -> dict[str, BaselineEstimate]:
        """Evaluate every optimistic CEG configuration for a query.

        Builds the optimistic CEG once and reuses it for all 9
        configurations, avoiding redundant graph construction.
        """
        optimistic_ceg = build_optimistic_ceg(query)
        candidate_paths = estimate_paths(optimistic_ceg)
        results: dict[str, BaselineEstimate] = {}
        for config in ALL_CEG_CONFIGS:
            selected = select_path_estimates(candidate_paths, config.path_length)
            agg_estimate = aggregate_estimates(selected, config.aggregator)
            hop_count = selected[0].hop_count
            results[config.name] = BaselineEstimate(
                estimate=agg_estimate,
                path_count=len(selected),
                hop_count=hop_count,
                config=config,
            )
        return results

    def oracle_best_config(
        self, query: Query, true_cardinality: float
    ) -> BaselineEstimate:
        """Return the baseline configuration with the smallest q-error."""
        if true_cardinality <= 0:
            raise ValueError("True cardinality must be positive")

        all_estimates = self.estimate_all_configs(query)
        best_name, best_estimate = min(
            all_estimates.items(),
            key=lambda item: q_error(item[1].estimate, true_cardinality),
        )
        _ = best_name
        return best_estimate
