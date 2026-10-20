"""Shared type definitions for Safe-CEG."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

RelationName: TypeAlias = str
AttributeName: TypeAlias = str


class PathLengthChoice(Enum):
    MIN_HOP = "min_hop"
    MAX_HOP = "max_hop"
    ALL_HOPS = "all_hops"


class AggregatorChoice(Enum):
    MIN = "min"
    AVG = "avg"
    MAX = "max"


@dataclass(frozen=True)
class CEGConfig:
    """One of the 9 CEG estimator configurations (path_length x aggregator)."""

    path_length: PathLengthChoice
    aggregator: AggregatorChoice

    @property
    def name(self) -> str:
        return f"{self.path_length.value}-{self.aggregator.value}"


ALL_CEG_CONFIGS: list[CEGConfig] = [
    CEGConfig(pl, agg) for pl in PathLengthChoice for agg in AggregatorChoice
]


@dataclass(frozen=True)
class Relation:
    name: RelationName
    attributes: frozenset[AttributeName]
    cardinality: int


@dataclass
class CEGNode:
    relation_subset: frozenset[RelationName]
    attributes: frozenset[AttributeName]


@dataclass
class CEGEdge:
    source: frozenset[RelationName]
    target: frozenset[RelationName]
    weight: float
    added_relation: RelationName


@dataclass
class MLPrediction:
    """GNN output: predicted optimal CEG configuration + per-edge alpha."""

    path_length: PathLengthChoice
    aggregator: AggregatorChoice
    per_edge_alpha: dict[tuple[frozenset[RelationName], frozenset[RelationName]], float]
    confidence: float


class FallbackDirection(Enum):
    """Direction of safety fallback, if any."""

    NONE = "none"
    UPPER = "upper"
    LOWER = "lower"


@dataclass
class SafeEstimationResult:
    """Safe-CEG estimation result with bidirectional fallback tracking."""

    estimate: float
    ml_estimate: float
    pessimistic_bound: float
    optimistic_lower_bound: float
    used_fallback: bool
    fallback_direction: FallbackDirection
    config_used: CEGConfig | None
    metadata: dict[str, object] = field(default_factory=dict)
