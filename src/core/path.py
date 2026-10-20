"""Path utilities for optimistic and pessimistic CEG traversal."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from statistics import fmean
from typing import TYPE_CHECKING, cast

import networkx as nx

from src.utils.types import AggregatorChoice, PathLengthChoice

if TYPE_CHECKING:
    from src.core.ceg import CEGGraph


@dataclass(frozen=True)
class PathEstimate:
    """A candidate CEG path and its multiplicative estimate."""

    nodes: tuple[frozenset[object], ...]
    estimate: float

    @property
    def hop_count(self) -> int:
        """Return the number of edges in the path."""
        return len(self.nodes) - 1


def enumerate_paths(ceg: CEGGraph) -> tuple[tuple[frozenset[object], ...], ...]:
    """Enumerate all simple source-to-terminal paths in a CEG."""
    paths = nx.all_simple_paths(ceg.graph, ceg.root, ceg.terminal)
    return tuple(
        tuple(cast("frozenset[object]", node) for node in path) for path in paths
    )


def estimate_paths(ceg: CEGGraph) -> tuple[PathEstimate, ...]:
    """Return every source-to-terminal path paired with its estimate."""
    path_estimates: list[PathEstimate] = []
    for path in enumerate_paths(ceg):
        estimate = 1.0
        for source_node, target_node in pairwise(path):
            estimate *= float(ceg.graph.edges[source_node, target_node]["weight"])
        path_estimates.append(PathEstimate(nodes=path, estimate=estimate))
    return tuple(path_estimates)


def select_path_estimates(
    path_estimates: tuple[PathEstimate, ...],
    path_length_choice: PathLengthChoice,
) -> tuple[PathEstimate, ...]:
    """Select path candidates according to the configured hop policy."""
    if not path_estimates:
        raise ValueError("At least one path estimate is required")

    if path_length_choice == PathLengthChoice.ALL_HOPS:
        return path_estimates

    hop_counts = [path_estimate.hop_count for path_estimate in path_estimates]
    if path_length_choice == PathLengthChoice.MIN_HOP:
        selected_hop_count = min(hop_counts)
    elif path_length_choice == PathLengthChoice.MAX_HOP:
        selected_hop_count = max(hop_counts)
    else:
        raise ValueError(f"Unsupported path length choice: {path_length_choice!r}")

    return tuple(
        path_estimate
        for path_estimate in path_estimates
        if path_estimate.hop_count == selected_hop_count
    )


def aggregate_estimates(
    path_estimates: tuple[PathEstimate, ...],
    aggregator_choice: AggregatorChoice,
) -> float:
    """Aggregate path estimates with the configured reducer."""
    if not path_estimates:
        raise ValueError("At least one path estimate is required")

    values = [path_estimate.estimate for path_estimate in path_estimates]
    if aggregator_choice == AggregatorChoice.MIN:
        return min(values)
    if aggregator_choice == AggregatorChoice.AVG:
        return float(fmean(values))
    if aggregator_choice == AggregatorChoice.MAX:
        return max(values)
    raise ValueError(f"Unsupported aggregator choice: {aggregator_choice!r}")


def shortest_path_weight(ceg: CEGGraph) -> float:
    """Return the shortest-path additive weight in a CEG."""
    path = nx.shortest_path(ceg.graph, ceg.root, ceg.terminal, weight="weight")
    total_weight = 0.0
    for source_node, target_node in pairwise(path):
        total_weight += float(ceg.graph.edges[source_node, target_node]["weight"])
    return total_weight
