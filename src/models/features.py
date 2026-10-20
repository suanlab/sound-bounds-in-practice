"""Feature extraction for Safe-CEG models."""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import TYPE_CHECKING, cast

import torch

from src.core.ceg import build_optimistic_ceg

if TYPE_CHECKING:
    from src.core.query import Query


@dataclass(frozen=True)
class FeatureBundle:
    """Tensor bundle representing one query-specific CEG instance."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    global_features: torch.Tensor


def extract_features(query: Query) -> FeatureBundle:
    """Extract node, edge, and global features from the optimistic CEG."""
    optimistic_ceg = build_optimistic_ceg(query)
    nodes = list(optimistic_ceg.graph.nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}

    node_features = torch.tensor(
        [
            _node_feature_vector(query, cast("frozenset[object]", node))
            for node in nodes
        ],
        dtype=torch.float32,
    )

    edge_pairs: list[list[int]] = [[], []]
    edge_feature_rows: list[list[float]] = []
    for source_node, target_node, edge_data in optimistic_ceg.graph.edges(data=True):
        typed_source = cast("frozenset[object]", source_node)
        typed_target = cast("frozenset[object]", target_node)
        edge_pairs[0].append(node_to_index[source_node])
        edge_pairs[1].append(node_to_index[target_node])
        edge_feature_rows.append(
            [
                float(log1p(float(edge_data["weight"]))),
                float(len(typed_target) - len(typed_source)),
            ]
        )

    edge_index = torch.tensor(edge_pairs, dtype=torch.long)
    edge_features = torch.tensor(edge_feature_rows, dtype=torch.float32)
    global_features = torch.tensor(
        [_global_feature_vector(query, len(nodes), len(edge_feature_rows))],
        dtype=torch.float32,
    )
    return FeatureBundle(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        global_features=global_features,
    )


def _node_feature_vector(query: Query, node: frozenset[object]) -> list[float]:
    relation_subset = frozenset(str(relation_name) for relation_name in node)
    relation_count = float(len(relation_subset))
    attribute_count = float(len(query.subset_attributes(relation_subset)))
    total_cardinality = float(
        sum(
            query.relation_statistics(relation_name).relation.cardinality
            for relation_name in relation_subset
        )
    )
    induced_join_count = float(len(query.induced_joins(relation_subset)))
    return [
        relation_count,
        attribute_count,
        log1p(total_cardinality),
        induced_join_count,
    ]


def _global_feature_vector(
    query: Query, node_count: int, edge_count: int
) -> list[float]:
    relation_count = float(len(query.relation_names))
    join_count = float(len(query.joins))
    attribute_count = float(len(query.all_attributes))
    cyclicity = 1.0 if len(query.joins) >= len(query.relation_names) else 0.0
    return [
        relation_count,
        join_count,
        attribute_count,
        float(node_count),
        float(edge_count),
        cyclicity,
    ]
