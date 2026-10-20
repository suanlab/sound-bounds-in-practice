"""Cardinality Estimation Graph construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import exp, log
from typing import TYPE_CHECKING

import networkx as nx

from src.core.query import Query, iter_supersets

if TYPE_CHECKING:
    from src.utils.types import RelationName


CEGNodeKey = frozenset[object]


@dataclass(frozen=True)
class CEGGraph:
    """A constructed cardinality estimation graph."""

    graph: nx.DiGraph[object]
    root: CEGNodeKey
    terminal: CEGNodeKey
    kind: str


def build_optimistic_ceg(query: Query) -> CEGGraph:
    """Build the optimistic CEG over connected relation subsets."""
    graph: nx.DiGraph[object] = nx.DiGraph()
    root: frozenset[RelationName] = frozenset()
    terminal = query.relation_names
    connected_subsets = query.connected_subsets()

    graph.add_node(root)
    for subset in connected_subsets:
        graph.add_node(subset)

    for target_subset in connected_subsets:
        if len(target_subset) != 1:
            continue
        graph.add_edge(
            root,
            target_subset,
            weight=_optimistic_edge_weight(query, root, target_subset),
            added_relations=tuple(sorted(target_subset)),
        )

    all_subsets = (root, *connected_subsets)
    for source_subset in connected_subsets:
        for target_subset in iter_supersets(all_subsets, source_subset):
            if not _is_valid_optimistic_extension(query, source_subset, target_subset):
                continue
            graph.add_edge(
                source_subset,
                target_subset,
                weight=_optimistic_edge_weight(query, source_subset, target_subset),
                added_relations=tuple(sorted(target_subset - source_subset)),
            )

    return CEGGraph(graph=graph, root=root, terminal=terminal, kind="optimistic")


def build_pessimistic_ceg(query: Query) -> CEGGraph:
    """Build a conservative pessimistic CEG over relation subsets."""
    graph: nx.DiGraph[object] = nx.DiGraph()
    root: frozenset[RelationName] = frozenset()
    terminal = query.relation_names

    relation_subsets = query.connected_subsets()
    graph.add_node(root)
    for subset in relation_subsets:
        graph.add_node(subset)

    for source_subset in (root, *relation_subsets):
        for relation_name in sorted(query.relation_names):
            target_subset = source_subset | frozenset((relation_name,))
            if target_subset == source_subset:
                continue
            if source_subset and not query.is_connected_subset(target_subset):
                continue
            graph.add_edge(
                source_subset,
                target_subset,
                weight=_pessimistic_edge_weight(query, relation_name, source_subset),
                relation_name=relation_name,
            )

    return CEGGraph(graph=graph, root=root, terminal=terminal, kind="pessimistic")


def pessimistic_bound(query: Query) -> float:
    """Return the pessimistic upper bound induced by the shortest CEG_M path."""
    pessimistic_ceg = build_pessimistic_ceg(query)
    shortest_path = nx.shortest_path(
        pessimistic_ceg.graph,
        source=pessimistic_ceg.root,
        target=pessimistic_ceg.terminal,
        weight="weight",
    )
    total_log_weight = 0.0
    for source_node, target_node in pairwise(shortest_path):
        total_log_weight += float(
            pessimistic_ceg.graph.edges[source_node, target_node]["weight"]
        )
    return float(exp(total_log_weight))


def _is_valid_optimistic_extension(
    query: Query,
    source_subset: frozenset[RelationName],
    target_subset: frozenset[RelationName],
) -> bool:
    if not source_subset < target_subset:
        return False
    if not query.is_connected_subset(target_subset):
        return False
    if not source_subset:
        return len(target_subset) == 1
    return bool(query.boundary_joins(source_subset, target_subset))


def _optimistic_edge_weight(
    query: Query,
    source_subset: frozenset[RelationName],
    target_subset: frozenset[RelationName],
) -> float:
    added_relations = target_subset - source_subset
    if not added_relations:
        raise ValueError("Optimistic edge must add at least one relation")

    relation_factor = 1.0
    for relation_name in added_relations:
        relation_factor *= float(
            query.relation_statistics(relation_name).relation.cardinality
        )

    if not source_subset:
        return relation_factor

    if not query.boundary_joins(source_subset, target_subset):
        raise ValueError("Non-root optimistic edges require at least one boundary join")

    # Every join the edge newly covers: those crossing from the source *and* those
    # among the added relations. Multiplying only the crossing joins (as before
    # 2026-10-04) dropped the internal ones and inflated multi-relation edges.
    selectivity_factor = 1.0
    new_joins = query.induced_joins(target_subset) - query.induced_joins(source_subset)
    for join in new_joins:
        left_statistics = query.relation_statistics(join.left_relation)
        right_statistics = query.relation_statistics(join.right_relation)
        max_distinct = max(
            left_statistics.distinct_count(join.left_attribute),
            right_statistics.distinct_count(join.right_attribute),
        )
        selectivity_factor *= 1.0 / float(max_distinct)

    return relation_factor * selectivity_factor


def _pessimistic_edge_weight(
    query: Query,
    relation_name: RelationName,
    source_subset: frozenset[RelationName],
) -> float:
    """Compute a pessimistic (upper-bound) edge weight.

    For the first relation added (from root), the weight is
    log(cardinality) — the trivial product bound.

    For subsequent relations, the weight is the minimum over all
    boundary joins of log(max_frequency) on the joining attribute
    of the newly added relation. This is the degree-based bound:
    the join result can grow by at most max_freq(join_attr) per
    existing tuple.
    """
    relation_statistics = query.relation_statistics(relation_name)
    cardinality_weight = log(float(relation_statistics.relation.cardinality))

    if not source_subset:
        return cardinality_weight

    target_subset = source_subset | frozenset((relation_name,))
    boundary_joins = query.boundary_joins(source_subset, target_subset)
    if not boundary_joins:
        return cardinality_weight

    degree_weights: list[float] = []
    for join in boundary_joins:
        if join.left_relation == relation_name:
            freq = relation_statistics.max_frequency(join.left_attribute)
        else:
            freq = relation_statistics.max_frequency(join.right_attribute)
        degree_weights.append(log(float(freq)))

    return min(cardinality_weight, min(degree_weights))


def tighter_pessimistic_bound(query: Query) -> float:
    """Return the tighter of the MOLP and geometric-mean bounds.

    Computes both the standard max-frequency bound and the geometric-mean
    bound, then returns the minimum, so the result is always at least as tight
    as MOLP.

    WARNING: unlike the MOLP/AGM max-degree bound, the geometric-mean variant is
    a *heuristic* tightening and is NOT proven to be a sound upper bound on the
    true cardinality. Experiments measure its empirical validity
    (``geo_bound_violation_rate``); it should not be relied upon as a safety
    certificate. The sound tighter-certificate upgrade path is LpBound.
    """
    molp = pessimistic_bound(query)
    geo_ceg = _build_l2_pessimistic_ceg(query)
    shortest_path = nx.shortest_path(
        geo_ceg.graph,
        source=geo_ceg.root,
        target=geo_ceg.terminal,
        weight="weight",
    )
    total_log_weight = 0.0
    for source_node, target_node in pairwise(shortest_path):
        total_log_weight += float(
            geo_ceg.graph.edges[source_node, target_node]["weight"]
        )
    geo_bound = float(exp(total_log_weight))
    return min(molp, geo_bound)


def _build_l2_pessimistic_ceg(query: Query) -> CEGGraph:
    """Build a pessimistic CEG using L2-norm edge weights."""
    graph: nx.DiGraph[object] = nx.DiGraph()
    root: frozenset[RelationName] = frozenset()
    terminal = query.relation_names

    relation_subsets = query.connected_subsets()
    graph.add_node(root)
    for subset in relation_subsets:
        graph.add_node(subset)

    for source_subset in (root, *relation_subsets):
        for relation_name in sorted(query.relation_names):
            target_subset = source_subset | frozenset((relation_name,))
            if target_subset == source_subset:
                continue
            if source_subset and not query.is_connected_subset(target_subset):
                continue
            graph.add_edge(
                source_subset,
                target_subset,
                weight=_l2_pessimistic_edge_weight(query, relation_name, source_subset),
                relation_name=relation_name,
            )

    return CEGGraph(graph=graph, root=root, terminal=terminal, kind="pessimistic_l2")


def _l2_pessimistic_edge_weight(
    query: Query,
    relation_name: RelationName,
    source_subset: frozenset[RelationName],
) -> float:
    """Compute a tighter pessimistic edge weight.

    Uses the geometric mean of max_frequency and average degree as the
    per-edge weight: ``sqrt(max_freq * avg_degree)``.  This is always
    between ``avg_degree`` and ``max_freq``, giving a tighter (but heuristic)
    bound than pure max_freq when the degree distribution is skewed
    (high max_freq, moderate average).

    NOTE: this is a heuristic, NOT a proven sound upper bound. Replacing
    ``f_max`` with the geometric mean is an intuition (interpolating between
    the worst-case and average extension factor), not a theorem; the resulting
    bound can in principle fall below the true cardinality. Its empirical
    validity is measured (``geo_bound_violation_rate``) rather than guaranteed,
    and it is not used as a safety certificate.
    """
    relation_statistics = query.relation_statistics(relation_name)
    cardinality_weight = log(float(relation_statistics.relation.cardinality))

    if not source_subset:
        return cardinality_weight

    target_subset = source_subset | frozenset((relation_name,))
    boundary_joins = query.boundary_joins(source_subset, target_subset)
    if not boundary_joins:
        return cardinality_weight

    tighter_weights: list[float] = []
    for join in boundary_joins:
        if join.left_relation == relation_name:
            attr = join.left_attribute
        else:
            attr = join.right_attribute
        f_max = float(relation_statistics.max_frequency(attr))
        n = float(relation_statistics.relation.cardinality)
        d = float(relation_statistics.distinct_count(attr))
        f_avg = n / d
        # Geometric mean: tighter than max_freq, looser than avg
        geo_mean = (f_max * f_avg) ** 0.5
        tighter_weights.append(log(max(1.0, geo_mean)))

    return min(cardinality_weight, min(tighter_weights))
