"""Alpha-CEG estimator with continuous per-edge interpolation."""

from __future__ import annotations

from itertools import pairwise
from math import log
from typing import TYPE_CHECKING

from src.core.ceg import build_optimistic_ceg, build_pessimistic_ceg
from src.core.path import enumerate_paths

if TYPE_CHECKING:
    from src.core.query import Query


def alpha_ceg_estimate(
    query: Query,
    per_edge_alpha: dict[tuple[frozenset[str], frozenset[str]], float],
) -> float:
    """Estimate cardinality by interpolating optimistic and pessimistic weights.

    For each edge in the optimistic CEG, the effective log-weight is:
        (1 - alpha) * log(optimistic_weight) + alpha * log(pessimistic_weight)
    where alpha in [0, 1] is learned per edge.

    Alpha = 0 yields the pure optimistic estimate.
    Alpha = 1 yields the pure pessimistic estimate.
    """
    optimistic_ceg = build_optimistic_ceg(query)
    pessimistic_ceg = build_pessimistic_ceg(query)

    pessimistic_weights = _pessimistic_weight_lookup(pessimistic_ceg, query)

    paths = enumerate_paths(optimistic_ceg)
    if not paths:
        raise ValueError("No paths found in optimistic CEG")

    path_estimates: list[float] = []
    for path in paths:
        log_estimate = 0.0
        for source_node, target_node in pairwise(path):
            edge_key = (
                frozenset(str(n) for n in source_node),
                frozenset(str(n) for n in target_node),
            )
            alpha = per_edge_alpha.get(edge_key, 0.5)
            alpha = max(0.0, min(1.0, alpha))

            opt_weight = float(
                optimistic_ceg.graph.edges[source_node, target_node]["weight"]
            )
            log_opt = log(max(opt_weight, 1e-30))

            added = frozenset(str(n) for n in target_node) - frozenset(
                str(n) for n in source_node
            )
            log_pes = sum(
                pessimistic_weights.get(relation_name, log_opt)
                for relation_name in added
            )

            log_estimate += (1.0 - alpha) * log_opt + alpha * log_pes

        from math import exp

        path_estimates.append(exp(log_estimate))

    return sum(path_estimates) / len(path_estimates)


def _pessimistic_weight_lookup(
    pessimistic_ceg: object,
    query: Query,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for relation_name in query.relation_names:
        stats = query.relation_statistics(relation_name)
        weights[relation_name] = log(float(stats.relation.cardinality))
    return weights
