"""Independence-assumption cardinality estimator (external baseline).

This estimator assumes attribute value independence (AVI) across all joins
and estimates join cardinality as:

    |R_1| * |R_2| * ... * |R_k| / prod(max(D_left, D_right))

where D_left and D_right are the distinct counts of the join attributes
on each side.  This is the standard textbook estimator used by most
traditional query optimizers (e.g., PostgreSQL before learned methods).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.query import Query


@dataclass(frozen=True)
class IndependenceEstimator:
    """Estimate join cardinality using attribute-value independence."""

    def estimate(self, query: Query) -> float:
        """Return the AVI estimate for the given join query."""
        # Product of all relation cardinalities
        numerator = 1.0
        for stats in query.relations.values():
            numerator *= stats.relation.cardinality

        # For each join predicate, divide by max(distinct_left, distinct_right)
        denominator = 1.0
        for join in query.joins:
            left_stats = query.relations[join.left_relation]
            right_stats = query.relations[join.right_relation]
            d_left = left_stats.attribute_distinct_counts.get(
                join.left_attribute, left_stats.relation.cardinality
            )
            d_right = right_stats.attribute_distinct_counts.get(
                join.right_attribute, right_stats.relation.cardinality
            )
            denominator *= max(d_left, d_right)

        return max(1.0, numerator / denominator)
