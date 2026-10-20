"""Plan cost model for measuring cardinality estimation impact.

WARNING: this is a HAND-CRAFTED SURROGATE, not a real query-optimizer plan cost.
It maps a q-error to a synthetic "plan cost ratio" via an exponent model; it does
NOT enumerate plans or call a real optimizer. Use it only as an illustrative
sensitivity surrogate and report it as such -- do not present surrogate plan-cost
numbers as real plan latency/cost. A real optimizer (e.g., PostgreSQL plan
injection) is needed to defend any plan-quality claim.

The surrogate assumes nested-loop joins with cost proportional to the product of
input cardinalities (cost-based-optimization intuition, cf. Leis et al., VLDB 2015),
to reflect that a q-error of X can cascade into a larger plan-cost ratio.
"""

from __future__ import annotations

from math import log2


def plan_cost_ratio(
    estimated_cardinality: float,
    true_cardinality: float,
    n_relations: int,
) -> float:
    """Compute the plan cost ratio induced by a cardinality estimate.

    Uses a simplified model where the cost of a join plan is proportional
    to the sum of intermediate result sizes. When the optimizer chooses
    a join order based on estimated cardinalities, the actual execution
    cost can differ dramatically from the estimated cost.

    For a query with k relations, the worst-case cost amplification
    from a q-error of η is approximately η^(k-1) because the error
    compounds across k-1 join steps.

    Returns the ratio actual_cost / optimal_cost, where 1.0 means
    the estimate led to the optimal plan.
    """
    if estimated_cardinality <= 0 or true_cardinality <= 0:
        return 1.0

    q_err = max(
        estimated_cardinality / true_cardinality,
        true_cardinality / estimated_cardinality,
    )

    # Simplified cascading model: each join step amplifies the error.
    # With k relations, there are k-1 join steps.
    # Conservative model: cost ratio ≈ q_error^(log2(k)) for bushy plans
    # or q_error^(k-1) for left-deep plans. We use the geometric mean.
    if n_relations <= 1:
        return 1.0
    exponent = (n_relations - 1 + log2(max(2, n_relations))) / 2.0
    return float(q_err**exponent)


def subplan_cost(
    estimated_sizes: list[float],
    true_sizes: list[float],
) -> tuple[float, float]:
    """Compute estimated and true total plan costs from intermediate sizes.

    Args:
        estimated_sizes: Estimated cardinalities of intermediate results.
        true_sizes: True cardinalities of intermediate results.

    Returns:
        (estimated_total_cost, true_total_cost) where cost = sum of sizes.
    """
    return (sum(estimated_sizes), sum(true_sizes))
