"""Tests for the ICDE 2027 analysis pipeline (joint certificates, injection helpers)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments.pg_injection.inject_plans import _strip, hint_text
from experiments.pg_injection.subset_counts import count_sql
from experiments.scripts.plan_certificate_joint import (
    incidence,
    joint_certificate,
    left_deep_orders,
)
from experiments.scripts.selective_sampling import joint_bounds

# a 4-relation chain a-b-c-d: connected sub-joins of size >= 2
SUBSETS = [
    frozenset("ab"),
    frozenset("bc"),
    frozenset("cd"),
    frozenset("abc"),
    frozenset("bcd"),
    frozenset("abcd"),
]
INDEX = {s: i for i, s in enumerate(SUBSETS)}


def _brute_force_bounds(
    inc: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """max over every corner s of the box of C(p,s) / min_p' C(p',s)."""
    best = np.zeros(inc.shape[0])
    for corner in itertools.product(*zip(lower, upper, strict=True)):
        cost = inc @ np.array(corner)
        best = np.maximum(best, cost / cost.min())
    return best


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("include_final", [True, False])
def test_joint_bound_equals_brute_force_over_the_box(
    seed: int, include_final: bool
) -> None:
    rng = np.random.default_rng(seed)
    orders = left_deep_orders(list("abcd"), INDEX)
    inc = incidence(orders, INDEX, include_final=include_final)
    truth = rng.uniform(1, 1e6, len(SUBSETS))
    lower = truth * rng.uniform(0.01, 1.0, len(SUBSETS))
    upper = truth * rng.uniform(1.0, 100.0, len(SUBSETS))
    bounds, _ = joint_bounds(inc, lower, upper)
    np.testing.assert_allclose(
        bounds, _brute_force_bounds(inc, lower, upper), rtol=1e-9
    )


@pytest.mark.parametrize("seed", range(20))
def test_joint_certificate_is_sound_for_any_truth_in_the_box(seed: int) -> None:
    rng = np.random.default_rng(seed)
    orders = left_deep_orders(list("abcd"), INDEX)
    inc = incidence(orders, INDEX, include_final=True)
    truth = rng.uniform(1, 1e6, len(SUBSETS))
    lower = truth * rng.uniform(0.05, 1.0, len(SUBSETS))
    upper = truth * rng.uniform(1.0, 50.0, len(SUBSETS))
    # raises AssertionError if the optimum is pruned or a true regret exceeds its bound
    result = joint_certificate(inc, lower, upper, truth)
    assert result["joint_bound_best"] >= 1.0
    assert result["G_decoupled"] >= result["joint_bound_best"] - 1e-9


def test_first_two_relations_commute_under_cout() -> None:
    orders = left_deep_orders(list("abcd"), INDEX)
    inc = incidence(orders, INDEX, include_final=True)
    rows = {o: inc[i] for i, o in enumerate(orders)}
    for order in orders:
        twin = (order[1], order[0], *order[2:])
        if twin in rows:
            np.testing.assert_array_equal(rows[order], rows[twin])


def test_hint_text_lists_one_rows_hint_per_subplan() -> None:
    text = hint_text({(1, ("b", "u")): 79851.4, (1, ("b", "p", "u")): 0.2})
    assert text == "/*+ Rows(b p u #1) Rows(b u #79851) */ "
    assert hint_text(None) == ""


def test_plan_fingerprint_ignores_estimates_and_costs() -> None:
    plan_a = {"Node Type": "Hash Join", "Plan Rows": 10, "Total Cost": 5.0, "Plans": []}
    plan_b = {"Node Type": "Hash Join", "Plan Rows": 99, "Total Cost": 7.0, "Plans": []}
    assert _strip(plan_a) == _strip(plan_b)


def test_count_sql_samples_only_the_named_alias() -> None:
    sql = "SELECT * FROM BADGES AS B, USERS AS U WHERE B.USERID=U.ID"
    sampled = count_sql(sql, ("u", 0.1, 0))
    assert sampled.startswith(
        "SELECT COUNT(*) FROM BADGES AS B, USERS AS U TABLESAMPLE"
    )
    assert "BERNOULLI (10) REPEATABLE (0)" in sampled
    assert sampled.count("TABLESAMPLE") == 1
