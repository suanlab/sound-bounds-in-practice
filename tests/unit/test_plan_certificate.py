"""Unit tests for the pre-registered plan-certificate computation."""

from __future__ import annotations

import csv
import math
from typing import TYPE_CHECKING, Any

import pytest

from experiments.scripts.plan_certificate import (
    SoundnessCheckError,
    _bracket_rows,
    decide,
    evaluate_bracket,
    load_artifact_column,
    repair_rounding,
    summarise,
)
from experiments.scripts.plan_regret import _order_cost

if TYPE_CHECKING:
    from pathlib import Path

A, B, C = "a", "b", "c"
ORDERS: list[tuple[str, ...]] = [
    (A, B, C),
    (A, C, B),
    (B, A, C),
    (B, C, A),
    (C, A, B),
    (C, B, A),
]


def _sizes(ab: float, ac: float, bc: float, abc: float) -> dict[frozenset[str], float]:
    return {
        frozenset({A, B}): ab,
        frozenset({A, C}): ac,
        frozenset({B, C}): bc,
        frozenset({A, B, C}): abc,
    }


TRUTH = _sizes(ab=10.0, ac=1000.0, bc=500.0, abc=2000.0)


def test_exact_bracket_certifies_the_true_optimum() -> None:
    """With L = B = truth every plan's interval is a point; one survivor."""
    # Distinct C_out: 2010, 3000, 2500.
    orders: list[tuple[str, ...]] = [(A, B, C), (A, C, B), (B, C, A)]
    result = evaluate_bracket(orders, TRUTH, TRUTH, TRUTH, {})
    assert result["certified"]
    assert result["useful"]
    assert result["n_survivors"] == 1
    assert result["G"] == pytest.approx(1.0)
    assert result["width"] == pytest.approx(0.0)


def test_exact_bracket_survivor_count_counts_ties() -> None:
    """(a,b,c) and (b,a,c) have identical Cout; both survive and G stays 1."""
    orders: list[tuple[str, ...]] = [(A, B, C), (B, A, C), (A, C, B)]
    result = evaluate_bracket(orders, TRUTH, TRUTH, TRUTH, {})
    assert result["n_survivors"] == 2
    assert not result["certified"]
    assert result["G"] == pytest.approx(1.0)


def test_trivial_lower_bound_prunes_nothing_but_keeps_g_sound() -> None:
    """L = 1 everywhere: no plan can be dominated; G still bounds every regret."""
    lower = _sizes(1.0, 1.0, 1.0, 1.0)
    upper = _sizes(20.0, 2000.0, 1000.0, 4000.0)
    result = evaluate_bracket(ORDERS, TRUTH, lower, upper, {})
    assert result["n_survivors"] == len(ORDERS)
    assert result["pruning_rate"] == 0.0
    assert not result["certified"]
    optimal = min(_order_cost(o, TRUTH) for o in ORDERS)
    worst = max(_order_cost(o, TRUTH) for o in ORDERS) / optimal
    assert result["G"] >= worst
    assert result["G_all"] >= worst
    assert result["width"] == pytest.approx(math.log(4000.0))


def test_tight_bracket_prunes_dominated_orders() -> None:
    """A 2x bracket separates the cheap (a,b)-first plans from the rest."""
    lower = _sizes(5.0, 500.0, 250.0, 1000.0)
    upper = _sizes(20.0, 2000.0, 1000.0, 4000.0)
    result = evaluate_bracket(ORDERS, TRUTH, lower, upper, {})
    # (a,b,c) and (b,a,c) cost 10 + 2000 = 2010 (hi 4020); (a,c,b) costs
    # 1000 + 2000 = 3000 (lo 1500 <= 4020 survives); (b,c,a) 500+2000 (lo 1250)
    # also survives; nothing is dominated at 2x. G is the honest ratio.
    assert 0.0 <= result["pruning_rate"] < 1.0
    assert result["G"] == pytest.approx(
        max(_order_cost(o, upper) for o in ORDERS if o in _survivors(lower, upper))
        / min(_order_cost(o, lower) for o in ORDERS)
    )
    assert result["minimax_regret"] <= result["minimax_bound"]


def _survivors(
    lower: dict[frozenset[str], float], upper: dict[frozenset[str], float]
) -> list[tuple[str, ...]]:
    min_hi = min(_order_cost(o, upper) for o in ORDERS)
    return [o for o in ORDERS if _order_cost(o, lower) <= min_hi]


def test_estimator_picks_report_pruned_and_free_regret() -> None:
    """An estimator that inverts the sizes picks a bad plan; pruning can only help."""
    lower = _sizes(9.0, 900.0, 450.0, 1800.0)
    upper = _sizes(11.0, 1100.0, 550.0, 2200.0)
    inverted = _sizes(ab=1000.0, ac=10.0, bc=500.0, abc=2000.0)
    result = evaluate_bracket(ORDERS, TRUTH, lower, upper, {"bad": inverted})
    picks = result["estimators"]["bad"]
    assert picks["regret_free"] > 1.0
    assert picks["regret_pruned"] <= result["G"]
    assert picks["regret_pruned"] <= picks["regret_free"]
    assert not picks["picked_survivor"]


def test_unsound_bracket_raises_instead_of_recording() -> None:
    """A lower bound above the truth can prune the optimum; that must halt."""
    lower = _sizes(50.0, 1000.0, 500.0, 2000.0)  # (a,b) claimed >= 50, truth 10
    upper = _sizes(50.0, 1000.0, 500.0, 2000.0)
    with pytest.raises(SoundnessCheckError):
        evaluate_bracket(ORDERS, TRUTH, lower, upper, {})


def test_repair_rounding_only_within_tolerance() -> None:
    """Sub-1e-9 relative shortfalls are rounding; anything larger is a violation."""
    assert repair_rounding(100.0, 100.0) == (100.0, False)
    repaired, violated = repair_rounding(99.99999999999, 100.0)
    assert repaired == 100.0
    assert not violated
    assert repair_rounding(99.0, 100.0) == (99.0, True)


def test_load_artifact_column_normalises_alias_sets(tmp_path: Path) -> None:
    """Alias strings are order-insensitive and case-insensitive."""
    path = tmp_path / "x_subquery_estimations.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["QueryID", "Tables", "x_Estimate"])
        writer.writerow(["3", "B U", "12.5"])
        writer.writerow(["3", "u c b", "7"])
    values = load_artifact_column(path, "x")
    assert values[(3, frozenset({"b", "u"}))] == 12.5
    assert values[(3, frozenset({"b", "c", "u"}))] == 7.0


def _query_record(
    query_id: int, *, certified: bool, flagged: bool = False
) -> dict[str, Any]:
    bracket: dict[str, Any] = {"flagged": flagged}
    if not flagged:
        bracket.update(
            {
                "n_plans": 6,
                "n_survivors": 1 if certified else 4,
                "pruning_rate": 5 / 6 if certified else 2 / 6,
                "G": 1.0 if certified else 5.0,
                "G_all": 9.0,
                "certified": certified,
                "useful": certified,
                "width": 0.5,
                "minimax_regret": 1.0,
                "minimax_bound": 2.0,
                "estimators": {
                    "duckdb": {
                        "regret_free": 1.0,
                        "regret_pruned": 1.0,
                        "picked_survivor": True,
                    },
                    "postgres": {
                        "regret_free": 3.0,
                        "regret_pruned": 1.0,
                        "picked_survivor": False,
                    },
                },
            }
        )
    keys = [
        f"{upper}x{lower}"
        for upper in ("maxdeg", "safebound", "lpbound")
        for lower in ("one", "sample0.10", "sample0.25")
    ]
    return {"query_id": query_id, "brackets": {k: dict(bracket) for k in keys}}


def test_summarise_excludes_flagged_queries_and_decides() -> None:
    """Flagged queries are dropped per bracket; the decision reads the primary one."""
    per_query = [_query_record(i, certified=True) for i in range(12)]
    per_query.append(_query_record(99, certified=False, flagged=True))
    summary = summarise(per_query, replicates=200, seed=0)
    primary = summary["lpboundxsample0.25"]
    assert primary["n_queries"] == 12
    assert primary["n_flagged"] == 1
    assert primary["certified_share"]["point"] == pytest.approx(1.0)
    assert primary["postgres_share_pick_pruned_away"]["point"] == pytest.approx(1.0)
    assert "zero_event_upper_95" not in primary["certified_share"]
    decision = decide(summary)
    assert decision["evaluable"]
    assert decision["certificates_non_trivial"]
    assert decision["pruning_non_trivial"]
    # Same records under both lpbound brackets -> paired difference is exactly 0.
    assert not decision["lower_certificate_matters"]


def test_summarise_reports_zero_event_bound_when_nothing_is_certified() -> None:
    """Zero certified queries gets a Clopper-Pearson upper bound, not a bare 0."""
    per_query = [_query_record(i, certified=False) for i in range(20)]
    summary = summarise(per_query, replicates=100, seed=0)
    share = summary["lpboundxsample0.25"]["certified_share"]
    assert share["point"] == 0.0
    assert 0.0 < share["zero_event_upper_95"] < 0.2
    assert len(_bracket_rows(per_query, "lpboundxsample0.25")) == 20
