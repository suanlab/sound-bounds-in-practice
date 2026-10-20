"""Tests for the certificate-mechanism analyses (exploratory scripts)."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments.pg_injection.tightness_sweep import policies
from experiments.scripts.certificate_mechanisms import auc
from experiments.scripts.price_of_certification import feasible, oracle_cost

if TYPE_CHECKING:
    from experiments.pg_injection.inject_plans import Key


def test_auc_counts_ties_as_half() -> None:
    assert auc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert auc([0.5, 0.5], [True, False]) == 0.5


def random_query(rng: np.random.Generator) -> dict[str, Any]:
    """Four plans over six sub-joins with a sound random bracket."""
    inc = (rng.random((4, 6)) < 0.6).astype(float)
    inc[:, 5] = 1.0  # a shared final result
    truth = rng.uniform(10, 1000, 6)
    return {
        "inc": inc,
        "truth": truth,
        "upper": truth * rng.uniform(1, 3, 6),
        "floor": np.maximum(truth * rng.uniform(0.2, 1, 6), 1.0),
    }


def test_oracle_cost_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(60):
        q = random_query(rng)
        if not feasible(q, np.ones(6, dtype=bool)):
            continue
        times = rng.uniform(1, 100, 6)
        cost, _, _ = oracle_cost(q, times)
        useful = [i for i in range(6) if q["floor"][i] > 1.0]
        best = min(
            sum(times[list(subset)])
            for r in range(len(useful) + 1)
            for subset in itertools.combinations(useful, r)
            if feasible(q, np.isin(np.arange(6), subset))
        )
        assert abs(cost - best) < 1e-9
        checked += 1
    assert checked > 10


def test_synthetic_guards_are_sound_and_tighten_with_k() -> None:
    keys = [(1, ("a", "b")), (1, ("a", "b", "c"))]
    raw = {keys[0]: 1e6, keys[1]: 2.0}
    truth = {keys[0]: 100.0, keys[1]: 50.0}
    out = policies(keys, raw, truth, real={})
    assert "lpbound_upper" not in out  # no real bound given
    assert out["upper_k2"][keys[0]] == 200.0  # overestimate clamped to 2|S|
    assert out["lower_k2"][keys[1]] == 25.0  # underestimate floored at |S|/2
    assert out["bi_k1"] == truth  # perfect bounds give the truth
    for t in (1.5, 2.0, 8.0, 64.0):  # both guards keep every value within k of |S|
        for k in keys:
            value = out[f"bi_k{t:g}"][k]
            assert max(value / truth[k], truth[k] / value) <= t + 1e-12


def test_joint_certificate_depends_only_on_the_bracket_ratio() -> None:
    from experiments.scripts.plan_certificate_joint import regret_bounds

    rng = np.random.default_rng(3)
    for _ in range(50):
        q = random_query(rng)
        c = rng.uniform(1.5, 4.0)
        tight_upper, _ = regret_bounds(q["inc"], q["floor"], q["upper"] / c)
        raised_floor, _ = regret_bounds(q["inc"], q["floor"] * c, q["upper"])
        assert np.allclose(tight_upper, raised_floor)


def test_funnel_sees_errors_only_on_the_guards_side() -> None:
    from experiments.pg_injection.guard_funnel import seen

    key: Key = (1, ("a", "b"))
    upper: dict[Key, float] = {key: 100.0}
    floor: dict[Key, float] = {key: 10.0}
    assert seen(key, ("upper",), {key: 500.0}, upper, floor)
    assert not seen(key, ("lower",), {key: 500.0}, upper, floor)
    assert seen(key, ("lower",), {key: 2.0}, upper, floor)
    assert not seen(key, ("upper", "lower"), {key: 50.0}, upper, floor)


def test_heldout_paragraph_reports_every_verdict() -> None:
    from experiments.scripts.render_heldout_section import render

    data = {
        "a_posteriori": {
            "n_queries": 150,
            "excluded": {"missing_count": 7, "fewer_than_3": 40, "missing_plan": 0},
            "shares": {
                "cout|none": 0.8,
                "cout|sample": 0.9,
                "cout_minus_final|none": 0.0,
                "cout_minus_final|sample": 0.3,
            },
            "violations": 0,
        },
        "n_sweep_queries": 190,
        "plan_changed_by_k": {"2": 0.3, "16": 0.1},
        "truth_gain": 0.99,
        "share_of_truth_gain_log": {"2": None, "16": None},
        "tests": {
            "C1": "confirmed",
            "C2": "confirmed",
            "C3": "confirmed",
            "C4": "confirmed",
            "C5": "confirmed",
            "C6": "falsified",
            "C7": "not testable",
        },
    }
    text = render(data)
    assert "7 for a sub-plan without an exact count" in text
    assert "missing native plan" not in text  # zero exclusions are not listed
    assert (
        "(C6, at least halving and never rising by more than 5 points: falsified)"
        in text
    )
    assert "C7 was not testable" in text
    assert "Of the 6 testable predictions, 5 were confirmed." in text
