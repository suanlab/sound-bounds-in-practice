"""Tests for the checks answering the simulated review (positivity, censoring)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from experiments.scripts.plan_certificate_joint import regret_bounds
from experiments.scripts.review_checks import censoring_case

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# Three plans; each costs one pair join plus the shared final result (C_out).
INC = np.array(
    [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
)
TRUTH = np.array([1.0, 3.0, 3.0, 0.0])  # an empty final result
UPPER = np.array([100.0, 3.0, 3.0, 0.0])


def test_raw_bracket_is_sound_with_an_empty_result() -> None:
    bound, _ = regret_bounds(INC, TRUTH, UPPER)
    cost = INC @ TRUTH
    assert (cost / cost.min() <= bound + 1e-9).all()


def test_flooring_at_one_is_unsound_when_a_sub_join_is_empty() -> None:
    """The review's counterexample: max(size, 1) certifies a plan of regret 3.

    The certificates floor every size at 1, which is sound only on non-empty
    sub-joins; ``review_checks.positivity`` verifies that every size used is >= 1.
    """
    bound, _ = regret_bounds(INC, np.maximum(TRUTH, 1.0), np.maximum(UPPER, 1.0))
    assert np.allclose(bound, [25.25, 2.0, 2.0])
    cost = INC @ TRUTH
    assert cost[1] / cost.min() == 3.0  # certified within 2x, true regret 3


def test_censoring_cases() -> None:
    record = {
        "policy_plan": {"raw": "a", "same": "a", "other": "b", "fast": "c"},
        "plan_censored": {"a": True, "b": True, "c": False},
    }
    assert censoring_case(record, "raw", "same") == "both_same_plan"
    assert censoring_case(record, "raw", "other") == "both_distinct_plans"
    assert censoring_case(record, "raw", "fast") == "base_only"
    assert censoring_case(record, "fast", "raw") == "policy_only"


def _record(qid: int, ms: dict[str, float], censored: set[str]) -> dict[str, object]:
    """An injection record whose policies each have their own plan."""
    return {
        "query_id": qid,
        "policy_plan": {p: f"plan_{p}" for p in ms},
        "plan_ms": {f"plan_{p}": t for p, t in ms.items()},
        "plan_censored": {f"plan_{p}": p in censored for p in ms},
    }


def test_censoring_counts_cases_and_leaves_out_unknown_directions() -> None:
    from experiments.scripts.review_checks import censoring

    records = [
        _record(1, {"x_raw": 200.0, "x_bi": 100.0}, set()),
        _record(2, {"x_raw": 300.0, "x_bi": 300.0}, {"x_raw", "x_bi"}),
    ]
    out = censoring(records, [("x_raw", "x_bi")])["x_raw->x_bi"]
    assert out["both_distinct_plans"] == [2]
    assert np.isclose(out["geo_all"], np.sqrt(2.0))
    assert np.isclose(out["geo_without_distinct_both"], 2.0)


def test_headroom_exceedance_uses_a_ten_percent_band() -> None:
    from experiments.scripts.review_checks import headroom_exceedance

    records = [
        _record(1, {"x_raw": 100.0, "truth": 50.0, "x_bi": 40.0}, set()),  # 1.25
        _record(2, {"x_raw": 100.0, "truth": 50.0, "x_bi": 47.0}, set()),  # 1.06
        _record(3, {"x_raw": 100.0, "truth": 120.0, "x_bi": 100.0}, set()),
    ]
    out = headroom_exceedance(records)["x"]
    assert out["bi_beats_truth"] == 2  # queries 1 and 3
    assert out["truth_slower_than_raw"] == 1


def test_coverage_counts_only_fully_compared_queries(tmp_path: Path) -> None:
    from experiments.scripts.review_checks import coverage

    rows = [
        {
            "query_id": 1,
            "aliases": ["a", "b"],
            "exact_censored": False,
            "truth_artifact": 5.0,
            "exact_pg": 5.0,
        },
        {
            "query_id": 2,
            "aliases": ["a", "b"],
            "exact_censored": False,
            "truth_artifact": 7.0,
            "exact_pg": 7.0,
        },
        {
            "query_id": 2,
            "aliases": ["a", "b", "c"],
            "exact_censored": True,
            "truth_artifact": 9.0,
            "exact_pg": None,
        },
    ]
    path = tmp_path / "counts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = coverage(path)
    assert (out["n_queries"], out["n_all_compared"]) == (2, 1)
    assert np.isclose(out["upper95_share_complete_queries"], 0.95)


def test_heldout_ranges_bracket_the_excluded_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.scripts import review_checks

    path = tmp_path / "confirmatory.json"
    data = {"n_queries": 10, "excluded": {"missing_count": 2}, "shares": {"c": 0.5}}
    path.write_text(json.dumps({"a_posteriori": data}))
    monkeypatch.setattr(review_checks, "HELDOUT", path)
    out = review_checks.heldout_ranges()["c"]
    assert np.isclose(out["if_none_certified"], 5 / 12)
    assert np.isclose(out["if_all_certified"], 7 / 12)
