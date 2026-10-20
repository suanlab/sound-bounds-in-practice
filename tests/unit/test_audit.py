"""Tests of the number audit's safeguards (not of its individual checks)."""

from __future__ import annotations

from pathlib import Path

from experiments.scripts.audit_icde_numbers import (
    is_superseded,
    normalise,
    superseded_entries,
)


def test_withdrawn_and_superseded_files_are_detected() -> None:
    entries = superseded_entries()
    withdrawn = Path("results/pg_injection/certified_plans_summary.json")
    assert is_superseded(withdrawn, entries) is not None
    old_native = Path("results/pg_injection/native_subquery_estimations.csv")
    assert is_superseded(old_native, entries) is not None


def test_current_sources_and_still_read_rows_are_not_flagged() -> None:
    entries = superseded_entries()
    current = Path("results/pg_injection/v2/native_incontext_estimations.csv")
    assert is_superseded(current, entries) is None
    still_read = Path("results/utility_table/real_stats_tau1.csv")
    assert is_superseded(still_read, entries) is None


def test_normalise_drops_comments_and_joins_lines() -> None:
    tex = "on 55.0\\% of\nqueries % 99.9\\% in a comment\n% 12.3 hidden\nnext"
    assert normalise(tex) == "on 55.0\\% of queries next"
    assert "12.3" not in normalise(tex)
