"""Tests of the PostgreSQL harness paths behind the paper's execution numbers.

The database is replaced by small fakes, so these run without PostgreSQL:
the injection policies, plan timing with censoring and crashes, the recovery
of PostgreSQL's in-context join estimates through the benchmark patch's hooks,
plan-shape checks for forced join orders, and the handling of plans whose true
C_out is undefined.
"""

from __future__ import annotations

import math
import re
from typing import Any

import psycopg2
import pytest

from experiments.pg_injection import analyze_injection, native_estimates_incontext
from experiments.pg_injection.inject_plans import (
    Key,
    PlanCrashedError,
    policy_values,
    run_interleaved,
)
from experiments.pg_injection.run_forced_orders import honours, left_deep_order
from experiments.scripts.bootstrap_ci import cluster_bootstrap

# --- injection policies -------------------------------------------------------

K1: Key = (1, ("a", "b"))
K2: Key = (1, ("a", "b", "c"))
EST: dict[str, dict[Key, float]] = {
    "truecardinality": {K1: 100.0, K2: 1_000.0},
    "lpbound": {K1: 200.0, K2: 5_000.0},
    "postgres": {K1: 900.0, K2: 50.0},  # over on K1, under on K2
    "duckdb": {K1: 10.0},  # does not cover K2
}


def test_guards_clamp_on_their_own_side_only() -> None:
    lower: dict[Key, tuple[float | None, float | None]] = {
        K1: (20.0, 200.0),
        K2: (300.0, 3_000.0),
    }
    out = policy_values([K1, K2], EST, lower, tau=1.0, estimators=("postgres",))
    assert out["postgres_upper"] == {K1: 200.0, K2: 50.0}  # only the overestimate
    assert out["postgres_lower"] == {K1: 900.0, K2: 300.0}  # only the underestimate
    assert out["postgres_bi"] == {K1: 200.0, K2: 300.0}
    assert out["postgres_bi_hat"] == {K1: 200.0, K2: 3_000.0}
    assert out["lhat_est"] == {K1: 200.0, K2: 3_000.0}
    assert "postgres_upper_t1" not in out  # only produced when tau != 1


def test_censored_sample_gives_no_floor_and_no_lhat_policy() -> None:
    lower: dict[Key, tuple[float | None, float | None]] = {
        K1: (20.0, 200.0),
        K2: (None, None),
    }
    out = policy_values([K1, K2], EST, lower, tau=1.0, estimators=("postgres",))
    assert out["postgres_lower"] == {K1: 900.0, K2: 50.0}
    assert "lhat_est" not in out


def test_estimator_that_misses_a_subplan_is_skipped() -> None:
    out = policy_values([K1, K2], EST, {}, tau=1.0, estimators=("duckdb",))
    assert not any(name.startswith("duckdb") for name in out)


def values_of(out: dict[str, Any], name: str) -> dict[Any, float]:
    """A policy's injected values, which must exist."""
    found = out[name]
    assert found is not None
    return dict(found)


def test_tau_scales_the_upper_gate() -> None:
    out = policy_values([K1, K2], EST, {}, tau=5.0, estimators=("postgres",))
    assert values_of(out, "postgres_upper")[K1] == 900.0  # 900 <= 5 * 200
    assert values_of(out, "postgres_upper_t1")[K1] == 200.0


# --- timing with censoring and crashes ----------------------------------------


class TimingCursor:
    """Executes named statements; some time out, one may crash the server."""

    def __init__(self, behaviour: dict[str, str]) -> None:
        self.behaviour = behaviour
        self.calls: list[str] = []
        self.connection = self

    def rollback(self) -> None:
        """Connection API used after a cancelled statement."""

    def execute(self, sql: str) -> None:
        if sql.startswith("SET"):
            return
        self.calls.append(sql)
        kind = self.behaviour.get(sql, "ok")
        if kind == "timeout":
            raise psycopg2.errors.QueryCanceled
        if kind == "crash":
            raise psycopg2.OperationalError("server closed the connection")

    def fetchone(self) -> tuple[int]:
        return (1,)


def test_timed_out_plan_is_censored_at_the_timeout_and_not_rerun() -> None:
    cursor = TimingCursor({"slow": "timeout"})
    out = run_interleaved(cursor, {"a": "fast", "b": "slow"}, 300_000, 3)
    assert out["b"] == {"ms": 300_000.0, "censored": True, "reps": 1}
    assert out["a"]["censored"] is False
    assert out["a"]["reps"] == 3
    assert cursor.calls.count("slow") == 1


def test_crash_names_the_plan_that_was_running() -> None:
    cursor = TimingCursor({"boom": "crash"})
    with pytest.raises(PlanCrashedError) as raised:
        run_interleaved(cursor, {"a": "fast", "b": "boom"}, 300_000, 3)
    assert raised.value.digest == "b"


def test_every_round_runs_plans_in_the_same_order() -> None:
    cursor = TimingCursor({})
    run_interleaved(cursor, {"n": "first", "p": "second"}, 1_000, 2, always_reps=True)
    assert cursor.calls == ["first", "second", "first", "second"]


# --- in-context estimates through the patch hooks -----------------------------


class PatchCursor:
    """Imitates the STATS-CEB patch's join-estimate hooks for one query.

    Each planning call ``i`` sizes the union of two inputs and logs the native
    estimate, then substitutes the next value of the estimate file. A wrong
    substitution perturbs every later native estimate, as in the real planner.
    """

    def __init__(self, calls: list[tuple[str, str, float]]) -> None:
        self.calls = calls  # (inner aliases, outer aliases, true native value)
        self.files = {"join_est_record_job.txt": "", "old_new_join_est.txt": ""}
        self.feed: list[float] = []
        self.result: Any = None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        if sql.startswith("COPY"):
            body = re.search(r"ARRAY\[(.*)\]::float8", sql)
            assert body is not None
            self.feed = [float(x) for x in body.group(1).split(",")]
        elif "pg_stat_file" in sql:
            self.result = (len(self.files[params[0]]),)
        elif "pg_read_file" in sql:
            name, offset = params
            self.result = (self.files[name][offset:],)
        elif sql.startswith("EXPLAIN"):
            perturbed = False
            for i, (inner, outer, native) in enumerate(self.calls):
                logged = native * 7 if perturbed else native
                self.files["join_est_record_job.txt"] += (
                    f"RELOPTINFO ({inner}): rows=1 width=4\n"
                    f"RELOPTINFO ({outer}): rows=1 width=4\n"
                )
                self.files["old_new_join_est.txt"] += f"{logged:.5f}:{self.feed[i]}\n"
                perturbed |= self.feed[i] != logged

    def fetchone(self) -> Any:
        return self.result

    def fetchall(self) -> list[Any]:
        return []


def test_in_context_recovery_returns_the_unperturbed_natives() -> None:
    calls = [
        ("a", "b", 10.0),
        ("b", "c", 20.0),
        ("a b", "c", 300.0),
        ("a", "b c", 999.0),  # a later call on the same set (parameterized path)
    ]
    out = native_estimates_incontext.in_context_estimates(PatchCursor(calls), "Q")
    assert out == {
        frozenset("ab"): 10.0,
        frozenset("bc"): 20.0,
        frozenset("abc"): 300.0,  # the first call on {a, b, c} sets its size
    }


def test_in_context_recovery_rejects_a_shrinking_call_count() -> None:
    class Shrinking(PatchCursor):
        def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
            if sql.startswith("EXPLAIN") and self.feed[1] != 1.0:
                self.calls = self.calls[:1]  # the third pass makes fewer calls
            super().execute(sql, params)

    cursor = Shrinking([("a", "b", 10.0), ("b", "c", 20.0), ("a b", "c", 30.0)])
    with pytest.raises(RuntimeError, match="join calls after"):
        native_estimates_incontext.in_context_estimates(cursor, "Q")


# --- plan shapes of forced orders ---------------------------------------------


def scan(alias: str) -> dict[str, Any]:
    return {"Node Type": "Seq Scan", "Alias": alias}


def join(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"Node Type": "Hash Join", "Plans": [left, right]}


LEFT_DEEP = join(join(scan("p"), scan("c")), scan("u"))
BUSHY = join(join(scan("p"), scan("c")), join(scan("u"), scan("v")))


def test_left_deep_order_of_a_left_deep_plan() -> None:
    assert left_deep_order(LEFT_DEEP) == ["c", "p", "u"]
    assert left_deep_order(BUSHY) is None


def test_honours_accepts_either_input_side_but_not_another_order() -> None:
    swapped = join(scan("u"), join(scan("c"), scan("p")))  # inner/outer swapped
    assert honours(swapped, ["p", "c", "u"])
    assert not honours(LEFT_DEEP, ["p", "u", "c"])


def test_parameterized_scan_inside_a_nested_loop_still_counts() -> None:
    index_side = {
        "Node Type": "Index Scan",
        "Alias": "u",
        "Parent Relationship": "Inner",
    }
    nested = {
        "Node Type": "Nested Loop",
        "Plans": [join(scan("p"), scan("c")), index_side],
    }
    assert honours(nested, ["p", "c", "u"])


# --- undefined true C_out ------------------------------------------------------


def test_true_cout_is_nan_for_a_join_outside_the_list() -> None:
    truth = {(1, ("c", "p")): 5.0, (1, ("c", "p", "u")): 7.0}
    assert analyze_injection.true_cout(LEFT_DEEP, 1, truth) == 12.0
    assert math.isnan(analyze_injection.true_cout(BUSHY, 1, truth))


def test_bootstrap_reports_dropped_replicates() -> None:
    rows = [{"query_id": i, "x": float(i)} for i in range(20)]

    def stat(rs: list[dict[str, Any]]) -> float:
        return math.nan if any(r["query_id"] == 0 for r in rs) else 1.0

    out = cluster_bootstrap(rows, stat, replicates=200, seed=0)
    assert out["replicates"] + out["replicates_dropped"] == 200
    assert out["replicates_dropped"] > 0


def test_count_from_the_first_run_survives_a_timed_out_second_run() -> None:
    from experiments.pg_injection.subset_counts import timed_count

    class SecondRunTimesOut:
        def __init__(self) -> None:
            self.runs = 0
            self.connection = self

        def rollback(self) -> None:
            """Connection API used after a cancelled statement."""

        def execute(self, sql: str) -> None:
            if sql.startswith("SET"):
                return
            self.runs += 1
            if self.runs == 2:
                raise psycopg2.errors.QueryCanceled

        def fetchone(self) -> tuple[int]:
            return (42,)

    value, ms = timed_count(SecondRunTimesOut(), "SELECT 1", 120_000)
    assert value == 42.0
    assert ms == 120_000.0


def test_hint_text_rejects_non_finite_values() -> None:
    from experiments.pg_injection.inject_plans import hint_text

    assert hint_text({K1: 10.4}) == "/*+ Rows(a b #10) */ "
    with pytest.raises(ValueError, match="not finite"):
        hint_text({K1: math.inf})
