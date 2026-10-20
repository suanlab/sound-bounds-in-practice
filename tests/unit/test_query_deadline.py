"""Tests for the SQLite query deadline used by the runtime experiments."""

from __future__ import annotations

import sqlite3
import time

import pytest

from experiments.scripts.plan_regret import QueryTimeoutError, time_limit
from experiments.scripts.plan_runtime import query_deadline

# Big enough that the cross join cannot finish within the deadline.
_ROWS = 3000


def _slow_connection() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (x INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(_ROWS)])
    return con


def test_deadline_aborts_a_long_sqlite_query() -> None:
    """The progress handler must stop a query that outruns its budget.

    This is the mechanism the runtime experiments depend on. A SIGALRM timeout
    cannot do it: SQLite stays inside a single C call, so the Python signal
    handler never runs until the query has already finished.
    """
    con = _slow_connection()
    start = time.perf_counter()

    with pytest.raises(sqlite3.OperationalError), query_deadline(con, 0.5):
        con.execute("SELECT count(*) FROM t AS a, t AS b, t AS c").fetchone()

    assert time.perf_counter() - start < 10.0


def test_sigalrm_alone_does_not_interrupt_sqlite() -> None:
    """Documents why the progress handler is needed rather than ``time_limit``."""
    con = _slow_connection()

    try:
        with time_limit(0.5):
            con.execute("SELECT count(*) FROM t AS a, t AS b").fetchone()
    except QueryTimeoutError:  # pragma: no cover - would mean SIGALRM sufficed
        pytest.skip("SIGALRM interrupted SQLite on this platform")

    # The query ran to completion despite the 0.5 s alarm.
    assert con.execute("SELECT count(*) FROM t").fetchone()[0] == _ROWS


def test_deadline_is_a_no_op_without_a_budget() -> None:
    """A ``None`` budget must leave the connection untouched."""
    con = _slow_connection()

    with query_deadline(con, None):
        result = con.execute("SELECT count(*) FROM t").fetchone()

    assert result[0] == _ROWS


def test_deadline_is_removed_afterwards() -> None:
    """The handler must not survive the block and abort later queries."""
    con = _slow_connection()

    with pytest.raises(sqlite3.OperationalError), query_deadline(con, 0.3):
        con.execute("SELECT count(*) FROM t AS a, t AS b, t AS c").fetchone()

    assert con.execute("SELECT count(*) FROM t").fetchone()[0] == _ROWS
