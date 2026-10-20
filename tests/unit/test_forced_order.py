"""Tests for the explicit-JOIN rewrite that forces a left-deep join order."""

from __future__ import annotations

import pytest

from experiments.pg_injection.forced_order import FORCE, forced_join_sql

SQL = (
    "SELECT COUNT(*) FROM votes as v, posts as p, users as u WHERE v.PostId = p.Id "
    "AND v.UserId = u.Id AND v.CreationDate<='2014-09-12 00:00:00'::timestamp "
    "AND p.Score>=-1 AND u.UpVotes>=0;"
)


def test_joins_follow_the_order_and_filters_stay_in_where() -> None:
    out = forced_join_sql(SQL, ["p", "v", "u"])
    assert out.startswith(FORCE)
    assert (
        "FROM posts AS p JOIN votes AS v ON v.PostId = p.Id "
        "JOIN users AS u ON v.UserId = u.Id WHERE " in out
    )
    assert "p.Score>=-1" in out.split(" WHERE ", 1)[1]
    assert "'2014-09-12 00:00:00'::timestamp" in out


def test_a_cross_product_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="no join predicate"):
        forced_join_sql(SQL, ["p", "u", "v"])


def test_order_must_be_a_permutation() -> None:
    with pytest.raises(ValueError, match="permutation"):
        forced_join_sql(SQL, ["p", "v"])


def test_or_is_rejected() -> None:
    with pytest.raises(ValueError, match="conjunctive"):
        forced_join_sql(SQL.replace(" AND u.UpVotes", " OR u.UpVotes"), ["p", "v", "u"])


def test_every_predicate_is_kept_exactly_once() -> None:
    out = forced_join_sql(SQL, ["u", "v", "p"])
    for conjunct in ("v.PostId = p.Id", "v.UserId = u.Id", "u.UpVotes>=0"):
        assert out.count(conjunct) == 1


def test_between_and_quoted_and_are_rejected() -> None:
    between = SQL.replace("p.Score>=-1", "p.Score BETWEEN 1 AND 5")
    with pytest.raises(ValueError, match="BETWEEN"):
        forced_join_sql(between, ["p", "v", "u"])
    quoted = SQL.replace("u.UpVotes>=0", "u.Location = 'Here AND There'")
    with pytest.raises(ValueError, match="quoted AND"):
        forced_join_sql(quoted, ["p", "v", "u"])


def test_count_sql_refuses_to_return_an_unsampled_count() -> None:
    from experiments.pg_injection.subset_counts import count_sql

    sql = "SELECT * FROM badges b, users AS u WHERE b.UserId = u.Id"
    assert "TABLESAMPLE" in count_sql(sql, ("u", 0.1, 0))
    with pytest.raises(ValueError, match="matched 0 FROM items"):
        count_sql(sql, ("b", 0.1, 0))  # no "AS": the alias is not recognised


def test_an_order_through_an_implied_equality_is_forced() -> None:
    sql = (
        "SELECT COUNT(*) FROM comments as c, badges as b, users as u "
        "WHERE u.Id = c.UserId AND c.UserId = b.UserId AND u.UpVotes>=0;"
    )
    out = forced_join_sql(sql, ["b", "u", "c"])  # b-u is joined only by implication
    assert "JOIN users AS u ON u.Id = b.UserId JOIN comments AS c ON" in out
    for conjunct in ("u.Id = c.UserId", "c.UserId = b.UserId", "u.UpVotes>=0"):
        assert conjunct in out


def test_plan_shape_ignores_conditions_but_not_join_methods() -> None:
    from experiments.pg_injection.summarize_forced_orders import shape

    def join(method: str, cond: str) -> dict[str, object]:
        return {
            "Node Type": method,
            "Join Type": "Inner",
            "Hash Cond": cond,
            "Total Cost": 1.0,
            "Plans": [
                {"Node Type": "Seq Scan", "Alias": "a", "Parent Relationship": "Outer"},
                {"Node Type": "Seq Scan", "Alias": "b", "Parent Relationship": "Inner"},
            ],
        }

    assert shape(join("Hash Join", "(a.x = b.x)")) == shape(
        join("Hash Join", "(b.x = a.x)")
    )
    assert shape(join("Hash Join", "")) != shape(join("Merge Join", ""))
