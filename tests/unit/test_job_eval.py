"""Tests for alias-aware exact cardinality of parsed JOB queries."""

from __future__ import annotations

from itertools import product

from src.core.ceg import pessimistic_bound
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.utils.job_eval import (
    build_job_query,
    compute_job_true_cardinality,
)
from src.utils.job_sql_parser import parse_job_sql


def _brute_force_self_join(rows: list[dict[str, str]]) -> int:
    """Reference count for ``t1.id = t2.episode_of_id AND t1.year > 2000``."""
    total = 0
    for t1, t2 in product(rows, rows):
        if int(t1["production_year"]) > 2000 and t1["id"] == t2["episode_of_id"]:
            total += 1
    return total


def test_self_join_is_alias_separated_and_exact() -> None:
    rows = [
        {"id": "1", "production_year": "2005", "episode_of_id": "0"},
        {"id": "2", "production_year": "1999", "episode_of_id": "1"},
        {"id": "3", "production_year": "2010", "episode_of_id": "1"},
        {"id": "4", "production_year": "2003", "episode_of_id": "2"},
    ]
    parsed = parse_job_sql(
        "SELECT 1 FROM title AS t1, title AS t2 "
        "WHERE t1.production_year > 2000 AND t1.id = t2.episode_of_id;",
        "self",
    )
    assert parsed.is_self_join
    assert parsed.aliases == {"t1", "t2"}
    assert parsed.base_relations == {"title"}

    expected = _brute_force_self_join(rows)
    assert expected == 2  # guards the fixture
    got = compute_job_true_cardinality(parsed, {"title": rows})
    assert got == float(expected)


def test_two_table_join_with_filter_is_exact() -> None:
    titles = [
        {"id": "1", "production_year": "2005"},
        {"id": "2", "production_year": "1990"},
        {"id": "3", "production_year": "2008"},
    ]
    keywords = [
        {"movie_id": "1", "keyword_id": "10"},
        {"movie_id": "1", "keyword_id": "11"},
        {"movie_id": "3", "keyword_id": "10"},
    ]
    parsed = parse_job_sql(
        "SELECT 1 FROM title AS t, movie_keyword AS mk "
        "WHERE t.production_year > 2000 AND t.id = mk.movie_id;",
        "two",
    )
    # t passes: ids 1 and 3; mk rows joining those: movie_id 1 (x2) + 3 (x1) = 3
    got = compute_job_true_cardinality(
        parsed, {"title": titles, "movie_keyword": keywords}
    )
    assert got == 3.0


def test_build_job_query_runs_estimators_on_self_join() -> None:
    rows = [
        {"id": "1", "production_year": "2005", "episode_of_id": "2"},
        {"id": "2", "production_year": "2010", "episode_of_id": "1"},
        {"id": "3", "production_year": "2003", "episode_of_id": "1"},
    ]
    parsed = parse_job_sql(
        "SELECT 1 FROM title AS t1, title AS t2 "
        "WHERE t1.production_year > 2000 AND t1.id = t2.episode_of_id;",
        "self",
    )
    query = build_job_query(parsed, {"title": rows})
    # the alias nodes are distinct, so the self-join is a valid query
    assert query.relation_names == {"t1", "t2"}

    # the optimistic CEG configs and the pessimistic bound all produce
    # finite, positive estimates -> Safe-CEG can be evaluated on this query
    estimates = BaselineCEGEstimator().estimate_all_configs(query)
    assert estimates
    assert all(est.estimate > 0 for est in estimates.values())
    assert pessimistic_bound(query) > 0


def test_null_join_keys_never_match() -> None:
    """SQL semantics: NULL join keys must not join (regression for STATS-CEB)."""
    from src.core.query import JoinPredicate
    from src.utils.join_counting import count_join

    left = [{"k": "1"}, {"k": ""}, {"k": ""}]  # two NULL keys
    right = [{"k": "1"}, {"k": ""}, {"k": ""}, {"k": ""}]  # three NULL keys
    joins = frozenset({JoinPredicate("L", "k", "R", "k")})
    # only the 1<->1 pair joins; 2x3 NULL pairs must be excluded
    assert count_join({"L": left, "R": right}, frozenset({"L", "R"}), joins) == 1.0


def test_null_join_keys_do_not_enter_degree_statistics() -> None:
    """Regression (2026-10-04): NULL was counted as a join value in max degree."""
    parsed = parse_job_sql(
        "SELECT COUNT(*) FROM users AS u, votes AS v WHERE v.UserId = u.Id;"
    )
    rows = {
        "users": [{"Id": "1"}, {"Id": "2"}],
        # 1 vote for user 1, 1 for user 2, five NULL voters
        "votes": [{"UserId": "1"}, {"UserId": "2"}] + [{"UserId": ""}] * 5,
    }
    query = build_job_query(parsed, rows)
    votes = query.relation_statistics("v")
    assert votes.max_frequency("UserId") == 1
    assert votes.distinct_count("UserId") == 2
    assert pessimistic_bound(query) >= 2.0  # still sound: the true join size is 2
