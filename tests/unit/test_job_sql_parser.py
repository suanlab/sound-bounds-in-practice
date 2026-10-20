"""Unit tests for the JOB SQL-subset parser."""

from __future__ import annotations

from src.core.query import JoinPredicate
from src.utils.job_sql_parser import (
    JobFilter,
    parse_job_sql,
)

_QUERY_1A = """
SELECT MIN(t.title) AS movie_title
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info_idx AS mi_idx,
     title AS t
WHERE ct.kind = 'production companies'
  AND it.info = 'top 250 rank'
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND (mc.note LIKE '%(co-production)%'
       OR mc.note LIKE '%(presents)%')
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND it.id = mi_idx.info_type_id;
"""


def test_parses_relations_joins_and_filters() -> None:
    parsed = parse_job_sql(_QUERY_1A, "1a")
    assert parsed.fully_parsed
    assert not parsed.is_self_join
    assert parsed.aliases == {"ct", "it", "mc", "mi_idx", "t"}
    assert parsed.base_relations == {
        "company_type",
        "info_type",
        "movie_companies",
        "movie_info_idx",
        "title",
    }
    assert len(parsed.joins) == 5
    # joins are expressed over aliases (the join-graph node ids)
    assert JoinPredicate("ct", "id", "mc", "company_type_id") in parsed.joins
    # two '=' filters + one NOT LIKE filter; the OR is a separate group
    assert len(parsed.filters) == 3
    assert len(parsed.or_groups) == 1
    assert parsed.or_groups[0].alias == "mc"
    assert len(parsed.or_groups[0].alternatives) == 2


def test_alias_resolves_to_base_relation() -> None:
    parsed = parse_job_sql(_QUERY_1A, "1a")
    assert parsed.alias_to_relation["t"] == "title"
    assert parsed.alias_to_relation["mi_idx"] == "movie_info_idx"


def test_equality_and_like_evaluation() -> None:
    eq = JobFilter("company_type", "kind", "=", ("production companies",))
    assert eq.matches({"kind": "production companies"})
    assert not eq.matches({"kind": "distributors"})

    like = JobFilter("title", "title", "LIKE", ("%sequel%",))
    assert like.matches({"title": "the sequel story"})
    assert not like.matches({"title": "prequel"})

    not_like = JobFilter("mc", "note", "NOT LIKE", ("%(MGM)%",))
    assert not_like.matches({"note": "(co-production)"})
    assert not not_like.matches({"note": "x (MGM) y"})


def test_in_between_null_and_numeric_ops() -> None:
    in_filter = JobFilter("keyword", "keyword", "IN", ("murder", "blood"))
    assert in_filter.matches({"keyword": "murder"})
    assert not in_filter.matches({"keyword": "comedy"})

    between = JobFilter("title", "production_year", "BETWEEN", ("2000", "2010"))
    assert between.matches({"production_year": "2005"})
    assert not between.matches({"production_year": "1999"})

    gt = JobFilter("title", "production_year", ">", ("2005",))
    assert gt.matches({"production_year": "2010"})
    assert not gt.matches({"production_year": "2001"})

    not_null = JobFilter("mc", "note", "IS NOT NULL", ())
    assert not_null.matches({"note": "something"})
    assert not not_null.matches({"note": ""})

    is_null = JobFilter("mc", "note", "IS NULL", ())
    assert is_null.matches({"note": ""})
    assert not is_null.matches({"note": "x"})


def test_in_list_with_quoted_values_and_commas() -> None:
    parsed = parse_job_sql(
        "SELECT 1 FROM cast_info AS ci WHERE ci.note IN ('(voice)', '(uncredited)');",
        "synthetic",
    )
    assert parsed.fully_parsed
    assert len(parsed.filters) == 1
    assert parsed.filters[0].op == "IN"
    assert parsed.filters[0].values == ("(voice)", "(uncredited)")


def test_between_and_is_not_consumed_as_conjunction_separator() -> None:
    parsed = parse_job_sql(
        "SELECT 1 FROM title AS t, movie_keyword AS mk "
        "WHERE t.production_year BETWEEN 1990 AND 2000 AND t.id = mk.movie_id;",
        "synthetic",
    )
    assert parsed.fully_parsed
    assert len(parsed.joins) == 1
    assert len(parsed.filters) == 1
    assert parsed.filters[0].op == "BETWEEN"
    assert parsed.filters[0].values == ("1990", "2000")


def test_row_passes_combines_filters_and_or_groups() -> None:
    parsed = parse_job_sql(_QUERY_1A, "1a")
    # the mc-aliased movie_companies row must clear the NOT LIKE filter AND OR group
    passing = {"note": "(co-production)", "company_type_id": "2", "movie_id": "1"}
    assert parsed.row_passes("mc", passing)
    failing_or = {"note": "(other)", "company_type_id": "2", "movie_id": "1"}
    assert not parsed.row_passes("mc", failing_or)


def test_null_cell_fails_every_comparison_but_is_null() -> None:
    """SQL: comparison with NULL is UNKNOWN, so the row is dropped (STATS-CEB)."""
    null_row = {"c": ""}
    for op, values in (
        ("<=", ("100",)),
        (">=", ("0",)),
        ("<", ("5",)),
        (">", ("5",)),
        ("=", ("5",)),
        ("!=", ("5",)),
        ("LIKE", ("%x%",)),
        ("NOT LIKE", ("%x%",)),
        ("IN", ("a", "b")),
        ("BETWEEN", ("1", "9")),
    ):
        assert not JobFilter("t", "c", op, values).matches(null_row), op
    # only the NULL-aware operators see it
    assert JobFilter("t", "c", "IS NULL", ()).matches(null_row)
    assert not JobFilter("t", "c", "IS NOT NULL", ()).matches(null_row)
    # non-NULL cells are unaffected
    assert JobFilter("t", "c", "<=", ("100",)).matches({"c": "50"})
