"""Exact cardinality for parsed JOB queries (alias-aware, filtered).

Bridges :mod:`src.utils.job_sql_parser` and
:func:`src.utils.join_counting.count_join`: each FROM *alias* becomes its own
join-graph node (so self-joins are handled), the alias's selection filters are
applied to its base-table rows, and the alias-keyed equi-join is counted exactly
without materialisation.

Operating on already-loaded ``rows_by_relation`` keeps this pure and unit-
testable; the real run loads those rows via :mod:`src.utils.imdb_schema`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from src.core.query import Query, RelationStatistics
from src.utils.join_counting import count_join
from src.utils.types import AttributeName, Relation

if TYPE_CHECKING:
    from src.utils.job_sql_parser import ParsedJobQuery


def build_alias_rows(
    parsed: ParsedJobQuery,
    rows_by_relation: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    """Return per-alias rows, each base table filtered by that alias's predicates.

    Args:
        parsed: The parsed JOB query (alias-keyed joins and filters).
        rows_by_relation: Rows for each *base table* referenced by the query.

    Returns:
        A mapping from alias (join-graph node id) to its filtered rows. Two
        aliases of the same base table get independently filtered copies.
    """
    alias_rows: dict[str, list[dict[str, str]]] = {}
    for alias, base in parsed.alias_to_relation.items():
        if base not in rows_by_relation:
            raise KeyError(f"missing rows for base relation {base!r} (alias {alias!r})")
        alias_rows[alias] = [
            row for row in rows_by_relation[base] if parsed.row_passes(alias, row)
        ]
    return alias_rows


def compute_job_true_cardinality(
    parsed: ParsedJobQuery,
    rows_by_relation: dict[str, list[dict[str, str]]],
) -> float:
    """Exact cardinality of a parsed JOB query (joins + filters, alias-aware)."""
    alias_rows = build_alias_rows(parsed, rows_by_relation)
    return count_join(alias_rows, parsed.aliases, parsed.joins)


def _alias_join_attributes(
    parsed: ParsedJobQuery,
) -> dict[str, set[AttributeName]]:
    """Map each alias to the join attributes it binds."""
    attrs: dict[str, set[AttributeName]] = defaultdict(set)
    for join in parsed.joins:
        attrs[join.left_relation].add(join.left_attribute)
        attrs[join.right_relation].add(join.right_attribute)
    return attrs


def build_job_query(
    parsed: ParsedJobQuery,
    rows_by_relation: dict[str, list[dict[str, str]]],
) -> Query:
    """Build a CEG :class:`Query` from a parsed JOB query and its base rows.

    Each alias becomes a relation node named by the alias (so self-joins are
    distinct nodes); its cardinality is the *filtered* row count and its
    join-attribute statistics (distinct counts, max frequencies) are computed
    from the filtered rows. This is what the optimistic/pessimistic CEGs consume
    to produce an estimate, so Safe-CEG can be evaluated on real JOB queries.
    """
    alias_rows = build_alias_rows(parsed, rows_by_relation)
    join_attrs = _alias_join_attributes(parsed)
    relations: dict[str, RelationStatistics] = {}
    for alias in parsed.aliases:
        rows = alias_rows[alias]
        attributes = join_attrs.get(alias, set())
        distinct_counts: dict[AttributeName, int] = {}
        max_frequencies: dict[AttributeName, int] = {}
        for attribute in attributes:
            # NULL (empty field / None) never satisfies an equi-join, so it is not a
            # join value: counting it made the NULL group the "max degree" of
            # mostly-NULL keys (votes.UserId) and loosened the bound (fixed
            # 2026-10-04 after internal review panel 4).
            value_counts = Counter(
                row[attribute] for row in rows if row[attribute] not in ("", None)
            )
            distinct_counts[attribute] = max(len(value_counts), 1)
            max_frequencies[attribute] = max(max(value_counts.values(), default=1), 1)
        relation = Relation(
            name=alias,
            attributes=frozenset(attributes),
            cardinality=max(len(rows), 1),
        )
        relations[alias] = RelationStatistics(
            relation=relation,
            attribute_distinct_counts=distinct_counts,
            attribute_max_frequencies=max_frequencies,
        )
    return Query(relations=relations, joins=parsed.joins)
