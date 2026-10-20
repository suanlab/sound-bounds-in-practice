"""Parser for the Join Order Benchmark (JOB) SQL subset.

JOB queries are select-project-join (SPJ) statements of the form::

    SELECT MIN(...) ...
    FROM rel1 AS a1, rel2 AS a2, ...
    WHERE <conjunction of join predicates and selection filters>;

This module turns one such query into a :class:`ParsedJobQuery` carrying the
base relations, the equi-join predicates (as :class:`~src.core.query.JoinPredicate`),
and the selection filters (single predicates and parenthesised ``OR`` groups).
The filter vocabulary covers everything the 113 JOB templates use:
``= != < <= > >=``, ``LIKE`` / ``NOT LIKE``, ``IN (...)``, ``BETWEEN x AND y``,
``IS NULL`` / ``IS NOT NULL``, and disjunctive ``(a OR b)`` groups.

Evaluation is included (:meth:`ParsedJobQuery.row_passes`) so true cardinalities
can be computed exactly by the count-based hash join, matching SQL ``WHERE``
semantics, without materialising the join. The parser is intentionally
self-contained (no I/O) so it is unit-testable against the query text alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.query import JoinPredicate
from src.utils.types import AttributeName, RelationName

if TYPE_CHECKING:
    from collections.abc import Mapping

# Selection operators understood by :meth:`JobFilter.matches`.
_COMPARISON_OPS = ("!=", "<>", "<=", ">=", "=", "<", ">")
_NULL_EMPTY = ""  # CSV representation of SQL NULL


def _as_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _like_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a SQL ``LIKE`` pattern (``%``/``_`` wildcards) to a regex."""
    parts: list[str] = []
    for ch in pattern:
        if ch == "%":
            parts.append(".*")
        elif ch == "_":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    return re.compile("^" + "".join(parts) + "$", re.DOTALL)


@dataclass(frozen=True)
class JobFilter:
    """A single selection predicate on one *alias*'s attribute.

    ``alias`` is the per-query table alias (the join-graph node id), not the base
    relation -- so self-joins, where one table appears under several aliases with
    different filters, stay separated.
    """

    alias: str
    attribute: AttributeName
    op: str
    values: tuple[str, ...]

    def matches(self, row: Mapping[str, str]) -> bool:
        """Return whether ``row`` satisfies this predicate (SQL semantics).

        **NULL handling.** Every operator except ``IS [NOT] NULL`` evaluates to
        UNKNOWN on a NULL operand, and a WHERE clause keeps only rows that
        evaluate to TRUE -- so a NULL cell fails the predicate. Falling back to
        string comparison instead (``"" <= "100"`` is lexicographically true)
        silently admits NULL rows; on STATS-CEB, where columns such as
        ``votes.BountyAmount`` and ``posts.FavoriteCount`` are largely NULL,
        that inflated exact counts by up to 97x.
        """
        cell = row.get(self.attribute, _NULL_EMPTY)
        op = self.op
        if op == "IS NULL":
            return cell == _NULL_EMPTY
        if op == "IS NOT NULL":
            return cell != _NULL_EMPTY
        if cell == _NULL_EMPTY:
            return False  # SQL: comparison with NULL is UNKNOWN, row is dropped
        if op == "LIKE":
            return _like_to_regex(self.values[0]).match(cell) is not None
        if op == "NOT LIKE":
            return _like_to_regex(self.values[0]).match(cell) is None
        if op == "IN":
            return cell in self.values
        if op == "BETWEEN":
            return self._compare(cell, ">=", self.values[0]) and self._compare(
                cell, "<=", self.values[1]
            )
        return self._compare(cell, op, self.values[0])

    @staticmethod
    def _compare(cell: str, op: str, target: str) -> bool:
        """Scalar comparison, numeric when both sides parse as numbers."""
        if op == "=":
            return cell == target
        if op == "!=":
            return cell != target
        left, right = _as_number(cell), _as_number(target)
        lhs: object
        rhs: object
        lhs, rhs = (cell, target) if left is None or right is None else (left, right)
        if op == "<":
            return bool(lhs < rhs)  # type: ignore[operator]
        if op == "<=":
            return bool(lhs <= rhs)  # type: ignore[operator]
        if op == ">":
            return bool(lhs > rhs)  # type: ignore[operator]
        if op == ">=":
            return bool(lhs >= rhs)  # type: ignore[operator]
        raise ValueError(f"Unsupported comparison operator: {op!r}")


@dataclass(frozen=True)
class JobOrGroup:
    """A disjunction of single filters, all on the same alias."""

    alias: str
    alternatives: tuple[JobFilter, ...]

    def matches(self, row: Mapping[str, str]) -> bool:
        """Return whether ``row`` satisfies at least one alternative."""
        return any(alt.matches(row) for alt in self.alternatives)


@dataclass(frozen=True)
class ParsedJobQuery:
    """A parsed JOB query: aliases, joins, filters, and parse residue.

    The join-graph nodes are the query *aliases* (so self-joins are distinct
    nodes); ``alias_to_relation`` recovers the base table for data loading.
    ``joins`` are expressed over aliases, ready for the alias-keyed count join.
    """

    query_id: str
    alias_to_relation: Mapping[str, RelationName]
    joins: frozenset[JoinPredicate]
    filters: tuple[JobFilter, ...]
    or_groups: tuple[JobOrGroup, ...]
    unsupported: tuple[str, ...]

    @property
    def aliases(self) -> frozenset[str]:
        """The join-graph node set (one per FROM alias)."""
        return frozenset(self.alias_to_relation)

    @property
    def base_relations(self) -> frozenset[RelationName]:
        """The distinct base tables referenced (for loading CSVs)."""
        return frozenset(self.alias_to_relation.values())

    @property
    def fully_parsed(self) -> bool:
        """True when every WHERE conjunct was recognised."""
        return not self.unsupported

    @property
    def is_self_join(self) -> bool:
        """True when some base table appears under more than one alias."""
        return len(self.base_relations) != len(self.alias_to_relation)

    def row_passes(self, alias: str, row: Mapping[str, str]) -> bool:
        """Return whether ``row`` of ``alias`` satisfies all of its filters."""
        for flt in self.filters:
            if flt.alias == alias and not flt.matches(row):
                return False
        for group in self.or_groups:
            if group.alias == alias and not group.matches(row):
                return False
        return True


class JobParseError(ValueError):
    """Raised when a JOB query falls outside the supported SPJ subset."""


def _split_top_level_and(where: str) -> list[str]:
    """Split a WHERE clause on top-level ``AND``, respecting strings, parens, and
    the ``AND`` that belongs to ``BETWEEN x AND y``."""
    conjuncts: list[str] = []
    depth = 0
    in_string = False
    between_pending = False
    token = ""
    i = 0
    upper = where.upper()
    length = len(where)
    while i < length:
        ch = where[i]
        if in_string:
            token += ch
            if ch == "'":
                if i + 1 < length and where[i + 1] == "'":  # escaped quote
                    token += where[i + 1]
                    i += 2
                    continue
                in_string = False
            i += 1
            continue
        if ch == "'":
            in_string = True
            token += ch
            i += 1
            continue
        if ch == "(":
            depth += 1
            token += ch
            i += 1
            continue
        if ch == ")":
            depth -= 1
            token += ch
            i += 1
            continue
        if depth == 0 and upper.startswith(" BETWEEN ", i):
            between_pending = True
            token += where[i : i + 9]
            i += 9
            continue
        if depth == 0 and upper.startswith(" AND ", i):
            if between_pending:
                between_pending = False
                token += where[i : i + 5]
                i += 5
                continue
            conjuncts.append(token.strip())
            token = ""
            i += 5
            continue
        token += ch
        i += 1
    tail = token.strip()
    if tail:
        conjuncts.append(tail)
    return conjuncts


def _parse_from(from_clause: str) -> dict[str, RelationName]:
    """Map each alias to its base relation from a comma-separated FROM list."""
    alias_to_relation: dict[str, RelationName] = {}
    for item in from_clause.split(","):
        tokens = item.split()
        if len(tokens) == 3 and tokens[1].upper() == "AS":
            table, alias = tokens[0], tokens[2]
        elif len(tokens) == 2:
            table, alias = tokens[0], tokens[1]
        elif len(tokens) == 1:
            table = alias = tokens[0]
        else:
            raise JobParseError(f"Unparseable FROM item: {item!r}")
        alias_to_relation[alias] = RelationName(table)
    return alias_to_relation


_QUALIFIED = re.compile(r"^([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)$")


def _strip_literal(value: str) -> str:
    """Strip surrounding quotes and unescape doubled quotes in a SQL literal."""
    value = value.strip()
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def _split_in_list(body: str) -> tuple[str, ...]:
    """Split the comma-separated literal list inside ``IN (...)``."""
    values: list[str] = []
    token = ""
    in_string = False
    i = 0
    while i < len(body):
        ch = body[i]
        if in_string:
            token += ch
            if ch == "'":
                if i + 1 < len(body) and body[i + 1] == "'":
                    token += body[i + 1]
                    i += 2
                    continue
                in_string = False
            i += 1
            continue
        if ch == "'":
            in_string = True
            token += ch
        elif ch == ",":
            values.append(_strip_literal(token))
            token = ""
        else:
            token += ch
        i += 1
    if token.strip():
        values.append(_strip_literal(token))
    return tuple(values)


def _parse_filter(
    conjunct: str, alias_to_relation: Mapping[str, RelationName]
) -> JobFilter:
    """Parse a single (non-join) selection predicate."""
    upper = conjunct.upper()

    def _resolve(qualified: str) -> tuple[str, AttributeName]:
        match = _QUALIFIED.match(qualified.strip())
        if match is None or match.group(1) not in alias_to_relation:
            raise JobParseError(f"Unresolved column: {qualified!r}")
        return match.group(1), AttributeName(match.group(2))

    for tag in (" IS NOT NULL", " IS NULL"):
        if upper.endswith(tag):
            alias, attr = _resolve(conjunct[: len(conjunct) - len(tag)])
            return JobFilter(alias, attr, tag.strip(), ())

    for tag, op in ((" NOT LIKE ", "NOT LIKE"), (" LIKE ", "LIKE")):
        idx = upper.find(tag)
        if idx != -1:
            alias, attr = _resolve(conjunct[:idx])
            literal = _strip_literal(conjunct[idx + len(tag) :])
            return JobFilter(alias, attr, op, (literal,))

    idx = upper.find(" IN ")
    if idx != -1:
        alias, attr = _resolve(conjunct[:idx])
        rest = conjunct[idx + 4 :].strip()
        if not (rest.startswith("(") and rest.endswith(")")):
            raise JobParseError(f"Malformed IN list: {conjunct!r}")
        return JobFilter(alias, attr, "IN", _split_in_list(rest[1:-1]))

    idx = upper.find(" BETWEEN ")
    if idx != -1:
        alias, attr = _resolve(conjunct[:idx])
        rest = conjunct[idx + 9 :]
        and_idx = rest.upper().find(" AND ")
        if and_idx == -1:
            raise JobParseError(f"Malformed BETWEEN: {conjunct!r}")
        low = _strip_literal(rest[:and_idx])
        high = _strip_literal(rest[and_idx + 5 :])
        return JobFilter(alias, attr, "BETWEEN", (low, high))

    for op in _COMPARISON_OPS:
        idx = conjunct.find(op)
        if idx != -1:
            alias, attr = _resolve(conjunct[:idx])
            normalised = "!=" if op == "<>" else op
            return JobFilter(
                alias, attr, normalised, (_strip_literal(conjunct[idx + len(op) :]),)
            )
    raise JobParseError(f"Unrecognised predicate: {conjunct!r}")


def _try_join(
    conjunct: str, alias_to_relation: Mapping[str, RelationName]
) -> JoinPredicate | None:
    """Return a JoinPredicate if ``conjunct`` is ``alias.col = alias.col``."""
    idx = conjunct.find("=")
    if idx == -1:
        return None
    left = _QUALIFIED.match(conjunct[:idx].strip())
    right = _QUALIFIED.match(conjunct[idx + 1 :].strip())
    if left is None or right is None:
        return None
    if (
        left.group(1) not in alias_to_relation
        or right.group(1) not in alias_to_relation
    ):
        return None
    return JoinPredicate(
        left.group(1),
        AttributeName(left.group(2)),
        right.group(1),
        AttributeName(right.group(2)),
    )


def parse_job_sql(sql: str, query_id: str = "") -> ParsedJobQuery:
    """Parse one JOB SQL query into its relations, joins, and filters.

    Args:
        sql: The full text of a single JOB ``*.sql`` template.
        query_id: Optional identifier carried into the result for reporting.

    Returns:
        A :class:`ParsedJobQuery`. Conjuncts that fall outside the supported
        subset are collected in ``unsupported`` rather than raising, so callers
        can report coverage and skip partially-parsed queries.

    Raises:
        JobParseError: If the FROM/WHERE block structure cannot be recovered.
    """
    text = re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip()
    from_match = re.search(r"\bFROM\b(.*?)\bWHERE\b", text, re.IGNORECASE)
    if from_match is None:
        raise JobParseError("Missing FROM ... WHERE block")
    alias_to_relation = _parse_from(from_match.group(1))
    where_clause = text[from_match.end() :].strip()

    joins: list[JoinPredicate] = []
    filters: list[JobFilter] = []
    or_groups: list[JobOrGroup] = []
    unsupported: list[str] = []

    for conjunct in _split_top_level_and(where_clause):
        if not conjunct:
            continue
        if conjunct.startswith("(") and conjunct.endswith(")"):
            try:
                or_groups.append(_parse_or_group(conjunct[1:-1], alias_to_relation))
            except JobParseError:
                unsupported.append(conjunct)
            continue
        join = _try_join(conjunct, alias_to_relation)
        if join is not None:
            joins.append(join)
            continue
        try:
            filters.append(_parse_filter(conjunct, alias_to_relation))
        except JobParseError:
            unsupported.append(conjunct)

    return ParsedJobQuery(
        query_id=query_id,
        alias_to_relation=dict(alias_to_relation),
        joins=frozenset(joins),
        filters=tuple(filters),
        or_groups=tuple(or_groups),
        unsupported=tuple(unsupported),
    )


def _parse_or_group(
    body: str, alias_to_relation: Mapping[str, RelationName]
) -> JobOrGroup:
    """Parse a parenthesised disjunction of single filters on one relation."""
    alternatives = tuple(
        _parse_filter(part.strip(), alias_to_relation)
        for part in re.split(r"\bOR\b", body, flags=re.IGNORECASE)
        if part.strip()
    )
    if not alternatives:
        raise JobParseError(f"Empty OR group: {body!r}")
    alias_set = {alt.alias for alt in alternatives}
    if len(alias_set) != 1:
        raise JobParseError(f"OR group spans multiple aliases: {body!r}")
    return JobOrGroup(next(iter(alias_set)), alternatives)
