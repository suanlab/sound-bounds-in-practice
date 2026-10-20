"""Force a left-deep join order in PostgreSQL without pg_hint_plan.

``pg_hint_plan``'s ``Leading`` hint does not work in our patched PostgreSQL
13.1 build: the hint is reported as used, yet the planner joins in another
order with disable-cost totals (~1e17) and row estimates of ~1.5e19. Plain
PostgreSQL offers a reliable alternative: write the joins explicitly, in the
wanted order, and set ``join_collapse_limit = 1`` so the planner keeps the join
tree as written. It still chooses each join's method and which input is inner.

``forced_join_sql`` rewrites a STATS-CEB query (``FROM t AS a, ...`` with a
conjunctive ``WHERE`` and no ``OR``) into that form: every join predicate is
attached to the first join at which all of its aliases are present, and
single-alias filters stay in ``WHERE``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

FROM_ITEM = re.compile(r"(\w+)\s+as\s+(\w+)", re.IGNORECASE)
COLUMN = re.compile(r"\b(\w+)\.\w+")
FORCE = "SET join_collapse_limit = 1; "
UNFORCE = "SET join_collapse_limit = 8; "


EQUALITY = re.compile(r"^\s*(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\s*$")


def equivalence_classes(conjuncts: Iterable[str]) -> list[set[tuple[str, str]]]:
    """Columns made equal by the equi-join conjuncts, as (alias, column) sets."""
    classes: list[set[tuple[str, str]]] = []
    for conjunct in conjuncts:
        match = EQUALITY.match(conjunct)
        if match is None:
            continue
        left = (match.group(1).lower(), match.group(2))
        right = (match.group(3).lower(), match.group(4))
        merged = {left, right}
        rest = []
        for cls in classes:
            if cls & merged:
                merged |= cls
            else:
                rest.append(cls)
        classes = [*rest, merged]
    return classes


def implied_equality(
    alias: str, present: set[str], classes: list[set[tuple[str, str]]]
) -> str | None:
    """An equality implied by the classes between ``alias`` and a present alias."""
    for cls in classes:
        mine = sorted(col for col in cls if col[0] == alias)
        theirs = sorted(col for col in cls if col[0] in present)
        if mine and theirs:
            (a, x), (b, y) = mine[0], theirs[0]
            return f"{a}.{x} = {b}.{y}"
    return None


def forced_join_sql(sql: str, order: list[str]) -> str:
    """Rewrite ``sql`` so that its relations join left-deep in ``order``.

    Raises:
        ValueError: if the query has an ``OR``, if ``order`` is not a
            permutation of the query's aliases, or if some prefix of ``order``
            has no join predicate to the next relation (a cross product).
    """
    body = sql.strip().rstrip(";")
    head, where = re.split(r"\s+WHERE\s+", body, maxsplit=1, flags=re.IGNORECASE)
    if re.search(r"\bOR\b", where, flags=re.IGNORECASE):
        msg = "only conjunctive WHERE clauses are supported"
        raise ValueError(msg)
    # Splitting on AND would break BETWEEN ... AND ... and quoted text.
    unquoted = re.sub(r"'[^']*'", "''", where)
    if re.search(r"\bBETWEEN\b", unquoted, flags=re.IGNORECASE) or any(
        re.search(r"\bAND\b", q, flags=re.IGNORECASE)
        for q in re.findall(r"'[^']*'", where)
    ):
        msg = "BETWEEN and quoted AND are not supported"
        raise ValueError(msg)
    select, from_list = re.split(r"\s+FROM\s+", head, maxsplit=1, flags=re.IGNORECASE)
    tables = {alias.lower(): table for table, alias in FROM_ITEM.findall(from_list)}
    wanted = [a.lower() for a in order]
    if sorted(wanted) != sorted(tables):
        msg = f"order {order} is not a permutation of {sorted(tables)}"
        raise ValueError(msg)

    filters: list[str] = []
    joins: list[tuple[frozenset[str], str]] = []
    for conjunct in re.split(r"\s+AND\s+", where, flags=re.IGNORECASE):
        aliases = frozenset(
            a.lower() for a in COLUMN.findall(conjunct) if a.lower() in tables
        )
        if len(aliases) >= 2:
            joins.append((aliases, conjunct.strip()))
        else:
            filters.append(conjunct.strip())

    classes = equivalence_classes(conjunct for _, conjunct in joins)
    first = wanted[0]
    text = f"{select} FROM {tables[first]} AS {first}"
    present = {first}
    pending = list(joins)
    for alias in wanted[1:]:
        present.add(alias)
        ready = [c for a, c in pending if a <= present]
        if not ready:
            # PostgreSQL derives joins from equivalence classes (a.x = b.y and
            # b.y = c.z imply a.x = c.z); an order it chooses may rely on one.
            implied = implied_equality(alias, present - {alias}, classes)
            if implied is None:
                msg = (
                    f"no join predicate connects {alias} to {sorted(present - {alias})}"
                )
                raise ValueError(msg)
            ready = [implied]
        pending = [(a, c) for a, c in pending if not a <= present]
        text += f" JOIN {tables[alias]} AS {alias} ON " + " AND ".join(ready)
    if filters:
        text += " WHERE " + " AND ".join(filters)
    return FORCE + text + ";"
