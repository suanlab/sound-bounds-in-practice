"""Exact equi-join cardinality counting via count-based hash joins.

Shared by the real-data harness and the synthetic workload generator so that
*true* cardinalities are computed by actually executing the join over generated
or loaded tuples -- never derived from the estimators under evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.query import JoinPredicate

# CSV representation of SQL NULL (``csv.DictReader`` yields "" for empty fields).
_NULL = ""


def _join_variable_map(
    joins: frozenset[JoinPredicate],
) -> dict[tuple[str, str], int]:
    """Union-find over ``(relation, attribute)`` pairs linked by equi-joins.

    Two ``(relation, attribute)`` pairs that are transitively equated by the
    join predicates are assigned the same integer *join-variable* id. This
    handles equalities across differently named attributes and transitive
    join chains.
    """
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(x: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: tuple[str, str], b: tuple[str, str]) -> None:
        parent[find(a)] = find(b)

    for j in joins:
        union(
            (j.left_relation, j.left_attribute),
            (j.right_relation, j.right_attribute),
        )

    roots = sorted({find(p) for p in parent})
    root_to_id = {root: i for i, root in enumerate(roots)}
    return {p: root_to_id[find(p)] for p in parent}


def count_join(
    rows_by_relation: dict[str, list[dict[str, str]]],
    relation_names: frozenset[str],
    joins: frozenset[JoinPredicate],
) -> float:
    """Exact equi-join cardinality of the given rows via count-based hash joins.

    Instead of materializing the join (which explodes on medium/large data),
    each relation is projected onto its *join-variable* columns and reduced to
    a multiplicity count per distinct value combination. Relations are then
    hash-joined while multiplying counts, so the work is bounded by the number
    of distinct join-key combinations, not by the size of the join result.
    The method is exact for arbitrary equi-join queries, including cyclic ones.

    **SQL NULL semantics.** A row whose join attribute is NULL can never satisfy
    an equi-join predicate (``NULL = NULL`` is unknown, not true), so such rows
    are dropped before counting. NULL is represented as the empty string, the
    convention produced by ``csv.DictReader`` on the real datasets. Without this
    rule, NULL-keyed rows on both sides would be matched to each other and
    inflate the count -- on STATS-CEB, where ``votes.UserId`` is 89% NULL, that
    error reaches 86x.
    """
    var_map = _join_variable_map(joins)

    # Per relation, the list of (join_variable_id, attribute) it binds.
    rel_bindings: dict[str, list[tuple[int, str]]] = {}
    for (rel, attr), var_id in var_map.items():
        if rel in relation_names:
            rel_bindings.setdefault(rel, []).append((var_id, attr))

    def relation_counts(rname: str) -> dict[tuple[str, ...], int]:
        """Map each tuple of join-variable values to its row multiplicity."""
        bindings = sorted(rel_bindings.get(rname, []))  # by var id
        counts: dict[tuple[str, ...], int] = {}
        for row in rows_by_relation[rname]:
            key = tuple(row[attr] for _, attr in bindings)
            if any(value == _NULL for value in key):
                continue  # SQL: a NULL join key never matches
            counts[key] = counts.get(key, 0) + 1
        return counts

    rel_list = sorted(relation_names)
    # Frontier: ordered join-variable ids -> {value tuple: count}.
    frontier_vars: list[int] = [v for v, _ in sorted(rel_bindings.get(rel_list[0], []))]
    frontier: dict[tuple[str, ...], int] = relation_counts(rel_list[0])

    for rname in rel_list[1:]:
        rel_vars = [v for v, _ in sorted(rel_bindings.get(rname, []))]
        rel_counts = relation_counts(rname)
        shared = [v for v in frontier_vars if v in set(rel_vars)]
        new_vars = frontier_vars + [v for v in rel_vars if v not in set(frontier_vars)]

        # Index the incoming relation by the shared join variables.
        f_pos = {v: i for i, v in enumerate(frontier_vars)}
        r_pos = {v: i for i, v in enumerate(rel_vars)}
        shared_r_idx = [r_pos[v] for v in shared]
        rel_index: dict[tuple[str, ...], list[tuple[tuple[str, ...], int]]] = {}
        for rk, rc in rel_counts.items():
            sub = tuple(rk[i] for i in shared_r_idx)
            rel_index.setdefault(sub, []).append((rk, rc))

        shared_f_idx = [f_pos[v] for v in shared]
        only_r_idx = [r_pos[v] for v in rel_vars if v not in set(frontier_vars)]
        new_frontier: dict[tuple[str, ...], int] = {}
        for fk, fc in frontier.items():
            sub = tuple(fk[i] for i in shared_f_idx)
            for rk, rc in rel_index.get(sub, []):
                new_key = fk + tuple(rk[i] for i in only_r_idx)
                new_frontier[new_key] = new_frontier.get(new_key, 0) + fc * rc
        frontier_vars = new_vars
        frontier = new_frontier

    total = sum(frontier.values())
    return float(max(1, total))
