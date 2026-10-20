"""Synthetic workload generation for Safe-CEG experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

import yaml

from src.core.query import JoinPredicate, Query, RelationStatistics
from src.utils.graph_io import query_from_payload, query_to_payload
from src.utils.join_counting import _join_variable_map, count_join
from src.utils.types import AttributeName, Relation, RelationName

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class WorkloadQuery:
    """One synthetic query paired with its target cardinality and family."""

    query: Query
    true_cardinality: float
    family: str


def generate_synthetic_workload(
    family: str,
    query_count: int,
    relation_count: int,
    seed: int,
) -> list[WorkloadQuery]:
    """Generate a deterministic synthetic workload for one query family."""
    if query_count <= 0:
        raise ValueError("Query count must be positive")
    if relation_count < 2:
        raise ValueError("Relation count must be at least 2")

    random = Random(seed)
    workload: list[WorkloadQuery] = []
    for query_index in range(query_count):
        if family == "path":
            workload.append(
                _build_path_workload_query(query_index, relation_count, random)
            )
        elif family == "star":
            workload.append(
                _build_star_workload_query(query_index, relation_count, random)
            )
        elif family == "cycle":
            workload.append(
                _build_cycle_workload_query(query_index, relation_count, random)
            )
        else:
            raise ValueError(f"Unsupported workload family: {family!r}")
    return workload


def save_workload_corpus(workload: list[WorkloadQuery], path: Path) -> None:
    """Persist a workload corpus to JSON or YAML."""
    payload = {
        "queries": [
            {
                "family": workload_query.family,
                "true_cardinality": workload_query.true_cardinality,
                "query": query_to_payload(workload_query.query),
            }
            for workload_query in workload
        ]
    }

    if path.suffix == ".json":
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return
    if path.suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        return
    raise ValueError(f"Unsupported workload corpus format: {path.suffix}")


def load_workload_corpus(path: Path) -> list[WorkloadQuery]:
    """Load a persisted workload corpus from JSON or YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Workload corpus does not exist: {path}")

    text = path.read_text()
    if path.suffix == ".json":
        payload = json.loads(text)
    elif path.suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported workload corpus format: {path.suffix}")

    if not isinstance(payload, dict):
        raise ValueError("Workload corpus root must be a mapping")
    queries_block = payload.get("queries")
    if not isinstance(queries_block, list):
        raise ValueError("Workload corpus must contain a 'queries' list")

    workload: list[WorkloadQuery] = []
    for query_payload in queries_block:
        if not isinstance(query_payload, dict):
            raise ValueError("Each workload query entry must be a mapping")
        family = query_payload.get("family")
        true_cardinality = query_payload.get("true_cardinality")
        raw_query = query_payload.get("query")
        if not isinstance(family, str) or not family:
            raise ValueError("Workload query family must be a non-empty string")
        if not isinstance(true_cardinality, (int, float)) or true_cardinality <= 0:
            raise ValueError("Workload true_cardinality must be positive")
        if not isinstance(raw_query, dict):
            raise ValueError("Workload query payload must be a mapping")
        workload.append(
            WorkloadQuery(
                query=query_from_payload(raw_query),
                true_cardinality=float(true_cardinality),
                family=family,
            )
        )
    return workload


# ---------------------------------------------------------------------------
# Data-grounded synthetic generation
#
# Join-key columns are sampled from shared per-edge domains so joins actually
# produce matches; relation statistics are DERIVED from the generated columns,
# and the true cardinality is computed by EXECUTING the join (count_join).
# Nothing here calls the estimators under evaluation, so labels are independent
# evidence (addresses the estimator-derived-label circularity).
# ---------------------------------------------------------------------------

# Per-family generation profile: (cardinality range, join-key skew, domain
# factor). The domain factor sets each join-variable domain relative to the
# mean cardinality of the relations touching it; smaller domains -> denser
# joins (lower selectivity), larger domains -> sparser joins (higher
# selectivity). Skew controls max-frequency on join keys.
_PROFILE_CARD: dict[str, tuple[int, int]] = {
    "path": (500, 2000),
    "star": (200, 1000),
    "cycle": (300, 1200),
}
_PROFILE_SKEW: dict[str, float] = {"path": 0.0, "star": 0.0, "cycle": 0.6}
# Within-relation join-key correlation (fraction of rows whose later join keys
# are tied to the first). This BREAKS the per-edge independence assumption that
# the optimistic CEG relies on, so optimistic estimates overestimate (as on real
# data) and config selection / the safety certificate carry genuine value.
# Cycles correlate most strongly (cycle-closing correlation is the canonical
# independence violation); true cardinalities remain executed-join labels.
_PROFILE_CORRELATION: dict[str, float] = {"path": 0.4, "star": 0.4, "cycle": 0.85}
_PROFILE_DOMAIN_FACTOR: dict[str, float] = {
    "path": 1.5,
    "star": 0.5,
    "cycle": 0.8,
}


def _sample_column(
    domain_size: int, n_rows: int, skew: float, random: Random
) -> list[str]:
    """Sample ``n_rows`` join-key values from ``[0, domain_size)``."""
    domain_size = max(1, domain_size)
    if skew <= 0.0:
        return [str(random.randrange(domain_size)) for _ in range(n_rows)]
    weights = [1.0 / ((i + 1) ** skew) for i in range(domain_size)]
    return [
        str(v) for v in random.choices(range(domain_size), weights=weights, k=n_rows)
    ]


def _stats_from_columns(
    name: RelationName,
    join_columns: dict[str, list[str]],
    cardinality: int,
) -> RelationStatistics:
    """Derive relation statistics from generated join-key columns."""
    from collections import Counter

    distinct_counts: dict[AttributeName, int] = {}
    max_frequencies: dict[AttributeName, int] = {}
    attributes: set[AttributeName] = set()
    for attr, values in join_columns.items():
        counter = Counter(values)
        attribute = AttributeName(attr)
        distinct_counts[attribute] = len(counter)
        max_frequencies[attribute] = max(counter.values()) if counter else 1
        attributes.add(attribute)
    # A near-key payload column (does not participate in joins).
    payload = AttributeName("payload")
    distinct_counts[payload] = cardinality
    max_frequencies[payload] = 1
    attributes.add(payload)
    return RelationStatistics(
        relation=Relation(
            name=name, attributes=frozenset(attributes), cardinality=cardinality
        ),
        attribute_distinct_counts=distinct_counts,
        attribute_max_frequencies=max_frequencies,
    )


def _family_structure(
    query_index: int, relation_count: int, family: str
) -> tuple[frozenset[JoinPredicate], dict[RelationName, set[str]]]:
    """Return the join predicates and per-relation join attributes for a family."""

    def rel(i: int) -> RelationName:
        return RelationName(f"R{query_index}_{i}")

    joins: set[JoinPredicate] = set()
    attrs: dict[RelationName, set[str]] = {rel(i): set() for i in range(relation_count)}
    if family == "path":
        for i in range(relation_count - 1):
            joins.add(
                JoinPredicate(
                    rel(i),
                    AttributeName("k_right"),
                    rel(i + 1),
                    AttributeName("k_left"),
                )
            )
            attrs[rel(i)].add("k_right")
            attrs[rel(i + 1)].add("k_left")
    elif family == "star":
        for j in range(1, relation_count):
            joins.add(
                JoinPredicate(
                    rel(0), AttributeName(f"k_{j}"), rel(j), AttributeName("k_left")
                )
            )
            attrs[rel(0)].add(f"k_{j}")
            attrs[rel(j)].add("k_left")
    elif family == "cycle":
        for i in range(relation_count):
            j = (i + 1) % relation_count
            joins.add(
                JoinPredicate(
                    rel(i), AttributeName("k_right"), rel(j), AttributeName("k_left")
                )
            )
            attrs[rel(i)].add("k_right")
            attrs[rel(j)].add("k_left")
    else:
        raise ValueError(f"Unsupported workload family: {family!r}")
    return frozenset(joins), attrs


def _generate_query(
    query_index: int, relation_count: int, family: str, random: Random
) -> WorkloadQuery:
    """Generate one query with data-derived stats and an executed-join label."""
    card_lo, card_hi = _PROFILE_CARD[family]
    base_skew = _PROFILE_SKEW[family]
    domain_factor = _PROFILE_DOMAIN_FACTOR[family]
    correlation = _PROFILE_CORRELATION[family]

    joins, rel_attrs = _family_structure(query_index, relation_count, family)

    # Per-relation cardinality and skew (the star center is skewed and large).
    rel_card: dict[RelationName, int] = {}
    rel_skew: dict[RelationName, float] = {}
    for i in range(relation_count):
        name = RelationName(f"R{query_index}_{i}")
        if family == "star" and i == 0:
            rel_card[name] = random.randint(2000, 5000)
            rel_skew[name] = 1.2
        else:
            rel_card[name] = random.randint(card_lo, card_hi)
            rel_skew[name] = base_skew

    # Shared domain size per join variable (so joined columns actually match).
    var_map = _join_variable_map(joins)
    var_cards: dict[int, list[int]] = {}
    for (rel_name, _attr), var_id in var_map.items():
        var_cards.setdefault(var_id, []).append(rel_card[RelationName(rel_name)])
    var_domain: dict[int, int] = {
        var_id: max(2, round(sum(cs) / len(cs) * domain_factor))
        for var_id, cs in var_cards.items()
    }

    # Generate join-key columns, derive stats, and assemble rows for counting.
    rows_by_relation: dict[str, list[dict[str, str]]] = {}
    relations: dict[RelationName, RelationStatistics] = {}
    for i in range(relation_count):
        name = RelationName(f"R{query_index}_{i}")
        card = rel_card[name]
        attr_list = sorted(rel_attrs[name])
        columns: dict[str, list[str]] = {}
        for attr in attr_list:
            var_id = var_map[(name, AttributeName(attr))]
            columns[attr] = _sample_column(
                var_domain[var_id], card, rel_skew[name], random
            )
        # Tie later join keys to the first for a fraction `correlation` of rows,
        # breaking per-edge independence so optimistic estimates overestimate.
        if correlation > 0.0 and len(attr_list) >= 2:
            first_col = columns[attr_list[0]]
            for other in attr_list[1:]:
                other_domain = var_domain[var_map[(name, AttributeName(other))]]
                other_col = columns[other]
                for row in range(card):
                    if random.random() < correlation:
                        other_col[row] = str(int(first_col[row]) % other_domain)
        rows_by_relation[str(name)] = [
            {attr: columns[attr][row] for attr in columns} for row in range(card)
        ]
        relations[name] = _stats_from_columns(name, columns, card)

    query = Query(relations=relations, joins=joins)
    true_cardinality = count_join(rows_by_relation, query.relation_names, joins)
    return WorkloadQuery(query=query, true_cardinality=true_cardinality, family=family)


def _build_path_workload_query(
    query_index: int, relation_count: int, random: Random
) -> WorkloadQuery:
    return _generate_query(query_index, relation_count, "path", random)


def _build_star_workload_query(
    query_index: int, relation_count: int, random: Random
) -> WorkloadQuery:
    return _generate_query(query_index, relation_count, "star", random)


def _build_cycle_workload_query(
    query_index: int, relation_count: int, random: Random
) -> WorkloadQuery:
    return _generate_query(query_index, relation_count, "cycle", random)
