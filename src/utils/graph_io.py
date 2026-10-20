"""I/O helpers for Safe-CEG query specifications and statistics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import yaml

from src.core.query import JoinPredicate, Query, RelationStatistics
from src.utils.types import AttributeName, Relation, RelationName

if TYPE_CHECKING:
    from pathlib import Path


def load_query_spec(path: Path) -> Query:
    """Load a query specification from JSON or YAML into a Query."""
    raw_spec = _load_mapping(path)
    return query_from_payload(raw_spec)


def query_from_payload(raw_spec: Mapping[str, object]) -> Query:
    """Construct a Query from an in-memory payload mapping."""
    relations_block = raw_spec.get("relations")
    joins_block = raw_spec.get("joins")

    if not isinstance(relations_block, Sequence) or isinstance(relations_block, str):
        raise ValueError("Query specification must contain a 'relations' sequence")
    if not isinstance(joins_block, Sequence) or isinstance(joins_block, str):
        raise ValueError("Query specification must contain a 'joins' sequence")

    relations = {
        relation_statistics.relation.name: relation_statistics
        for relation_statistics in (
            _parse_relation_statistics(relation_spec)
            for relation_spec in relations_block
        )
    }
    joins = frozenset(_parse_join(join_spec) for join_spec in joins_block)
    return Query(relations=relations, joins=joins)


def dump_query_spec(query: Query, path: Path) -> None:
    """Persist a Query specification to JSON or YAML."""
    payload = query_to_payload(query)

    if path.suffix == ".json":
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return
    if path.suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        return
    raise ValueError(f"Unsupported query specification format: {path.suffix}")


def query_to_payload(query: Query) -> dict[str, object]:
    """Serialize a Query to an in-memory payload mapping."""
    return {
        "relations": [
            {
                "name": relation_name,
                "attributes": sorted(relation_statistics.relation.attributes),
                "cardinality": relation_statistics.relation.cardinality,
                "distinct_counts": dict(relation_statistics.attribute_distinct_counts),
                "max_frequencies": dict(relation_statistics.attribute_max_frequencies),
            }
            for relation_name, relation_statistics in sorted(query.relations.items())
        ],
        "joins": [
            {
                "left_relation": join.left_relation,
                "left_attribute": join.left_attribute,
                "right_relation": join.right_relation,
                "right_attribute": join.right_attribute,
            }
            for join in sorted(
                query.joins,
                key=lambda join: (
                    join.left_relation,
                    join.left_attribute,
                    join.right_relation,
                    join.right_attribute,
                ),
            )
        ],
    }


def _load_mapping(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Query specification does not exist: {path}")

    text = path.read_text()
    if path.suffix == ".json":
        payload = json.loads(text)
    elif path.suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported query specification format: {path.suffix}")

    if not isinstance(payload, Mapping):
        raise ValueError("Query specification root must be a mapping")
    return payload


def _parse_relation_statistics(relation_spec: object) -> RelationStatistics:
    if not isinstance(relation_spec, Mapping):
        raise ValueError("Each relation specification must be a mapping")

    name = _require_string(relation_spec, "name")
    cardinality = _require_positive_int(relation_spec, "cardinality")
    attributes_block = relation_spec.get("attributes")
    if not isinstance(attributes_block, Sequence) or isinstance(attributes_block, str):
        raise ValueError("Relation attributes must be a sequence of attribute names")
    attributes = frozenset(
        _ensure_attribute_name(attribute) for attribute in attributes_block
    )

    distinct_counts_block = relation_spec.get("distinct_counts")
    max_frequencies_block = relation_spec.get("max_frequencies")
    if not isinstance(distinct_counts_block, Mapping):
        raise ValueError("Relation must contain a 'distinct_counts' mapping")
    if not isinstance(max_frequencies_block, Mapping):
        raise ValueError("Relation must contain a 'max_frequencies' mapping")

    distinct_counts = _parse_int_mapping(distinct_counts_block)
    max_frequencies = _parse_int_mapping(max_frequencies_block)

    return RelationStatistics(
        relation=Relation(
            name=RelationName(name),
            attributes=attributes,
            cardinality=cardinality,
        ),
        attribute_distinct_counts=distinct_counts,
        attribute_max_frequencies=max_frequencies,
    )


def _parse_join(join_spec: object) -> JoinPredicate:
    if not isinstance(join_spec, Mapping):
        raise ValueError("Each join specification must be a mapping")

    return JoinPredicate(
        left_relation=RelationName(_require_string(join_spec, "left_relation")),
        left_attribute=AttributeName(_require_string(join_spec, "left_attribute")),
        right_relation=RelationName(_require_string(join_spec, "right_relation")),
        right_attribute=AttributeName(_require_string(join_spec, "right_attribute")),
    )


def _parse_int_mapping(
    raw_mapping: Mapping[object, object],
) -> dict[AttributeName, int]:
    parsed: dict[AttributeName, int] = {}
    for key, value in raw_mapping.items():
        if not isinstance(key, str):
            raise ValueError("Statistic mapping keys must be strings")
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Statistic mapping values must be positive integers")
        parsed[AttributeName(key)] = value
    return parsed


def _require_string(raw_mapping: Mapping[str, object], key: str) -> str:
    value = raw_mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Field {key!r} must be a non-empty string")
    return value


def _require_positive_int(raw_mapping: Mapping[str, object], key: str) -> int:
    value = raw_mapping.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Field {key!r} must be a positive integer")
    return value


def _ensure_attribute_name(value: object) -> AttributeName:
    if not isinstance(value, str) or not value:
        raise ValueError("Attribute names must be non-empty strings")
    return AttributeName(value)
