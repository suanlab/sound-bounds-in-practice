"""Query and statistics data structures for Safe-CEG."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from src.utils.types import AttributeName, Relation, RelationName


@dataclass(frozen=True)
class JoinPredicate:
    """A join predicate connecting two relation attributes."""

    left_relation: RelationName
    left_attribute: AttributeName
    right_relation: RelationName
    right_attribute: AttributeName

    def relation_names(self) -> frozenset[RelationName]:
        """Return the relation names referenced by the predicate."""
        return frozenset((self.left_relation, self.right_relation))

    def references_relation(self, relation_name: RelationName) -> bool:
        """Return whether the predicate references a relation."""
        return relation_name in self.relation_names()

    def other_relation(self, relation_name: RelationName) -> RelationName:
        """Return the other relation participating in the predicate."""
        if relation_name == self.left_relation:
            return self.right_relation
        if relation_name == self.right_relation:
            return self.left_relation
        raise ValueError(f"Relation {relation_name!r} is not part of the predicate")


@dataclass(frozen=True)
class RelationStatistics:
    """Statistics required to build CEG edge weights for one relation."""

    relation: Relation
    attribute_distinct_counts: Mapping[AttributeName, int]
    attribute_max_frequencies: Mapping[AttributeName, int]
    attribute_l2_norms: Mapping[AttributeName, float] | None = None

    def distinct_count(self, attribute_name: AttributeName) -> int:
        """Return the distinct count for an attribute."""
        if attribute_name not in self.attribute_distinct_counts:
            raise ValueError(
                f"Missing distinct-count statistic for attribute {attribute_name!r}"
            )
        value = self.attribute_distinct_counts[attribute_name]
        if value <= 0:
            raise ValueError(
                f"Distinct count must be positive for attribute {attribute_name!r}"
            )
        return value

    def max_frequency(self, attribute_name: AttributeName) -> int:
        """Return the maximum frequency for an attribute."""
        if attribute_name not in self.attribute_max_frequencies:
            raise ValueError(
                f"Missing max-frequency statistic for attribute {attribute_name!r}"
            )
        value = self.attribute_max_frequencies[attribute_name]
        if value <= 0:
            raise ValueError(
                f"Max frequency must be positive for attribute {attribute_name!r}"
            )
        return value

    def l2_norm(self, attribute_name: AttributeName) -> float:
        """Return the L2 norm of the degree sequence for an attribute.

        If exact L2 norms are stored, returns them directly. Otherwise,
        computes a conservative upper bound from cardinality, distinct
        count, and max frequency using Cauchy-Schwarz:
        ``L2 <= sqrt(f_max² + (|R| - f_max)² / (d - 1))``
        """
        if (
            self.attribute_l2_norms is not None
            and attribute_name in self.attribute_l2_norms
        ):
            return self.attribute_l2_norms[attribute_name]
        # Conservative approximation
        from math import sqrt

        n = self.relation.cardinality
        d = self.distinct_count(attribute_name)
        f_max = self.max_frequency(attribute_name)
        if d <= 1:
            return float(n)
        remainder_sq = (n - f_max) ** 2 / (d - 1)
        return sqrt(f_max**2 + remainder_sq)


@dataclass(frozen=True)
class Query:
    """A join query with relation-level statistics."""

    relations: Mapping[RelationName, RelationStatistics]
    joins: frozenset[JoinPredicate]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def relation_names(self) -> frozenset[RelationName]:
        """Return all relation names in the query."""
        return frozenset(self.relations)

    @property
    def all_attributes(self) -> frozenset[AttributeName]:
        """Return the union of relation attributes in the query."""
        attributes: set[AttributeName] = set()
        for relation_statistics in self.relations.values():
            attributes.update(relation_statistics.relation.attributes)
        return frozenset(attributes)

    def validate(self) -> None:
        """Validate internal consistency of query relations and joins."""
        if not self.relations:
            raise ValueError("Query must contain at least one relation")

        for relation_name, relation_statistics in self.relations.items():
            if relation_statistics.relation.name != relation_name:
                raise ValueError(
                    "Relation statistics key must match relation name: "
                    f"{relation_name!r} != {relation_statistics.relation.name!r}"
                )

            for attribute_name in relation_statistics.attribute_distinct_counts:
                if attribute_name not in relation_statistics.relation.attributes:
                    raise ValueError(
                        "Distinct-count statistic references an unknown attribute: "
                        f"{attribute_name!r}"
                    )

            for attribute_name in relation_statistics.attribute_max_frequencies:
                if attribute_name not in relation_statistics.relation.attributes:
                    raise ValueError(
                        "Max-frequency statistic references an unknown attribute: "
                        f"{attribute_name!r}"
                    )

        for join in self.joins:
            if join.left_relation not in self.relations:
                raise ValueError(f"Unknown relation in join: {join.left_relation!r}")
            if join.right_relation not in self.relations:
                raise ValueError(f"Unknown relation in join: {join.right_relation!r}")
            if join.left_relation == join.right_relation:
                raise ValueError(
                    "Self-joins are not supported in the current query model"
                )

            left_relation = self.relations[join.left_relation].relation
            right_relation = self.relations[join.right_relation].relation

            if join.left_attribute not in left_relation.attributes:
                raise ValueError(
                    "Join references an unknown left attribute: "
                    f"{join.left_relation}.{join.left_attribute}"
                )
            if join.right_attribute not in right_relation.attributes:
                raise ValueError(
                    "Join references an unknown right attribute: "
                    f"{join.right_relation}.{join.right_attribute}"
                )

        if len(self.relations) > 1 and not self.is_connected_subset(
            self.relation_names
        ):
            raise ValueError("Query join graph must be connected")

    def relation_statistics(self, relation_name: RelationName) -> RelationStatistics:
        """Return statistics for a named relation."""
        if relation_name not in self.relations:
            raise ValueError(f"Unknown relation: {relation_name!r}")
        return self.relations[relation_name]

    def joins_touching_subset(
        self, subset: frozenset[RelationName]
    ) -> frozenset[JoinPredicate]:
        """Return joins incident to at least one relation in a subset."""
        return frozenset(
            join
            for join in self.joins
            if join.left_relation in subset or join.right_relation in subset
        )

    def induced_joins(
        self, subset: frozenset[RelationName]
    ) -> frozenset[JoinPredicate]:
        """Return joins whose endpoints are fully contained in a subset."""
        return frozenset(
            join
            for join in self.joins
            if join.left_relation in subset and join.right_relation in subset
        )

    def boundary_joins(
        self,
        source_subset: frozenset[RelationName],
        target_subset: frozenset[RelationName],
    ) -> frozenset[JoinPredicate]:
        """Return joins crossing from source relations to added target relations."""
        if not source_subset <= target_subset:
            raise ValueError("Source subset must be contained in target subset")

        added_relations = target_subset - source_subset
        return frozenset(
            join
            for join in self.joins
            if (
                join.left_relation in source_subset
                and join.right_relation in added_relations
            )
            or (
                join.right_relation in source_subset
                and join.left_relation in added_relations
            )
        )

    def subset_attributes(
        self, subset: frozenset[RelationName]
    ) -> frozenset[AttributeName]:
        """Return the union of attributes for a relation subset."""
        attributes: set[AttributeName] = set()
        for relation_name in subset:
            attributes.update(
                self.relation_statistics(relation_name).relation.attributes
            )
        return frozenset(attributes)

    def neighbors(self, relation_name: RelationName) -> frozenset[RelationName]:
        """Return join-neighbors for a relation."""
        adjacent: set[RelationName] = set()
        for join in self.joins:
            if join.left_relation == relation_name:
                adjacent.add(join.right_relation)
            elif join.right_relation == relation_name:
                adjacent.add(join.left_relation)
        return frozenset(adjacent)

    def is_connected_subset(self, subset: frozenset[RelationName]) -> bool:
        """Return whether the induced relation subset is connected."""
        if not subset:
            return True

        start = next(iter(subset))
        visited: set[RelationName] = {start}
        frontier: deque[RelationName] = deque([start])

        while frontier:
            current = frontier.popleft()
            for neighbor in self.neighbors(current):
                if neighbor in subset and neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)

        return visited == set(subset)

    def connected_subsets(self) -> tuple[frozenset[RelationName], ...]:
        """Return every connected relation subset in increasing cardinality order."""
        relation_names = tuple(sorted(self.relation_names))
        subsets: list[frozenset[RelationName]] = []
        total_relations = len(relation_names)

        for mask in range(1, 1 << total_relations):
            subset = frozenset(
                relation_name
                for index, relation_name in enumerate(relation_names)
                if mask & (1 << index)
            )
            if self.is_connected_subset(subset):
                subsets.append(subset)

        subsets.sort(key=lambda subset: (len(subset), tuple(sorted(subset))))
        return tuple(subsets)


def iter_supersets(
    subsets: Iterable[frozenset[RelationName]],
    source_subset: frozenset[RelationName],
) -> tuple[frozenset[RelationName], ...]:
    """Return every strict superset of a source subset from an iterable."""
    supersets = [subset for subset in subsets if source_subset < subset]
    supersets.sort(key=lambda subset: (len(subset), tuple(sorted(subset))))
    return tuple(supersets)
