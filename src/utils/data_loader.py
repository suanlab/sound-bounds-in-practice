"""Loaders for real graph datasets and statistics extraction."""

from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.query import RelationStatistics
from src.utils.types import AttributeName, Relation, RelationName

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedRelation:
    """A relation loaded from disk with raw rows and computed statistics."""

    statistics: RelationStatistics
    row_count: int


@dataclass
class DatasetCatalog:
    """A collection of loaded relations from one graph dataset directory."""

    name: str
    relations: dict[RelationName, LoadedRelation] = field(default_factory=dict)

    def relation_names(self) -> frozenset[RelationName]:
        """Return all loaded relation names."""
        return frozenset(self.relations)

    def get_statistics(self, relation_name: RelationName) -> RelationStatistics:
        """Return statistics for a named relation."""
        if relation_name not in self.relations:
            raise ValueError(f"Unknown relation in catalog: {relation_name!r}")
        return self.relations[relation_name].statistics


def load_dataset_catalog(dataset_dir: Path, dataset_name: str) -> DatasetCatalog:
    """Load all CSV/TSV relation files from a dataset directory."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    catalog = DatasetCatalog(name=dataset_name)
    relation_files = sorted(dataset_dir.glob("*.csv")) + sorted(
        dataset_dir.glob("*.tsv")
    )
    if not relation_files:
        raise ValueError(f"No CSV/TSV relation files found in {dataset_dir}")

    for relation_file in relation_files:
        relation_name = RelationName(relation_file.stem)
        loaded = load_relation_file(relation_file, relation_name)
        catalog.relations[relation_name] = loaded
        logger.info(
            "Loaded relation %s: %d rows, %d attributes",
            relation_name,
            loaded.row_count,
            len(loaded.statistics.relation.attributes),
        )

    return catalog


def load_relation_file(
    file_path: Path,
    relation_name: RelationName,
) -> LoadedRelation:
    """Load one CSV/TSV file as a relation with computed statistics."""
    delimiter = "\t" if file_path.suffix == ".tsv" else ","
    rows: list[dict[str, str]] = []

    with file_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Relation file has no header: {file_path}")
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Relation file is empty: {file_path}")

    attribute_names = frozenset(
        AttributeName(column_name) for column_name in (reader.fieldnames or [])
    )
    cardinality = len(rows)

    distinct_counts: dict[AttributeName, int] = {}
    max_frequencies: dict[AttributeName, int] = {}

    l2_norms: dict[AttributeName, float] = {}
    for attribute_name in attribute_names:
        column_values = [row[str(attribute_name)] for row in rows]
        value_counter: Counter[str] = Counter(column_values)
        distinct_counts[attribute_name] = len(value_counter)
        max_frequencies[attribute_name] = max(value_counter.values())
        l2_norms[attribute_name] = sum(c * c for c in value_counter.values()) ** 0.5

    return LoadedRelation(
        statistics=RelationStatistics(
            relation=Relation(
                name=relation_name,
                attributes=attribute_names,
                cardinality=cardinality,
            ),
            attribute_distinct_counts=distinct_counts,
            attribute_max_frequencies=max_frequencies,
            attribute_l2_norms=l2_norms,
        ),
        row_count=cardinality,
    )


def catalog_to_query_statistics(
    catalog: DatasetCatalog,
    relation_names: frozenset[RelationName],
) -> dict[RelationName, RelationStatistics]:
    """Extract statistics for a subset of relations from a catalog."""
    missing = relation_names - catalog.relation_names()
    if missing:
        raise ValueError(f"Relations not found in catalog: {sorted(missing)}")
    return {
        relation_name: catalog.get_statistics(relation_name)
        for relation_name in relation_names
    }
