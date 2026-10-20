"""Run Safe-CEG evaluation on JOB-light real dataset fixtures."""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import torch

from src.core.ceg import pessimistic_bound
from src.core.query import JoinPredicate, Query
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.models.features import extract_features
from src.models.gnn import SafeCEGGNN
from src.training.dataset import SafeCEGDataset, build_training_example
from src.training.evaluate import evaluate_model
from src.training.train import TrainingConfig, train_model
from src.utils.data_loader import (
    DatasetCatalog,
    catalog_to_query_statistics,
    load_dataset_catalog,
)
from src.utils.join_counting import count_join
from src.utils.metrics import q_error

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# JOB-light join schema: which relations join on which attributes
JOIN_SCHEMA: list[tuple[str, str, str, str]] = [
    # title-based joins (title_id)
    ("title", "title_id", "cast_info", "title_id"),
    ("title", "title_id", "movie_keyword", "title_id"),
    ("title", "title_id", "movie_companies", "title_id"),
    ("title", "title_id", "movie_info", "title_id"),
    ("cast_info", "title_id", "movie_keyword", "title_id"),
    ("cast_info", "title_id", "movie_companies", "title_id"),
    ("cast_info", "title_id", "movie_info", "title_id"),
    ("movie_keyword", "title_id", "movie_companies", "title_id"),
    ("movie_keyword", "title_id", "movie_info", "title_id"),
    ("movie_companies", "title_id", "movie_info", "title_id"),
    # person-based joins (person_id)
    ("cast_info", "person_id", "person_info", "person_id"),
    # company-based joins (company_id)
    ("movie_companies", "company_id", "company_name", "company_id"),
    # keyword-based joins (keyword_id)
    ("movie_keyword", "keyword_id", "keyword", "keyword_id"),
]


def _load_relation_rows(
    data_dir: Path, relation_names: frozenset[str]
) -> dict[str, list[dict[str, str]]]:
    """Load raw CSV rows for each relation in the query."""
    import csv

    rows_by_relation: dict[str, list[dict[str, str]]] = {}
    for rname in relation_names:
        with (data_dir / f"{rname}.csv").open(newline="") as f:
            rows_by_relation[rname] = list(csv.DictReader(f))
    return rows_by_relation


def compute_true_cardinality(
    data_dir: Path,
    relation_names: frozenset[str],
    joins: frozenset[JoinPredicate],
) -> float:
    """Exact join cardinality over the full relations (see ``count_join``)."""
    rows = _load_relation_rows(data_dir, relation_names)
    return count_join(rows, relation_names, joins)


# A selection predicate ``(relation, attribute, op, value)``. Real JOB queries
# carry such filters in addition to joins; the harness applies them per relation
# before counting, so filtered true cardinalities match the SQL semantics.
SelectionFilter = tuple[str, str, str, str]


def _as_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _passes(cell: str, op: str, target: str) -> bool:
    """Evaluate one comparison, using numeric order when both sides are numbers."""
    if op == "=":
        return cell == target
    if op == "!=":
        return cell != target
    left, right = _as_number(cell), _as_number(target)
    if left is None or right is None:
        left_s: object = cell
        right_s: object = target
    else:
        left_s, right_s = left, right
    if op == "<":
        return bool(left_s < right_s)  # type: ignore[operator]
    if op == "<=":
        return bool(left_s <= right_s)  # type: ignore[operator]
    if op == ">":
        return bool(left_s > right_s)  # type: ignore[operator]
    if op == ">=":
        return bool(left_s >= right_s)  # type: ignore[operator]
    raise ValueError(f"Unsupported filter operator: {op!r}")


def apply_selection_filters(
    rows_by_relation: dict[str, list[dict[str, str]]],
    filters: list[SelectionFilter],
) -> dict[str, list[dict[str, str]]]:
    """Return a copy of the rows with per-relation selection predicates applied."""
    filtered = {rel: list(rows) for rel, rows in rows_by_relation.items()}
    for relation, attribute, op, value in filters:
        if relation not in filtered:
            continue
        filtered[relation] = [
            row for row in filtered[relation] if _passes(row[attribute], op, value)
        ]
    return filtered


def compute_true_cardinality_filtered(
    data_dir: Path,
    relation_names: frozenset[str],
    joins: frozenset[JoinPredicate],
    filters: list[SelectionFilter],
) -> float:
    """Exact cardinality of a select-project-join query (joins + filters).

    Filters are applied per relation before the count-based hash join, so the
    result matches SQL ``WHERE`` semantics. This is what real JOB queries need
    (they combine joins with selection predicates).
    """
    rows = _load_relation_rows(data_dir, relation_names)
    rows = apply_selection_filters(rows, filters)
    return count_join(rows, relation_names, joins)


def build_job_light_queries(
    catalog: DatasetCatalog, data_dir: Path
) -> list[tuple[Query, float]]:
    """Build join queries from the JOB-light schema with true cardinalities."""
    all_relations = sorted(catalog.relation_names())
    queries: list[tuple[Query, float]] = []

    # Generate 2-way through 5-way joins
    for size in (2, 3, 4, 5):
        for relation_combo in combinations(all_relations, size):
            relation_set = frozenset(relation_combo)
            # Find applicable joins
            applicable = []
            for left_r, left_a, right_r, right_a in JOIN_SCHEMA:
                if left_r in relation_set and right_r in relation_set:
                    applicable.append(JoinPredicate(left_r, left_a, right_r, right_a))
            if not applicable:
                continue

            join_set = frozenset(applicable)
            stats = catalog_to_query_statistics(catalog, relation_set)

            try:
                query = Query(relations=stats, joins=join_set)
            except ValueError:
                continue

            true_card = compute_true_cardinality(data_dir, relation_set, join_set)
            queries.append((query, true_card))

    return queries


def _run_single_seed(
    query_pairs: list[tuple[Query, float]],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    """Run real data evaluation for one seed."""
    torch.manual_seed(seed)
    estimator = BaselineCEGEstimator()

    examples = []
    for query, true_card in query_pairs:
        examples.append(build_training_example(query, true_card, estimator))

    if len(examples) < 4:
        return {"error": "too few examples"}

    # Shuffle with seed for different train/test splits
    import random as rng_mod

    rng = rng_mod.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    split = max(2, int(len(shuffled) * 0.7))
    train_dataset = SafeCEGDataset(shuffled[:split])
    eval_dataset = SafeCEGDataset(shuffled[split:])

    features = extract_features(train_dataset[0].query)
    model = SafeCEGGNN(
        node_feature_dim=features.node_features.shape[1],
        global_feature_dim=features.global_features.shape[1],
    )
    train_model(model, train_dataset, TrainingConfig(epochs=args.epochs, tau=args.tau))

    summary = evaluate_model(model, eval_dataset, tau=args.tau)

    # Pessimistic and oracle baselines
    pes_qerrors = []
    oracle_qerrors = []
    for i in range(len(eval_dataset)):
        ex = eval_dataset[i]
        bound = pessimistic_bound(ex.query)
        pes_qerrors.append(q_error(bound, ex.true_cardinality))
        oracle_est = estimator.oracle_best_config(ex.query, ex.true_cardinality)
        oracle_qerrors.append(q_error(oracle_est.estimate, ex.true_cardinality))

    from src.utils.metrics import summarize_q_errors

    summary["pes_mean"] = summarize_q_errors(pes_qerrors)["mean"]
    summary["oracle_mean"] = summarize_q_errors(oracle_qerrors)["mean"]
    summary["train_size"] = len(train_dataset)
    summary["eval_size"] = len(eval_dataset)
    return summary


def main() -> None:
    """Run Safe-CEG on real data fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/job_light"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/real_data/job_results.json"),
    )
    args = parser.parse_args()

    dataset_name = args.data_dir.name
    logger.info(f"Loading {dataset_name} dataset...")
    catalog = load_dataset_catalog(args.data_dir, dataset_name)
    logger.info(f"Relations: {sorted(catalog.relation_names())}")

    logger.info("Building queries with true cardinalities...")
    query_pairs = build_job_light_queries(catalog, args.data_dir)
    logger.info(f"Generated {len(query_pairs)} join queries")

    # Per-query analysis (once, before seeds)
    estimator = BaselineCEGEstimator()
    logger.info("\n=== Per-Query Analysis ===")
    for qi, (query, true_card) in enumerate(query_pairs[:15]):
        bound = pessimistic_bound(query)
        all_est = estimator.estimate_all_configs(query)
        best_n = min(all_est, key=lambda n: q_error(all_est[n].estimate, true_card))
        best_qe = q_error(all_est[best_n].estimate, true_card)
        pes_qe = q_error(bound, true_card)
        n_rels = len(query.relation_names)
        valid = "Y" if true_card <= bound else "N"
        logger.info(
            f"  Q{qi:>3d} rels={n_rels} true={true_card:>8.0f}"
            f" bound={bound:>12.0f} oracle_qe={best_qe:>6.2f}"
            f" pes_qe={pes_qe:>8.2f} best={best_n:>16s} valid={valid}"
        )

    seed_list = [args.seed + i * 1000 for i in range(args.seeds)]
    all_seed_results: list[dict[str, object]] = []

    for i, seed in enumerate(seed_list):
        logger.info(f"\n--- Seed {i + 1}/{len(seed_list)} (seed={seed}) ---")
        result = _run_single_seed(query_pairs, args, seed)
        all_seed_results.append(result)
        logger.info(
            f"  q-error={result.get('mean', 'N/A')}"
            f"  config_acc={result.get('config_accuracy', 'N/A')}"
            f"  pes={result.get('pes_mean', 'N/A')}"
            f"  oracle={result.get('oracle_mean', 'N/A')}"
        )

    from src.utils.metrics import aggregate_multi_seed, format_mean_std

    agg_keys = [
        "mean",
        "config_accuracy",
        "fallback_rate",
        "safety_violation_rate",
        "pes_mean",
        "oracle_mean",
    ]
    agg = aggregate_multi_seed(all_seed_results, agg_keys)

    logger.info(f"\n=== {dataset_name} Summary ({len(seed_list)} seeds) ===")
    for key in agg_keys:
        if key in agg:
            d = agg[key]
            logger.info(f"  {key}: {format_mean_std(d['mean'], d['std'])}")

    results = {
        "dataset": dataset_name,
        "num_queries": len(query_pairs),
        "n_seeds": len(seed_list),
        "aggregated": agg,
        "per_seed": all_seed_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
