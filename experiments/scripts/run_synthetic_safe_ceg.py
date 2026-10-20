"""Run config-driven synthetic Safe-CEG experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.models.features import extract_features
from src.models.gnn import SafeCEGGNN
from src.training.dataset import build_dataset_from_workload
from src.training.evaluate import evaluate_model
from src.training.train import TrainingConfig, train_model
from src.training.workload import generate_synthetic_workload, load_workload_corpus

if TYPE_CHECKING:
    from src.training.workload import WorkloadQuery


@dataclass(frozen=True)
class SyntheticExperimentConfig:
    """Configuration for a synthetic Safe-CEG experiment run."""

    name: str
    seed: int
    train_family: str
    eval_family: str
    train_count: int
    eval_count: int
    relation_count: int
    epochs: int
    learning_rate: float
    tau: float
    safety_lambda: float
    output_path: Path
    train_corpus_path: Path | None = None
    eval_corpus_path: Path | None = None


def main() -> None:
    """Parse CLI arguments and run a synthetic Safe-CEG experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path for a synthetic experiment run",
    )
    parser.add_argument(
        "--train-family", default="path", choices=["path", "star", "cycle"]
    )
    parser.add_argument(
        "--eval-family", default="cycle", choices=["path", "star", "cycle"]
    )
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=16)
    parser.add_argument("--relation-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--safety-lambda", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/synthetic_safe_ceg_summary.json"),
    )
    parser.add_argument("--train-corpus", type=Path, default=None)
    parser.add_argument("--eval-corpus", type=Path, default=None)
    args = parser.parse_args()

    config = (
        _load_config(args.config)
        if args.config is not None
        else SyntheticExperimentConfig(
            name="synthetic-safe-ceg-cli",
            seed=args.seed,
            train_family=args.train_family,
            eval_family=args.eval_family,
            train_count=args.train_count,
            eval_count=args.eval_count,
            relation_count=args.relation_count,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            tau=args.tau,
            safety_lambda=args.safety_lambda,
            output_path=args.output,
            train_corpus_path=args.train_corpus,
            eval_corpus_path=args.eval_corpus,
        )
    )
    payload = run_experiment(config)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_experiment(config: SyntheticExperimentConfig) -> dict[str, object]:
    """Execute one synthetic Safe-CEG experiment from a resolved config."""
    train_workload = _resolve_workload(
        family=config.train_family,
        query_count=config.train_count,
        relation_count=config.relation_count,
        seed=config.seed,
        corpus_path=config.train_corpus_path,
    )
    eval_workload = _resolve_workload(
        family=config.eval_family,
        query_count=config.eval_count,
        relation_count=config.relation_count,
        seed=config.seed + 1,
        corpus_path=config.eval_corpus_path,
    )
    train_dataset = build_dataset_from_workload(train_workload)
    eval_dataset = build_dataset_from_workload(eval_workload)

    feature_bundle = extract_features(train_dataset[0].query)
    model = SafeCEGGNN(
        node_feature_dim=feature_bundle.node_features.shape[1],
        global_feature_dim=feature_bundle.global_features.shape[1],
    )
    training_losses = train_model(
        model,
        train_dataset,
        TrainingConfig(
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            safety_lambda=config.safety_lambda,
            tau=config.tau,
        ),
    )
    evaluation_summary = evaluate_model(model, eval_dataset, tau=config.tau)

    return {
        "name": config.name,
        "train_family": config.train_family,
        "eval_family": config.eval_family,
        "train_count": len(train_workload),
        "eval_count": len(eval_workload),
        "relation_count": config.relation_count,
        "seed": config.seed,
        "epochs": config.epochs,
        "training_losses": training_losses,
        "evaluation": evaluation_summary,
        "train_corpus_path": None
        if config.train_corpus_path is None
        else str(config.train_corpus_path),
        "eval_corpus_path": None
        if config.eval_corpus_path is None
        else str(config.eval_corpus_path),
    }


def _resolve_workload(
    family: str,
    query_count: int,
    relation_count: int,
    seed: int,
    corpus_path: Path | None,
) -> list[WorkloadQuery]:
    if corpus_path is not None:
        return load_workload_corpus(corpus_path)
    return generate_synthetic_workload(
        family=family,
        query_count=query_count,
        relation_count=relation_count,
        seed=seed,
    )


def _load_config(path: Path) -> SyntheticExperimentConfig:
    raw_payload = yaml.safe_load(path.read_text())
    if not isinstance(raw_payload, dict):
        raise ValueError("Synthetic experiment config root must be a mapping")

    experiment_block = _require_mapping(raw_payload, "experiment")
    workload_block = _require_mapping(raw_payload, "synthetic_workload")
    training_block = _require_mapping(raw_payload, "training")
    output_block = _require_mapping(raw_payload, "output")

    return SyntheticExperimentConfig(
        name=_require_string(experiment_block, "name"),
        seed=_require_int(experiment_block, "seed"),
        train_family=_require_string(workload_block, "train_family"),
        eval_family=_require_string(workload_block, "eval_family"),
        train_count=_require_int(workload_block, "train_count"),
        eval_count=_require_int(workload_block, "eval_count"),
        relation_count=_require_int(workload_block, "relation_count"),
        epochs=_require_int(training_block, "epochs"),
        learning_rate=_require_float(training_block, "learning_rate"),
        tau=_require_float(training_block, "safety_tau"),
        safety_lambda=_require_float(training_block, "safety_lambda"),
        output_path=Path(_require_string(output_block, "summary_path")),
        train_corpus_path=_optional_path(workload_block.get("train_corpus_path")),
        eval_corpus_path=_optional_path(workload_block.get("eval_corpus_path")),
    )


def _require_mapping(raw_payload: dict[str, object], key: str) -> dict[str, object]:
    value = raw_payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Field {key!r} must be a mapping")
    return value


def _require_string(raw_payload: dict[str, object], key: str) -> str:
    value = raw_payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Field {key!r} must be a non-empty string")
    return value


def _require_int(raw_payload: dict[str, object], key: str) -> int:
    value = raw_payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Field {key!r} must be an integer")
    return value


def _require_float(raw_payload: dict[str, object], key: str) -> float:
    value = raw_payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Field {key!r} must be numeric")
    return float(value)


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Corpus path must be a non-empty string when provided")
    return Path(value)


if __name__ == "__main__":
    main()
