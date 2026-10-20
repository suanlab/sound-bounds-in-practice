"""Run distribution shift evaluation experiments from YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from src.models.features import extract_features
from src.models.gnn import SafeCEGGNN
from src.training.checkpoint import save_checkpoint
from src.training.dataset import build_dataset_from_workload
from src.training.evaluate import evaluate_model
from src.training.train import TrainingConfig, train_model
from src.training.workload import WorkloadQuery, generate_synthetic_workload
from src.utils.metrics import aggregate_multi_seed, format_mean_std

FAMILY_MAP: dict[str, str] = {
    "acyclic": "path",
    "cyclic_large": "cycle",
    "complex": "cycle",
    "size_3_4": "path",
    "size_5_6_7": "path",
    "synthetic_path_A": "path",
    "synthetic_path_D": "path",
    "synthetic_star_B": "star",
    "synthetic_star_E": "star",
    "synthetic_cycle_C": "cycle",
}

RELATION_COUNT_MAP: dict[str, int] = {
    "acyclic": 4,
    "cyclic_large": 6,
    "complex": 7,
    "size_3_4": 4,
    "size_5_6_7": 6,
    "synthetic_path_A": 4,
    "synthetic_path_D": 5,
    "synthetic_star_B": 4,
    "synthetic_star_E": 5,
    "synthetic_cycle_C": 5,
}

# Deterministic per-scenario seed offset so that scenarios sharing a family and
# relation count still draw *distinct* query streams (prevents the bug where
# differently named shift scenarios were byte-identical workloads).
NAME_SEED_OFFSET: dict[str, int] = {
    name: index * 7919 for index, name in enumerate(sorted(FAMILY_MAP))
}


def _run_single_seed(
    scenarios: list[dict[str, Any]],
    args: argparse.Namespace,
    seed: int,
    tau: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Run all shift scenarios for one seed."""
    torch.manual_seed(seed)
    result: dict[str, Any] = {"seed": seed, "scenarios": {}}

    for scenario in scenarios:
        scenario_name = scenario.get("name", "unnamed")
        train_on = scenario.get("train_on", [])
        test_on = scenario.get("test_on", [])

        train_workload = _build_workload_from_names(train_on, args.train_count, seed)
        if not train_workload:
            continue

        train_dataset = build_dataset_from_workload(train_workload)
        feature_bundle = extract_features(train_dataset[0].query)
        model = SafeCEGGNN(
            node_feature_dim=feature_bundle.node_features.shape[1],
            global_feature_dim=feature_bundle.global_features.shape[1],
        )
        train_model(
            model,
            train_dataset,
            TrainingConfig(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                safety_lambda=args.safety_lambda,
                tau=tau,
            ),
        )

        checkpoint_path = output_dir / f"{scenario_name}_model_s{seed}.pt"
        save_checkpoint(model, checkpoint_path)

        scenario_results: dict[str, dict[str, object]] = {
            "train_on": {"names": train_on},
            "test_on": {},
        }
        for test_name in test_on:
            test_workload = _build_workload_from_names(
                [test_name], args.eval_count, seed + 100
            )
            if not test_workload:
                continue
            test_dataset = build_dataset_from_workload(test_workload)
            summary = evaluate_model(model, test_dataset, tau=tau)
            scenario_results["test_on"][test_name] = summary

        result["scenarios"][scenario_name] = scenario_results

    return result


def main() -> None:
    """Parse CLI arguments and run distribution shift experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/dist_shift.yaml"),
    )
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--safety-lambda", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")

    base_seed = config.get("experiment", {}).get("seed", 42)
    tau = 2.0
    output_dir = Path(config.get("output", {}).get("results_dir", "results/shift"))
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = config.get("shift_scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("shift_scenarios must be a list")

    seed_list = [base_seed + i * 1000 for i in range(args.seeds)]
    all_seed_results: list[dict[str, Any]] = []

    for i, seed in enumerate(seed_list):
        print(f"\n--- Seed {i + 1}/{len(seed_list)} (seed={seed}) ---")
        result = _run_single_seed(scenarios, args, seed, tau, output_dir)
        all_seed_results.append(result)

    # Save per-seed results
    (output_dir / "shift_per_seed.json").write_text(
        json.dumps(all_seed_results, indent=2) + "\n"
    )

    # Aggregate across seeds
    agg_keys = [
        "mean",
        "median",
        "fallback_rate",
        "upper_fallback_rate",
        "lower_fallback_rate",
        "safety_violation_rate",
        "config_accuracy",
    ]
    aggregated: dict[str, Any] = {"n_seeds": len(seed_list), "scenarios": {}}

    # Get scenario/test structure from first seed
    first = all_seed_results[0]
    for sn, sd in first.get("scenarios", {}).items():
        aggregated["scenarios"][sn] = {"test_on": {}}
        for tn in sd.get("test_on", {}):
            per_seed = [
                cast("dict[str, object]", r["scenarios"][sn]["test_on"][tn])
                for r in all_seed_results
                if sn in r.get("scenarios", {})
                and tn in r["scenarios"][sn].get("test_on", {})
            ]
            agg = aggregate_multi_seed(per_seed, agg_keys)
            aggregated["scenarios"][sn]["test_on"][tn] = agg

    (output_dir / "shift_results.json").write_text(
        json.dumps(aggregated, indent=2) + "\n"
    )

    # Print summary
    for sn, sd in aggregated.get("scenarios", {}).items():
        for tn, agg in sd.get("test_on", {}).items():
            m = agg.get("mean", {})
            fb = agg.get("fallback_rate", {})
            ca = agg.get("config_accuracy", {})
            print(
                f"{sn:15s} {tn:22s}"
                f"  qe={format_mean_std(m.get('mean', 0), m.get('std', 0)):>16s}"
                f"  fb={format_mean_std(fb.get('mean', 0), fb.get('std', 0)):>12s}"
                f"  ca={format_mean_std(ca.get('mean', 0), ca.get('std', 0)):>12s}"
            )


def _build_workload_from_names(
    names: list[str],
    count_per_name: int,
    seed: int,
) -> list[WorkloadQuery]:
    workload: list[WorkloadQuery] = []
    for index, name in enumerate(names):
        family = FAMILY_MAP.get(name, "path")
        relation_count = RELATION_COUNT_MAP.get(name, 4)
        workload.extend(
            generate_synthetic_workload(
                family=family,
                query_count=count_per_name,
                relation_count=relation_count,
                seed=seed + index + NAME_SEED_OFFSET.get(name, 0),
            )
        )
    return workload


if __name__ == "__main__":
    main()
