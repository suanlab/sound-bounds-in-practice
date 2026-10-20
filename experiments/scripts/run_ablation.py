from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

import torch
from torch import nn

from src.core.ceg import pessimistic_bound
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.estimators.independence_estimator import IndependenceEstimator
from src.estimators.safe_estimator import SafeCEGEstimator
from src.models.features import extract_features
from src.models.gnn import GNNOutput, SafeCEGGNN
from src.models.predictor import CEGPredictor
from src.training.dataset import SafeCEGDataset, build_dataset_from_workload
from src.training.evaluate import evaluate_model
from src.training.train import TrainingConfig, train_model
from src.training.workload import generate_synthetic_workload
from src.utils.metrics import q_error, safety_violation_rate, summarize_q_errors
from src.utils.types import (
    ALL_CEG_CONFIGS,
    AggregatorChoice,
    CEGConfig,
    MLPrediction,
    PathLengthChoice,
)

MLP_MAX_NODES = 16


def _add_bound_split(
    summary: dict[str, object],
    dataset: SafeCEGDataset,
    per_query_qerrors: list[float],
) -> None:
    """Add bound-valid/invalid conditional q-errors to a summary dict."""
    from statistics import fmean, median

    bv, bi = [], []
    for i in range(len(dataset)):
        ex = dataset[i]
        bound = pessimistic_bound(ex.query)
        if ex.true_cardinality <= bound:
            bv.append(per_query_qerrors[i])
        else:
            bi.append(per_query_qerrors[i])
    summary["bound_valid_fraction"] = len(bv) / len(dataset)
    summary["bound_valid_mean"] = fmean(bv) if bv else 0.0
    summary["bound_valid_median"] = median(bv) if bv else 0.0
    summary["bound_invalid_mean"] = fmean(bi) if bi else 0.0
    summary["median"] = median(per_query_qerrors)


class BucketedSelector:
    def __init__(
        self,
        bucket_to_config: dict[tuple[int, int], CEGConfig],
        fallback: CEGConfig,
    ):
        self.bucket_to_config = bucket_to_config
        self.fallback = fallback

    def predict(self, relation_count: int, cyclicity: int) -> CEGConfig:
        return self.bucket_to_config.get((relation_count, cyclicity), self.fallback)


class FlatMLP(nn.Module):
    def __init__(
        self,
        flatten_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 64,
        num_configs: int = 9,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_configs),
        )

    def forward(self, features: object) -> object:
        node_feats = getattr(features, "node_features", None)
        global_feat = getattr(features, "global_features", None)
        if node_feats is None or global_feat is None:
            raise ValueError("FlatMLP requires FeatureBundle input")
        n = node_feats.shape[0]
        padded = torch.zeros(MLP_MAX_NODES, node_feats.shape[1])
        padded[: min(n, MLP_MAX_NODES)] = node_feats[:MLP_MAX_NODES]
        flat = padded.reshape(-1)
        full = torch.cat((flat, global_feat.squeeze(0)))
        logits = self.net(full.unsqueeze(0))
        return GNNOutput(
            logits=logits,
            confidence=torch.tensor([0.5]),
            graph_embedding=torch.zeros(1, 64),
        )


def _evaluate_ml_only(
    dataset: SafeCEGDataset, model: SafeCEGGNN, tau: float
) -> Mapping[str, object]:
    predictor = CEGPredictor(model)
    q_errors: list[float] = []
    ml_violations = 0
    for i in range(len(dataset)):
        ex = dataset[i]
        out = predictor.predict(ex.query)
        config_idx = int(torch.argmax(out.logits, dim=1).item())
        config = CEGConfig(PathLengthChoice.MAX_HOP, AggregatorChoice.MAX)
        all_cfgs = [CEGConfig(p, a) for p in PathLengthChoice for a in AggregatorChoice]
        config = all_cfgs[config_idx]
        est = BaselineCEGEstimator().estimate(ex.query, config)
        bound = pessimistic_bound(ex.query)
        q_errors.append(q_error(est.estimate, ex.true_cardinality))
        if est.estimate > tau * bound:
            ml_violations += 1
    summary = summarize_q_errors(q_errors)
    summary["safety_violation_rate"] = ml_violations / len(dataset)
    summary["fallback_rate"] = 0.0
    summary["ml_safety_violation_rate"] = summary["safety_violation_rate"]
    _add_bound_split(summary, dataset, q_errors)
    return summary


def _evaluate_pes_only(dataset: SafeCEGDataset, tau: float) -> Mapping[str, object]:
    q_errors: list[float] = []
    for i in range(len(dataset)):
        ex = dataset[i]
        bound = pessimistic_bound(ex.query)
        q_errors.append(q_error(bound, ex.true_cardinality))
    summary = summarize_q_errors(q_errors)
    summary["fallback_rate"] = 1.0
    summary["safety_violation_rate"] = 0.0
    summary["ml_safety_violation_rate"] = 0.0
    _add_bound_split(summary, dataset, q_errors)
    return summary


def _evaluate_independence(dataset: SafeCEGDataset, tau: float) -> Mapping[str, object]:
    """Evaluate the attribute-value independence baseline."""
    ind = IndependenceEstimator()
    q_errors: list[float] = []
    estimates: list[float] = []
    bounds: list[float] = []
    for i in range(len(dataset)):
        ex = dataset[i]
        est = ind.estimate(ex.query)
        q_errors.append(q_error(est, ex.true_cardinality))
        estimates.append(est)
        bounds.append(pessimistic_bound(ex.query))
    summary = summarize_q_errors(q_errors)
    summary["fallback_rate"] = 0.0
    summary["safety_violation_rate"] = safety_violation_rate(estimates, bounds, tau=tau)
    summary["ml_safety_violation_rate"] = summary["safety_violation_rate"]
    _add_bound_split(summary, dataset, q_errors)
    return summary


def _select_best_fixed_config(train_dataset: SafeCEGDataset) -> CEGConfig:
    estimator = BaselineCEGEstimator()
    best_config = ALL_CEG_CONFIGS[0]
    best_mean_q_error = float("inf")

    for config in ALL_CEG_CONFIGS:
        q_errors: list[float] = []
        for i in range(len(train_dataset)):
            ex = train_dataset[i]
            estimate = estimator.estimate(ex.query, config)
            q_errors.append(q_error(estimate.estimate, ex.true_cardinality))
        mean_q = sum(q_errors) / len(q_errors)
        if mean_q < best_mean_q_error:
            best_mean_q_error = mean_q
            best_config = config

    return best_config


def _evaluate_fixed_optimistic(
    dataset: SafeCEGDataset, config: CEGConfig, tau: float
) -> Mapping[str, object]:
    estimator = BaselineCEGEstimator()
    q_errors: list[float] = []
    ml_violations = 0

    for i in range(len(dataset)):
        ex = dataset[i]
        estimate = estimator.estimate(ex.query, config)
        bound = pessimistic_bound(ex.query)
        q_errors.append(q_error(estimate.estimate, ex.true_cardinality))
        if estimate.estimate > tau * bound:
            ml_violations += 1

    summary = summarize_q_errors(q_errors)
    summary["fallback_rate"] = 0.0
    summary["safety_violation_rate"] = ml_violations / len(dataset)
    summary["ml_safety_violation_rate"] = summary["safety_violation_rate"]
    _add_bound_split(summary, dataset, q_errors)
    return summary


def _evaluate_oracle_best(dataset: SafeCEGDataset, tau: float) -> Mapping[str, object]:
    """Pure oracle: q-error of the best optimistic config's *own* estimate.

    This is the configuration-selection ceiling and is deliberately NOT routed
    through ``SafeCEGEstimator`` -- wrapping it in the fallback logic would
    conflate the oracle's best-config performance with the safety mechanism
    (the previous version reported a nonzero oracle fallback rate). Fallback
    rate is therefore structurally 0 / N/A for the pure oracle.
    """
    q_errors: list[float] = []
    for i in range(len(dataset)):
        ex = dataset[i]
        oracle_estimate = float(ex.estimate_tensor()[ex.best_config_index].item())
        q_errors.append(q_error(oracle_estimate, ex.true_cardinality))
    summary = summarize_q_errors(q_errors)
    summary["fallback_rate"] = 0.0  # pure oracle: no fallback wrapper
    summary["safety_violation_rate"] = 0.0
    summary["ml_safety_violation_rate"] = 0.0
    _add_bound_split(summary, dataset, q_errors)
    return summary


def _evaluate_safe_ceg_mlp(
    dataset: SafeCEGDataset, model: nn.Module, tau: float
) -> Mapping[str, object]:
    safe_estimator = SafeCEGEstimator(tau=tau)
    q_errors: list[float] = []
    final_estimates: list[float] = []
    ml_estimates: list[float] = []
    pessimistic_bounds: list[float] = []
    fallback_count = 0
    for i in range(len(dataset)):
        ex = dataset[i]
        features = extract_features(ex.query)
        output = model(features)
        logits = output.logits if hasattr(output, "logits") else output
        config_idx = int(torch.argmax(logits, dim=1).item())
        config = ALL_CEG_CONFIGS[config_idx]
        prediction = MLPrediction(
            path_length=config.path_length,
            aggregator=config.aggregator,
            per_edge_alpha={},
            confidence=0.5,
        )
        result = safe_estimator.estimate(ex.query, prediction)
        q_errors.append(q_error(result.estimate, ex.true_cardinality))
        final_estimates.append(result.estimate)
        ml_estimates.append(result.ml_estimate)
        pessimistic_bounds.append(result.pessimistic_bound)
        fallback_count += int(result.used_fallback)

    summary = summarize_q_errors(q_errors)
    summary["fallback_rate"] = fallback_count / len(dataset)
    ml_viol = sum(
        1
        for ml, pb in zip(ml_estimates, pessimistic_bounds, strict=True)
        if ml > tau * pb
    )
    summary["ml_safety_violation_rate"] = ml_viol / len(dataset)
    from src.utils.metrics import safety_violation_rate

    summary["safety_violation_rate"] = safety_violation_rate(
        final_estimates, pessimistic_bounds, tau=tau
    )
    _add_bound_split(summary, dataset, q_errors)
    return summary


def _train_bucketed_selector(train_dataset: SafeCEGDataset) -> BucketedSelector:
    bucket_counts: dict[tuple[int, int], Counter[CEGConfig]] = defaultdict(Counter)
    global_counts: Counter[CEGConfig] = Counter()

    for i in range(len(train_dataset)):
        ex = train_dataset[i]
        features = extract_features(ex.query)
        relation_count = int(features.global_features[0, 0].item())
        cyclicity = int(features.global_features[0, 5].item())
        bucket_counts[(relation_count, cyclicity)][ex.best_config] += 1
        global_counts[ex.best_config] += 1

    fallback = global_counts.most_common(1)[0][0]
    bucket_to_config = {
        bucket: counts.most_common(1)[0][0] for bucket, counts in bucket_counts.items()
    }
    return BucketedSelector(bucket_to_config=bucket_to_config, fallback=fallback)


def _evaluate_safe_ceg_bucketed(
    dataset: SafeCEGDataset, selector: BucketedSelector, tau: float
) -> Mapping[str, object]:
    safe_estimator = SafeCEGEstimator(tau=tau)
    q_errors: list[float] = []
    final_estimates: list[float] = []
    ml_estimates: list[float] = []
    pessimistic_bounds: list[float] = []
    fallback_count = 0

    for i in range(len(dataset)):
        ex = dataset[i]
        features = extract_features(ex.query)
        relation_count = int(features.global_features[0, 0].item())
        cyclicity = int(features.global_features[0, 5].item())
        config = selector.predict(relation_count=relation_count, cyclicity=cyclicity)
        prediction = MLPrediction(
            path_length=config.path_length,
            aggregator=config.aggregator,
            per_edge_alpha={},
            confidence=0.5,
        )
        result = safe_estimator.estimate(ex.query, prediction)
        q_errors.append(q_error(result.estimate, ex.true_cardinality))
        final_estimates.append(result.estimate)
        ml_estimates.append(result.ml_estimate)
        pessimistic_bounds.append(result.pessimistic_bound)
        fallback_count += int(result.used_fallback)

    summary = summarize_q_errors(q_errors)
    summary["fallback_rate"] = fallback_count / len(dataset)
    ml_viol = sum(
        1
        for ml, pb in zip(ml_estimates, pessimistic_bounds, strict=True)
        if ml > tau * pb
    )
    summary["ml_safety_violation_rate"] = ml_viol / len(dataset)
    summary["safety_violation_rate"] = safety_violation_rate(
        final_estimates, pessimistic_bounds, tau=tau
    )
    _add_bound_split(summary, dataset, q_errors)
    return summary


def _train_generic_model(
    model: nn.Module,
    dataset: SafeCEGDataset,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    tau: float = 2.0,
    safety_lambda: float = 0.5,
    ce_lambda: float = 0.5,
) -> list[float]:
    """Train a non-GNN selector with the SAME objective as ``train_model``.

    Using ``expected_risk_loss`` (identical loss, safety_lambda, ce_lambda, tau)
    ensures the GNN-vs-MLP comparison reflects architecture only, not a different
    training objective.
    """
    from src.training.loss import expected_risk_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _epoch in range(epochs):
        total_loss = 0.0
        for i in range(len(dataset)):
            ex = dataset[i]
            features = extract_features(ex.query)
            output = model(features)
            logits = output.logits if hasattr(output, "logits") else output
            loss = expected_risk_loss(
                logits=logits,
                config_q_errors=ex.q_error_tensor().unsqueeze(0),
                config_estimates=ex.estimate_tensor().unsqueeze(0),
                pessimistic_bounds=torch.tensor(
                    [ex.pessimistic_bound], dtype=torch.float32
                ),
                safety_lambda=safety_lambda,
                tau=tau,
                oracle_labels=torch.tensor([ex.best_config_index], dtype=torch.long),
                ce_lambda=ce_lambda,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        losses.append(total_loss / len(dataset))
    return losses


def _run_single_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    """Run a complete ablation study for one seed and return results dict."""
    torch.manual_seed(seed)
    count_per_family = args.train_count // 3
    train_workload = (
        generate_synthetic_workload("path", count_per_family, 4, seed=seed)
        + generate_synthetic_workload("star", count_per_family, 4, seed=seed + 10)
        + generate_synthetic_workload("cycle", count_per_family, 4, seed=seed + 20)
    )
    eval_workloads = {
        "in_distribution": (
            generate_synthetic_workload("path", args.eval_count // 3, 4, seed=seed + 1)
            + generate_synthetic_workload(
                "star", args.eval_count // 3, 4, seed=seed + 11
            )
            + generate_synthetic_workload(
                "cycle", args.eval_count // 3, 4, seed=seed + 21
            )
        ),
        "cross_workload": generate_synthetic_workload(
            "cycle", args.eval_count, 5, seed=seed + 100
        ),
    }

    train_dataset = build_dataset_from_workload(train_workload)
    feature_bundle = extract_features(train_dataset[0].query)
    node_dim = feature_bundle.node_features.shape[1]
    global_dim = feature_bundle.global_features.shape[1]

    gnn_model = SafeCEGGNN(node_feature_dim=node_dim, global_feature_dim=global_dim)
    train_model(
        gnn_model,
        train_dataset,
        TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            safety_lambda=args.safety_lambda,
            tau=args.tau,
        ),
    )

    sample = extract_features(train_dataset[0].query)
    actual_node_dim = sample.node_features.shape[1]
    actual_global_dim = sample.global_features.shape[1]
    mlp_flat_dim = actual_node_dim * MLP_MAX_NODES + actual_global_dim
    mlp_model = FlatMLP(flatten_dim=mlp_flat_dim, global_feature_dim=actual_global_dim)
    _train_generic_model(
        mlp_model,
        train_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        tau=args.tau,
        safety_lambda=args.safety_lambda,
        ce_lambda=0.5,
    )
    bucketed_selector = _train_bucketed_selector(train_dataset)
    fixed_config = _select_best_fixed_config(train_dataset)

    results: dict[str, Any] = {
        "metadata": {"seed": seed, "fixed_optimistic_config": fixed_config.name},
    }
    for eval_name, eval_wl in eval_workloads.items():
        eval_dataset = build_dataset_from_workload(eval_wl)
        results[eval_name] = {
            "ml_only_gnn": _evaluate_ml_only(eval_dataset, gnn_model, args.tau),
            "fixed_optimistic": _evaluate_fixed_optimistic(
                eval_dataset, fixed_config, args.tau
            ),
            "pes_only": _evaluate_pes_only(eval_dataset, args.tau),
            "independence": _evaluate_independence(eval_dataset, args.tau),
            "safe_ceg_gnn": evaluate_model(gnn_model, eval_dataset, tau=args.tau),
            "safe_ceg_bucketed": _evaluate_safe_ceg_bucketed(
                eval_dataset, bucketed_selector, args.tau
            ),
            "safe_ceg_mlp": _evaluate_safe_ceg_mlp(eval_dataset, mlp_model, args.tau),
            "oracle_best": _evaluate_oracle_best(eval_dataset, args.tau),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation study for Safe-CEG.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to run")
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--safety-lambda", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ablation"),
    )
    args = parser.parse_args()

    from src.utils.metrics import aggregate_multi_seed, format_mean_std

    seed_list = [args.seed + i * 1000 for i in range(args.seeds)]
    all_seed_results: list[dict[str, Any]] = []

    for i, seed in enumerate(seed_list):
        print(f"\n--- Seed {i + 1}/{len(seed_list)} (seed={seed}) ---")
        result = _run_single_seed(args, seed)
        all_seed_results.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate across seeds per eval_name and method
    eval_names = [k for k in all_seed_results[0] if k != "metadata"]
    methods = list(cast("dict[str, Any]", all_seed_results[0][eval_names[0]]).keys())
    agg_keys = [
        "mean",
        "median",
        "fallback_rate",
        "safety_violation_rate",
        "config_accuracy",
        "bound_valid_mean",
        "bound_valid_median",
        "bound_invalid_mean",
        "bound_valid_fraction",
    ]

    aggregated: dict[str, Any] = {"n_seeds": len(seed_list)}
    for eval_name in eval_names:
        aggregated[eval_name] = {}
        for method in methods:
            per_seed = [
                cast("dict[str, object]", r[eval_name][method])
                for r in all_seed_results
            ]
            agg = aggregate_multi_seed(per_seed, agg_keys)
            aggregated[eval_name][method] = agg

    (args.output_dir / "ablation.json").write_text(
        json.dumps(aggregated, indent=2) + "\n"
    )
    # Also save per-seed raw results
    (args.output_dir / "ablation_per_seed.json").write_text(
        json.dumps(all_seed_results, indent=2) + "\n"
    )

    # Print summary
    for eval_name in eval_names:
        print(f"\n=== {eval_name} ({len(seed_list)} seeds) ===")
        typed_methods = cast(
            "dict[str, dict[str, dict[str, float]]]", aggregated[eval_name]
        )
        for method, agg in typed_methods.items():
            m_data = agg.get("mean", {})
            sv_data = agg.get("safety_violation_rate", {})
            m_str = format_mean_std(m_data.get("mean", 0), m_data.get("std", 0))
            sv_str = format_mean_std(sv_data.get("mean", 0), sv_data.get("std", 0))
            bv = agg.get("bound_valid_mean", {})
            bi = agg.get("bound_invalid_mean", {})
            bv_str = format_mean_std(bv.get("mean", 0), bv.get("std", 0))
            bi_str = format_mean_std(bi.get("mean", 0), bi.get("std", 0))
            print(
                f"  {method:20s}  q-error={m_str:>16s}"
                f"  bv={bv_str:>12s}  bi={bi_str:>12s}"
                f"  viol={sv_str}"
            )


if __name__ == "__main__":
    main()
