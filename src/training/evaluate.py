"""Evaluation utilities for Safe-CEG models and baselines."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from src.estimators.safe_estimator import SafeCEGEstimator
from src.models.predictor import CEGPredictor
from src.utils.metrics import q_error, safety_violation_rate, summarize_q_errors
from src.utils.types import CEGConfig, FallbackDirection

if TYPE_CHECKING:
    from src.models.gnn import SafeCEGGNN
    from src.training.dataset import SafeCEGDataset


def evaluate_model(
    model: SafeCEGGNN,
    dataset: SafeCEGDataset,
    tau: float = 2.0,
) -> dict[str, object]:
    """Evaluate Safe-CEG predictions on an in-memory dataset."""
    if len(dataset) == 0:
        raise ValueError("Dataset must contain at least one example")

    predictor = CEGPredictor(model)
    safe_estimator = SafeCEGEstimator(tau=tau)
    # Evaluate deterministically: switch to eval mode and disable autograd so
    # any future stochastic layer (dropout/batchnorm) does not perturb metrics.
    was_training = model.training
    grad_was_enabled = torch.is_grad_enabled()
    model.eval()
    torch.set_grad_enabled(False)
    q_errors: list[float] = []
    final_estimates: list[float] = []
    ml_estimates: list[float] = []
    pessimistic_bounds: list[float] = []
    correct_configs = 0
    fallback_count = 0
    upper_fallback_count = 0
    lower_fallback_count = 0
    final_differs_from_pessimistic_count = 0
    final_better_than_ml_count = 0
    final_better_than_pessimistic_count = 0
    ml_better_than_pessimistic_count = 0
    helpful_fallback_count = 0
    expected_q_errors: list[float] = []
    expected_safety_penalties: list[float] = []
    ml_only_q_errors_safe: list[float] = []
    ml_only_q_errors_violating: list[float] = []
    # Bound-valid/invalid conditional tracking
    bound_valid_q_errors: list[float] = []
    bound_invalid_q_errors: list[float] = []

    for index in range(len(dataset)):
        example = dataset[index]
        prediction_output = predictor.predict(example.query)
        probabilities = torch.softmax(prediction_output.logits, dim=1).squeeze(0)
        result = safe_estimator.estimate(example.query, prediction_output.prediction)
        final_qe = q_error(result.estimate, example.true_cardinality)
        ml_qe = q_error(result.ml_estimate, example.true_cardinality)
        pessimistic_qe = q_error(result.pessimistic_bound, example.true_cardinality)

        # Bound validity: is pessimistic bound >= true cardinality?
        bound_is_valid = example.true_cardinality <= result.pessimistic_bound
        if bound_is_valid:
            bound_valid_q_errors.append(final_qe)
        else:
            bound_invalid_q_errors.append(final_qe)

        q_errors.append(final_qe)
        final_estimates.append(result.estimate)
        ml_estimates.append(result.ml_estimate)
        pessimistic_bounds.append(result.pessimistic_bound)
        fallback_count += int(result.used_fallback)
        upper_fallback_count += int(
            result.fallback_direction == FallbackDirection.UPPER
        )
        lower_fallback_count += int(
            result.fallback_direction == FallbackDirection.LOWER
        )
        final_differs_from_pessimistic_count += int(
            abs(result.estimate - result.pessimistic_bound) > 1e-12
        )
        final_better_than_ml_count += int(final_qe < ml_qe)
        final_better_than_pessimistic_count += int(final_qe < pessimistic_qe)
        ml_better_than_pessimistic_count += int(ml_qe < pessimistic_qe)
        helpful_fallback_count += int(result.used_fallback and final_qe < ml_qe)

        is_violating = result.ml_estimate > tau * result.pessimistic_bound
        if is_violating:
            ml_only_q_errors_violating.append(ml_qe)
        else:
            ml_only_q_errors_safe.append(ml_qe)
        expected_q_errors.append(
            float(torch.sum(probabilities * example.q_error_tensor()).item())
        )
        expected_safety_penalties.append(
            float(
                torch.sum(
                    probabilities
                    * torch.relu(
                        example.estimate_tensor() - tau * example.pessimistic_bound
                    )
                ).item()
            )
        )
        correct_configs += int(
            _predicted_config(prediction_output.prediction) == example.best_config
        )

    summary: dict[str, object] = cast("dict[str, object]", summarize_q_errors(q_errors))
    summary["fallback_rate"] = fallback_count / len(dataset)
    summary["upper_fallback_rate"] = upper_fallback_count / len(dataset)
    summary["lower_fallback_rate"] = lower_fallback_count / len(dataset)
    summary["config_accuracy"] = correct_configs / len(dataset)
    summary["expected_q_error"] = sum(expected_q_errors) / len(expected_q_errors)
    summary["expected_safety_penalty"] = sum(expected_safety_penalties) / len(
        expected_safety_penalties
    )
    summary["final_differs_from_pessimistic_rate"] = (
        final_differs_from_pessimistic_count / len(dataset)
    )
    summary["final_better_than_ml_rate"] = final_better_than_ml_count / len(dataset)
    summary["final_better_than_pessimistic_rate"] = (
        final_better_than_pessimistic_count / len(dataset)
    )
    summary["ml_better_than_pessimistic_rate"] = ml_better_than_pessimistic_count / len(
        dataset
    )
    summary["helpful_fallback_rate"] = helpful_fallback_count / len(dataset)
    summary["safety_violation_rate"] = safety_violation_rate(
        final_estimates,
        pessimistic_bounds,
        tau=tau,
    )
    summary["ml_safety_violation_rate"] = safety_violation_rate(
        ml_estimates,
        pessimistic_bounds,
        tau=tau,
    )

    if ml_only_q_errors_violating:
        violating_summary = summarize_q_errors(ml_only_q_errors_violating)
        summary["ml_only_violating_q_error"] = violating_summary
        summary["ml_only_violating_count"] = len(ml_only_q_errors_violating)
    else:
        summary["ml_only_violating_q_error"] = {"median": 0.0, "p90": 0.0, "max": 0.0}
        summary["ml_only_violating_count"] = 0

    if ml_only_q_errors_safe:
        safe_summary = summarize_q_errors(ml_only_q_errors_safe)
        summary["ml_only_safe_q_error"] = safe_summary
        summary["ml_only_safe_count"] = len(ml_only_q_errors_safe)
    else:
        summary["ml_only_safe_q_error"] = {"median": 0.0, "p90": 0.0, "max": 0.0}
        summary["ml_only_safe_count"] = 0

    # Bound-valid/invalid conditional breakdown
    n_total = len(dataset)
    summary["bound_valid_fraction"] = len(bound_valid_q_errors) / n_total
    if bound_valid_q_errors:
        bv = summarize_q_errors(bound_valid_q_errors)
        summary["bound_valid_mean"] = bv["mean"]
        summary["bound_valid_median"] = bv["median"]
    else:
        summary["bound_valid_mean"] = 0.0
        summary["bound_valid_median"] = 0.0
    if bound_invalid_q_errors:
        bi = summarize_q_errors(bound_invalid_q_errors)
        summary["bound_invalid_mean"] = bi["mean"]
        summary["bound_invalid_median"] = bi["median"]
    else:
        summary["bound_invalid_mean"] = 0.0
        summary["bound_invalid_median"] = 0.0

    # Plan cost impact (simplified cascading model)
    from statistics import fmean as _fmean

    from src.estimators.plan_cost import plan_cost_ratio

    plan_costs = [
        plan_cost_ratio(
            final_estimates[i],
            dataset[i].true_cardinality,
            len(dataset[i].query.relation_names),
        )
        for i in range(n_total)
    ]
    summary["plan_cost_ratio_mean"] = _fmean(plan_costs)

    # Restore the caller's autograd/training state.
    torch.set_grad_enabled(grad_was_enabled)
    if was_training:
        model.train()

    return summary


def main() -> None:
    """Evaluate a saved Safe-CEG checkpoint on a synthetic eval workload."""
    import argparse
    import json
    import logging
    from pathlib import Path

    import yaml

    from src.models.features import extract_features
    from src.training.checkpoint import load_checkpoint
    from src.training.dataset import build_dataset_from_workload
    from src.training.train import _build_workload_from_config, set_seeds

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Evaluate a Safe-CEG checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seeds(seed)

    eval_cfg = cfg.get("eval_workload", {})
    eval_seed = int(eval_cfg.get("seed", seed + 100))
    workload = _build_workload_from_config(eval_cfg, eval_seed)
    dataset = build_dataset_from_workload(workload)

    features = extract_features(dataset[0].query)
    checkpoint_path = Path(cfg["checkpoint"])
    model = load_checkpoint(
        checkpoint_path,
        node_feature_dim=features.node_features.shape[1],
        global_feature_dim=features.global_features.shape[1],
    )
    tau = float(cfg.get("tau", 2.0))
    summary = evaluate_model(model, dataset, tau=tau)

    out_path = Path(
        cfg.get("output", {}).get("results", "results/eval/eval_results.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info(
        "Evaluated %d queries (tau=%.2f): mean q-error %.3f, fallback %.1f%%; -> %s",
        len(dataset),
        tau,
        float(cast("float", summary.get("mean", 0.0))),
        100 * float(cast("float", summary.get("fallback_rate", 0.0))),
        out_path,
    )


def _predicted_config(prediction: object) -> CEGConfig:
    from src.utils.types import MLPrediction

    if not isinstance(prediction, MLPrediction):
        raise ValueError("Prediction must be an MLPrediction instance")
    return CEGConfig(
        path_length=prediction.path_length,
        aggregator=prediction.aggregator,
    )


if __name__ == "__main__":
    main()
