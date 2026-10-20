"""Run tau-sweep experiment measuring safety-accuracy tradeoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments.scripts.plot_results import plot_tau_tradeoff
from src.models.features import extract_features
from src.models.gnn import SafeCEGGNN
from src.training.dataset import build_dataset_from_workload
from src.training.evaluate import evaluate_model
from src.training.train import TrainingConfig, train_model
from src.training.workload import generate_synthetic_workload
from src.utils.metrics import format_mean_std

FAMILY_MAP: dict[str, str] = {
    "acyclic": "path",
    "cyclic_large": "cycle",
    "complex": "cycle",
    "cross_dataset": "path",
}
RELATION_COUNT_MAP: dict[str, int] = {
    "acyclic": 4,
    "cyclic_large": 5,
    "complex": 5,
    "cross_dataset": 4,
}


def _run_single_seed(args: argparse.Namespace, seed: int) -> list[dict[str, float]]:
    """Run tau sweep for one seed, return per-tau results."""
    torch.manual_seed(seed)
    train_family = FAMILY_MAP.get(args.train_workload, "path")
    train_rel = RELATION_COUNT_MAP.get(args.train_workload, 4)
    eval_family = FAMILY_MAP.get(args.eval_workload, "cycle")
    eval_rel = RELATION_COUNT_MAP.get(args.eval_workload, 5)

    train_wl = generate_synthetic_workload(
        train_family,
        args.train_count,
        train_rel,
        seed=seed,
    )
    eval_wl = generate_synthetic_workload(
        eval_family,
        args.eval_count,
        eval_rel,
        seed=seed + 100,
    )

    train_ds = build_dataset_from_workload(train_wl)
    eval_ds = build_dataset_from_workload(eval_wl)

    fb = extract_features(train_ds[0].query)
    model = SafeCEGGNN(
        node_feature_dim=fb.node_features.shape[1],
        global_feature_dim=fb.global_features.shape[1],
    )
    train_model(
        model,
        train_ds,
        TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            safety_lambda=args.safety_lambda,
            tau=2.0,
        ),
    )

    rows = []
    for tau in args.tau_values:
        s = evaluate_model(model, eval_ds, tau=tau)
        rows.append(
            {
                "tau": tau,
                "mean_q_error": float(s["mean"]),
                "fallback_rate": float(s["fallback_rate"]),
                "ml_violation_rate": float(s["ml_safety_violation_rate"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tau-values",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
    )
    parser.add_argument("--train-workload", default="acyclic")
    parser.add_argument("--eval-workload", default="cyclic_large")
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--eval-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--safety-lambda", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tau_sweep"))
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path("figures/tau_tradeoff.png"),
    )
    args = parser.parse_args()

    seed_list = [args.seed + i * 1000 for i in range(args.seeds)]
    all_seed_rows: list[list[dict[str, float]]] = []

    for i, seed in enumerate(seed_list):
        print(f"\n--- Seed {i + 1}/{len(seed_list)} (seed={seed}) ---")
        rows = _run_single_seed(args, seed)
        all_seed_rows.append(rows)

    # Aggregate per tau across seeds
    n_taus = len(args.tau_values)
    aggregated: list[dict[str, Any]] = []
    for ti in range(n_taus):
        tau = args.tau_values[ti]
        qes = [all_seed_rows[si][ti]["mean_q_error"] for si in range(len(seed_list))]
        fbs = [all_seed_rows[si][ti]["fallback_rate"] for si in range(len(seed_list))]
        from statistics import fmean, stdev

        qe_m, qe_s = fmean(qes), (stdev(qes) if len(qes) > 1 else 0.0)
        fb_m, fb_s = fmean(fbs), (stdev(fbs) if len(fbs) > 1 else 0.0)
        aggregated.append(
            {
                "tau": tau,
                "mean_q_error": qe_m,
                "std_q_error": qe_s,
                "mean_fallback": fb_m,
                "std_fallback": fb_s,
            }
        )
        print(
            f"  tau={tau:.1f}  qe={format_mean_std(qe_m, qe_s):>16s}"
            f"  fb={format_mean_std(fb_m, fb_s):>12s}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_seeds": len(seed_list),
        "per_tau": aggregated,
        "per_seed": all_seed_rows,
    }
    (args.output_dir / "tau_sweep.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    # Plot using mean values
    tau_vals = [r["tau"] for r in aggregated]
    qe_vals = [r["mean_q_error"] for r in aggregated]
    fb_vals = [r["mean_fallback"] for r in aggregated]
    plot_tau_tradeoff(tau_vals, qe_vals, fb_vals, args.figure_path)


if __name__ == "__main__":
    main()
