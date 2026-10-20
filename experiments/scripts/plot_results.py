"""Plotting utilities for Safe-CEG experiment figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(result_path: Path, output_path: Path) -> None:
    """Plot training loss curve from a result JSON."""
    payload = json.loads(result_path.read_text())
    losses = payload.get("training_losses", [])
    if not losses:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_tau_tradeoff(
    tau_values: list[float],
    q_errors: list[float],
    fallback_rates: list[float],
    output_path: Path,
) -> None:
    """Plot the tau vs q-error/fallback-rate tradeoff curve."""
    fig, ax1 = plt.subplots(figsize=(6, 4))
    color_q = "#1f77b4"
    color_fb = "#d62728"

    ax1.plot(tau_values, q_errors, marker="o", color=color_q, label="Mean q-error")
    ax1.set_xlabel(r"Safety threshold $\tau$")
    ax1.set_ylabel("Mean q-error", color=color_q)
    ax1.tick_params(axis="y", labelcolor=color_q)

    ax2 = ax1.twinx()
    ax2.plot(
        tau_values,
        fallback_rates,
        marker="s",
        color=color_fb,
        linestyle="--",
        label="Fallback rate",
    )
    ax2.set_ylabel("Fallback rate", color=color_fb)
    ax2.tick_params(axis="y", labelcolor=color_fb)
    ax2.set_ylim(-0.05, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(r"Safety-Accuracy Tradeoff ($\tau$)")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_shift_comparison(
    shift_result_path: Path,
    output_path: Path,
) -> None:
    """Plot a grouped bar chart comparing q-error across shift scenarios."""
    payload = json.loads(shift_result_path.read_text())
    scenarios = payload.get("scenarios", {})
    if not scenarios:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    scenario_names: list[str] = []
    test_names_all: set[str] = set()

    for scenario_name, scenario_data in scenarios.items():
        scenario_names.append(scenario_name)
        test_names_all.update(scenario_data.get("test_on", {}).keys())

    test_names = sorted(test_names_all)
    x = np.arange(len(scenario_names))
    width = 0.8 / max(len(test_names), 1)

    for idx, test_name in enumerate(test_names):
        values = []
        for scenario_name in scenario_names:
            test_data = scenarios[scenario_name].get("test_on", {}).get(test_name, {})
            values.append(test_data.get("mean", 0.0))
        offset = (idx - len(test_names) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=test_name)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=15, ha="right")
    ax.set_ylabel("Mean q-error")
    ax.set_title("Distribution Shift: q-error by Scenario")
    ax.legend(title="Test set", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_loss(args.result, args.output_dir / "training_loss.png")
