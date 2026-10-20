"""Figures for the ICDE 2027 draft.

* ``icde_cost.pdf`` -- in-engine sampled-count cost relative to the exact count
  of the same sub-plan, against how long the exact count takes (one dot per
  sub-plan, median per log-bin as a line).
* ``icde_certificates.pdf`` -- per-query certified regret bound on the primary
  bracket (LpBound x PostgreSQL's in-engine sample, r = 0.25): the decoupled
  certificate ``G`` against the joint certificate, with and without the shared
  final result.
* ``icde_execution.pdf`` -- per STATS-CEB query, the speedup a guard gives
  against the speedup of planning on the true join sizes (the headroom), for
  the guard on the side each source errs on and for the other one.

Colours: first three categorical slots of the validated reference palette
(blue, orange, aqua; all-pairs CVD and normal-vision checks pass); aqua is below
3:1 contrast on white, so every curve is direct-labelled and also differs in
line style.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.plot_icde_figures
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WIDTH = 3.45  # IEEE column, inches
LEADER_TEXT = {"joint, $C_{out}$": (2.5e2, 0.76)}
LABEL_SHARE = {
    "joint, $C_{out}$": 0.90,
    "joint, no final result": 0.62,
    "decoupled $G$": 0.40,
}


def cost_figure(counts: Path, output: Path) -> None:
    """Sampled-count cost ratio against exact-count time, per sub-plan."""
    xs, ys = [], []
    for line in counts.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record["exact_censored"] or record["sample_censored"]:
            continue
        xs.append(record["exact_ms"])
        ys.append(record["sample_ms"] / record["exact_ms"])
    x, y = np.array(xs), np.array(ys)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    ax.scatter(x, y, s=4, color=BLUE, alpha=0.18, linewidths=0, rasterized=True)
    edges = np.logspace(math.log10(x.min()), math.log10(x.max()), 14)
    centres, medians = [], []
    for low, high in itertools.pairwise(edges):
        mask = (x >= low) & (x < high)
        if mask.sum() >= 15:
            centres.append(math.sqrt(low * high))
            medians.append(float(np.median(y[mask])))
    ax.plot(centres, medians, color=BLUE, linewidth=2, solid_capstyle="round")
    ax.axhline(0.10, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(
        x.min(),
        0.10 * 0.62,
        "sampling rate $r=0.10$",
        ha="left",
        va="top",
        color=MUTED,
        fontsize=7,
    )
    ax.text(centres[1], medians[1] * 1.35, "median", color=INK, fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("exact count of the sub-plan (ms)")
    ax.set_ylabel("sampled / exact time")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(output)
    plt.close(fig)
    logger.info("wrote %s (%d sub-plans)", output, len(x))


def ecdf(
    values: list[float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sorted values and their empirical cumulative shares."""
    data = np.sort(np.array(values))
    return data, np.arange(1, len(data) + 1) / len(data)


def certificate_figure(joint: Path, output: Path) -> None:
    """Per-query certified regret bounds: decoupled G versus joint R."""
    payload = json.loads(joint.read_text())
    cout = payload["per_query"]["lpboundxsample|cout"]
    minus = payload["per_query"]["lpboundxsample|cout_minus_final"]
    series = [
        ("joint, $C_{out}$", [r["joint_bound_best"] for r in cout], BLUE, "solid"),
        (
            "joint, no final result",
            [r["joint_bound_best"] for r in minus],
            ORANGE,
            (0, (5, 2)),
        ),
        ("decoupled $G$", [r["G_decoupled"] for r in cout], AQUA, (0, (1, 1.5))),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    for label, values, colour, style in series:
        xs, ys = ecdf(values)
        ax.step(
            xs,
            ys,
            where="post",
            color=colour,
            linewidth=2,
            linestyle=style,
            label=label,
        )
        share = LABEL_SHARE[label]  # a distinct height per curve: no collisions
        idx = int(np.searchsorted(ys, share))
        if label in LEADER_TEXT:  # crowded: put the label in free space, with a leader
            ax.annotate(
                label,
                xy=(xs[idx], share),
                xytext=LEADER_TEXT[label],
                color=INK,
                fontsize=7,
                va="center",
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
            )
        else:
            ax.text(xs[idx] * 1.6, share, label, color=INK, fontsize=7, va="center")
    ax.axvline(2.0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(
        2.0 * 0.92, 0.97, "$2\\times$", color=MUTED, fontsize=7, ha="right", va="top"
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("certified regret bound (log scale)")
    ax.set_ylabel("share of queries")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(output)
    plt.close(fig)
    logger.info("wrote %s", output)


def per_query_speedups(
    runs: Path, base: str, policies: tuple[str, ...]
) -> dict[str, npt.NDArray[np.float64]]:
    """Per-query time ratio base / policy for each policy (one value per query)."""
    out: dict[str, list[float]] = {p: [] for p in policies}
    for line in runs.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        plan, ms = record["policy_plan"], record["plan_ms"]
        for policy in policies:
            out[policy].append(ms[plan[base]] / max(ms[plan[policy]], 1e-9))
    return {p: np.array(v) for p, v in out.items()}


def execution_figure(runs: Path, output: Path) -> None:
    """Guard speedup against headroom, per query, on and off the error's side."""
    sources = (
        ("PostgreSQL (published)", "postgres", BLUE, "o", "upper", "lower"),
        ("DuckDB", "duckdb", ORANGE, "^", "lower", "upper"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.0), sharex=True, sharey=True)
    titles = ("guard on the error's side", "guard on the other side")
    for panel, (ax, title) in enumerate(zip(axes, titles, strict=True)):
        for label, source, colour, marker, own, other in sources:
            guard = f"{source}_{own if panel == 0 else other}"
            values = per_query_speedups(runs, f"{source}_raw", ("truth", guard))
            ax.scatter(
                values["truth"],
                values[guard],
                s=9,
                marker=marker,
                color=colour,
                alpha=0.55,
                linewidths=0,
                label=label,
            )
        ax.plot(
            [0.03, 5e4], [0.03, 5e4], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3))
        )
        ax.axhline(1.0, color=MUTED, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.04, 3e4)  # wide enough for every query of both runs (v2, v3)
        ax.set_ylim(0.04, 3e4)
        ax.set_title(title, fontsize=7, color=INK)
        ax.set_xlabel("headroom (true sizes)")
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("guard speedup")
    axes[0].text(200, 2500, "$y=x$", color=MUTED, fontsize=7, ha="right")
    axes[1].legend(loc="upper left", frameon=False, fontsize=6, handletextpad=0.2)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(output)
    plt.close(fig)
    logger.info("wrote %s", output)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts", type=Path, default=Path("results/pg_injection/subset_counts.jsonl")
    )
    parser.add_argument(
        "--joint",
        type=Path,
        default=Path("results/plan_certificate/plan_certificate_joint_pg_r025.json"),
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/pg_injection/v3/stats_inject.jsonl"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    cost_figure(args.counts, args.figure_dir / "icde_cost.pdf")
    certificate_figure(args.joint, args.figure_dir / "icde_certificates.pdf")
    execution_figure(args.runs, args.figure_dir / "icde_execution.pdf")


if __name__ == "__main__":
    main()
