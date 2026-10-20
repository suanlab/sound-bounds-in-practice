"""Figure 1: the certificate-utility regime map, from real-benchmark measurements.

Left panel: intervention rate per (certificate family x error direction) on real
STATS-CEB --- the complementarity result. Upper certificates fire only on
overestimates, the sampling lower certificate only on underestimates, and each is
blind to the other direction *by construction*, not by looseness.

Right panel: the cost axis --- what each certificate costs relative to executing
the join it is meant to avoid, with the fraction of queries where the certificate
itself is intractable.

Together the two panels state the paper's thesis in one figure: a certificate is
useful only when its direction matches the error and its cost is affordable.

Usage::

    python -m experiments.scripts.plot_regime_map \
        --table results/utility_table/real_stats_bidirectional.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Embed TrueType (Type 42), not Type 3, fonts so the PDF prints cleanly.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_LABELS = {
    "lpbound": "LpBound\n(upper)",
    "safebound": "SafeBound\n(upper)",
    "sample_lb": "sample-LB\n(lower)",
}
_ORDER = ["lpbound", "safebound", "sample_lb"]


def fire_rates(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Fire rate per certificate, split by error direction."""
    out: dict[str, dict[str, float]] = {}
    for cert in _ORDER:
        subset = [r for r in rows if r["certificate"] == cert]
        if not subset:
            continue
        entry: dict[str, float] = {}
        for regime, sign in (("over", "1"), ("under", "-1")):
            part = [r for r in subset if r["direction"] == sign]
            fired = sum(1 for r in part if r["guard_fired"] == "True")
            entry[regime] = fired / len(part) if part else float("nan")
        out[cert] = entry
    return out


def cost_ratios(rows: list[dict[str, str]]) -> dict[str, float]:
    """Median measured cost ratio per certificate (0 for statistics lookups)."""
    out: dict[str, float] = {}
    for cert in _ORDER:
        values = [
            float(r["cost_ratio"])
            for r in rows
            if r["certificate"] == cert and r["cost_ratio"] not in ("", "nan")
        ]
        out[cert] = float(np.median(values)) if values else 0.0
    return out


def render(
    rates: dict[str, dict[str, float]],
    costs: dict[str, float],
    intractable: float,
    output_dir: Path,
) -> None:
    """Draw and save the two-panel regime map."""
    certs = [c for c in _ORDER if c in rates]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4), width_ratios=[1.6, 1])

    x = np.arange(len(certs))
    width = 0.38
    over = [100 * rates[c]["over"] for c in certs]
    under = [100 * rates[c]["under"] for c in certs]
    ax1.bar(x - width / 2, over, width, label="overestimates", color="#4C72B0")
    ax1.bar(x + width / 2, under, width, label="underestimates", color="#C44E52")
    for xi, (o, u) in enumerate(zip(over, under, strict=True)):
        ax1.text(xi - width / 2, o + 1.5, f"{o:.0f}%", ha="center", fontsize=8)
        ax1.text(xi + width / 2, u + 1.5, f"{u:.0f}%", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([_LABELS[c] for c in certs], fontsize=9)
    ax1.set_ylabel("intervention rate (%)", fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.set_title(
        "Direction decides whether a certificate can fire\n"
        "(all three are valid on 100% of queries)",
        fontsize=9.5,
    )
    ax1.legend(fontsize=8, loc="upper center")
    ax1.grid(axis="y", alpha=0.3)

    values = [costs[c] for c in certs]
    colors = ["#55A868" if v < 0.05 else "#DD8452" for v in values]
    ax2.bar(x, values, 0.55, color=colors)
    for xi, v in enumerate(values):
        ax2.text(
            xi,
            v + 0.02,
            "~0 (lookup)" if v < 0.05 else f"{v:.2f}",
            ha="center",
            fontsize=8,
        )
    ax2.axhline(1.0, ls="--", lw=1, color="grey")
    ax2.text(
        len(certs) - 0.5,
        1.02,
        "cost of executing the join",
        fontsize=7.5,
        ha="right",
        color="grey",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([_LABELS[c] for c in certs], fontsize=9)
    ax2.set_ylabel("cost / full join execution", fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_title(
        f"and cost decides whether it is affordable\n"
        f"({intractable:.0f}% of queries: lower cert. intractable)",
        fontsize=9.5,
    )
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_dir / f"regime_map.{suffix}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("wrote %s", path)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("results/utility_table/real_stats_bidirectional.csv"),
    )
    parser.add_argument("--total-queries", type=int, default=146)
    parser.add_argument("--completed-queries", type=int, default=130)
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    with args.table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rates = fire_rates(rows)
    costs = cost_ratios(rows)
    intractable = 100 * (
        (args.total_queries - args.completed_queries) / args.total_queries
    )
    for cert, entry in rates.items():
        logger.info(
            "%-10s over=%.1f%% under=%.1f%% cost=%.3f",
            cert,
            100 * entry["over"],
            100 * entry["under"],
            costs[cert],
        )
    render(rates, costs, intractable, args.figure_dir)


if __name__ == "__main__":
    main()
