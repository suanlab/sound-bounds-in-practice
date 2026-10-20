"""Figure: how bracket tightness turns into plan-level certificates.

Reads ``results/plan_certificate/plan_certificate.json`` (pre-registered C4,
``docs/c4_preregistration.md``) and draws two panels over the nine
upper x lower brackets.

Left: median bracket width $w = \\max_S \\ln(B/L)$ on the x axis against the
share of queries whose plan is certified (single survivor) and the mean
dominance-pruning rate, with 95% query-clustered bootstrap intervals. This is
the tightness-to-utility curve: it shows how much narrower a sound bracket
would have to be before it decides plans, not just cardinalities.

Right: the certified regret bound $G(Q)$ per bracket (median, with the p95 as a
whisker) on a log axis against the ``useful'' line $G \\le 2$, beside the true
regret of the bracket's own minimax plan (post-hoc; the plan the bracket
recommends is near-optimal, but the bracket cannot prove it).

Usage::

    python -m experiments.scripts.plot_plan_certificate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Embed TrueType (Type 42), not Type 3, fonts so the PDF prints cleanly.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

UPPERS = ("maxdeg", "safebound", "lpbound")
LOWERS = ("one", "sample0.10", "sample0.25")
UPPER_LABEL = {"maxdeg": "max-degree", "safebound": "SafeBound", "lpbound": "LpBound"}
LOWER_LABEL = {"one": "L=1", "sample0.10": "sample 10%", "sample0.25": "sample 25%"}
UPPER_COLOR = {"maxdeg": "#DD8452", "safebound": "#4C72B0", "lpbound": "#55A868"}
LOWER_MARKER = {"one": "o", "sample0.10": "s", "sample0.25": "^"}
USEFUL_G = 2.0


def _ci(entry: dict[str, Any]) -> tuple[float, float, float]:
    return float(entry["point"]), float(entry["ci_low"]), float(entry["ci_high"])


def render(
    summary: dict[str, Any],
    per_query: list[dict[str, Any]],
    n_eligible: int,
    output_dir: Path,
) -> None:
    """Draw and save the certified-bound-versus-achieved-regret figure."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    keys = [
        (u, lo, f"{u}x{lo}")
        for u in UPPERS
        for lo in LOWERS
        if "certified_share" in summary.get(f"{u}x{lo}", {})
    ]
    x = np.arange(len(keys))
    for xi, (upper, lower, key) in enumerate(keys):
        entry = summary[key]
        color = UPPER_COLOR[upper]
        med = float(entry["median_G"]["point"])
        p95 = float(entry["p95_G"]["point"])
        ax.plot([xi - 0.12, xi - 0.12], [med, p95], color=color, lw=1.2)
        ax.plot(xi - 0.12, med, LOWER_MARKER[lower], color=color, ms=7, zorder=3)
        ax.plot(xi - 0.12, p95, "_", color=color, ms=9, mew=1.5)
        rows = [
            q["brackets"][key] for q in per_query if not q["brackets"][key]["flagged"]
        ]
        regrets = sorted(r["minimax_regret"] for r in rows)
        r_med = float(np.median(regrets))
        r_p95 = float(np.percentile(regrets, 95))
        ax.plot([xi + 0.12, xi + 0.12], [r_med, r_p95], color=color, lw=1.2, ls=":")
        ax.plot(
            xi + 0.12,
            r_med,
            LOWER_MARKER[lower],
            color=color,
            mfc="white",
            ms=7,
            zorder=3,
        )
        ax.plot(xi + 0.12, r_p95, "_", color=color, ms=9, mew=1.5)
        ax.text(
            xi,
            6e-2,
            f"{100 * float(entry['mean_pruning_rate']['point']):.1f}%",
            ha="center",
            fontsize=7,
            color="dimgrey",
        )
    ax.text(-0.45, 6e-2, "pruned:", ha="right", fontsize=7, color="dimgrey")
    ax.axhline(USEFUL_G, ls="--", lw=1, color="grey")
    ax.text(
        -0.45,
        USEFUL_G * 1.3,
        "useful: G $\\leq$ 2",
        fontsize=7.5,
        ha="left",
        color="grey",
    )
    for upper in UPPERS:
        ax.plot([], [], "s", color=UPPER_COLOR[upper], label=UPPER_LABEL[upper])
    for lower in LOWERS:
        ax.plot([], [], LOWER_MARKER[lower], color="grey", label=LOWER_LABEL[lower])
    ax.plot([], [], "o-", color="grey", label="certified bound $G(Q)$")
    ax.plot([], [], "o:", color="grey", mfc="white", label="minimax plan's true regret")
    ax.set_yscale("log")
    ax.set_ylim(3e-2, 3e14)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{UPPER_LABEL[u]}\n{LOWER_LABEL[lo]}" for u, lo, _ in keys],
        fontsize=6.5,
        rotation=55,
        ha="right",
    )
    ax.set_ylabel("plan regret (median, p95 whisker)", fontsize=9)
    ax.set_title(
        f"What a sound bracket certifies vs. what its plan achieves\n"
        f"(STATS-CEB, {n_eligible} queries, 0 certified on every bracket)",
        fontsize=9.5,
    )
    ax.legend(fontsize=6.5, ncol=3, loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.3, which="major")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_dir / f"plan_certificate.{suffix}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("wrote %s", path)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/plan_certificate/plan_certificate.json"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    summary = payload["summary"]
    for upper in UPPERS:
        for lower in LOWERS:
            entry = summary.get(f"{upper}x{lower}", {})
            if "certified_share" not in entry:
                continue
            logger.info(
                "%-9s x %-11s n=%3d width %5.2f  certified %5.1f%%  pruning %5.1f%%"
                "  G med %9.2f p95 %11.2f",
                upper,
                lower,
                entry["n_queries"],
                entry["median_width"]["point"],
                100 * entry["certified_share"]["point"],
                100 * entry["mean_pruning_rate"]["point"],
                entry["median_G"]["point"],
                entry["p95_G"]["point"],
            )
    render(
        summary,
        payload["per_query"],
        int(payload["n_queries_eligible"]),
        args.figure_dir,
    )


if __name__ == "__main__":
    main()
