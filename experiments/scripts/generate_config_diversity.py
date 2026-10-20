"""Generate the oracle-best-config diversity heatmap (Figure 1).

Reproducible companion to Table 1: uses the *same* synthetic query set
(3 families x relation counts 2..7 x 200 queries, executed-join labels) and
records, per family, the fraction of queries for which each of the 9 CEG
configurations is oracle-best. The resulting heatmap motivates per-query
selection (different families favor different configs).

Run with ``PYTHONHASHSEED=0`` for bit-reproducible numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.training.dataset import build_training_example
from src.training.workload import generate_synthetic_workload
from src.utils.types import ALL_CEG_CONFIGS

logger = logging.getLogger(__name__)

FAMILIES = ("path", "star", "cycle")
RELATION_COUNTS = range(2, 8)
_FAMILY_OFFSET = {"path": 0, "star": 1, "cycle": 2}


def compute_config_diversity(
    queries_per_family: int = 200, seed: int = 42
) -> dict[str, list[float]]:
    """Return, per family, the oracle-best fraction for each of the 9 configs.

    The query set and seeding match ``verify_bound_validity`` so the figure and
    the bound-validity table describe the identical workload.
    """
    config_names = [config.name for config in ALL_CEG_CONFIGS]
    diversity: dict[str, list[float]] = {}
    for family in FAMILIES:
        counts = np.zeros(len(ALL_CEG_CONFIGS), dtype=np.float64)
        total = 0
        for rel_count in RELATION_COUNTS:
            workload = generate_synthetic_workload(
                family=family,
                query_count=queries_per_family,
                relation_count=rel_count,
                seed=seed + rel_count * 100 + _FAMILY_OFFSET[family] * 10000,
            )
            for wq in workload:
                example = build_training_example(wq.query, wq.true_cardinality)
                counts[example.best_config_index] += 1.0
                total += 1
        diversity[family] = (counts / max(total, 1)).tolist()
    logger.info("Computed config diversity over %d configs", len(config_names))
    return diversity


def render_heatmap(diversity: dict[str, list[float]], output_dir: Path) -> None:
    """Render the family x config oracle-best heatmap to PDF and PNG."""
    config_names = [config.name for config in ALL_CEG_CONFIGS]
    families = list(diversity.keys())
    matrix = np.array([diversity[family] for family in families])

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(config_names)))
    ax.set_xticklabels(config_names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels([family.capitalize() for family in families], fontsize=9)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value >= 0.005:
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.6 else "black",
                    fontsize=7,
                )

    ax.set_xlabel("CEG configuration (path_length-aggregator)", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Oracle-best fraction", fontsize=8)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_dir / f"config_diversity.{suffix}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Wrote %s", path)
    plt.close(fig)


def main() -> None:
    """CLI entry point: compute diversity, write JSON, render heatmap."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-per-family", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/config_diversity/config_diversity.json"),
    )
    args = parser.parse_args()

    diversity = compute_config_diversity(
        queries_per_family=args.queries_per_family, seed=args.seed
    )
    payload = {
        "config_names": [config.name for config in ALL_CEG_CONFIGS],
        "queries_per_family": args.queries_per_family,
        "relation_counts": list(RELATION_COUNTS),
        "seed": args.seed,
        "oracle_best_fraction": diversity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", args.output)

    render_heatmap(diversity, args.figure_dir)


if __name__ == "__main__":
    main()
