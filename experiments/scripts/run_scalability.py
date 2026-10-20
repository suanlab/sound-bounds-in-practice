"""Measure Safe-CEG runtime breakdown by relation count.

Times each pipeline component across *all* queries of each size (with a warmup),
and reports median / mean / p95 plus hardware metadata, rather than relying on a
single first query.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.core.ceg import build_optimistic_ceg, build_pessimistic_ceg, pessimistic_bound
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.models.features import extract_features
from src.models.gnn import SafeCEGGNN
from src.training.workload import generate_synthetic_workload
from src.utils.metrics import _percentile

if TYPE_CHECKING:
    from collections.abc import Callable


def _time_once(fn: Callable[..., object], *args: object, **kwargs: object) -> float:
    """Wall-clock time of a single call, in milliseconds."""
    start = time.perf_counter()
    fn(*args, **kwargs)
    return (time.perf_counter() - start) * 1000.0


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median": _percentile(ordered, 0.5),
        "mean": sum(ordered) / len(ordered),
        "p95": _percentile(ordered, 0.95),
        "n": float(len(ordered)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--relation-counts",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8],
    )
    parser.add_argument("--queries-per-size", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("results/scalability/runtime.json")
    )
    args = parser.parse_args()

    estimator = BaselineCEGEstimator()
    from src.utils.types import AggregatorChoice, CEGConfig, PathLengthChoice

    cfg = CEGConfig(PathLengthChoice.MAX_HOP, AggregatorChoice.MAX)
    rows = []

    for n_rels in args.relation_counts:
        print(f"\n=== {n_rels} relations ===")
        wl = generate_synthetic_workload("path", args.queries_per_size, n_rels, seed=42)
        queries = [w.query for w in wl]
        n_subsets = len(queries[0].connected_subsets())

        # Warm up caches / lazy imports on the first query (untimed).
        estimator.estimate_all_configs(queries[0])
        warm_features = extract_features(queries[0])
        model = SafeCEGGNN(
            node_feature_dim=warm_features.node_features.shape[1],
            global_feature_dim=warm_features.global_features.shape[1],
        )
        model.eval()
        with torch.no_grad():
            model(warm_features)

        all9_ms: list[float] = []
        bound_ms: list[float] = []
        ml_ms: list[float] = []
        full_ms: list[float] = []
        for query in queries:
            t_all9 = _time_once(estimator.estimate_all_configs, query)
            t_bound = _time_once(pessimistic_bound, query)
            feats = extract_features(query)
            with torch.no_grad():
                t_ml = _time_once(model, feats)
            all9_ms.append(t_all9)
            bound_ms.append(t_bound)
            ml_ms.append(t_ml)
            full_ms.append(t_all9 + t_bound + t_ml)

        row = {
            "n_relations": n_rels,
            "n_connected_subsets": n_subsets,
            "queries_measured": len(queries),
            "ceg_optimistic_ms_median": round(
                _percentile(
                    sorted(_time_once(build_optimistic_ceg, q) for q in queries), 0.5
                ),
                3,
            ),
            "ceg_pessimistic_ms_median": round(
                _percentile(
                    sorted(_time_once(build_pessimistic_ceg, q) for q in queries), 0.5
                ),
                3,
            ),
            "single_config_ms_median": round(
                _percentile(
                    sorted(_time_once(estimator.estimate, q, cfg) for q in queries), 0.5
                ),
                3,
            ),
            "all_9_configs_ms": {k: round(v, 3) for k, v in _stats(all9_ms).items()},
            "pessimistic_bound_ms": {
                k: round(v, 3) for k, v in _stats(bound_ms).items()
            },
            "ml_inference_ms": {k: round(v, 3) for k, v in _stats(ml_ms).items()},
            "full_pipeline_ms": {k: round(v, 3) for k, v in _stats(full_ms).items()},
        }
        rows.append(row)
        full = row["full_pipeline_ms"]
        ml = row["ml_inference_ms"]
        print(
            f"  subsets={n_subsets:>6d}  "
            f"full[median/mean/p95]={full['median']}/{full['mean']}/{full['p95']}ms  "
            f"ml_median={ml['median']}ms  (n={len(queries)})"
        )

    results = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch_threads": torch.get_num_threads(),
        },
        "queries_per_size": args.queries_per_size,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
