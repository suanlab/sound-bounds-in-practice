"""Summarise the forced-order runs (``run_forced_orders``).

For each forced variant, the geometric-mean speedup of the variant over
PostgreSQL's native plan (native time / variant time, so < 1 means slower),
with query-clustered intervals, the share of queries faster and slower beyond a
+-10% band, and censoring counts:

* ``native_lo`` vs ``native``: the cost of forcing itself (queries whose native
  plan is left-deep);
* ``cout_opt`` vs ``native`` and vs ``native_lo``: what the true-C_out-optimal
  left-deep order is worth in the engine;
* ``certified`` vs ``native`` and vs ``cout_opt``: the certified order, on
  certified queries, also charging the sampling time.

It also reports how often the certified order *is* the C_out-optimal one (the
``certified -> cout_opt`` comparison is then a repeatability check), the
certified comparison split by whether plan choice matters (worst over best true
C_out of the left-deep orders above 2, as in ``certificate_context``), and on
how many queries forcing PostgreSQL's own left-deep order changes its physical
plan (join methods, inner/outer sides).

Usage::

    PYTHONHASHSEED=0 python -m experiments.pg_injection.summarize_forced_orders
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import _read_jsonl
from experiments.scripts.bootstrap_ci import cluster_bootstrap
from experiments.scripts.certificate_context import true_spread

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COMPARISONS = (
    ("native", "native_lo"),
    ("native", "cout_opt"),
    ("native_lo", "cout_opt"),
    ("native", "certified"),
    ("cout_opt", "certified"),
)


def geo(rows: list[dict[str, Any]]) -> float:
    """Geometric mean of base / variant."""
    return math.exp(sum(math.log(r["base"] / r["variant"]) for r in rows) / len(rows))


def shape(node: dict[str, Any]) -> object:
    """Physical plan shape: operators, join types, sides and scanned aliases."""
    return (
        node.get("Node Type"),
        node.get("Join Type"),
        node.get("Parent Relationship"),
        node.get("Alias"),
        tuple(shape(child) for child in node.get("Plans", [])),
    )


def speedups(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    """Geometric-mean speedup with and without sampling on a subset of rows."""
    return {
        "n": len(rows),
        "geo_speedup": cluster_bootstrap(rows, geo, replicates=replicates, seed=seed),
        "geo_speedup_with_sampling": cluster_bootstrap(
            [{**r, "variant": r["end_to_end"]} for r in rows],
            geo,
            replicates=replicates,
            seed=seed,
        ),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=Path("results/pg_injection/v3/forced_orders.jsonl")
    )
    parser.add_argument(
        "--sizes",
        type=Path,
        default=Path("results/plan_guard/subjoin_sizes_maxdeg_fixed.jsonl"),
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pg_injection/v3/forced_orders_summary.json"),
    )
    args = parser.parse_args()

    records = _read_jsonl(args.runs)
    out: dict[str, Any] = {
        "n_queries": len(records),
        "n_high_load": sum(r["high_load"] for r in records),
        "n_native_left_deep": sum("native_lo" in r["ms"] for r in records),
        "n_certified": sum("certified" in r["ms"] for r in records),
        "comparisons": {},
    }
    for base, variant in COMPARISONS:
        rows = [
            {
                "query_id": r["query_id"],
                "base": max(r["ms"][base], 1e-3),
                "variant": max(r["ms"][variant], 1e-3),
                "censored": r["censored"][base] or r["censored"][variant],
                "end_to_end": max(r["ms"][variant] + r["sampling_ms"], 1e-3),
            }
            for r in records
            if base in r["ms"] and variant in r["ms"]
        ]
        if not rows:
            continue
        entry: dict[str, Any] = {
            "n": len(rows),
            "geo_speedup": cluster_bootstrap(
                rows, geo, replicates=args.replicates, seed=args.seed
            ),
            "share_faster": sum(r["base"] / r["variant"] > 1.1 for r in rows)
            / len(rows),
            "share_slower": sum(r["base"] / r["variant"] < 1 / 1.1 for r in rows)
            / len(rows),
            "n_censored": sum(r["censored"] for r in rows),
        }
        if variant == "certified" and base == "native":
            with_sampling = [{**r, "variant": r["end_to_end"]} for r in rows]
            entry["geo_speedup_with_sampling"] = cluster_bootstrap(
                with_sampling, geo, replicates=args.replicates, seed=args.seed
            )
        out["comparisons"][f"{base}->{variant}"] = entry
        g = entry["geo_speedup"]
        logger.info(
            "%-22s n=%3d geo %.2fx [%.2f, %.2f] faster %.0f%% slower %.0f%% cens %d",
            f"{base}->{variant}",
            entry["n"],
            g["point"],
            g["ci_low"],
            g["ci_high"],
            100 * entry["share_faster"],
            100 * entry["share_slower"],
            entry["n_censored"],
        )
    sizes = {
        r["query_id"]: r
        for r in (json.loads(x) for x in args.sizes.read_text().splitlines() if x)
    }
    certified = [r for r in records if "certified" in r["ms"]]
    same = [r for r in certified if r["orders"]["certified"] == r["orders"]["cout_opt"]]
    out["n_certified_is_cout_opt"] = len(same)
    rows = [
        {
            "query_id": r["query_id"],
            "base": max(r["ms"]["native"], 1e-3),
            "variant": max(r["ms"]["certified"], 1e-3),
            "end_to_end": max(r["ms"]["certified"] + r["sampling_ms"], 1e-3),
            "matters": true_spread(sizes[r["query_id"]]) > 2.0,
        }
        for r in certified
    ]
    out["certified_by_choice"] = {
        label: speedups(
            [r for r in rows if r["matters"] is flag], args.replicates, args.seed
        )
        for label, flag in (("matters", True), ("does_not_matter", False))
    }
    left_deep = [r for r in records if "native_lo" in r["ms"]]
    changed = [
        r
        for r in left_deep
        if shape(r["plans"]["native_lo"]) != shape(r["plans"]["native"])
    ]
    out["n_forcing_changes_physical_plan"] = len(changed)
    for label, subset in (
        ("forcing_changed", changed),
        ("forcing_unchanged", [r for r in left_deep if r not in changed]),
    ):
        out[f"native->native_lo|{label}"] = {
            "n": len(subset),
            "geo": geo(
                [
                    {
                        "base": max(r["ms"]["native"], 1e-3),
                        "variant": max(r["ms"]["native_lo"], 1e-3),
                    }
                    for r in subset
                ]
            ),
        }
    logger.info(
        "certified = cout_opt on %d of %d; forcing changes the plan on %d of %d",
        len(same),
        len(certified),
        len(changed),
        len(left_deep),
    )
    for label, entry in out["certified_by_choice"].items():
        logger.info(
            "certified vs native, choice %s: n=%d %.2fx, with sampling %.2fx",
            label,
            entry["n"],
            entry["geo_speedup"]["point"],
            entry["geo_speedup_with_sampling"]["point"],
        )
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
