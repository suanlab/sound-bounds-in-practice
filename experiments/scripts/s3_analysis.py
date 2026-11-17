"""Analyse the S3 blocks against the registered predictions.

Registered, with its thresholds, in ``docs/vldb_2027/PREDICTIONS_S3.md``
(hashed before any S3 block ran). Inputs are the blocks
``results/pg_injection/s3/block_<b>.jsonl`` written by
``experiments.pg_injection.run_blocks_s3``. The output is
``results/pg_injection/s3/s3_summary.json``.

Per block, query and plan, the time is the median of the timed repetitions,
capped at 300 s (censored = 300 s). A comparison's per-query value is
log(base / variant). Per block it is summarised by the geometric mean over its
queries. Pooled over blocks, it is summarised by the geometric mean over all
(block, query) cells, with two intervals:

* a two-way bootstrap that resamples queries and blocks independently, so that
  both levels of variation are in the interval;
* a query-only bootstrap, comparable with the paper's.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.s3_analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DESIGN = Path("docs/vldb_2027/S3_DESIGN.json")
S3 = Path("results/pg_injection/s3")
REPLICATES = 10_000
MIN_VALID_BLOCKS = 4
GUARDS = ("native_upper", "native_lower", "native_bi")
COMPARISONS: dict[str, tuple[str, str, str]] = {
    # name: (base, variant, query set)
    "native_upper": ("native_raw", "native_upper", "A+B"),
    "native_lower": ("native_raw", "native_lower", "A+B"),
    "native_bi": ("native_raw", "native_bi", "A+B"),
    "truth": ("native_raw", "truth", "A+B"),
    "bi_k2": ("native_raw", "bi_k2", "A+B"),
    "bi_k8": ("native_raw", "bi_k8", "A+B"),
    "certified": ("native", "certified", "A"),
    "certified_with_sampling": ("native", "chain_total", "A"),
    "certified_with_warm_sampling": ("native", "chain_warm", "A"),
}


def load_blocks(directory: Path) -> dict[int, dict[str, Any]]:
    """Header and per-query rows of every block file."""
    blocks: dict[int, dict[str, Any]] = {}
    for path in sorted(directory.glob("block_*.jsonl")):
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        header = next(r for r in rows if "block_header" in r)
        blocks[int(header["block_header"])] = {
            "header": header,
            "rows": {int(r["query_id"]): r for r in rows if "query_id" in r},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return blocks


def times(row: dict[str, Any]) -> dict[str, float]:
    """Per-policy time (ms, capped) of one query in one block, plus the chain."""
    t = {
        p: max(float(row["timings"][d]["median_ms"]), 1e-3)
        for p, d in row["policy_plan"].items()
    }
    chain = row.get("chain")
    if chain and chain.get("certified") and "certified" in t:
        fixed = float(chain["certificate_ms"]) + float(chain.get("plan_ms", 0.0))
        t["chain_total"] = float(chain["sampling_first_ms"]) + fixed + t["certified"]
        t["chain_warm"] = float(chain["sampling_second_ms"]) + fixed + t["certified"]
    return t


def two_way(
    cells: npt.NDArray[np.float64], rng: np.random.Generator
) -> tuple[float, float]:
    """95% percentile interval of exp(mean) resampling rows (queries) and columns."""
    nq, nb = cells.shape
    stats = np.empty(REPLICATES)
    for i in range(REPLICATES):
        qi = rng.integers(0, nq, nq)
        bi = rng.integers(0, nb, nb)
        stats[i] = cells[np.ix_(qi, bi)].mean()
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(math.exp(lo)), float(math.exp(hi))


def query_only(
    cells: npt.NDArray[np.float64], rng: np.random.Generator
) -> tuple[float, float]:
    """95% percentile interval resampling queries only (blocks fixed)."""
    nq = cells.shape[0]
    stats = np.array([cells[rng.integers(0, nq, nq)].mean() for _ in range(REPLICATES)])
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(math.exp(lo)), float(math.exp(hi))


def summarise(
    blocks: dict[int, dict[str, Any]], design: dict[str, Any]
) -> dict[str, Any]:
    """Every comparison per block and pooled, the checks, and the verdicts."""
    set_a, set_b = design["set_a"], design["set_b"]
    wanted = set(set_a) | set(set_b)
    valid = sorted(b for b, v in blocks.items() if wanted <= set(v["rows"]))
    out: dict[str, Any] = {
        "blocks": {
            str(b): {
                "sha256": v["sha256"],
                "n_queries": len(v["rows"]),
                "valid": b in valid,
                "start": v["header"]["start"],
            }
            for b, v in blocks.items()
        },
        "valid_blocks": valid,
    }
    if len(valid) < 2:
        out["status"] = "fewer than two valid blocks"
        return out
    per: dict[int, dict[int, dict[str, float]]] = {
        b: {q: times(blocks[b]["rows"][q]) for q in wanted} for b in valid
    }
    rng = np.random.default_rng(0)
    comps: dict[str, Any] = {}
    for name, (base, variant, which) in COMPARISONS.items():
        ids = sorted(set_a if which == "A" else wanted)
        ids = [
            q
            for q in ids
            if all(variant in per[b][q] and base in per[b][q] for b in valid)
        ]
        if not ids:
            continue
        cells = np.array(
            [
                [math.log(per[b][q][base] / per[b][q][variant]) for b in valid]
                for q in ids
            ]
        )
        by_block = np.exp(cells.mean(axis=0))
        comps[name] = {
            "n_queries": len(ids),
            "per_block": {
                str(b): float(g) for b, g in zip(valid, by_block, strict=True)
            },
            "pooled": float(math.exp(cells.mean())),
            "two_way_95": two_way(cells, rng),
            "query_only_95": query_only(cells, rng),
            "sd_log_block": float(np.std(np.log(by_block), ddof=1)),
            "share_faster_cells": float((cells > math.log(1.1)).mean()),
            "share_slower_cells": float((cells < -math.log(1.1)).mean()),
        }
    for k in ("bi_k2", "bi_k8"):
        if k in comps and "truth" in comps:
            comps[k]["share_of_truth_log_pooled"] = math.log(
                comps[k]["pooled"]
            ) / math.log(comps["truth"]["pooled"])
            comps[k]["share_of_truth_log_per_block"] = {
                b: math.log(comps[k]["per_block"][b])
                / math.log(comps["truth"]["per_block"][b])
                for b in comps[k]["per_block"]
            }
    out["comparisons"] = comps

    # Repeatability of the same plan across blocks: native_raw.
    nat = np.array(
        [[math.log(per[b][q]["native_raw"]) for b in valid] for q in sorted(wanted)]
    )
    centred = nat - nat.mean(axis=1, keepdims=True)
    out["native_raw_block_drift"] = {
        str(b): float(math.exp(centred[:, i].mean())) for i, b in enumerate(valid)
    }
    out["native_raw_median_max_over_min"] = float(
        np.median(np.exp(nat.max(axis=1) - nat.min(axis=1)))
    )

    # Checks: live sample counts and certified orders.
    draws = [
        d
        for b in valid
        for q in wanted
        if (c := blocks[b]["rows"][q].get("chain"))
        for d in c["draws"]
    ]
    out["live_counts"] = {
        "n_draws": len(draws),
        "n_censored": sum(d["censored"] for d in draws),
        "n_mismatch": sum(d["matches_stored"] is False for d in draws),
    }
    stored = {
        r["query_id"]: r
        for r in json.loads(
            Path("results/plan_certificate/selective_sampling_v3.json").read_text()
        )["per_query"]
    }
    out["certified_orders"] = {
        "n_a_cells": len(set_a) * len(valid),
        "n_live_certified": sum(
            bool((c := blocks[b]["rows"][q].get("chain")) and c["certified"])
            for b in valid
            for q in set_a
        ),
        "n_order_differs_from_v3": sum(
            bool(
                (c := blocks[b]["rows"][q].get("chain"))
                and c["certified"]
                and c["order"] != stored[q]["selective_order"]
            )
            for b in valid
            for q in set_a
        ),
    }
    censored = {
        str(b): sum(
            t["censored"]
            for q in wanted
            for t in blocks[b]["rows"][q]["timings"].values()
        )
        for b in valid
    }
    out["censored_plans_per_block"] = censored
    out["verdicts"] = verdicts(comps, out, len(valid))
    out["status"] = (
        "valid"
        if len(valid) >= MIN_VALID_BLOCKS
        else "inconclusive (too few valid blocks)"
    )
    return out


def verdicts(comps: dict[str, Any], out: dict[str, Any], nv: int) -> dict[str, Any]:
    """The registered predictions P1-P6, evaluated with their registered thresholds."""

    def blocks_of(name: str) -> list[float]:
        return list(comps[name]["per_block"].values())

    v: dict[str, Any] = {}
    v["P1_native_guards_small"] = all(
        0.95 <= comps[g]["pooled"] <= 1.12
        and all(0.90 <= x <= 1.18 for x in blocks_of(g))
        for g in GUARDS
    )
    v["P2_headroom"] = comps["truth"]["pooled"] >= 1.15 and all(
        x >= 1.08 for x in blocks_of("truth")
    )
    ordered = sum(
        comps["truth"]["per_block"][b] >= 0.99 * comps["bi_k2"]["per_block"][b]
        and comps["bi_k2"]["per_block"][b] >= 0.99 * comps["bi_k8"]["per_block"][b]
        for b in comps["truth"]["per_block"]
    )
    v["P3_tightness_order"] = (
        ordered >= nv - 1
        and 0.45 <= comps["bi_k2"]["share_of_truth_log_pooled"] <= 0.95
        and 0.05 <= comps["bi_k8"]["share_of_truth_log_pooled"] <= 0.55
    )
    v["P3_blocks_ordered"] = ordered
    if "certified" not in comps or "certified_with_sampling" not in comps:
        # No set-A query certified in every valid block: P4-P6 cannot hold.
        v["P4_certified_faster"] = v["P5_sampling_dominates"] = False
        v["P6_reproducible_chain"] = False
        return v
    c = comps["certified"]
    v["P4_certified_faster"] = (
        c["pooled"] >= 1.10
        and c["two_way_95"][0] > 1.0
        and sum(x >= 1.0 for x in blocks_of("certified")) >= nv - 1
    )
    s = comps["certified_with_sampling"]
    v["P5_sampling_dominates"] = s["two_way_95"][1] < 1.0 and all(
        x < 1.0 for x in blocks_of("certified_with_sampling")
    )
    v["P6_reproducible_chain"] = (
        out["live_counts"]["n_mismatch"] == 0
        and out["certified_orders"]["n_order_differs_from_v3"] == 0
        and out["certified_orders"]["n_live_certified"]
        == out["certified_orders"]["n_a_cells"]
    )
    return v


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=S3)
    parser.add_argument("--output", type=Path, default=S3 / "s3_summary.json")
    parser.add_argument("--design", type=Path, default=DESIGN, help="tests only")
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    out = summarise(load_blocks(args.dir), design)
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    logger.info(
        json.dumps({k: out.get(k) for k in ("status", "valid_blocks", "verdicts")})
    )


if __name__ == "__main__":
    main()
