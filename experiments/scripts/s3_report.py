"""Descriptive S3 numbers outside the registered analysis, and the supplement table.

``s3_analysis`` (registered, hashed) evaluates the predictions. This script adds
descriptive figures the registration did not cover but the paper cites:

* the certificate computation time per query, over all chains and blocks;
* the live sampling time of set A (first and second run).

It also renders the per-block table for the supplement. It reads the blocks
and ``s3_summary.json``; written after the analysis, not registered.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.s3_report
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

S3 = Path("results/pg_injection/s3")
DESIGN = Path("docs/vldb_2027/S3_DESIGN.json")
LABELS = {
    "native_upper": "Upper guard (A+B)",
    "native_lower": "Floor (A+B)",
    "native_bi": "Both guards (A+B)",
    "truth": "True sizes (A+B)",
    "bi_k2": "Synthetic bracket, k = 2 (A+B)",
    "bi_k8": "Synthetic bracket, k = 8 (A+B)",
    "certified": "Certified order (A)",
    "certified_with_sampling": "Certified, with same-block sampling (A)",
}


def stats(values: list[float]) -> dict[str, float]:
    """Median, max and count."""
    return {
        "median": statistics.median(values),
        "max": max(values),
        "n": float(len(values)),
    }


def main() -> None:
    """Write ``s3_report.json`` and ``S3_TABLE.md``."""
    blocks = sorted(S3.glob("block_*.jsonl"))
    rows = [
        json.loads(x)
        for p in blocks
        for x in p.read_text().splitlines()
        if '"query_id"' in x
    ]
    chains = [r["chain"] for r in rows if r.get("chain")]
    set_a = [r["chain"] for r in rows if r["set"] == "A" and r.get("chain")]
    summary = json.loads((S3 / "s3_summary.json").read_text())
    ref = json.loads(DESIGN.read_text())["reference_v3"]
    out: dict[str, Any] = {
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*blocks, S3 / "s3_summary.json"]
        },
        "post_registration": True,
        "certificate_ms_all": stats([c["certificate_ms"] for c in chains]),
        "certificate_ms_set_a": stats([c["certificate_ms"] for c in set_a]),
        "sampling_first_ms_set_a": stats([c["sampling_first_ms"] for c in set_a]),
        "sampling_second_ms_set_a": stats([c["sampling_second_ms"] for c in set_a]),
    }
    (S3 / "s3_report.json").write_text(json.dumps(out, indent=1) + "\n")

    valid = [str(b) for b in summary["valid_blocks"]]
    lines = [
        "# S3: six-block re-measurement (STATS-CEB, PostgreSQL 13.1)",
        "",
        "Registered in `docs/vldb_2027/PREDICTIONS_S3.md` before any block ran.",
        "Each value is a geometric-mean speedup (base / policy) over the set's",
        "queries.",
        "The pooled interval resamples queries and blocks independently",
        "(10,000 replicates). The v3 column is the main run on the same queries.",
        "",
        "| Comparison | v3 | Pooled [95%] | "
        + " | ".join(f"B{b}" for b in valid)
        + " |",
        "|---|---|---|" + "---|" * len(valid),
    ]
    for name, label in LABELS.items():
        c = summary["comparisons"][name]
        part = "A" if name.startswith("certified") else "A+B"
        r = ref[part][name]["geo_speedup"]["point"]
        lo, hi = c["two_way_95"]
        cells = " | ".join(f"{c['per_block'][b]:.2f}" for b in valid)
        lines.append(
            f"| {label} | {r:.2f} | {c['pooled']:.2f} [{lo:.2f}, {hi:.2f}] | {cells} |"
        )
    drift = summary["native_raw_block_drift"]
    lines += [
        "",
        "Native-plan drift per block (block mean / mean over blocks): "
        + ", ".join(f"B{b} {drift[b]:.2f}" for b in valid)
        + f"; median per-query max/min over blocks "
        f"{summary['native_raw_median_max_over_min']:.2f}.",
        f"Verdicts: {json.dumps(summary['verdicts'])}.",
        f"Live sample counts: {json.dumps(summary['live_counts'])}; certified orders: "
        f"{json.dumps(summary['certified_orders'])}.",
        "Certificate computation per query: median "
        f"{out['certificate_ms_all']['median']:.2f} ms, max "
        f"{out['certificate_ms_all']['max']:.0f} ms "
        f"({out['certificate_ms_all']['n']:.0f} chains).",
    ]
    (S3 / "S3_TABLE.md").write_text("\n".join(lines) + "\n")
    logger.info("\n".join(lines))


if __name__ == "__main__":
    main()
