"""Fix the query sets and reference values of the S3 block re-measurement.

S3 (``REVIEW_VLDB.md``, R3-M2 / R4-M1 / R5-M3) asks whether the small runtime
effects and the sampling cost of the paper survive independent time blocks with
a randomised plan order. This script is run once, before any S3 measurement.
Its output, ``docs/vldb_2027/S3_DESIGN.json``, is hashed together with
``docs/vldb_2027/PREDICTIONS_S3.md``. It reads only the stored v3 runs; no
S3 measurement exists when it runs.

* Set A: the STATS-CEB queries whose certified forced order was executed and
  where the choice matters (true spread of the left-deep orders above 2), as in
  ``summarize_forced_orders`` (21 queries).
* Excluded: queries whose native and true-size plans both hit the 300 s cap in
  v3 (no runtime information; ``REVIEW_VLDB.md`` R5-m1).
* Set B: 40 of the remaining queries, allocated to strata of the number of
  relations in proportion to stratum size (largest remainder), drawn with a
  fixed seed.
* Reference values: the v3 geometric-mean speedups of every S3 comparison on
  A, B and A+B, with query-bootstrap intervals, which S3 is to reproduce.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.s3_design
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from experiments.scripts.bootstrap_ci import cluster_bootstrap
from experiments.scripts.certificate_context import true_spread

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

V3 = Path("results/pg_injection/v3")
INJECT = V3 / "stats_inject.jsonl"
TIGHT = V3 / "tightness_points_stats.jsonl"
FORCED = V3 / "forced_orders.jsonl"
SIZES = Path("results/plan_guard/subjoin_sizes_maxdeg_fixed.jsonl")
OUT = Path("docs/vldb_2027/S3_DESIGN.json")
SEED = 20261116
N_B = 40
REPLICATES = 10_000

# comparison -> (base policy, variant policy); policies as named in S3.
COMPARISONS: dict[str, tuple[str, str]] = {
    "native_upper": ("native_raw", "native_upper"),
    "native_lower": ("native_raw", "native_lower"),
    "native_bi": ("native_raw", "native_bi"),
    "truth": ("native_raw", "truth"),
    "bi_k2": ("native_raw", "bi_k2"),
    "bi_k8": ("native_raw", "bi_k8"),
    "certified": ("native", "certified"),
    "certified_with_sampling": ("native", "certified_chain"),
}


def _jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows = (json.loads(x) for x in path.read_text().splitlines() if x.strip())
    return {int(r["query_id"]): r for r in rows}


def v3_times(
    query_id: int, inj: dict[int, Any], tig: dict[int, Any], forced: dict[int, Any]
) -> dict[str, float]:
    """The v3 median time (ms, capped) of every S3 policy of one query."""
    rec = inj[query_id]
    out = {
        p: float(rec["plan_ms"][rec["policy_plan"][p]])
        for p in (
            "native",
            "native_raw",
            "native_upper",
            "native_lower",
            "native_bi",
            "truth",
        )
    }
    t = tig[query_id]
    out["tight_raw"] = float(t["plan_ms"][t["policy_plan"]["raw"]])
    for p in ("bi_k2", "bi_k8"):
        out[p] = float(t["plan_ms"][t["policy_plan"][p]])
    f = forced.get(query_id)
    if f is not None and "certified" in f["ms"]:
        out["forced_native"] = float(f["ms"]["native"])
        out["certified"] = float(f["ms"]["certified"])
        out["certified_chain"] = float(f["ms"]["certified"]) + float(f["sampling_ms"])
    return out


def references(ids: list[int], times: dict[int, dict[str, float]]) -> dict[str, Any]:
    """v3 geometric-mean speedups (query bootstrap) of every comparison on ``ids``."""
    out: dict[str, Any] = {}
    for name, (base, variant) in COMPARISONS.items():
        rows = []
        for q in ids:
            t = times[q]
            if variant not in t:
                continue
            # The tightness run timed its own raw plan; its points compare to it.
            b = t["tight_raw"] if variant.startswith("bi_k") else t[base]
            if variant.startswith("certified"):
                b = t["forced_native"]
            rows.append(
                {"query_id": q, "log": math.log(max(b, 1e-3) / max(t[variant], 1e-3))}
            )
        if not rows:
            continue

        def stat(rs: list[dict[str, Any]]) -> float:
            return math.exp(sum(r["log"] for r in rs) / len(rs))

        out[name] = {
            "n": len(rows),
            "geo_speedup": cluster_bootstrap(rows, stat, replicates=REPLICATES, seed=0),
        }
    for k in ("bi_k2", "bi_k8"):
        if k in out and "truth" in out:
            out[f"{k}_share_of_truth_log"] = math.log(
                out[k]["geo_speedup"]["point"]
            ) / math.log(out["truth"]["geo_speedup"]["point"])
    return out


def main() -> None:
    """Write ``S3_DESIGN.json``."""
    inj, tig, forced = _jsonl(INJECT), _jsonl(TIGHT), _jsonl(FORCED)
    sizes = _jsonl(SIZES)
    set_a = sorted(
        q
        for q, r in forced.items()
        if "certified" in r["ms"] and true_spread(sizes[q]) > 2.0
    )
    excluded = sorted(
        q
        for q, r in inj.items()
        if r["plan_censored"][r["policy_plan"]["native"]]
        and r["plan_censored"][r["policy_plan"]["truth"]]
    )
    rest = sorted(q for q in inj if q not in set_a and q not in excluded)
    strata: dict[str, list[int]] = {}
    for q in rest:
        n = int(sizes[q]["n_aliases"]) if q in sizes else 2
        strata.setdefault(str(min(n, 6)) if n < 6 else "6+", []).append(q)
    quota = {s: N_B * len(v) / len(rest) for s, v in strata.items()}
    alloc = {s: math.floor(x) for s, x in quota.items()}
    for s in sorted(quota, key=lambda s: quota[s] - alloc[s], reverse=True)[
        : N_B - sum(alloc.values())
    ]:
        alloc[s] += 1
    rng = random.Random(SEED)
    set_b = sorted(q for s in sorted(strata) for q in rng.sample(strata[s], alloc[s]))
    times = {q: v3_times(q, inj, tig, forced) for q in set_a + set_b}
    pass_ms = 0.0
    for q in set_a + set_b:
        t = times[q]
        distinct = {
            inj[q]["policy_plan"][p]: t[p]
            for p in (
                "native",
                "native_raw",
                "native_upper",
                "native_lower",
                "native_bi",
                "truth",
            )
        }
        distinct.update({tig[q]["policy_plan"][p]: t[p] for p in ("bi_k2", "bi_k8")})
        pass_ms += sum(distinct.values()) + t.get("certified_chain", 0.0)
    out: dict[str, Any] = {
        "frozen": "2026-11-16",
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (INJECT, TIGHT, FORCED, SIZES)
        },
        "seed": SEED,
        "set_a": set_a,
        "excluded_both_capped": excluded,
        "strata_sizes": {s: len(v) for s, v in sorted(strata.items())},
        "set_b_allocation": dict(sorted(alloc.items())),
        "set_b": set_b,
        "estimated_pass_hours_v3": pass_ms / 3.6e6,
        "comparisons": {k: list(v) for k, v in COMPARISONS.items()},
        "reference_v3": {
            "A": references(set_a, times),
            "B": references(set_b, times),
            "A+B": references(sorted(set_a + set_b), times),
        },
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    logger.info(
        "A %s\nB %s\nexcluded %s\npass %.2f h",
        set_a,
        set_b,
        excluded,
        out["estimated_pass_hours_v3"],
    )
    for part, ref in out["reference_v3"].items():
        logger.info(
            "%s %s",
            part,
            {
                k: round(v["geo_speedup"]["point"], 3)
                if isinstance(v, dict)
                else round(v, 3)
                for k, v in ref.items()
            },
        )


if __name__ == "__main__":
    main()
