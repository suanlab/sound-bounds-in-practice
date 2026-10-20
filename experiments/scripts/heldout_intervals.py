"""Query-clustered intervals for the held-out JOB-range shares (C1-C4).

Re-derives the per-query a-posteriori certificates with the registered
inclusion rules (``heldout_jobrange.included``, unchanged and hashed in the
addendum) and adds 95% percentile intervals (10,000 replicates, seed 0). Also
reports the relation counts of the included queries. Not part of the
registered analysis; it only adds intervals to its shares.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.heldout_intervals
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments.pg_injection.inject_plans import _read_jsonl
from experiments.scripts.bootstrap_ci import cluster_bootstrap
from experiments.scripts.heldout_jobrange import COUNTS, NATIVE_PLANS, TARGET, included
from experiments.scripts.plan_certificate_joint import incidence, left_deep_orders

if TYPE_CHECKING:
    from collections.abc import Callable

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CELLS = ("cout|none", "cout|sample", "cout_minus_final|none", "cout_minus_final|sample")


def per_query() -> list[dict[str, Any]]:
    """Certified flags per included 3+-relation query, as the registered analysis."""
    keep, _ = included(_read_jsonl(COUNTS))
    plans = {r["query_id"]: r for r in _read_jsonl(NATIVE_PLANS)}
    rows = []
    for qid, subplans in keep.items():
        full: frozenset[str] = max(list(subplans), key=len)
        aliases = sorted(full)
        if len(aliases) < 3:
            continue
        sets = {frozenset(s) for s in plans[qid]["join_sets"]}
        index = {s: i for i, s in enumerate(subplans)}
        truth = np.array([max(float(subplans[s]["exact_pg"]), 1.0) for s in index])
        floor = np.array(
            [
                1.0
                if subplans[s]["sample_censored"]
                else max(float(subplans[s]["lower_sound"]), 1.0)
                for s in index
            ]
        )
        orders = left_deep_orders(aliases, index)
        final = index[full]
        row: dict[str, Any] = {"query_id": qid, "n_relations": len(aliases)}
        for variant in ("cout", "cout_minus_final"):
            include = variant == "cout"
            inc = incidence(orders, index, include_final=include)
            observed = np.zeros(len(index), dtype=bool)
            observed[[index[s] for s in sets]] = True
            if not include:
                observed[final] = False
            c_native = float(sum(truth[index[s]] for s in sets)) - (
                0.0 if include else float(truth[final])
            )
            for name, lower in (("sample", floor), ("none", np.ones_like(truth))):
                low = np.where(observed, truth, lower)
                bound = c_native / min(c_native, float((inc @ low).min()))
                row[f"{variant}|{name}"] = bound <= TARGET
        rows.append(row)
    return rows


def share(cell: str) -> Callable[[list[dict[str, Any]]], float]:
    """Statistic: share of queries certified in one cell."""

    def stat(rows: list[dict[str, Any]]) -> float:
        return sum(bool(r[cell]) for r in rows) / len(rows)

    return stat


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/heldout_jobrange/heldout_intervals.json"),
    )
    args = parser.parse_args()
    rows = per_query()
    out = {
        "n_queries": len(rows),
        "n_relations_min": min(r["n_relations"] for r in rows),
        "n_relations_max": max(r["n_relations"] for r in rows),
        **{
            cell: cluster_bootstrap(rows, share(cell), replicates=10_000, seed=0)
            for cell in CELLS
        },
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("%s", json.dumps(out))


if __name__ == "__main__":
    main()
