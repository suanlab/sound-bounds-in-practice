"""Guard policies against the baselines a reviewer would ask for (STATS-CEB).

Post-hoc (added after internal review panel 4). On the same 780 (query, proposal)
pairs as an earlier draft's policy table (the 130 queries with a completed sampling
certificate), compare the guard policies with *using a certificate as the
estimate*: LpBound, SafeBound, the sound sample count ``L`` and the unbiased
``L_hat = L / r``. Policies are reported at tau = 1 (for a sound bound every
estimate above ``B`` is a certified overestimate) and tau = 2, pooled and per
estimator, with query-clustered bootstrap intervals.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.cardinality_baselines
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.scripts.bootstrap_ci import cluster_bootstrap, percentile

if TYPE_CHECKING:
    from collections.abc import Callable

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def q_error(estimate: float, truth: float) -> float:
    """Symmetric ratio error with both sides floored at one."""
    e, t = max(estimate, 1.0), max(truth, 1.0)
    return max(e / t, t / e)


def load_pairs(path: Path, *, require_sample: bool = True) -> list[dict[str, Any]]:
    """One record per (query, proposal) with every certificate value."""
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["query_id"], row["proposal"])
            entry = pairs.setdefault(
                key,
                {
                    "query_id": row["query_id"],
                    "proposal": row["proposal"],
                    "estimate": float(row["estimate"]),
                    "truth": float(row["true_cardinality"]),
                },
            )
            if row["certificate"] in ("lpbound", "safebound"):
                entry[row["certificate"]] = float(row["bound"])
            elif row["certificate"] == "sample_lb":
                entry["l_sound"] = float(row["bound"])
                # tightness_proxy = log(e / L_hat)  =>  L_hat = e / exp(proxy)
                entry["l_hat"] = float(row["estimate"]) / math.exp(
                    float(row["tightness_proxy"])
                )
    needed = {"lpbound", "safebound"} | ({"l_hat"} if require_sample else set())
    return [p for p in pairs.values() if needed <= set(p)]


def override_lower_from_pg(
    pairs: list[dict[str, Any]], path: Path
) -> list[dict[str, Any]]:
    """Replace the sample columns with PostgreSQL's in-engine sample of the full
    query (the largest sub-plan of each query in ``subset_counts.jsonl``);
    queries whose sampled count timed out are dropped."""
    best: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = str(record["query_id"])
        if qid not in best or len(record["aliases"]) > len(best[qid]["aliases"]):
            best[qid] = record
    kept: list[dict[str, Any]] = []
    for pair in pairs:
        record = best.get(pair["query_id"])
        if record is None or record["sample_censored"]:
            continue
        pair = dict(pair)
        pair["l_sound"] = float(record["lower_sound"])
        pair["l_hat"] = float(record["lower_hat"])
        kept.append(pair)
    return kept


def finals(pair: dict[str, Any]) -> dict[str, float]:
    """The estimate each policy returns for one pair."""
    e, upper = pair["estimate"], pair["lpbound"]
    l_sound, l_hat = pair["l_sound"], pair["l_hat"]

    def gate(value: float, tau: float, floor: float | None) -> float:
        out = upper if value > tau * upper else value
        if floor is not None and out < floor / tau:
            out = floor
        return out

    return {
        "none": e,
        "upper_t2": gate(e, 2.0, None),
        "upper_t1": gate(e, 1.0, None),
        "bi_sound_t2": gate(e, 2.0, l_sound),
        "bi_sound_t1": gate(e, 1.0, l_sound),
        "bi_hat_t2": gate(e, 2.0, l_hat),
        "lpbound_as_estimate": upper,
        "safebound_as_estimate": pair["safebound"],
        "l_sound_as_estimate": l_sound,
        "l_hat_as_estimate": l_hat,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("results/utility_table/real_stats_bidirectional.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/utility_table/cardinality_baselines.json"),
    )
    parser.add_argument(
        "--pg-lower",
        type=Path,
        default=None,
        help="use PostgreSQL's in-engine sample (subset_counts.jsonl) for L, L_hat",
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs = load_pairs(args.table, require_sample=args.pg_lower is None)
    if args.pg_lower is not None:
        pairs = override_lower_from_pg(pairs, args.pg_lower)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        row = {"query_id": pair["query_id"], "proposal": pair["proposal"]}
        row["under"] = pair["estimate"] < pair["truth"]
        for name, value in finals(pair).items():
            row[name] = q_error(value, pair["truth"])
        rows.append(row)
    policies = [k for k in rows[0] if k not in {"query_id", "proposal", "under"}]

    def stat(field: str, fraction: float) -> Callable[[list[dict[str, Any]]], float]:
        return lambda rs: percentile([r[field] for r in rs], fraction)

    pooled: dict[str, Any] = {}
    for name in policies:
        pooled[name] = {
            "median": cluster_bootstrap(
                rows, stat(name, 0.5), replicates=args.replicates, seed=args.seed
            ),
            "p95": cluster_bootstrap(
                rows, stat(name, 0.95), replicates=args.replicates, seed=args.seed
            ),
            "max": max(r[name] for r in rows),
        }
    by_proposal: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["proposal"]].append(row)
    for proposal, subset in sorted(grouped.items()):
        by_proposal[proposal] = {
            "n": len(subset),
            "under_share": sum(r["under"] for r in subset) / len(subset),
            **{
                f"p95_{name}": percentile([r[name] for r in subset], 0.95)
                for name in policies
            },
        }
    payload = {
        "experiment": "cardinality_baselines",
        "preregistered": False,
        "n_pairs": len(rows),
        "n_queries": len({r["query_id"] for r in rows}),
        "lower_source": str(args.pg_lower) if args.pg_lower else "python sampler (CSV)",
        "pooled": pooled,
        "by_proposal": by_proposal,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for name in policies:
        entry = pooled[name]
        logger.info(
            "%-22s median %8.3f  p95 %10.2f [%.2f, %.2f]  max %.3g",
            name,
            entry["median"]["point"],
            entry["p95"]["point"],
            entry["p95"]["ci_low"],
            entry["p95"]["ci_high"],
            entry["max"],
        )
    for proposal, entry in by_proposal.items():
        logger.info(
            "%-11s under %.2f  p95 none %10.1f  upper_t1 %9.1f  bi_sound_t1 %7.1f  "
            "l_hat %.2f",
            proposal,
            entry["under_share"],
            entry["p95_none"],
            entry["p95_upper_t1"],
            entry["p95_bi_sound_t1"],
            entry["p95_l_hat_as_estimate"],
        )


if __name__ == "__main__":
    main()
