"""Query-clustered intervals and influence checks for the mechanism analyses.

Reads the per-query outputs of ``certificate_mechanisms``,
``price_of_certification``, ``forced_orders`` and ``tightness_sweep`` and adds
95% percentile intervals from a query-clustered bootstrap (10,000 replicates,
seed 0), as for the paper's other numbers:

* H1: AUC of rho, and of the number of relations as a baseline;
* H4: a-posteriori certified shares;
* H5: end-to-end speedup of the certified plans over the native plan with the
  oracle's and the online rule's sampling charged (forced-order run times);
* tightness sweep: geometric C_out gain and share of the true-size gain (log
  scale), with the largest change any single query's removal makes to the
  share (leave-one-out) and the share of the true-size gain carried by the
  three queries that contribute most.

Exploratory.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.mechanism_intervals
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.pg_injection.inject_plans import _read_jsonl
from experiments.scripts.bootstrap_ci import cluster_bootstrap
from experiments.scripts.certificate_mechanisms import auc

if TYPE_CHECKING:
    from collections.abc import Callable

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MECH = Path("results/plan_certificate/certificate_mechanisms.json")
POC = Path("results/plan_certificate/price_of_certification.json")
FORCED = Path("results/pg_injection/v3/forced_orders.jsonl")
SWEEPS = {
    "stats": Path("results/pg_injection/v3/tightness_sweep_stats.jsonl"),
    "joblight": Path("results/pg_injection/v3/tightness_sweep_joblight.jsonl"),
}
TIGHTNESS = ("1", "1.5", "2", "4", "8", "16", "32", "64")
TARGET = 2.0


def boot(
    rows: list[dict[str, Any]], stat: Callable[[list[dict[str, Any]]], float], seed: int
) -> dict[str, Any]:
    """Query-clustered bootstrap with the paper's settings."""
    return cluster_bootstrap(rows, stat, replicates=10_000, seed=seed)


def share_of(field: str) -> Callable[[list[dict[str, Any]]], float]:
    """Statistic: share of rows whose ``field`` is at most the 2x target."""

    def stat(rows: list[dict[str, Any]]) -> float:
        return sum(float(r[field]) <= TARGET for r in rows) / len(rows)

    return stat


def h1(mech: dict[str, Any], seed: int) -> dict[str, Any]:
    """AUC of rho and of the (negated) number of relations, with intervals."""
    out = {}
    for name, entry in mech["h1_shared_mass"].items():
        rows = entry["per_query"]

        def auc_rho(rs: list[dict[str, Any]]) -> float:
            labels = [r["certified"] for r in rs]
            if all(labels) or not any(labels):
                return math.nan
            return auc([r["rho"] for r in rs], labels)

        out[name] = {
            "auc_rho": boot(rows, auc_rho, seed),
            "auc_n_relations": auc(
                [-r["n_aliases"] for r in rows], [r["certified"] for r in rows]
            ),
        }
    return out


def h4(mech: dict[str, Any], seed: int) -> dict[str, Any]:
    """A-posteriori certified shares with intervals."""
    out: dict[str, Any] = {}
    for bench, key in (
        ("stats", "h4_a_posteriori"),
        ("joblight", "h4_a_posteriori_joblight"),
    ):
        for variant, entry in mech[key].items():
            rows = entry["per_query"]
            out[f"{bench}|{variant}"] = {
                field: boot(rows, share_of(field), seed)
                for field in ("bound_sample", "bound_none", "true_regret")
            }
            matters = [r for r in rows if r["matters"]]
            out[f"{bench}|{variant}|choice_matters"] = {
                field: boot(matters, share_of(field), seed)
                for field in ("bound_sample", "bound_none", "true_regret")
            }
    return out


def h5(seed: int) -> dict[str, Any]:
    """End-to-end certified-plan speedup with oracle and online sampling."""
    forced = {r["query_id"]: r for r in _read_jsonl(FORCED)}
    rows = []
    for r in json.loads(POC.read_text())["per_query"]:
        run = forced.get(r["query_id"])
        if run is None or "certified" not in run["ms"]:
            continue
        native = max(run["ms"]["native"], 1e-3)
        rows.append(
            {
                "query_id": r["query_id"],
                "matters": r["matters"],
                "oracle": math.log(
                    native / max(run["ms"]["certified"] + r["oracle_ms"], 1e-3)
                ),
                "greedy": math.log(
                    native / max(run["ms"]["certified"] + r["greedy_ms"], 1e-3)
                ),
            }
        )

    def geo(field: str) -> Callable[[list[dict[str, Any]]], float]:
        def stat(rs: list[dict[str, Any]]) -> float:
            return math.exp(sum(r[field] for r in rs) / len(rs))

        return stat

    out: dict[str, Any] = {}
    for label, subset in (
        ("all", rows),
        ("choice_matters", [r for r in rows if r["matters"]]),
    ):
        out[label] = {"n": len(subset)}
        for field in ("oracle", "greedy"):
            out[label][f"{field}_geo"] = boot(subset, geo(field), seed)
            out[label][f"{field}_share_faster"] = sum(
                r[field] > math.log(1.1) for r in subset
            ) / len(subset)
    return out


def sweep(path: Path, seed: int) -> dict[str, Any]:
    """Gain and share of the true-size gain per guard and k, with influence."""
    records = _read_jsonl(path)
    out: dict[str, Any] = {}
    for source in ("native", "postgres"):
        rows = [
            {"query_id": r["query_id"], **r[source]["cout"]}
            for r in records
            if source in r
        ]
        if not rows:
            continue
        logs = [
            {
                "query_id": r["query_id"],
                **{
                    k: math.log(r["raw"] / v)
                    for k, v in r.items()
                    if k not in ("query_id", "raw")
                },
            }
            for r in rows
        ]
        truth_total = sum(r["truth"] for r in logs)
        top3 = sorted((r["truth"] for r in logs), reverse=True)[:3]
        entry: dict[str, Any] = {
            "n": len(logs),
            "truth_gain": math.exp(truth_total / len(logs)),
            "top3_share_of_truth_gain": sum(top3) / truth_total
            if truth_total > 0
            else math.nan,
        }
        for guard in ("upper", "lower", "bi"):
            for k in TIGHTNESS:
                name = f"{guard}_k{k}"

                def share(rs: list[dict[str, Any]], name: str = name) -> float:
                    total = sum(r["truth"] for r in rs)
                    return sum(r[name] for r in rs) / total if total > 0 else math.nan

                def gain(rs: list[dict[str, Any]], name: str = name) -> float:
                    return math.exp(sum(r[name] for r in rs) / len(rs))

                point = share(logs)
                loo = max(
                    abs(share(logs[:i] + logs[i + 1 :]) - point)
                    for i in range(len(logs))
                )
                entry[name] = {
                    "gain": boot(logs, gain, seed),
                    "share_of_truth_gain": boot(logs, share, seed),
                    "max_leave_one_out_change": loo,
                }
        out[source] = entry
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/mechanism_intervals.json"),
    )
    args = parser.parse_args()
    mech = json.loads(MECH.read_text())
    out = {
        "experiment": "mechanism_intervals",
        "preregistered": False,
        "h1": h1(mech, args.seed),
        "h4": h4(mech, args.seed),
        "h5": h5(args.seed),
        "sweep": {name: sweep(path, args.seed) for name, path in SWEEPS.items()},
    }
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    logger.info("%s", json.dumps({k: out[k] for k in ("h1", "h5")})[:2000])


if __name__ == "__main__":
    main()
