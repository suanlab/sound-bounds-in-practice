"""Guard-direction ablation: no-guard vs upper-only vs lower-only vs bidirectional.

Pre-registered as TOP_PLAN Phase 3.2 ablation #2. Using the real STATS-CEB
utility table, we replay each (query, proposal) pair under four deployment
policies and report the resulting error distribution, how often each policy
intervenes, and what it costs:

  * ``no_guard``       -- accept the proposal as-is (the deployment baseline),
  * ``upper_only``     -- reject and fall back to $B(Q)$ when $e > \\tau B(Q)$,
  * ``lower_only``     -- raise to $\\hat{L}$ when $e < \\hat{L}/\\tau$,
  * ``bidirectional``  -- apply both gates.

The point of the ablation is not to crown a winner but to separate *which
direction* each policy can act on, and at what cost: the upper gate is free but
blind to underestimation, the lower gate covers underestimation but must
partially execute the join.

Usage::

    python -m experiments.scripts.guard_policy_ablation \
        --table results/utility_table/real_stats_bidirectional.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from statistics import median
from typing import Any

from src.utils.metrics import q_error

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_UPPER_CERT = "lpbound"
_LOWER_CERT = "sample_lb"


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile (numpy-compatible), for small samples."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def load_pairs(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Pivot the utility table into one record per (query, proposal)."""
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["query_id"], row["proposal"])
            entry = pairs.setdefault(
                key,
                {
                    "estimate": float(row["estimate"]),
                    "true": float(row["true_cardinality"]),
                },
            )
            if row["certificate"] == _UPPER_CERT:
                entry["upper"] = float(row["bound"])
            elif row["certificate"] == _LOWER_CERT:
                # tightness_proxy = log(e / L_hat)  =>  L_hat = e / exp(proxy)
                entry["l_hat"] = float(row["estimate"]) / math.exp(
                    float(row["tightness_proxy"])
                )
                entry["cost_ratio"] = float(row["cost_ratio"])
                # The SOUND lower bound L(Q) as stored, alongside the unbiased
                # L_hat the deployed policy actually floors with.
                entry["l_sound"] = float(row["bound"])
    return {k: v for k, v in pairs.items() if "upper" in v and "l_hat" in v}


def apply_policies(
    pairs: dict[tuple[str, str], dict[str, float]], tau: float
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Replay every pair under the four guard policies.

    Returns the aggregate summary and the per-pair record it was computed from.
    The per-pair record is what makes query-clustered resampling possible: the
    same query appears under several proposals, so the rows are not independent
    and any interval computed as if they were would be too narrow.
    """
    policies = (
        "no_guard",
        "upper_only",
        "lower_only",
        "bidirectional",
        "lower_only_sound",
        "bidirectional_sound",
    )
    errors: dict[str, list[float]] = {p: [] for p in policies}
    fired: dict[str, int] = dict.fromkeys(policies, 0)
    worsened: dict[str, int] = dict.fromkeys(policies, 0)
    costs: list[float] = []

    records: list[dict[str, Any]] = []
    for (query_id, proposal), entry in pairs.items():
        y = max(entry["true"], 1.0)
        e = max(entry["estimate"], 1.0)
        upper = max(entry["upper"], 1.0)
        l_hat = max(entry["l_hat"], 1.0)
        l_sound = max(entry["l_sound"], 1.0)
        costs.append(entry["cost_ratio"])
        base_q = q_error(e, y)

        finals = {"no_guard": e}
        finals["upper_only"] = upper if e > tau * upper else e
        finals["lower_only"] = l_hat if e < l_hat / tau else e
        both = finals["upper_only"]
        if both < l_hat / tau:
            both = l_hat
        finals["bidirectional"] = both

        # The same two policies floored with the SOUND bound instead. This is
        # what a deployment gets if it refuses to floor with an unbiased
        # estimate, and it is the price of that refusal.
        finals["lower_only_sound"] = l_sound if e < l_sound / tau else e
        both_sound = finals["upper_only"]
        if both_sound < l_sound / tau:
            both_sound = l_sound
        finals["bidirectional_sound"] = both_sound

        record: dict[str, Any] = {"query_id": query_id, "proposal": proposal}
        for policy in policies:
            q = q_error(finals[policy], y)
            errors[policy].append(q)
            record[f"qerror_{policy}"] = q
            if policy != "no_guard" and finals[policy] != e:
                fired[policy] += 1
                if q > base_q:
                    worsened[policy] += 1
        records.append(record)

    n = len(pairs)
    out: dict[str, dict[str, Any]] = {}
    for policy in policies:
        out[policy] = {
            "n_pairs": n,
            "qerror_median": median(errors[policy]),
            "qerror_p95": _percentile(errors[policy], 0.95),
            "intervention_rate": fired[policy] / n if n else float("nan"),
            "interventions_that_worsened": worsened[policy],
            "certificate_cost_ratio_median": (
                0.0 if policy in ("no_guard", "upper_only") else median(costs)
            ),
            "floor": (
                "sound L(Q)"
                if policy.endswith("_sound")
                else ("unbiased L-hat" if "lower" in policy or "bi" in policy else "--")
            ),
        }
    return out, records


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("results/utility_table/real_stats_bidirectional.csv"),
    )
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/utility_table/guard_policy.json")
    )
    args = parser.parse_args()

    pairs = load_pairs(args.table)
    summary, records = apply_policies(pairs, args.tau)
    payload = {
        "experiment": "guard_direction_ablation",
        "preregistered_as": "TOP_PLAN Phase 3.2 ablation #2",
        "dataset": "real STATS-CEB (bidirectional utility table)",
        "tau": args.tau,
        "n_pairs": len(pairs),
        "policies": summary,
        "per_pair": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    logger.info(
        "%-14s %10s %10s %11s %10s",
        "policy",
        "med q-err",
        "p95 q-err",
        "interv.",
        "cost",
    )
    for policy, stats in summary.items():
        logger.info(
            "%-14s %10.2f %10.1f %11.1f%% %10.3f  (worsened: %d)",
            policy,
            stats["qerror_median"],
            stats["qerror_p95"],
            100 * stats["intervention_rate"],
            stats["certificate_cost_ratio_median"],
            stats["interventions_that_worsened"],
        )
    logger.info("n=%d (query, proposal) pairs -> %s", len(pairs), args.output)


if __name__ == "__main__":
    main()
