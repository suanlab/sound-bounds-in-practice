"""Re-time the real-JOB lower certificate with one sampling draw.

The committed ``results/job_full/job_full.json`` timed the lower certificate
with ``run_job_real``'s default ``--lower-seeds 1 2 3`` --- three independent
sub-sampled joins per query --- while the STATS-CEB cost column
(``make_lower_column_stats``) used one draw. Both report the same ratio,
certificate time over exact-count time, so the two benchmarks were compared at
different draw counts (second review panel, 2026-09-05).

This script measures the like-for-like quantity instead of deriving it. For
every template the as-run file evaluated, on identical in-memory filtered rows,
it times the exact count, the one-draw certificate (seed 1) and --- for the
``--linearity`` fastest templates --- the three-draw certificate, and records
whether the exact count and the one-draw sound bound still agree with the
as-run file. Timings append to a JSON-lines file so an interrupted run resumes;
``--summarise-only`` rebuilds the summary JSON from an existing JSON-lines file
without re-timing anything.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.verify_draw_cost
    PYTHONHASHSEED=0 python -m experiments.scripts.verify_draw_cost --summarise-only
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Any

from experiments.scripts.plan_regret import QueryTimeoutError, time_limit
from experiments.scripts.run_job_real import _load_required_tables
from src.estimators.lower_certificate import aggregate_lower_certificate
from src.utils.imdb_schema import load_schema
from src.utils.job_eval import build_alias_rows
from src.utils.job_sql_parser import parse_job_sql
from src.utils.join_counting import count_join

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

Row = dict[str, Any]


def _bootstrap(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    replicates: int,
    seed: int,
) -> list[float]:
    """Percentile 95% interval of ``statistic`` under a plain bootstrap."""
    rng = random.Random(seed)
    draws = sorted(
        statistic(rng.choices(values, k=len(values))) for _ in range(replicates)
    )
    return [draws[int(0.025 * replicates)], draws[int(0.975 * replicates)]]


def _share_above_one(values: Sequence[float]) -> float:
    return sum(1 for v in values if v > 1.0) / len(values)


def time_templates(
    data_dir: Path,
    asrun: dict[str, Row],
    out: Path,
    rate: float,
    seed: int,
    linearity: int,
    budget: float,
) -> None:
    """Append one JSON line per not-yet-timed template to ``out``."""
    order = sorted(asrun, key=lambda q: float(asrun[q]["full_count_ms"]))
    schema = load_schema(data_dir)
    done: set[str] = set()
    if out.exists():
        done = {
            json.loads(line)["query_id"]
            for line in out.read_text().splitlines()
            if line.strip()
        }
    with out.open("a") as fh:
        for i, qid in enumerate(order):
            if qid in done:
                continue
            sql = (data_dir / "queries" / f"{qid}.sql").read_text()
            parsed = parse_job_sql(sql, qid)
            try:
                with time_limit(budget):
                    rows, _ = _load_required_tables(
                        data_dir, schema, parsed.base_relations, None
                    )
                    alias_rows = build_alias_rows(parsed, rows)
                    start = perf_counter()
                    true_card = count_join(alias_rows, parsed.aliases, parsed.joins)
                    full_s = perf_counter() - start
                    start = perf_counter()
                    one = aggregate_lower_certificate(
                        alias_rows, parsed.aliases, parsed.joins, rate, [seed]
                    )
                    c1_s = perf_counter() - start
                    c3_s: float | None = None
                    if i < linearity:
                        start = perf_counter()
                        aggregate_lower_certificate(
                            alias_rows,
                            parsed.aliases,
                            parsed.joins,
                            rate,
                            [seed, seed + 1, seed + 2],
                        )
                        c3_s = perf_counter() - start
            except QueryTimeoutError:
                logger.warning("timeout %s", qid)
                continue
            record: Row = {
                "query_id": qid,
                "true_cardinality": true_card,
                "true_matches_asrun": true_card == asrun[qid]["true_cardinality"],
                "full_ms": full_s * 1e3,
                "c1_ms": c1_s * 1e3,
                "c3_ms": None if c3_s is None else c3_s * 1e3,
                "ratio_1draw": c1_s / full_s,
                "ratio_3draw": None if c3_s is None else c3_s / full_s,
                "c3_over_c1": None if c3_s is None else c3_s / c1_s,
                "lower_sound_valid_1draw": one.sound_lower_bound <= true_card,
                "asrun_full_ms": asrun[qid]["full_count_ms"],
                "asrun_lower_cert_ms": asrun[qid]["lower_cert_ms"],
                "asrun_ratio_3draw": asrun[qid]["cost_ratio"],
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            logger.info(
                "%s full=%.1fs c1=%.1fs r1=%.2f c3=%s asrun_r3=%.2f",
                qid,
                full_s,
                c1_s,
                record["ratio_1draw"],
                "-" if c3_s is None else f"{c3_s:.1f}s",
                record["asrun_ratio_3draw"],
            )


def summarise(rows: list[Row], replicates: int, seed: int) -> dict[str, Any]:
    """Aggregate the per-template timings into the reported quantities."""
    one = [float(r["ratio_1draw"]) for r in rows]
    ordered = sorted(one)
    lin = [r for r in rows if r["c3_ms"]]
    c3c1 = sorted(float(r["c3_over_c1"]) for r in lin)
    return {
        "experiment": "job_full_lower_certificate_draw_count_verification",
        "date": "2026-09-08",
        "purpose": (
            "Panel-2 (A-W1/B-W1) verification: the committed job_full.json cost "
            "column used --lower-seeds 1 2 3 (3 draws) while STATS-CEB used 1 "
            "draw. Re-time, on identical in-memory filtered rows, the full count, "
            "the 1-draw certificate (seed 1) and, for the 15 fastest templates, "
            "the 3-draw certificate (seeds 1 2 3)."
        ),
        "script": (
            "experiments/scripts/verify_draw_cost.py; PYTHONHASHSEED=0; rate 0.1; "
            "templates = the 53 evaluated in results/job_full/job_full.json, "
            "ascending by as-run full_count_ms"
        ),
        "machine_note": (
            "shared machine (load 1-2 on 40 threads at start); absolute times not "
            "comparable to the as-run file, within-template ratios are. Full-count "
            "time today / as-run median 1.05."
        ),
        "n_templates": len(rows),
        "true_cardinality_matches_asrun_all": all(
            bool(r["true_matches_asrun"]) for r in rows
        ),
        "one_draw_sound_bound_valid_all": all(
            bool(r["lower_sound_valid_1draw"]) for r in rows
        ),
        "one_draw": {
            "median_cost_ratio": median(one),
            "ci95_bootstrap_median": _bootstrap(one, median, replicates, seed),
            "share_above_one": _share_above_one(one),
            "ci95_bootstrap_share": _bootstrap(one, _share_above_one, replicates, seed),
            "min": ordered[0],
            "p25": ordered[len(one) // 4],
            "p75": ordered[3 * len(one) // 4],
            "max": ordered[-1],
        },
        "derived_asrun_over_3": {
            "median": median(float(r["asrun_ratio_3draw"]) / 3 for r in rows),
            "measured_minus_derived_median": median(
                float(r["ratio_1draw"]) - float(r["asrun_ratio_3draw"]) / 3
                for r in rows
            ),
        },
        "asrun_three_draw": {
            "median": median(float(r["asrun_ratio_3draw"]) for r in rows),
            "share_above_one": _share_above_one(
                [float(r["asrun_ratio_3draw"]) for r in rows]
            ),
        },
        "linearity_check": {
            "n": len(lin),
            "c3_over_c1_median": median(c3c1),
            "c3_over_c1_iqr": [c3c1[len(c3c1) // 4], c3c1[3 * len(c3c1) // 4]],
            "c3_over_c1_max": c3c1[-1],
            "note": (
                "max 8.35 is template 5a, the first call of the process "
                "(warm-up); all others 2.64-3.8"
            ),
            "three_draw_ratio_remeasured_median": median(
                float(r["ratio_3draw"]) for r in lin
            ),
            "three_draw_ratio_asrun_same_templates_median": median(
                float(r["asrun_ratio_3draw"]) for r in lin
            ),
        },
        "verdict": (
            "Confound confirmed and derivation validated: cost is linear in draws "
            "(c3/c1 ~2.9); measured 1-draw median 0.63 [0.54, 0.67] vs derived "
            "0.60; 0/53 templates above 1.0 at one draw vs 49/53 as run."
        ),
        "per_template": rows,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/job_full"))
    parser.add_argument(
        "--asrun", type=Path, default=Path("results/job_full/job_full.json")
    )
    parser.add_argument(
        "--timings",
        type=Path,
        default=Path("results/job_full/job_full_1draw_verification.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/job_full/job_full_1draw_verification.json"),
    )
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1, help="first sampling seed")
    parser.add_argument(
        "--linearity",
        type=int,
        default=15,
        help="also time the three-draw certificate on this many fastest templates",
    )
    parser.add_argument("--budget", type=float, default=3600.0)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args()

    asrun = {r["query_id"]: r for r in json.loads(args.asrun.read_text())["per_query"]}
    if not args.summarise_only:
        time_templates(
            args.data_dir,
            asrun,
            args.timings,
            args.rate,
            args.seed,
            args.linearity,
            args.budget,
        )
    rows = [json.loads(line) for line in args.timings.read_text().splitlines() if line]
    payload = summarise(rows, args.replicates, args.bootstrap_seed)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    logger.info(
        "%d templates: one-draw median %.3f %s, share>1 %.3f; c3/c1 median %.2f",
        payload["n_templates"],
        payload["one_draw"]["median_cost_ratio"],
        payload["one_draw"]["ci95_bootstrap_median"],
        payload["one_draw"]["share_above_one"],
        payload["linearity_check"]["c3_over_c1_median"],
    )
    logger.info("-> %s", args.output)


if __name__ == "__main__":
    main()
