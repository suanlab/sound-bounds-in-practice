"""Compute the sample-based lower certificate column on REAL STATS-CEB.

This is the missing *lower* side of the certificate-utility table. The external
table (``make_utility_table_external``) covers strong sound **upper**
certificates (LpBound, SafeBound) but those are statistics lookups with no
measurable per-query join cost, which makes the pre-registered ``cost`` variable
non-discriminative (see ``results/c1_pilot/c1_pilot.json`` caveats).

Here we run the reference lower instrument on the real STATS-CEB tables shipped
with the LpBound artifact: for each of the 146 queries we parse the SPJ query,
apply its selection filters per alias, and measure

  * ``sound_lower_bound`` = size of the sub-sampled join (subset of the true
    join, hence a sound lower bound with probability 1),
  * ``l_hat``             = the rate-scaled unbiased point estimate,
  * ``full_count_ms`` / ``lower_cert_ms`` / ``cost_ratio`` = the honest cost of
    the certificate relative to just executing the join it avoids,
  * ``true_cardinality``  = the exact count (also cross-checked against the
    artifact's ground truth when available).

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.make_lower_column_stats \
        --rate 0.1 --seeds 1 --output results/utility_table/stats_lower.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import signal
import sys
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from src.estimators.lower_certificate import aggregate_lower_certificate
from src.utils.job_eval import build_alias_rows
from src.utils.job_sql_parser import parse_job_sql
from src.utils.join_counting import count_join

if TYPE_CHECKING:
    from collections.abc import Iterator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FIELDS = [
    "query_id",
    "n_aliases",
    "true_cardinality",
    "artifact_true_cardinality",
    "truth_source",
    "sound_lower_bound",
    "l_hat",
    "lower_valid",
    "sample_rate",
    "full_count_ms",
    "full_count_censored",
    "lower_cert_ms",
    "cost_ratio",
    "cost_ratio_is_upper_bound",
]


class _TimeoutError(Exception):
    """Raised when a per-query computation exceeds its wall-clock budget."""


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Abort the enclosed block after ``seconds`` (SIGALRM; main thread only).

    A per-query guard is essential here: the exact count is a Python hash join
    whose frontier can blow up on a single heavy STATS query, which would
    otherwise stall the whole sweep.
    """

    def _handler(signum: int, frame: object) -> None:
        raise _TimeoutError

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def load_tables(
    data_dir: Path, tables: set[str], cache: dict[str, list[dict[str, str]]]
) -> dict[str, list[dict[str, str]]]:
    """Load (and memoise) the headered STATS CSVs for the given base tables."""
    for table in tables:
        if table in cache:
            continue
        path = data_dir / f"{table}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            cache[table] = list(csv.DictReader(handle))
        logger.info("  loaded %s: %d rows", table, len(cache[table]))
    return {table: cache[table] for table in tables}


def load_artifact_truth(path: Path) -> dict[int, float]:
    """Read the artifact's ground-truth cardinalities, if present."""
    if not path.exists():
        return {}
    truth: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            values = [v for k, v in row.items() if k != "QueryID"]
            truth[int(row["QueryID"])] = float(values[0])
    return truth


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("external/LpBound/benchmarks/workloads/stats/statsQueries.sql"),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("external/LpBound/data/datasets/stats")
    )
    parser.add_argument(
        "--artifact-truth",
        type=Path,
        default=Path(
            "external/LpBound/results/accuracy_acyclic/stats/"
            "truecardinality_stats_full_estimations.csv"
        ),
    )
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--query-timeout-s",
        type=float,
        default=20.0,
        help="Per-query wall-clock budget for the exact count / certificate.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/utility_table/stats_lower.csv")
    )
    args = parser.parse_args()

    queries = [
        line.strip() for line in args.queries.read_text().splitlines() if line.strip()
    ]
    if args.limit:
        queries = queries[: args.limit]
    artifact_truth = load_artifact_truth(args.artifact_truth)
    logger.info("STATS-CEB: %d queries, rate=%.2f", len(queries), args.rate)

    cache: dict[str, list[dict[str, str]]] = {}
    rows_out: list[dict[str, Any]] = []
    skipped: list[str] = []
    budget = perf_counter()

    for index, sql in enumerate(queries, start=1):
        parsed = parse_job_sql(sql, str(index))
        if not parsed.fully_parsed:
            skipped.append(f"{index}:unparsed")
            continue
        try:
            base_rows = load_tables(args.data_dir, set(parsed.base_relations), cache)
            alias_rows = build_alias_rows(parsed, base_rows)

            # Exact count: censored on timeout (we then trust the artifact truth).
            censored = False
            t0 = perf_counter()
            try:
                with time_limit(args.query_timeout_s):
                    true_card = count_join(alias_rows, parsed.aliases, parsed.joins)
                full_s = max(perf_counter() - t0, 1e-9)
            except _TimeoutError:
                censored = True
                full_s = args.query_timeout_s  # a LOWER bound on the real cost
                if index not in artifact_truth:
                    logger.warning("  q%d: count timed out, no artifact truth", index)
                    skipped.append(f"{index}:count-timeout-no-truth")
                    continue
                true_card = artifact_truth[index]

            with time_limit(args.query_timeout_s):
                t1 = perf_counter()
                cert = aggregate_lower_certificate(
                    alias_rows, parsed.aliases, parsed.joins, args.rate, args.seeds
                )
                cert_s = perf_counter() - t1
        except _TimeoutError:
            logger.warning("  q%d: certificate timed out", index)
            skipped.append(f"{index}:cert-timeout")
            continue
        except (KeyError, ValueError) as exc:
            logger.warning("  q%d skipped: %s", index, exc)
            skipped.append(f"{index}:{type(exc).__name__}")
            continue

        rows_out.append(
            {
                "query_id": index,
                "n_aliases": len(parsed.aliases),
                "true_cardinality": true_card,
                "artifact_true_cardinality": artifact_truth.get(index, ""),
                "truth_source": "artifact" if censored else "exact_count",
                "sound_lower_bound": cert.sound_lower_bound,
                "l_hat": cert.point_estimate,
                "lower_valid": cert.sound_lower_bound <= true_card,
                "sample_rate": cert.sample_rate,
                "full_count_ms": full_s * 1000.0,
                "full_count_censored": censored,
                "lower_cert_ms": cert_s * 1000.0,
                # When censored, full_s underestimates the real full-count time,
                # so this ratio is an UPPER bound on the true cost ratio.
                "cost_ratio": cert_s / full_s,
                "cost_ratio_is_upper_bound": censored,
            }
        )
        logger.info(
            "  q%-3d aliases=%d true=%.0f L=%.0f cost_ratio=%.3f",
            index,
            len(parsed.aliases),
            true_card,
            cert.sound_lower_bound,
            cert_s / full_s,
        )
        if args.timeout_s and perf_counter() - budget > args.timeout_s:
            logger.warning(
                "time budget %.0fs exhausted after %d queries", args.timeout_s, index
            )
            skipped.append(f"budget-stop-after-{index}")
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    if rows_out:
        valid = sum(1 for r in rows_out if r["lower_valid"])
        exact_rows = [r for r in rows_out if not r["full_count_censored"]]
        ratios = sorted(r["cost_ratio"] for r in exact_rows) or [float("nan")]
        logger.info(
            "\ncensored (exact count timed out, artifact truth used): %d/%d",
            len(rows_out) - len(exact_rows),
            len(rows_out),
        )
        matched = [
            r
            for r in rows_out
            if r["artifact_true_cardinality"] not in ("", None)
            and abs(float(r["artifact_true_cardinality"]) - r["true_cardinality"]) < 0.5
        ]
        logger.info(
            "\n%d queries | lower validity %.1f%% | cost_ratio median %.3f | "
            "exact-count agrees with artifact truth on %d/%d",
            len(rows_out),
            100 * valid / len(rows_out),
            ratios[len(ratios) // 2],
            len(matched),
            sum(
                1 for r in rows_out if r["artifact_true_cardinality"] not in ("", None)
            ),
        )
    if skipped:
        logger.info("skipped/stopped: %s", skipped[:10])
    logger.info("Wrote %d rows -> %s", len(rows_out), args.output)
    if not rows_out:
        sys.exit(1)


if __name__ == "__main__":
    main()
