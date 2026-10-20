"""Evaluate Safe-CEG components on the real Join Order Benchmark (JOB).

For each JOB ``*.sql`` template this script:
  1. parses it into aliases, equi-joins, and selection filters
     (:mod:`src.utils.job_sql_parser`),
  2. loads the required headerless IMDb tables by schema
     (:mod:`src.utils.imdb_schema`),
  3. computes the *exact* filtered, alias-aware true cardinality via a
     count-based hash join (:func:`src.utils.join_counting.count_join`),
  4. builds the CEG :class:`Query` and records, per query, the optimistic
     oracle/floored q-errors, the pessimistic and AVI baselines, two-sided
     soundness flags, and the **C3 cost ratio** (lower-certificate time vs the
     full-join time it is meant to avoid).

Coverage is reported honestly (parsed / run / skipped with reasons); nothing is
silently dropped. This is a **local** run: IMDb is several GB, so heavy tables
can be capped with ``--max-rows`` (which makes those cardinalities a subsample
estimate -- reported in the output) or excluded with ``--skip-tables``.

Usage (local)::

    python -m experiments.scripts.prepare_real_job --out data/job_full
    PYTHONHASHSEED=0 python -m experiments.scripts.run_job_real --data-dir data/job_full
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from experiments.scripts.plan_regret import time_limit
from src.estimators.independence_estimator import IndependenceEstimator
from src.estimators.lower_certificate import aggregate_lower_certificate
from src.training.dataset import build_training_example
from src.utils.imdb_schema import load_relation_rows, load_schema
from src.utils.job_eval import build_alias_rows, build_job_query
from src.utils.job_sql_parser import ParsedJobQuery, parse_job_sql
from src.utils.join_counting import count_join
from src.utils.metrics import q_error

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


_TABLE_CACHE: dict[tuple[str, int | None], list[dict[str, str]]] = {}


def _load_required_tables(
    data_dir: Path,
    schema: dict[str, list[str]],
    tables: frozenset[str],
    max_rows: int | None,
) -> tuple[dict[str, list[dict[str, str]]], bool]:
    """Load each base table; return (rows_by_relation, capped?).

    Tables are cached across queries. The 113 JOB templates draw repeatedly on
    the same handful of large tables, so reloading per query dominated the run:
    ``cast_info`` alone is tens of millions of rows. The cache changes nothing
    about the results --- the rows are identical --- only how often they are
    parsed. It does keep every table touched so far resident, which is why this
    runner is documented as a local, memory-heavy job.
    """
    rows: dict[str, list[dict[str, str]]] = {}
    capped = False
    for table in tables:
        if table not in schema:
            raise KeyError(f"no schema for table {table!r}")
        key = (table, max_rows)
        if key not in _TABLE_CACHE:
            _TABLE_CACHE[key] = load_relation_rows(
                data_dir / f"{table}.csv", schema[table], max_rows
            )
        loaded = _TABLE_CACHE[key]
        if max_rows is not None and len(loaded) >= max_rows:
            capped = True
        rows[table] = loaded
    return rows, capped


def evaluate_query(
    parsed: ParsedJobQuery,
    rows_by_relation: dict[str, list[dict[str, str]]],
    lower_rate: float,
    lower_seeds: list[int],
) -> dict[str, Any]:
    """Compute true cardinality, estimator q-errors, and C3 cost metrics.

    Times the *full* join (the execution the estimate is meant to avoid) against
    the sample-based lower certificate, so ``cost_ratio`` answers the key
    question: is the certificate meaningfully cheaper than just running the join?
    Both start from the same per-alias filtered rows, so the ratio isolates the
    join cost, not the (shared) scan/filter cost.
    """
    alias_rows = build_alias_rows(parsed, rows_by_relation)

    t0 = perf_counter()
    true_card = count_join(alias_rows, parsed.aliases, parsed.joins)
    full_count_s = perf_counter() - t0
    safe_true = max(true_card, 1.0)

    t1 = perf_counter()
    cert = aggregate_lower_certificate(
        alias_rows, parsed.aliases, parsed.joins, lower_rate, lower_seeds
    )
    lower_cert_s = perf_counter() - t1

    query = build_job_query(parsed, rows_by_relation)
    example = build_training_example(query, safe_true)
    avi = IndependenceEstimator().estimate(query)
    oracle_estimate = example.config_estimates[example.best_config_index]
    floored = max(oracle_estimate, cert.point_estimate)  # lower-cert rescue (C2)

    return {
        "query_id": parsed.query_id,
        "n_relations": len(parsed.aliases),
        "self_join": parsed.is_self_join,
        "true_cardinality": true_card,
        "oracle_qerror": min(example.config_q_errors),
        "floored_qerror": q_error(floored, safe_true),
        "lhat_qerror": q_error(cert.point_estimate, safe_true),
        "pessimistic_qerror": q_error(example.pessimistic_bound, safe_true),
        "avi_qerror": q_error(avi, safe_true),
        "best_config": example.best_config.name,
        "lower_sound_valid": cert.sound_lower_bound <= true_card,
        # 1e-9 tolerance absorbs log-space exp() rounding on exactly-tight bounds.
        "upper_sound_valid": true_card <= example.pessimistic_bound * (1.0 + 1e-9),
        "full_count_ms": full_count_s * 1000.0,
        "lower_cert_ms": lower_cert_s * 1000.0,
        "cost_ratio": lower_cert_s / full_count_s if full_count_s > 0 else float("nan"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Parse, evaluate, and aggregate over all JOB templates in ``--data-dir``."""
    data_dir = Path(args.data_dir)
    schema = load_schema(data_dir)
    skip_tables = frozenset(args.skip_tables or ())
    query_paths = sorted((data_dir / "queries").glob("*.sql"))

    records: list[dict[str, Any]] = []
    skipped: dict[str, list[str]] = {
        "unparsed": [],
        "skipped_tables": [],
        "error": [],
        "timeout": [],
    }
    any_capped = False

    for path in query_paths:
        parsed = parse_job_sql(path.read_text(), path.stem)
        if not parsed.fully_parsed:
            skipped["unparsed"].append(parsed.query_id)
            continue
        if parsed.base_relations & skip_tables:
            skipped["skipped_tables"].append(parsed.query_id)
            continue
        try:
            with time_limit(args.query_timeout_s):
                rows, capped = _load_required_tables(
                    data_dir, schema, parsed.base_relations, args.max_rows
                )
                any_capped = any_capped or capped
                records.append(
                    evaluate_query(parsed, rows, args.lower_rate, args.lower_seeds)
                )
            logger.info("evaluated %s (%d rel)", parsed.query_id, len(parsed.aliases))
        except TimeoutError:
            # Exact counting on the full IMDb is unbounded for the largest
            # joins. Censored queries are listed, never silently dropped, so
            # coverage stays reportable.
            logger.warning(
                "timeout on %s after %.0fs", parsed.query_id, args.query_timeout_s
            )
            skipped["timeout"].append(parsed.query_id)
        except (KeyError, ValueError) as exc:
            logger.warning("error on %s: %s", parsed.query_id, exc)
            skipped["error"].append(parsed.query_id)
        if args.limit_queries and len(records) >= args.limit_queries:
            break

    return _aggregate(records, skipped, len(query_paths), any_capped, args.max_rows)


def _aggregate(
    records: list[dict[str, Any]],
    skipped: dict[str, list[str]],
    total: int,
    capped: bool,
    max_rows: int | None,
) -> dict[str, Any]:
    """Summarise per-query records into the result payload."""

    def med(key: str) -> float:
        values = [r[key] for r in records if r[key] == r[key]]  # drop NaN
        return median(values) if values else float("nan")

    def rate(key: str) -> float:
        if not records:
            return float("nan")
        return sum(1 for r in records if r[key]) / len(records)

    return {
        "benchmark": "JOB (real IMDb)",
        "templates_total": total,
        "evaluated": len(records),
        "coverage": f"{len(records)}/{total}",
        "subsampled": capped,
        "max_rows": max_rows,
        "skipped": {key: ids for key, ids in skipped.items() if ids},
        "aggregated": {
            "oracle_qerror_median": med("oracle_qerror"),
            "floored_qerror_median": med("floored_qerror"),
            "lhat_qerror_median": med("lhat_qerror"),
            "pessimistic_qerror_median": med("pessimistic_qerror"),
            "avi_qerror_median": med("avi_qerror"),
            "lower_sound_validity": rate("lower_sound_valid"),
            "upper_sound_validity": rate("upper_sound_valid"),
            "cost_ratio_median": med("cost_ratio"),
            "full_count_ms_median": med("full_count_ms"),
            "lower_cert_ms_median": med("lower_cert_ms"),
        },
        "per_query": records,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/job_full"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/real_job/job_real.json")
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap rows per table (subsample; makes cardinalities estimates).",
    )
    parser.add_argument(
        "--skip-tables",
        nargs="*",
        default=[],
        help="Base tables to exclude (queries touching them are skipped).",
    )
    parser.add_argument(
        "--query-timeout-s",
        type=float,
        default=600.0,
        help="Per-query budget for loading plus exact counting; censored on exceed.",
    )
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument(
        "--lower-rate",
        type=float,
        default=0.1,
        help="Sample rate for the lower certificate (C3 cost measurement).",
    )
    parser.add_argument("--lower-seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    agg = payload["aggregated"]
    logger.info(
        "\ncoverage %s (subsampled=%s)\n"
        "  q-err median: oracle %.2f | floored %.2f | pessimistic %.2f | AVI %.2f\n"
        "  soundness: lower %.1f%% | upper %.1f%%\n"
        "  C3 cost: lower-cert/full-count median %.3f "
        "(cert %.1f ms vs full %.1f ms) @ rate %.2f -> %s",
        payload["coverage"],
        payload["subsampled"],
        agg["oracle_qerror_median"],
        agg["floored_qerror_median"],
        agg["pessimistic_qerror_median"],
        agg["avi_qerror_median"],
        100 * agg["lower_sound_validity"],
        100 * agg["upper_sound_validity"],
        agg["cost_ratio_median"],
        agg["lower_cert_ms_median"],
        agg["full_count_ms_median"],
        args.lower_rate,
        args.output,
    )


if __name__ == "__main__":
    main()
