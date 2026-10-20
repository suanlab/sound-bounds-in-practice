"""Sub-join records for JOB-light, in the format ``plan_certificate_joint`` reads.

For every JOB-light query: its aliases and, for every sub-plan of at least two
relations in the LpBound artifact's list, the true size and the LpBound and
SafeBound upper bounds the artifact publishes, plus PostgreSQL's in-engine
sampled count from ``subset_counts`` (``--samples``). No max-degree bound is
computed for this benchmark.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.build_joblight_subjoins
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.pg_injection.inject_plans import WORKLOAD_PATHS, load_estimates
from experiments.pg_injection.subset_counts import _read_jsonl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("results/pg_injection/joblight/subset_counts.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_certificate/joblight_subjoin_sizes.jsonl"),
    )
    args = parser.parse_args()

    directory = WORKLOAD_PATHS["joblight"][0]
    truth = load_estimates("truecardinality", directory=directory)
    uppers = {
        name: load_estimates(name, directory=directory)
        for name in ("lpbound", "safebound")
    }
    samples = {
        (r["query_id"], tuple(sorted(r["aliases"]))): r
        for r in _read_jsonl(args.samples)
    }

    by_query: dict[int, list[tuple[str, ...]]] = defaultdict(list)
    for query_id, aliases in truth:
        by_query[query_id].append(aliases)
    records: list[dict[str, Any]] = []
    missing = 0
    for query_id in sorted(by_query):
        keys = sorted(by_query[query_id], key=lambda a: (len(a), a))
        query_aliases = sorted({a for key in keys for a in key})
        subjoins = []
        for key in keys:
            sample = samples.get((query_id, key))
            if sample is None:
                missing += 1
                continue
            subjoins.append(
                {
                    "aliases": list(key),
                    "truth": truth[(query_id, key)],
                    "upper": {n: u[(query_id, key)] for n, u in uppers.items()},
                    "sample": {
                        "rate": sample["rate"],
                        "sound": sample["lower_sound"],
                        "censored": sample["sample_censored"],
                    },
                }
            )
        records.append(
            {
                "query_id": query_id,
                "n_aliases": len(query_aliases),
                "aliases": query_aliases,
                "n_subjoins": len(subjoins),
                "subjoins": subjoins,
            }
        )
    if missing:
        # A sub-join without a sample would silently drop every order through it.
        msg = f"{missing} sub-joins have no sample in {args.samples}"
        raise RuntimeError(msg)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in records))
    logger.info(
        "%d queries, %d sub-joins (%d without a sample) -> %s",
        len(records),
        sum(r["n_subjoins"] for r in records),
        missing,
        args.output,
    )


if __name__ == "__main__":
    main()
