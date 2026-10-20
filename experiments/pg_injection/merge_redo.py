"""Merge a re-measurement of timed-out samples into the original sample file.

Rows of ``--redo`` replace the rows of ``--base`` with the same (query,
sub-plan) and are marked ``"redo": true``; every other row is kept unchanged.
The original file is not modified.

Usage::

    python -m experiments.pg_injection.merge_redo \
        --base results/pg_injection/subset_counts_r025.jsonl \
        --redo results/pg_injection/v2/subset_counts_r025_redo.jsonl \
        --output results/pg_injection/v2/subset_counts_r025_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from experiments.pg_injection.subset_counts import _read_jsonl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--redo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    redo = {(r["query_id"], tuple(r["aliases"])): r for r in _read_jsonl(args.redo)}
    merged, replaced = [], 0
    for row in _read_jsonl(args.base):
        key = (row["query_id"], tuple(row["aliases"]))
        if key in redo:
            row = {**redo[key], "redo": True}
            replaced += 1
        merged.append(row)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in merged))
    still = sum(r["sample_censored"] for r in merged)
    logger.info(
        "%d rows, %d replaced, %d samples still censored -> %s",
        len(merged),
        replaced,
        still,
        args.output,
    )


if __name__ == "__main__":
    main()
