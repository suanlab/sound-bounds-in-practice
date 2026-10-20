"""Prepare the real Join Order Benchmark (JOB) for the Safe-CEG harness.

Downloads the real IMDb dataset and the 113 JOB query templates, then lays them
out so the corrected real-data harness (``run_real_data``) can consume them.

This is intended to be run **locally** -- the IMDb archive is ~1.26 GB
compressed (several GB extracted) and the full JOB joins reach billions of
rows, so it is impractical to run inside a constrained sandbox. The count-based
hash join (``run_real_data.count_join``) and the new selection-filter support
(``compute_true_cardinality_filtered``) make exact filtered cardinalities
feasible without materializing the join.

Usage (local):
    python -m experiments.scripts.prepare_real_job --out data/job_full
    # then evaluate (after wiring the parsed JOB queries):
    python -m src.training.evaluate ...   # or run_real_data on data/job_full

Status / remaining work (documented honestly):
  * Download + extract IMDb CSVs and fetch JOB *.sql -- implemented here.
  * Parsing each JOB *.sql into (relations, JoinPredicate set, SelectionFilter
    list) -- a SQL-subset parser is the remaining piece; the harness side
    (joins + filters -> exact cardinality) is already in place via
    ``compute_true_cardinality_filtered``.
"""

from __future__ import annotations

import argparse
import logging
import tarfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

IMDB_ARCHIVE_URL = "https://event.cwi.nl/da/job/imdb.tgz"
JOB_QUERIES_BASE = (
    "https://raw.githubusercontent.com/gregrahn/join-order-benchmark/master"
)
# The 113 JOB query templates (families 1--33, variants a--d).
JOB_QUERY_IDS = [
    f"{n}{v}" for n in range(1, 34) for v in ("a", "b", "c", "d", "e", "f")
]


def download(url: str, dest: Path, *, skip_existing: bool = True) -> bool:
    """Download ``url`` to ``dest``; return True on success."""
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        logger.info("exists, skipping: %s", dest)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("downloading %s -> %s", url, dest)
        urllib.request.urlretrieve(url, dest)
        return True
    except OSError as exc:
        logger.warning("failed to download %s: %s", url, exc)
        return False


def fetch_job_queries(out_dir: Path) -> int:
    """Download the JOB *.sql templates that exist on the mirror."""
    queries_dir = out_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for qid in JOB_QUERY_IDS:
        dest = queries_dir / f"{qid}.sql"
        if download(f"{JOB_QUERIES_BASE}/{qid}.sql", dest):
            fetched += 1
    logger.info("fetched %d JOB query templates into %s", fetched, queries_dir)
    return fetched


def fetch_and_extract_imdb(out_dir: Path, *, skip_download: bool) -> None:
    """Download and extract the IMDb CSVs used by JOB."""
    archive = out_dir / "imdb.tgz"
    if not skip_download and not download(IMDB_ARCHIVE_URL, archive):
        logger.warning("IMDb archive download failed; skipping extraction")
        return
    if not archive.exists():
        logger.info("no IMDb archive present (use without --skip-download to fetch)")
        return
    logger.info("extracting %s (this is large) ...", archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(out_dir, filter="data")
    logger.info("extracted IMDb CSVs into %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/job_full"))
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the 1.26 GB IMDb download (use an archive already on disk).",
    )
    parser.add_argument(
        "--queries-only",
        action="store_true",
        help="Fetch only the JOB *.sql templates (small), not the IMDb data.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fetch_job_queries(args.out)
    if not args.queries_only:
        fetch_and_extract_imdb(args.out, skip_download=args.skip_download)

    logger.info(
        "\nNext: parse %s/queries/*.sql into (relations, joins, filters) and "
        "evaluate with compute_true_cardinality_filtered (joins + filters are "
        "already supported).",
        args.out,
    )


if __name__ == "__main__":
    main()
