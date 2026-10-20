#!/usr/bin/env bash
# Sequential PostgreSQL pipeline after phase 2 (STATS raw-estimate injection) ends.
# Each step is resumable; host Python is kept off the container's cores (36-39).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0
PY="taskset -c 0-35 python3 -m"
R=results/pg_injection
log() { echo "[$(date '+%F %T')] $*"; }

log "load IMDb (JOB-light tables)";   $PY experiments.pg_injection.load_imdb
log "JOB-light subset counts";        $PY experiments.pg_injection.subset_counts \
    --workload joblight --output $R/joblight/subset_counts.jsonl
log "JOB-light native estimates";     $PY experiments.pg_injection.native_estimates \
    --workload joblight --output $R/joblight/native_subquery_estimations.csv
log "JOB-light injection";            $PY experiments.pg_injection.inject_plans \
    --workload joblight --estimators postgres,duckdb,native \
    --lower $R/joblight/subset_counts.jsonl \
    --native-csv $R/joblight/native_subquery_estimations.csv \
    --output $R/joblight/inject_plans.jsonl --max-load 60
log "pipeline done"
