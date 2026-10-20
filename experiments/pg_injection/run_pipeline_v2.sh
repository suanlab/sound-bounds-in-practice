#!/usr/bin/env bash
# ICDE re-measurement (panel 5): one run per benchmark with every estimate source,
# tau = 1, every plan timed --reps times; then in-engine samples at r = 0.25.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0
PY="taskset -c 0-35 python3 -m"
R=results/pg_injection
log() { echo "[$(date '+%F %T')] $*"; }

log "STATS-CEB re-run (postgres, duckdb, native; tau 1; always 3 reps)"
$PY experiments.pg_injection.inject_plans --estimators postgres,duckdb,native \
    --tau 1 --always-reps --max-load 60 --output $R/v2/stats_inject.jsonl
log "JOB-light re-run (postgres, duckdb, factorjoin, bayescard, native)"
$PY experiments.pg_injection.inject_plans --workload joblight \
    --estimators postgres,duckdb,factorjoin,bayescard,native --tau 1 --always-reps \
    --lower $R/joblight/subset_counts.jsonl \
    --native-csv $R/joblight/native_subquery_estimations.csv \
    --max-load 60 --output $R/v2/joblight_inject.jsonl
log "STATS-CEB in-engine samples at r = 0.25"
$PY experiments.pg_injection.subset_counts --rate 0.25 --output $R/subset_counts_r025.jsonl
log "pipeline v2 done"
