#!/usr/bin/env bash
# Cross-version execution check on PostgreSQL 16 (stock server, container ceb-pg16
# on port 55433; experiments/pg_injection/docker16). Same protocol as v3 STATS-CEB:
# tau = 1, every distinct plan three times interleaved, 300 s timeout, load gate 60.
# PostgreSQL 16's own in-context estimates come from joinrel_log; the sampled
# floors are the v3 ones (a sampled count is a sound floor on any engine). The
# stats database has temp_file_limit = 8GB (data on /projects; v3 had no cap).
# Usage (repo root): bash experiments/pg_injection/run_pipeline_pg16.sh
set -euo pipefail
export PYTHONHASHSEED=0
OUT=results/pg_injection/pg16
DSN="host=127.0.0.1 port=55433 dbname=stats user=postgres password=postgres"
PY="taskset -c 0-35 python3 -m"
log() { echo "[$(date -Is)] $*"; }

log "1. execution run (published PostgreSQL, DuckDB, PostgreSQL 16's own estimates)"
$PY experiments.pg_injection.inject_plans --dsn "$DSN" \
    --estimators postgres,duckdb,native \
    --native-csv $OUT/native_incontext_estimations_stats.csv \
    --tau 1 --always-reps --max-load 60 --output $OUT/stats_inject.jsonl
log "2. tightness points (both synthetic guards, k = 2 and 8)"
$PY experiments.pg_injection.execute_tightness_points --dsn "$DSN" \
    --native-csv $OUT/native_incontext_estimations_stats.csv \
    --output $OUT/tightness_points_stats.jsonl
log "done"
