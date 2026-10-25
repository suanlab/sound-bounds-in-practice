#!/usr/bin/env bash
# PostgreSQL 16 execution re-run with JIT off (2026-10-25), after an independent
# review found that the first PostgreSQL 16 run (results/pg_injection/pg16/) had
# JIT on while PostgreSQL 13.1 has none. Same protocol, estimates and floors as
# run_pipeline_pg16.sh; the statistics are those of the first run (no new
# ANALYZE; exported in results/pg_injection/pg16/pg_stats_stats_db.csv). Waits,
# like the v3 pipeline, for 10 consecutive minutes below a load of 60 first.
# Usage (repo root): bash experiments/pg_injection/run_pipeline_pg16_jitoff.sh
set -euo pipefail
export PYTHONHASHSEED=0
OUT=results/pg_injection/pg16_jitoff
EST=results/pg_injection/pg16/native_incontext_estimations_stats.csv
DSN="host=127.0.0.1 port=55433 dbname=stats user=postgres password=postgres"
PY="taskset -c 0-35 python3 -m"
log() { echo "[$(date -Is)] $*"; }
mkdir -p $OUT

jit=$(docker exec ceb-pg16 psql -U postgres -d stats -At -c "show jit")
[[ "$jit" == "off" ]] || { echo "JIT is $jit; set jit = off first" >&2; exit 1; }
quiet=0
while [ "$quiet" -lt 10 ]; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$load" 'BEGIN{exit !(l < 60)}'; then quiet=$((quiet + 1)); else quiet=0; fi
    sleep 60
done

log "1. execution run, JIT off"
$PY experiments.pg_injection.inject_plans --dsn "$DSN" \
    --estimators postgres,duckdb,native --native-csv $EST \
    --tau 1 --always-reps --max-load 60 --output $OUT/stats_inject.jsonl
log "2. tightness points, JIT off"
$PY experiments.pg_injection.execute_tightness_points --dsn "$DSN" \
    --native-csv $EST --output $OUT/tightness_points_stats.jsonl
log "done"
