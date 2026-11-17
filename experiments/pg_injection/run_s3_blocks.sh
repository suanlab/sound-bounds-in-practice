#!/usr/bin/env bash
# S3 block re-measurement driver (registered in docs/vldb_2027/PREDICTIONS_S3.md).
#
# Runs blocks 1..N_BLOCKS of experiments.pg_injection.run_blocks_s3, one at a
# time. Before each block it:
#   * waits until START_GAP_H hours have passed since the previous block started;
#   * waits for a quiet host (1-min load < 60 for 10 consecutive minutes, as v3);
#   * restarts the PostgreSQL 13.1 container (empty shared buffers; the OS page
#     cache is not dropped, which needs root).
# No block starts after DEADLINE. Outputs are staged on the root disk (STAGE; the
# server's data directory is on that disk too) and copied to results/ after each
# block. Resumable: a finished block is skipped and an unfinished one continues.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0
N_BLOCKS=${N_BLOCKS:-6}
START_GAP_H=${START_GAP_H:-3}
DEADLINE=${DEADLINE:-"2026-11-22 18:00 +0900"}
STAGE=${STAGE:-/home/suan/ceg_s3_stage}
FINAL=results/pg_injection/s3
CONTAINER=ceb-icde
mkdir -p "$STAGE" "$FINAL"
log() { echo "[$(date '+%F %T %Z')] $*"; }
deadline_s=$(date -d "$DEADLINE" +%s)

wait_quiet() {
  local quiet=0 load
  while [ "$quiet" -lt 10 ]; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$load" 'BEGIN{exit !(l < 60)}'; then quiet=$((quiet + 1)); else quiet=0; fi
    sleep 60
  done
}

restart_server() {
  docker restart "$CONTAINER" >/dev/null
  for _ in $(seq 1 60); do
    if python3 -c "import psycopg2; psycopg2.connect('host=127.0.0.1 port=55432 dbname=stats user=postgres password=postgres').close()" 2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  log "server did not come back"; return 1
}

block_done() { [ -f "$STAGE/block_$1.done" ]; }

last_start=0
for b in $(seq 1 "$N_BLOCKS"); do
  if block_done "$b"; then
    last_start=$(cat "$STAGE/block_$b.start")
    continue
  fi
  if [ ! -f "$STAGE/block_$b.start" ]; then
    next=$((last_start + START_GAP_H * 3600))
    while [ "$(date +%s)" -lt "$next" ]; do sleep 300; done
    if [ "$(date +%s)" -gt "$deadline_s" ]; then log "deadline passed; block $b not started"; break; fi
    wait_quiet
    restart_server
    date +%s >"$STAGE/block_$b.start"
    log "block $b started"
  else
    log "block $b resumed"
    restart_server
  fi
  last_start=$(cat "$STAGE/block_$b.start")
  taskset -c 0-35 python3 -m experiments.pg_injection.run_blocks_s3 \
    --block "$b" --output "$STAGE/block_$b.jsonl" 2>&1 | tee -a "$STAGE/block_$b.log"
  touch "$STAGE/block_$b.done"
  cp -p "$STAGE/block_$b.jsonl" "$STAGE/block_$b.log" "$FINAL"/
  log "block $b done"
done
log "S3 driver finished"
