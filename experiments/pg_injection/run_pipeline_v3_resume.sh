#!/usr/bin/env bash
# Resume of run_pipeline_v3.sh from step 2 (2026-10-17): step 2 stopped on an order
# PostgreSQL forms through an implied equality, which forced_order now derives.
# Step 1 (STATS-CEB injection) had finished. Same steps and settings otherwise.
# ICDE re-measurement after panel 6 (2026-10-16). Runs only with >= 10 GB free on
# the output disk and a quiet host (1-min load < 60 for 10 consecutive minutes).
#   1. STATS-CEB injection, every source, PostgreSQL's own estimates read in context
#   2. forced left-deep orders (certified, true-C_out-optimal, native-order control)
#   3. JOB-light injection, same protocol
#   4. sampling cost against query execution, timed in one run
#   5. r = 0.25 samples that timed out under load, re-measured and merged
#   6. four more sampling seeds at r = 0.10 (sampled count only)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0
PY="taskset -c 0-35 python3 -m"
R=results/pg_injection
# Outputs are staged on a disk with >= 10 GB free (STAGE, default: the repository's
# own disk) and copied into results/pg_injection/v3 after every step. The guard
# and every harness check free space on the disk they actually write to.
V3=${STAGE:-$R/v3}
FINAL=$R/v3
mkdir -p "$V3" "$FINAL"
log() { echo "[$(date '+%F %T')] $*"; }
publish() { cp -p "$V3"/*.jsonl "$FINAL"/ 2>/dev/null || true; }
trap publish EXIT
wait_for_space_and_quiet() {
  publish
  until [ "$(df --output=avail -B1G "$V3" | tail -1 | tr -d ' ')" -ge 10 ]; do sleep 300; done
  local quiet=0 load
  while [ "$quiet" -lt 10 ]; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$load" 'BEGIN{exit !(l < 60)}'; then quiet=$((quiet + 1)); else quiet=0; fi
    sleep 60
  done
}

wait_for_space_and_quiet
log "2. forced left-deep orders"
$PY experiments.pg_injection.run_forced_orders --output $V3/forced_orders.jsonl

wait_for_space_and_quiet
log "3. JOB-light injection (postgres, duckdb, factorjoin, bayescard, native in context)"
$PY experiments.pg_injection.inject_plans --workload joblight \
    --estimators postgres,duckdb,factorjoin,bayescard,native --tau 1 --always-reps \
    --lower $R/joblight/subset_counts.jsonl \
    --native-csv $R/v2/native_incontext_estimations_joblight.csv \
    --max-load 60 --output $V3/joblight_inject.jsonl

wait_for_space_and_quiet
log "4. sampling cost against query execution, one run"
$PY experiments.pg_injection.sampling_overhead_inrun --output $V3/sampling_overhead_inrun.jsonl

wait_for_space_and_quiet
log "5. r = 0.25 samples that timed out under load"
$PY experiments.pg_injection.subset_counts --rate 0.25 \
    --redo-censored $R/subset_counts_r025.jsonl \
    --output $V3/subset_counts_r025_redo.jsonl
python3 -m experiments.pg_injection.merge_redo --base $R/subset_counts_r025.jsonl \
    --redo $V3/subset_counts_r025_redo.jsonl --output $V3/subset_counts_r025_merged.jsonl

for seed in 1 2 3 4; do
  wait_for_space_and_quiet
  log "6. sampling seed $seed at r = 0.10 (sampled count only)"
  $PY experiments.pg_injection.subset_counts --rate 0.10 --seed "$seed" --skip-exact \
      --output "$V3/subset_counts_r010_seed$seed.jsonl"
done
log "pipeline v3 done"
