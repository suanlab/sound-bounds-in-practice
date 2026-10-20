#!/usr/bin/env bash
# Held-out JOB-range confirmation (docs/icde_2027/PREREGISTRATION_JOBRANGE.md).
# Usage (repo root): bash experiments/scripts/run_heldout_jobrange.sh [PID-to-wait-for]
# Waits for the given process (e.g. a running execution measurement on the same
# container) to exit, then measures and analyses the pre-registered sample.
set -euo pipefail
export PYTHONHASHSEED=0
H=results/heldout_jobrange
PY="taskset -c 0-35 python3 -m"
log() { echo "[$(date -Is)] $*"; }

if [[ -n "${1:-}" ]]; then
    log "waiting for PID $1"
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
IDS=$(python3 -c "import json; print(','.join(map(str, json.load(open('$H/sample.json'))['query_ids'])))")

log "1. exact and sampled counts (r = 0.10, seed 0)"
$PY experiments.pg_injection.subset_counts --workload jobrange_heldout \
    --rate 0.10 --seed 0 --output $H/subset_counts.jsonl
log "2. truth file"
$PY experiments.scripts.heldout_jobrange truth
log "3. in-context native estimates"
$PY experiments.pg_injection.native_estimates_incontext --workload jobrange_heldout \
    --queries "$IDS" --output $H/native_incontext_estimations.csv
log "4. native plans"
$PY experiments.scripts.heldout_jobrange native-plans
log "5. tightness sweep (EXPLAIN only)"
$PY experiments.pg_injection.tightness_sweep --workload jobrange_heldout \
    --native-csv $H/native_incontext_estimations.csv --output $H/tightness_sweep.jsonl
log "6. pre-registered analysis"
$PY experiments.scripts.heldout_jobrange analyze
log "done"
