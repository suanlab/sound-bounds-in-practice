#!/usr/bin/env bash
# Wait for the r = 0.25 sample run to exit and for the shared host to be quiet
# (1-min load below 60 for 10 consecutive minutes), then time sampling against
# query execution in one run (sampling_overhead_inrun).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0
# Anchored at the start of the command line: an unanchored pattern also
# matches the shell that launched this script (its text contains the pattern).
while pgrep -f "^python3 -m experiments[.]pg_injection[.]subset_counts" >/dev/null; do
  sleep 60
done
until [ "$(df --output=avail -B1G . | tail -1 | tr -d " ")" -ge 10 ]; do
  sleep 300  # the harness stops below 10 GB free; wait for space instead
done
quiet=0
while [ "$quiet" -lt 10 ]; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$load" 'BEGIN{exit !(l < 60)}'; then quiet=$((quiet + 1)); else quiet=0; fi
  sleep 60
done
echo "[$(date '+%F %T')] in-run sampling overhead (load $(cut -d' ' -f1 /proc/loadavg))"
taskset -c 0-35 python3 -m experiments.pg_injection.sampling_overhead_inrun
echo "[$(date '+%F %T')] re-measure the r = 0.25 samples that timed out under load"
taskset -c 0-35 python3 -m experiments.pg_injection.subset_counts --rate 0.25 \
    --redo-censored results/pg_injection/subset_counts_r025.jsonl \
    --output results/pg_injection/v2/subset_counts_r025_redo.jsonl
python3 -m experiments.pg_injection.merge_redo \
    --base results/pg_injection/subset_counts_r025.jsonl \
    --redo results/pg_injection/v2/subset_counts_r025_redo.jsonl \
    --output results/pg_injection/v2/subset_counts_r025_merged.jsonl
echo "[$(date '+%F %T')] done"
