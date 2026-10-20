#!/usr/bin/env bash
# Cardinality results and r = 0.10 joint certificates for sampling seeds 1-4
# (seed 0 is the original r = 0.10 run). Analysis only; no database access.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0
PY="taskset -c 0-35 python3 -m"
for seed in 1 2 3 4; do
  samples=results/pg_injection/v3/subset_counts_r010_seed$seed.jsonl
  $PY experiments.scripts.cardinality_baselines --pg-lower "$samples" \
      --output "results/utility_table/cardinality_baselines_pg_seed$seed.json"
  $PY experiments.scripts.plan_certificate_joint \
      --sizes results/plan_guard/subjoin_sizes_maxdeg_fixed.jsonl --pg-samples "$samples" \
      --output "results/plan_certificate/plan_certificate_joint_pg_r010_seed$seed.json"
  echo "seed $seed done"
done
