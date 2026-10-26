#!/usr/bin/env bash
# Analyse the PostgreSQL 16 run (after run_pipeline_pg16.sh): the same summaries
# as for 13.1, then the side-by-side comparison and the frozen R1-R6 verdicts.
# Usage (repo root): bash experiments/pg_injection/analyze_pg16.sh [RUN_DIR]
#   RUN_DIR: results/pg_injection/pg16 (default, JIT on) or .../pg16_jitoff
set -euo pipefail
export PYTHONHASHSEED=0
OUT=${1:-results/pg_injection/pg16}
PY="taskset -c 0-35 python3 -m"
$PY experiments.pg_injection.analyze_injection --runs $OUT/stats_inject.jsonl \
    --output $OUT/stats_summary.json
$PY experiments.pg_injection.protocol_summary --runs $OUT/stats_inject.jsonl \
    --output $OUT/stats_protocol.json
$PY experiments.pg_injection.timing_sensitivity --runs $OUT/stats_inject.jsonl \
    --output $OUT/stats_timing_sensitivity.json
$PY experiments.pg_injection.summarize_tightness_points \
    --runs $OUT/tightness_points_stats.jsonl \
    --output $OUT/tightness_points_summary.json
$PY experiments.pg_injection.pg16_comparison --dir $OUT
