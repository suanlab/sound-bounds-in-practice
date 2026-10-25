# PostgreSQL 16 replication against the frozen predictions (`PREDICTIONS_PG16.md`)

Run: 2026-10-20 20:00 to 10-21 04:38 KST, STATS-CEB, 146 queries
(`results/pg_injection/pg16/`); comparison `pg16_comparison.json`. Predictions
frozen 2026-10-20 22:02 (hash verified before analysis).

| ID | Prediction (13.1) | PostgreSQL 16 | Verdict |
|---|---|---|---|
| R1 | own estimates reproduce the native plan on >= 138/146 (143) | 146/146 | holds |
| R2 | headroom in [1.15, 1.60] (1.35x) | 1.42x [1.25, 1.64] | holds |
| R3 | sound guards on own estimates all <= 1.12x (1.02/1.05/1.07) | 1.07 / 1.06 / **1.13** [1.04, 1.26] | **fails** (both guards) |
| R4 | published-PG upper guard >= 2.0x (3.08x) | 2.42x [1.92, 3.11] | holds |
| R5 | planning on LpBound < 0.95x of the true sizes (0.81x) | 0.81x [0.75, 0.88] | holds |
| R6 | synthetic both guards: >= 50% at k = 2, < 50% at k = 8 (71%, 36%) | 81% [69, 89], **51%** [32, 65] | **fails** (k = 8) |

Run notes: one plan (query 58) hit the 8 GB temp_file_limit and is censored with
the 22 timed-out plans; 49 queries started at a one-minute load above 30 (none
above the gate of 60); statistics from PostgreSQL 16's own ANALYZE.
