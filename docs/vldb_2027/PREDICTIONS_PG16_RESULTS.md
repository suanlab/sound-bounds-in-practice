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

## JIT-off re-run (2026-10-25 22:28 to 10-26 07:21; `results/pg_injection/pg16_jitoff/`)

An independent review (`reviews/pvldb_2026-10-25/report.md`) found JIT on in the
run above (`pg_jit_available()` true, `jit = on`) while PostgreSQL 13.1 has no
JIT. We re-ran with `jit = off` (`ALTER SYSTEM`), the same statistics (no new
ANALYZE; exported in `results/pg_injection/pg16/pg_stats_stats_db.csv`), after 10
quiet minutes; median start load 6.4, maximum 22.1. The predictions above were
frozen for the first run and are scored on it; on the re-run the values are
(predictions not re-scored): headroom 1.40x [1.23, 1.62]; guards 1.07 / 1.06 /
1.14x [1.04, 1.27]; published-PG upper guard 2.48x; LpBound vs truth 0.82x;
synthetic shares 81% [69, 89] at k = 2 and 47% [27, 62] at k = 8; 146/146 native
plans reproduced; 24 censored plans, one of them at the 8 GB temp cap. JIT does
not explain the difference from 13.1. The paper reports the JIT-off run as
primary and the JIT-on run beside it.

Note: the header of `PREDICTIONS_PG16.md` says 22:05 KST; the hash file's
timestamp, 22:02:00, is authoritative. The file was frozen with 67 of 146
queries measured and per-query native and true-size times visible in the
pipeline log (which bear on R2); no aggregate had been computed.
