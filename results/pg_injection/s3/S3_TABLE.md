# S3: six-block re-measurement (STATS-CEB, PostgreSQL 13.1)

Registered in `docs/vldb_2027/PREDICTIONS_S3.md` before any block ran.
Each value is a geometric-mean speedup (base / policy) over the set's
queries.
The pooled interval resamples queries and blocks independently
(10,000 replicates). The v3 column is the main run on the same queries.

| Comparison | v3 | Pooled [95%] | B1 | B2 | B3 | B4 | B5 | B6 |
|---|---|---|---|---|---|---|---|---|
| Upper guard (A+B) | 1.01 | 1.01 [0.97, 1.05] | 1.01 | 1.01 | 1.01 | 1.01 | 1.01 | 1.01 |
| Floor (A+B) | 1.03 | 1.03 [0.99, 1.08] | 1.04 | 1.03 | 1.03 | 1.03 | 1.03 | 1.03 |
| Both guards (A+B) | 1.04 | 1.04 [0.98, 1.11] | 1.05 | 1.04 | 1.04 | 1.05 | 1.04 | 1.04 |
| True sizes (A+B) | 1.30 | 1.31 [1.18, 1.46] | 1.33 | 1.31 | 1.31 | 1.31 | 1.31 | 1.29 |
| Synthetic bracket, k = 2 (A+B) | 1.20 | 1.20 [1.10, 1.33] | 1.21 | 1.20 | 1.21 | 1.20 | 1.20 | 1.19 |
| Synthetic bracket, k = 8 (A+B) | 1.07 | 1.07 [1.02, 1.14] | 1.07 | 1.07 | 1.07 | 1.07 | 1.07 | 1.07 |
| Certified order (A) | 1.32 | 1.37 [1.14, 1.69] | 1.38 | 1.36 | 1.38 | 1.37 | 1.39 | 1.36 |
| Certified, with same-block sampling (A) | 0.34 | 0.35 [0.18, 0.61] | 0.37 | 0.36 | 0.35 | 0.35 | 0.35 | 0.34 |

Native-plan drift per block (block mean / mean over blocks): B1 1.25, B2 0.93, B3 0.91, B4 0.93, B5 1.10, B6 0.92; median per-query max/min over blocks 1.36.
Verdicts: {"P1_native_guards_small": true, "P2_headroom": true, "P3_tightness_order": true, "P3_blocks_ordered": 6, "P4_certified_faster": true, "P5_sampling_dominates": true, "P6_reproducible_chain": true}.
Live sample counts: {"n_draws": 834, "n_censored": 12, "n_mismatch": 0}; certified orders: {"n_a_cells": 126, "n_live_certified": 126, "n_order_differs_from_v3": 0}.
Certificate computation per query: median 1.16 ms, max 160 ms (354 chains).
