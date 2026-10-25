# Predictions for the PostgreSQL 16 replication (frozen 2026-10-20 22:05 KST)

Frozen while the PostgreSQL 16 execution run (`results/pg_injection/pg16/`) was in
progress: 67 of 146 STATS-CEB queries had been measured. No aggregate had been
computed or seen; the pipeline log had shown the native and true-size plan times
of a handful of individual queries. The tightness points (k = 2, 8) had not
started. Hash and time in `PREDICTIONS_PG16.sha256`. Whatever the outcome, each
row is reported. Development values are the PostgreSQL 13.1 (v3) numbers.

| ID | Prediction for PostgreSQL 16 (13.1 value) | Holds if |
|---|---|---|
| R1 | Injecting PostgreSQL 16's own in-context estimates reproduces its native plan on almost every query (143/146) | >= 138 of 146 |
| R2 | Our instance's headroom (true sizes over its own injected estimates) stays moderate (1.35x) | geometric mean in [1.15, 1.60] |
| R3 | The sound guards on its own estimates change little (upper 1.02x, floor 1.05x, both 1.07x) | all three <= 1.12x |
| R4 | The upper guard on the published PostgreSQL estimates still gives a large speedup (3.08x) | >= 2.0x |
| R5 | Planning on LpBound is slower than planning on the true sizes (0.81x) | < 0.95x |
| R6 | Both synthetic guards recover at least half of the headroom at k = 2 and less than half at k = 8 (71%, 36%) | both hold |
