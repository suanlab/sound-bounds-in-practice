# S3: block re-measurement of the small runtime effects (registered 2026-11-16, KST)

Frozen before any S3 block ran. The hashes of this file, `S3_DESIGN.json`, the runner, the driver and the analysis script are in `PREDICTIONS_S3.sha256`. The runner refuses to start if any of them changed.

**Smoke tests run before freezing.** They covered queries 1, 11 and 99, none of them registered, and only checked the runner. For query 99, every plan fingerprint matched v3. For query 11, the live sample counts matched the stored ones. No registered query has been run under S3.

Every row below is reported, whatever its outcome.

## Why

The simulated review (`REVIEW_VLDB.md`) raised three related concerns:
- **R3-M2 and R4-M1 (both blocking):** the paper's small runtime effects come from one run per benchmark. In that run the plans of a query ran in a fixed order, native first, and only queries were resampled.
- **R5-M3:** the sampling cost combined times from different runs.

S3 re-measures these effects in independent time blocks with randomised order. It also runs the sampling, the certificate and the certified execution in the same block.

## Design (fixed in `S3_DESIGN.json`, produced by `experiments/scripts/s3_design.py` from the stored v3 runs only)

- **Set A (21 queries):** STATS-CEB queries whose certified forced order was executed in v3 and where the choice matters (true spread > 2). These are 7, 8, 12, 14, 16, 17, 19, 21, 22, 23, 24, 32, 33, 40, 43, 44, 56, 71, 92, 135, 138.
- **Excluded (4 queries):** 58, 120, 122 and 126. Their native and true-size plans both hit the 300 s cap in v3, so they carry no runtime information.
- **Set B (40 queries):** drawn from the other 121 queries.
  - Allocation is proportional to strata of the number of relations, using largest remainders.
  - The seed is 20261116.
  - The drawn list is in `S3_DESIGN.json`.
- **Policies per query (as in v3):**
  - Native plan.
  - Our instance's in-context estimates injected (`native_raw`).
  - Sound upper guard, floor and both guards (τ = 1).
  - True sizes.
  - Synthetic two-sided brackets at k = 2 and k = 8.
  - For queries with ≥ 3 relations, the certified order of the live certify-then-stop chain (target 2, LpBound, r = 0.10, seed 0).
- **Blocks:**
  - Six blocks (`run_s3_blocks.sh`), each starting at least 3 h after the previous one, and only after 10 consecutive minutes of one-minute load below 60.
  - The PostgreSQL 13.1 server is restarted before each block (empty shared buffers). The OS page cache is not dropped.
  - Queries run in a seeded random order per block.
  - Every distinct plan gets one untimed warm-up, then 3 timed rounds. Each round uses a fresh seeded random permutation, and every plan is timed in every round.
  - Each repetition's time, its position in the round, a timestamp and the load before and after are stored.
  - Timeout is 300 s. A plan that times out is capped and not re-run in that block. Temporary files are capped at 20 GB.
- **Chain:**
  - Each sample is drawn live (`BERNOULLI(10) REPEATABLE(0)`) and run twice. The first run is the deployment cost; the second is the warm time, comparable with v3. Each live count is checked against the stored count.
  - Certificate computation and planning time are timed.
  - The certified plan is executed among the other plans.
  - `chain_total` is first-run sampling + certificate + planning + execution.
  - Within each query, the chain runs before the warm-up pass, as a deployment would: sample first, then execute.
  - A block interrupted and resumed (driver log `block b resumed`, with a server restart) is reported as resumed. Its header records only the first server start.
- **Validity:**
  - A block is valid if all 61 registered queries completed.
  - At least 4 valid blocks are needed. Otherwise S3 is *inconclusive*.
  - No block starts after 2026-11-22 18:00 KST.

## Analysis (`experiments/scripts/s3_analysis.py`, fixed now)

- **Cell value:** for each block, query and plan, the median of the 3 timed repetitions, capped at 300 s. A comparison's cell is log(base / variant).
- **Per-block value:** the geometric mean over queries.
- **Pooled value:** the geometric mean over all (block, query) cells, with two 95% intervals:
  - a two-way bootstrap (queries and blocks resampled independently, 10,000 replicates, seed 0);
  - a query-only bootstrap.
- **Comparisons:**
  - On A+B (61 queries): the guards, true sizes, k = 2 and k = 8, each against `native_raw`.
  - On A (21 queries): `certified` and `chain_total` against `native`.
- **Descriptive measures:**
  - block drift of `native_raw`;
  - the median per-query max/min ratio across blocks;
  - censored plans per block.

## Predictions (reference: v3 on the same queries, `S3_DESIGN.json` → `reference_v3`)

| ID | Prediction (v3 value on the same set) | Holds if |
|---|---|---|
| P1 | The sound guards on our estimates change little. Upper 1.009×, floor 1.031×, both 1.041× (A+B). | For each guard: pooled in [0.95, 1.12] and every valid block in [0.90, 1.18]. |
| P2 | The headroom to the true sizes is moderate but real: 1.303× (A+B). | Pooled ≥ 1.15 and every valid block ≥ 1.08. |
| P3 | A tighter synthetic bracket recovers more: truth ≥ k = 2 ≥ k = 8 (1.303, 1.199, 1.069). The log share of the headroom is 0.68 at k = 2 and 0.25 at k = 8. | The order holds (1% tolerance) in ≥ V−1 of V valid blocks, the pooled k = 2 share is in [0.45, 0.95], and the pooled k = 8 share is in [0.05, 0.55]. |
| P4 | Certified plans run faster than native where the choice matters: 1.32× (A). | Pooled ≥ 1.10, two-way lower bound > 1.0, and ≥ 1.0 in ≥ V−1 valid blocks. |
| P5 | With same-block sampling, certification costs more than it saves: 0.34× (A, cross-run). | Two-way upper bound < 1.0, and < 1.0 in every valid block. |
| P6 | The chain reproduces. Every live sample count equals the stored count, every A query is certified in every valid block, and its order equals v3's. | No mismatch. |

## Consequences for the paper (fixed now)

- **If S3 is valid:**
  - The block table goes in the supplement.
  - In the Limitations section, the sentence "one run per benchmark and version … mitigate host noise but do not remove it" is replaced by one sentence giving the S3 result. Text is replaced, not added: the paper has no free space.
- **If P1, P4 or P5 fails:** the corresponding runtime statement in the abstract, the execution section and the conclusion is restated as a single-run description. The plan-change counts become the primary evidence, as R4-M1 asks.
- **If P2 or P3 fails:** the headroom and the synthetic-bracket recoveries (71%, 36%) are marked as not reproduced across blocks.
- **If P6 fails:** the mismatch is reported and the cause traced before anything else.
- **If S3 is inconclusive:** the paper adopts the review's fallback. The small runtime effects are stated as one-run descriptive statistics.
