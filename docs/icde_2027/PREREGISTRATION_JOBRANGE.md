# Pre-registration: held-out confirmation on JOB-range (frozen 2026-10-19, before any measurement)

Status at freezing: no JOB-range query has been counted, estimated, planned or
executed by us. Development analyses (STATS-CEB, JOB-light) are in
`results/plan_certificate/certificate_mechanisms.json` and
`results/pg_injection/v3/tightness_sweep_*.summary.json`; the predictions below
were set from them. This file's SHA-256 and timestamp are recorded in
`PREREGISTRATION_JOBRANGE.sha256`; the author is asked to deposit both publicly
(e.g. OSF) before the measurements finish. Whatever the outcome, every test
below is reported, labelled confirmatory.

## Sample and measurements
- Population: the 1,000 JOB-range queries of the LpBound artifact
  (`benchmarks/workloads/jobrange/jobrangeQueries.sql`, query id = line number),
  with the artifact's sub-plan list (`jobrange_subqueries.csv`).
- Sample: 200 query ids drawn with `numpy.random.default_rng(20261019)
  .choice(1000, 200, replace=False) + 1`; listed in
  `results/heldout_jobrange/sample.json`.
- Measured with the existing harness on the same PostgreSQL 13.1 instance
  (database `imdb`): exact and in-engine sampled count of every sub-plan
  (`subset_counts`, r = 0.10, seed 0, 120 s timeout), PostgreSQL's own
  in-context sub-plan estimates (`native_estimates_incontext`), PostgreSQL's
  native plan (`EXPLAIN`, no hints), and the tightness sweep (`tightness_sweep`,
  EXPLAIN only).
- Exclusions (fixed now): a query is excluded from an analysis if any of its
  sub-plans has no exact count (timeout or crash), if any in-context estimate is
  missing, or if its native plan joins a set absent from the sub-plan list.
  Certificate analyses use queries with 3+ relations. Counts of exclusions are
  reported.
- No upper bound from statistics is available for JOB-range sub-plans (the
  artifact ships none), so H1, H2 and H6 are not tested here.

## Definitions (as in the development code)
- Plans: left-deep orders whose prefixes are in the sub-plan list; cost
  `C_out` = sum of the true sizes of the join nodes, with or without the final
  result. Native plan: PostgreSQL's own plan for the query without hints.
- A-posteriori bound of the native plan p: `C(p) / min(C(p), min_p' C_lo(p'))`,
  where `C_lo(p')` takes a sub-plan at its exact size if p computes it and at the
  floor otherwise (floor = sound sampled count, or 1 for "no sampling").
  Certified: bound <= 2.
- Tightness sweep: synthetic sound bounds `B_k = k |S|`, `L_k = |S| / k`, both
  guards applied to PostgreSQL's own in-context estimates; gain = geometric mean
  over queries of `C_out(raw plan) / C_out(guarded plan)`; plan changed = plan
  fingerprint differs from the raw-estimate plan.

## Confirmatory hypotheses and decision rules

| ID | Prediction (development values in brackets) | Confirmed if | Falsified if |
|---|---|---|---|
| C1 | With the final result in `C_out` and **no sampling**, the native plan is certified within 2x a posteriori on most queries [STATS 83.9%, JOB-light 97.0%] | share >= 70% | share < 50% |
| C2 | With the final result and the sampled floor [STATS 86.1%, JOB-light 100%] | share >= 80% | share < 60% |
| C3 | The zero-cost certificates come from the final result: without it and without sampling almost nothing is certified [STATS 0%, JOB-light 1.5%] | share <= 5% | share > 10% |
| C4 | Without the final result but with the floor, a minority is certified [STATS 21.2%, JOB-light 41.8%] | 10% <= share <= 70% | share < 5% or > 80% |
| C5 | Soundness: on every included query, the native plan's true regret over the left-deep orders is at most its a-posteriori bound | no violation | any violation (a bug) |
| C6 | Plan changes under both synthetic guards fall with looseness [STATS 0.47 at k=2, 0.16 at k=16; JOB-light 0.20, 0.07] | share(k=2) >= 2 x share(k=16), and no increase > 0.05 between consecutive k | otherwise |
| C7 | If planning on the true sizes lowers `C_out` by >= 5% (geometric mean), both synthetic guards recover at least half of that gain (log scale) at k = 2 and less than half at k = 16 [STATS 0.69 and 0.41] | both hold | either fails; declared *not testable* if the true-size gain is < 1.05x [JOB-light 0.98x] |

Analyses beyond these are exploratory and labelled so.
