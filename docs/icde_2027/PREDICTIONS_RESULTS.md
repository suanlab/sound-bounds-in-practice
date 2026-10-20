# Exploratory results against the frozen predictions (2026-10-19, revised after review)

Predictions: `PREDICTIONS.md` (SHA-256 in `PREDICTIONS.sha256`, frozen 21:58 KST,
before any of these analyses ran). Development data only (STATS-CEB, JOB-light);
the confirmatory test is `PREREGISTRATION_JOBRANGE.md` (+ `_ADDENDUM.md`).
Intervals: query-clustered bootstrap, 10,000 replicates, seed 0
(`results/plan_certificate/mechanism_intervals.json`). Revised after an
independent review (`reviews/mechanisms_2026-10-19/report.md`), which found two
of the first explanations (H2, H5) false; verdicts did not change.

| H | Prediction | Result | Verdict |
|---|---|---|---|
| H1 | rho predicts certification, AUC >= 0.9 (STATS), >= 0.85 (JOB-light, STATS threshold); certification **and rho** fall with the number of relations | AUC 0.93 [0.88, 0.98] STATS, 0.93 [0.87, 0.98] JOB-light (accuracy 91% and 82% at the STATS threshold); the number of relations alone gives 0.87 and 0.64. Certification falls with size on both (STATS 94/44/14/4/0% for 3-7 relations; JOB-light 66/43/33%), but on JOB-light median rho does not (0.31/0.29/0.31) | **AUC part holds; the "rho falls with size" part fails on JOB-light**, so rho does not explain the size trend there |
| H2 | geometric tightening of upper bounds by 2.5x (whole-query median) at r = 0.10 gives 55.0% +- 5 pp | 45.0%; the r = 0.25 share is reached between 4x and 8x (interpolated 5.75x). Explanation: the joint certificate depends on the bracket only through B/L, so dividing every upper bound by 2.5 gives exactly the share of multiplying every floor by 2.5 (both 55.0%, the r = 0.25 share); the geometric tightening, which keeps the bounds sound, shrinks the median *sub-join* bound only 1.36x; dividing by 2.5 is unsound on 658 of 1,560 sub-joins, and its tightest sound version (clipped at the truth) certifies 50.7%. The scale invariance holds only for a common factor on every L and B, not per sub-join | **fails as operationalised**; the underlying exchange is exactly one-for-one in the ratio B/L (an identity, not a finding) |
| H3 | synthetic **upper** guard B_k = k|S| on our estimates: gain <= 1.05x for k >= 8; >= 50% of the true-size gain only for k <= 4 (threshold-shaped) | STATS (true-size C_out gain 1.20x, 53% of it from three queries): upper guard 1.03x for k = 4-32; share of the true-size gain 46% [17, 82] at k = 1, 43% at k = 2, 15% [0, 52] at k = 4 (leave-one-out changes up to 0.21). Both guards (exploratory): 69% [51, 87] at k = 2, 45% [15, 75] at k = 4, 38-41% for k = 8-32. Published PG estimates (robust, leave-one-out <= 0.03): both guards 73% [59, 84] even at k = 32. JOB-light: no true-size C_out gain for our estimates (0.98x), not testable | **first clause holds; the 50% clause fails** (the upper guard never reaches half even with exact bounds). The apparent step between k = 2 and 4 is within the intervals; the robust contrast is between estimators: the bound a guard needs scales with the errors it meets. **Executed (exploratory, both guards, 146 STATS queries, `v3/tightness_points_summary.json`):** true sizes 1.34x [1.23, 1.50]; k = 2 recovers 71% [59, 82] (1.23x), k = 8 36% [20, 49] (1.11x), the real bracket 23%; the time intervals separate k = 2 from k = 8, which the C_out sweep did not |
| H4 | a-posteriori: >= 55% certified with sample, >= 25% with no sampling, the latter mostly where choice does not matter | STATS (137 of 140 queries; 3 lack sub-plan truth): 86.1% with sample, 83.9% with none; JOB-light: 100% and 97.0%. Without the final result: STATS 21.2% / 0%, JOB-light 41.8% / 1.5%. No-sampling certificates: 64 of 115 where the choice does not matter (by count), but the rate is high in both groups (91% vs 76%) | **holds numerically; "mostly where choice does not matter" holds by count, fails by rate**. The zero-cost certificates exist only because the shared final result dominates C_out. Sizes come from our offline exact counts (what EXPLAIN ANALYZE would report), not from executed plans |
| H5 | oracle sampling cost > native execution on >= 50% of certified choice-matters queries | 19-33% (three native-time references); median oracle / execution 0.39-0.49; the online rule is optimal on 77% of certified queries (62% where the choice matters). **End to end** (certified-plan execution + sampling, forced-order run), the certified plans still run 0.51x [0.28, 0.78] as fast as native where the choice matters with oracle sampling (0.34x [0.17, 0.57] online), faster on 14% | **fails** for the cost of sampling alone; but end to end certification still loses even with an oracle |
| H6 | calibrated floor L-hat / c (c from seed 0): >= 55% certified, violations <= 5%, certified regret <= 2 for >= 95% | c = 1.057; certified 79-81% on seeds 1-4 (sound floor 40-41%) and 91% on JOB-light (52%); every certified plan truly within 2x; violations 8-13% on seeds 1-4 (5.6% JOB-light). Leave-one-draw-out: seed 0 gives the lowest c of five (others 1.10-1.13); calibrated on any other draw, violations on the rest are 1.5-6.4% | **fails on its own criterion**; the cause is the variability of the calibration constant across draws (seed 0 was the low draw), not correlation within a query as first written |
| H7 | witness `LIMIT k` floors <= 0.2x the sampled count | median 0.96x [0.80, 1.04] over 70 sub-joins of the oracle's certifying sets (57 queries, none censored), timed back to back with the sampled count; cheaper on 56%; total 228 s against 203 s. Deviation stated beforehand in `witness_floor.py`: the floors are the sampled counts, not floors computed from bounds | **fails** (> 0.5x): proving a sub-join large costs about as much as sampling it |

Also exploratory: the query-level funnel (`v3/guard_funnel.json`), both guards
for both sources. Our instance: 91 queries with a > 10x sub-plan error, 86 seen
by the bracket, 32 plans changed, 5 with > 10% less true work, 17 faster (12 of
them without less work). Published PostgreSQL estimates: 132 -> 131 -> 121 ->
88 less work -> 100 faster. Timings of censored plans are kept at the timeout.

## Confirmatory: held-out JOB-range (pre-registered; `results/heldout_jobrange/confirmatory_results.json`)

200 sampled queries; exclusions 9 (sub-plan without an exact count) and 12
(fewer than three relations); 179 queries for the certificates, 191 for the
sweep; no query hit the 2 GB temp cap (Addendum 2). Verdicts as computed by the
registered code (hashes in the addendum, unchanged):

| Test | Result | Verdict |
|---|---|---|
| C1 final included, no sample >= 70% | 86.6% | confirmed |
| C2 final included, sampled floor >= 80% | 88.8% | confirmed |
| C3 final excluded, no sample <= 5% (falsified > 10%) | 7.3% | inconclusive |
| C4 final excluded, sampled floor in [10%, 70%] | 35.8% | confirmed |
| C5 soundness (code check) | 0 violations | confirmed |
| C6 plan changes halve from k = 2 to k = 16, no rise > 0.05 | 19.4% -> 5.8%, monotone | confirmed |
| C7 dose-response if true-size C_out gain >= 1.05x | 1.03x | not testable |

Of the five empirical predictions that were testable (C5 is a code check), four
were confirmed and C3 was inconclusive.
