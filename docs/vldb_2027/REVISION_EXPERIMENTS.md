# Revision analyses after the first freeze

All post-hoc, on stored results. The 2026-10-26 freeze
(`freeze/vldb2027_submission_2026-10-26_2222.*`, supplement copy
`freeze/vldb2027_supplement_2026-10-26.zip`, tag `pvldb2027-submission`) is
superseded. The paper was revised on 2026-11-09 in response to the simulated
review in `REVIEW_VLDB.md`, and its re-freeze is pending. The threshold sweep
below (R1) and the review checks (R2) are now in the revised paper. They were
first prepared for the revision-prep doc
(https://claude.ai/artifact/7ZAJPYKjfXQ6uUsWP1pkyW).

## R3. S3 block re-measurement (registered 2026-11-16, run 2026-11-16/17)

Registration: `PREDICTIONS_S3.md` (with `S3_DESIGN.json`, hashes in `PREDICTIONS_S3.sha256`).
Runner: `experiments/pg_injection/run_blocks_s3.py`, driven by `run_s3_blocks.sh`.
Analysis: `experiments/scripts/s3_analysis.py` → `results/pg_injection/s3/s3_summary.json`.
Table: `experiments/scripts/s3_report.py` → `results/pg_injection/s3/S3_TABLE.md`.

All six blocks are valid, and all six predictions hold. Each pooled ratio is within 6% of the main run on the same queries:
- both guards 1.04×;
- true sizes 1.31×;
- k = 2 and k = 8 recover 68.5% and 25.2% of the headroom;
- certified 1.37× [1.14, 1.69];
- with same-block sampling 0.35× [0.18, 0.61].

Ratios are stable across blocks (range ≤ 0.04), although the native plans' absolute times drift by up to 1.25× between blocks. The two-way intervals (queries and blocks) are as wide as the query-only ones, so the uncertainty comes from the queries, not from the runs.

## R2. Review checks (2026-11-09)

Script: `experiments/scripts/review_checks.py`.
Output: `results/review/review_checks.json`.
Tests: `tests/unit/test_review_checks.py`.

It covers:
- the censoring case of every speedup comparison, including forced orders, with a sensitivity mean;
- headroom exceedance;
- agreement coverage by fully compared query;
- the positivity of every certified sub-join;
- the held-out exclusion ranges.

`plan_certificate_joint.py` was deliberately left at the hash recorded in
`docs/icde_2027/PREREGISTRATION_JOBRANGE_ADDENDUM.sha256`. Its non-empty
precondition is checked by `review_checks` and the audit instead.

## R1. Certification threshold t (2026-11-09; revision-plan item 1, Q5)

Script: `experiments/scripts/certificate_threshold_sweep.py`.
Output: `results/plan_certificate/certificate_threshold_sweep.json`.
Setup: LpBound upper and in-engine sampled floor under C_out, using the stored
per-query joint bounds (nothing re-measured). Intervals are query-clustered
bootstrap, 10k replicates, seed 0. At t = 2 the script reproduces
`certificate_context.json` exactly, which it asserts.

Certified share among the queries where the choice matters (true spread of the
left-deep orders above t, so certifying within t is not trivial):

| case | t = 1.5 | t = 2 (paper) | t = 3 | t = 4 |
| --- | --- | --- | --- | --- |
| STATS-CEB, r = 0.25 | 35.6% [25.3, 46.0] (n=87) | 40.6% [29.0, 52.2] (69) | 45.5% [32.7, 58.2] (55) | 54.3% [39.1, 67.4] (46) |
| STATS-CEB, r = 0.10 | 24.1% [14.9, 33.3] (87) | 30.4% [20.3, 42.0] (69) | 40.0% [27.3, 52.7] (55) | 43.5% [30.4, 58.7] (46) |
| JOB-light, r = 0.10 | 28.8% [18.2, 39.4] (66) | 52.4% [39.7, 65.1] (63) | 64.2% [50.9, 75.5] (53) | 71.1% [57.8, 84.4] (45) |

On the paper's fixed set (spread above 2), STATS-CEB at r = 0.25 gives 37.7%,
40.6%, 46.4% and 55.1% for t = 1.5, 2, 3 and 4. On JOB-light it gives 30.2%,
52.4%, 69.8% and 79.4%.

Reading:
- The shares rise monotonically with t. On STATS-CEB there is no cliff at 2:
  going from t = 2 to t = 1.5 costs 5 to 6 points.
- JOB-light is the exception. At t = 1.5 its share falls from 52% to 29%
  (33 to 19 queries). Many JOB-light certificates sit between 1.5 and 2, so the
  JOB-light headline depends on t = 2 more than the STATS-CEB one does.
- Answer to "is 2 arbitrary?": two things hold for every t from 1.5 to 4. The
  certificate never covers all the queries where the choice matters, and it
  certifies nothing without a lower bound. Coverage varies with t: from 24% to
  54% on STATS-CEB and from 29% to 71% on JOB-light (spread > t). (Corrected
  2026-11-09 after the simulated review's AUD-07: an earlier version said
  "minority-to-half", which t = 3 and 4 on JOB-light and t = 4 on STATS-CEB at
  r = 0.25 contradict.) The JOB-light number should be reported with its
  sensitivity to t. The revised paper now reports it in the joint-certificate
  subsection.
- "Certifies nothing without a lower bound" holds at every t. Every upper bound
  paired with L = 1 (the `xone` rows of the three `plan_certificate_joint_*pg*`
  files) certifies 0 queries even at t = 4. The smallest joint bound is 298 on
  STATS-CEB and 5.5 on JOB-light.
