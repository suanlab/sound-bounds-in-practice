# Addendum 1 to the JOB-range pre-registration (filed 2026-10-19, before the JOB-range measurements)

Filed after an independent review of the analysis code, while the held-out
pipeline was still waiting to start (`results/heldout_jobrange/run.log`: no
JOB-range sweep had been produced). Exceptions, from smoke tests of the
pipeline at 22:22-22:24: the exact and sampled counts of two sub-plans of query
2 (`results/heldout_jobrange/smoke/counts.jsonl`; their values were displayed),
PostgreSQL's in-context estimates of query 2 (written outside the repository),
and the native plans of all 200 queries (planning only; these are inputs, not
outcomes). No test statistic was computed or seen. It changes no prediction and no decision threshold; it
fixes how the registered rules are implemented and states what the original
text left implicit. Code hashes below identify the analysis that will be run.

1. **Exclusions are applied as registered, to every analysis (C1-C7).** A query
   is excluded, in this order, if: its full query is absent from the registered
   sub-plan list; a registered sub-plan has no exact count (missing row,
   timeout or crash); an in-context estimate of a registered sub-plan is
   missing; its native plan is missing; or the native plan joins a set outside
   the registered list. (The first version checked sub-plans against the count
   rows rather than the registered list, did not apply the estimate rule to
   C1-C5, and would have failed rather than excluded on a missing plan.)
2. **C6 and C7 use the same included queries,** and additionally only queries
   whose every guarded plan joins sets with known sizes (a guarded plan's true
   `C_out` is otherwise undefined); their number is reported.
3. **C6's grid** is k in {1, 1.5, 2, 4, 8, 16, 32, 64}; "no increase > 0.05"
   applies to consecutive grid points.
4. **"Both guards" in C6 and C7** is the two-sided policy: every sub-plan value
   `v` becomes `max(min(v, k|S|), |S|/k)`.
5. **C5 is a soundness check of our code,** not an empirical prediction (it
   holds by construction); a violation means a bug.
6. Known bias, stated in advance: 120 s timeouts on exact counts exclude the
   most expensive queries, which tend to be the largest; C1 and C2 may be
   favoured by this. The number excluded per rule is reported.

Validation before the run: on JOB-light (development data, paths substituted),
the corrected code certifies 100% / 96.9% (with / without sampling, final
included) and 43.1% / 1.5% (final excluded) of 65 included queries, consistent
with the development analysis (which used the artifact's truth for the two
queries whose own counts timed out).

Code SHA-256 at filing (`ADDENDUM.sha256` lists them with this file's hash):
`experiments/scripts/heldout_jobrange.py`, `experiments/pg_injection/tightness_sweep.py`,
`experiments/pg_injection/subset_counts.py`,
`experiments/pg_injection/native_estimates_incontext.py`,
`experiments/pg_injection/inject_plans.py`,
`experiments/scripts/plan_certificate_joint.py`,
`experiments/scripts/run_heldout_jobrange.sh`.
