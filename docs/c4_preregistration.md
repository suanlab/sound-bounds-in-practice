# C4 Pre-registration — Do Sound Brackets Certify Plans?

**Frozen: 2026-09-03.** Registered *before* `experiments/scripts/plan_certificate.py`
exists and before any bracket has been evaluated at the plan level. Nothing
below may change after results are seen; any deviation is reported in the paper
as a deviation, next to the number it affects. Companion design note:
`docs/edbt_2027/BREAKTHROUGH_STRATEGY.md` §3.1.

## Question

The paper's plan-level section (RQ6) reports that guarding subplan estimates
leaves p95 plan regret unchanged and that executed runtime has an unresolved
sign. Mandarapu and Kunkunuru (arXiv:2606.15600) show independently that
q-error does not predict plan regret; what does is the log-distance from the
estimate to the nearest plan-switch boundary. We ask the decidable version of
"when does a certificate help at the plan level":

> Given a **sound bracket** $[L(S), B(S)]$ on every sub-join $S$ of a query,
> and a cost model monotone in cardinalities, how often does the bracket
> (a) prune plans soundly, (b) bound plan regret non-vacuously, and
> (c) certify a single plan — using today's producers (LpBound, SafeBound,
> max-degree; sampling lower certificate)?

We claim **no new estimator and no new bound**. The brackets are the artifact's
and ours as already used in the paper; the contribution is the measurement.

## Definitions (fixed)

Let $Q$ have aliases $A$, $|A| = n$, join graph from the parsed SQL.

1. **Plans.** All left-deep alias orders $p$ whose every prefix is a connected
   sub-join (`plan_regret._left_deep_orders`). Cross products are never
   enumerated.
2. **Cost model.** $C_{\text{out}}$ exactly as in `plan_regret._order_cost`:
   $C(p, x) = \sum_{i=2}^{n} x(\text{prefix}_i(p))$, the sum of the sizes of the
   intermediate results of all prefixes of length $2..n$ (the final result is
   included, as in the existing script; it is a common additive term).
   $C$ is monotone non-decreasing in every $x(S)$.
3. **Truth.** $y(S)$ = the LpBound artifact's `truecardinality` sub-query value
   (`external/LpBound/results/evaluation_time/subquery_estimations/stats/truecardinality_subquery_estimations.csv`),
   keyed by (`QueryID`, `frozenset(Tables)`), floored at 1. Where our
   `count_join` finishes within the per-subquery budget it is cross-checked
   against the artifact; disagreements are counted and reported.
4. **Upper certificates** $B(S)$ (three): `maxdeg` = our `pessimistic_bound`
   on the sub-query (the paper's max-degree bound); `safebound` and `lpbound`
   = the artifact's sub-query columns. Each floored at 1.
5. **Lower certificates** $L(S)$ (three): `one` = the trivial $L(S) = 1$
   (the "upper-only" world of today's pessimistic-CE literature);
   `sample0.10` and `sample0.25` = the paper's sampling certificate
   (`aggregate_lower_certificate`, one seed, `seed=0`) run on the sub-join,
   using its **sound** value $L(S) = S$ (the sub-sampled join size), never the
   unbiased $\hat L$. A sub-join whose certificate exceeds the budget gets
   $L(S) = 1$ and is counted as censored.
6. **Brackets.** The $3 \times 3 = 9$ (upper, lower) pairs. A bracket is
   *sound on $Q$* iff $L(S) \le y(S) \le B(S)$ for every enumerated sub-join.
   Violations are counted per certificate and per query; a query with any
   violation is **flagged** and excluded from the soundness checks in §Checks
   (its plan-level numbers are still reported, separately).
7. **Plan intervals.** $C_{\text{lo}}(p) = C(p, L)$, $C_{\text{hi}}(p) = C(p, B)$.
   Soundness of the bracket implies $C_{\text{lo}}(p) \le C(p, y) \le C_{\text{hi}}(p)$.
8. **Dominance pruning.** $p$ is pruned iff
   $C_{\text{lo}}(p) > \min_{p'} C_{\text{hi}}(p')$. Survivors $\mathcal{S}(Q)$
   are the unpruned plans; $|\mathcal{S}(Q)| \ge 1$ always.
   **Pruning rate** $= 1 - |\mathcal{S}(Q)| / |\text{plans}(Q)|$.
9. **Certified regret bound.**
   $G(Q) = \max_{p \in \mathcal{S}(Q)} C_{\text{hi}}(p) \,/\, \min_{p'} C_{\text{lo}}(p')$.
   For any plan $p \in \mathcal{S}(Q)$, sound brackets give
   $\text{regret}(p) = C(p,y)/\min_{p'} C(p',y) \le G(Q)$.
   $G_{\text{all}}(Q)$ is the same ratio over all plans (no pruning); it bounds
   the regret of *any* plan.
10. **Minimax plan.** $p^\star = \arg\min_p C_{\text{hi}}(p)$ (ties: lexicographic
    order). Its regret is bounded by $\min_p C_{\text{hi}}(p) / \min_p C_{\text{lo}}(p)$.
11. **Plan certified.** $|\mathcal{S}(Q)| = 1$. Secondary, weaker:
    **useful** iff $G(Q) \le 2$.
12. **Bracket width.** $w(Q) = \max_S \ln(B(S)/L(S))$ over enumerated sub-joins.
13. **Estimators.** The artifact's sub-query estimates for `bayescard`,
    `factorjoin`, `duckdb`, `postgres` (floored at 1). For each estimator $E$:
    `pick_free` = $\arg\min_p C(p, E)$ over all plans; `pick_pruned` =
    $\arg\min_{p \in \mathcal{S}(Q)} C(p, E)$. Regret of each is scored with
    $y$. A sub-query missing from an estimator's file drops that estimator
    for that query (counted).

## Query set (fixed)

All STATS-CEB queries (146) that parse fully (`parse_job_sql`), have
$3 \le n \le 7$ aliases, and have artifact truth for **every** connected
sub-join. Queries with $n \le 2$ have one plan and are excluded (trivially
certified; reported as a count, not in any rate). Exclusions are reported with
denominators.

Budgets: 30 s per sub-join for the sampling certificate (censor → $L=1$), 30 s
per sub-join for the `count_join` cross-check (censor → no cross-check).
No timing of any kind is a result of this experiment: the machine is shared,
and the paper's cost axis already covers the sampling certificate's cost.

## Outputs (fixed)

`results/plan_certificate/plan_certificate.json` with, per query and bracket:
`n_plans`, `n_survivors`, `pruning_rate`, `G`, `G_all`, `certified`, `useful`,
`width`, `minimax_regret`, `minimax_bound`, and per estimator `regret_free`,
`regret_pruned`, `picked_survivor` (whether `pick_free` is a survivor);
plus per-bracket violation and censoring counts; plus the query list and all
exclusions.

Aggregates per bracket over unflagged queries: certified share, useful share,
mean pruning rate, median and p95 of $G$, median minimax regret, and for each
estimator the share of queries where `pick_free` was pruned (i.e. the estimator
chose a provably sub-optimal plan) and the median regret of `pick_free` vs
`pick_pruned`.

## Statistics (fixed)

Each query is one unit. Percentile bootstrap over queries, $B = 10{,}000$,
`seed=0`, 95% intervals, via the existing `bootstrap_ci.py` conventions.
Zero-event shares get exact one-sided Clopper–Pearson 95% upper bounds.
Paired comparisons between brackets (e.g. `lpbound×sample0.25` vs
`lpbound×one`) use the same resampled query indices for both arms.

## Checks that must pass (a failure is a code bug, not a finding)

On every unflagged query and every bracket:

* $\text{regret}(p) \le G(Q)$ for every survivor $p$ (checked over all
  survivors, not only the picked ones) — **0 violations required**;
* $\text{regret}(\text{pick\_pruned}_E) \le G(Q)$ for every estimator;
* $\text{regret}(\text{any } p) \le G_{\text{all}}(Q)$;
* $\text{regret}(p^\star) \le \min C_{\text{hi}} / \min C_{\text{lo}}$;
* the true-optimal plan is never pruned.

Any violation halts the write-out and is debugged before any number is read.

## Pre-registered decision rule

Primary bracket: **`lpbound × sample0.25`** (the tightest bracket we can
supply today).

| Outcome | Rule |
|---|---|
| Plan certificates are **non-trivial** today | certified-share 95% CI lower bound $> 10\%$ |
| Plan certificates are **vacuous** today | otherwise — reported as such in RQ6, with the curve across the 9 brackets showing what tightness would be needed |
| Dominance pruning is **non-trivial** | mean pruning-rate 95% CI lower bound $> 50\%$ on the primary bracket |
| The lower certificate **matters at plan level** | paired difference in certified share, `lpbound×sample0.25` minus `lpbound×one`, has 95% CI excluding 0 |

The honest prior, written before running: certificates rare ($< 10\%$),
$G$ mostly vacuous ($> 10$), pruning non-trivial. A vacuous result is reported
as the answer, not softened.

## What this does and does not establish

Establishes: cost-model-level statements (Proposition 2 candidate) that are
sound given sound brackets and a monotone cost model; a first measurement of
whether today's sound-interval producers supply what robust-QO consumers
(plan bouquets, minmax-regret, optimality ranges) assume.

Does not establish: anything about executed runtime (engine costs are not
monotone in cardinalities, and the paper's own runtime section shows the
cost-model/runtime gap); anything about bushy plans or physical operators.

## Pre-run amendments (2026-09-03, recorded after inspecting artifact files only, before any plan-level number)

1. **Artifact estimator coverage.** The artifact's `bayescard` and `factorjoin`
   sub-query files (1,596 rows each) do not align with the truth file's
   (`QueryID`, `Tables`) keying — only 5/140 eligible queries are fully
   covered and the files contain keys absent from the truth file. Both
   estimators are therefore dropped *before running*; the estimator set is
   `duckdb` and `postgres` (2,577/2,577 sub-joins covered each).
2. **Floating-point upper bounds.** The artifact's `lpbound` column is below
   the integer truth on 51/2,577 sub-joins by at most $4 \times 10^{-15}$
   relative (e.g. `79850.99999999975` vs `79851`). A violation at or below
   $10^{-9}$ relative is treated as rounding and the bound is ceiled to the next
   integer; anything larger is a real violation and flags the query for that
   bracket. `safebound` is genuinely below truth on 7 sub-joins (up to 2.5\%),
   which will flag those queries for the two `safebound` brackets — reported.
3. **Flagging granularity.** A query is flagged *per bracket* (the bracket
   whose certificate violates), not for all nine, so an unsound `safebound`
   value does not remove the query from the `lpbound` rows.
4. **Eligibility count.** 140/146 queries are eligible (6 have $n \le 2$;
   0 unparsed; 0 missing sub-join truth). Plans per query: 4–312, 6,344 total.
5. **Intervals.** Computed inside `plan_certificate.py` by importing
   `bootstrap_ci.cluster_bootstrap` / `zero_event_upper_bound` (same
   $B = 10{,}000$, `seed=0`, query as cluster) rather than by adding jobs to
   `bootstrap_ci.py`'s headline list; identical procedure.

## Implementation log (2026-09-03, after freeze)

- `experiments/scripts/plan_certificate.py` implemented against the
  definitions above; `tests/unit/test_plan_certificate.py` covers the bracket
  evaluation, rounding rule, artifact keying, flagging and the decision rule on
  synthetic fixtures. ruff / mypy `--strict` clean.
- **Smoke test seen before the full run.** An 8-query `--limit` smoke run
  (queries 1–8, of which only q7 and q8 are eligible; 20 s budgets; 200
  bootstrap replicates) was executed to check the pipeline end to end. Its
  output was inspected (both queries: 4/4 survivors on every bracket, no
  violations, truth cross-check agreeing 3/3 sub-joins each) and then
  discarded. This is disclosed because the decision rule was fixed before it
  and nothing was changed after it.
- Full run launched 2026-09-03 14:17 KST with `PYTHONHASHSEED=0`, default
  budgets (30 s), $B = 10{,}000$, seed 0; the machine is shared and loaded, so
  wall time is not reported and no timing claim is made.
