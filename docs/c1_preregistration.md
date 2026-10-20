# C1 Pre-registration — Certificate Utility Predictability

**Frozen: 2026-07-11.** These definitions and thresholds are registered *before*
the W2 pilot data is collected and before any abstract/contribution rewrite
(TOP_PLAN Phase 2 / 지금 당장 할 일 #3). They may not be changed after looking
at results; any deviation must be reported in the paper as a deviation.

## Claim under test (C1, main claim)

> Certificate utility is predictable from three per-query observables:
> **error direction**, **normalized tightness**, and **relative certificate
> cost**. A simple model on these three variables predicts whether a
> certificate intervention helps, on held-out workloads.

## Variable definitions (fixed)

For query $Q$, proposal estimate $e(Q)$, true cardinality $y(Q)$, certificate
bound $B(Q)$ (upper) or $L(Q)$ (lower), and threshold $\tau$ (default 2.0):

1. **direction** $= \mathrm{sign}(\log e(Q) - \log y(Q))$ — over (+1) / under (−1).
2. **normalized tightness** $= \log(B(Q)/y(Q))$ for upper; $\log(y(Q)/L(Q))$
   for lower. (0 = exactly tight; larger = more vacuous.)
   *Deployment-time proxy* (no $y$): $\log(B(Q)/e(Q))$, resp. $\log(e(Q)/L(Q))$ —
   both variants are computed; the *proxy* is what the predictor may use.
3. **relative cost** $=$ certificate wall-clock / full exact-count wall-clock
   (as instrumented in `run_job_real.py::cost_ratio`).

**intervention_help label (binary):** an intervention occurs when the guard
fires (upper: $e > \tau B$; lower: $e < \hat{L}/\tau$). It is *helpful* iff the
post-intervention q-error is strictly smaller than the proposal's q-error:
$q(\text{final}, y) < q(e, y)$. Queries where no guard fires are labeled
not-helpful (the certificate bought nothing on that query).

## Model class (fixed)

Logistic regression on the three variables (plus intercept), no interaction
terms, no additional features. Train on one dataset, evaluate on the held-out
other(s) (leave-one-dataset-out).

## Success thresholds (fixed; from TOP_PLAN Phase 2)

| Outcome | Threshold |
|---|---|
| C1 supported | pooled held-out AUROC **≥ 0.80** AND per-dataset AUROC **≥ 0.70** |
| C1 weakened (per-dataset taxonomy only) | pooled ≥ 0.70 but a dataset < 0.70 |
| C1 rejected → failure-atlas PIVOT (TOP_PLAN W4 gate) | held-out AUROC < 0.70 |
| C4 method claim supported | ≥ 20% p95 plan-regret reduction at ≤ 10% median overhead on ≥ 1 real benchmark |
| C4 rejected | otherwise → sample-LB demoted to expensive reference |

## Datasets (planned)

JOB (real IMDb; 111/113 parsed, coverage reported), STATS-CEB (W5–6 gate),
job_light / job_medium fixtures (secondary, labeled as fixtures). CardBench
subset only if budget allows (TOP_PLAN 3.5).

## Statistics (fixed)

Per TOP_PLAN 3.4: 10 seeds for learned components; deterministic baselines get
query-level bootstrap (10,000 resamples, 95% CI); paired comparisons use
Wilcoxon signed-rank on per-query log q-error / plan regret with Holm
correction; all exclusions reported with denominators.
