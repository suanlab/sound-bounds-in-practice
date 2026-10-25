# ICDE 2027 (EAB) — claims and their evidence

Every number below is re-derived from its source and found in `paper.tex` by
`PYTHONHASHSEED=0 python -m experiments.scripts.audit_icde_numbers` (sentence-anchored checks).
Intervals: 95% query-clustered bootstrap, 10,000 replicates, seed 0. **Post-hoc** = designed after
we saw a related result; nothing in this paper is pre-registered except the failed plan-certificate
study, whose two rules are reported as unattainable.

## Ground truth and instruments
| Claim | Value | Source | Status |
|---|---|---|---|
| PostgreSQL exact counts equal the artifact's truth | 2,510 / 2,510 STATS-CEB sub-plans (67 exact counts > 120 s) | `results/pg_injection/subset_counts_summary.json` | Supported |
| In-engine sampled lower bound is sound | 0 violations, 2,550 sampled sub-plans (27 censored) | same | Supported (by construction, checked) |
| Truth injection reaches every join of the chosen plan | 146 / 146 queries (both runs) | `inject_plans{,_native}.jsonl` | Supported |
| ~~Injecting PostgreSQL's own estimates reproduces its native plan~~ | ~~only 90 / 146; injected plans 1.12× [1.02, 1.24] faster~~ | `injection_summary_native.json` (superseded, stand-alone estimates) | **Superseded** by the v3 in-context row below (3/146, 1.00×); not in the paper |

## Cardinality level (STATS-CEB, 870 pairs / 145 queries; `cardinality_baselines_pg.json`)
| Claim | Value | Status |
|---|---|---|
| Underestimate share differs by estimator | 5% (NeuroCard) … 97% (DuckDB) | Measured; the "0% of underestimates caught" is by construction, not a finding |
| Upper guard helps only overestimators | NeuroCard p95 2.6×10⁸ → 1,135; DuckDB unchanged 304,835 | Supported |
| Floor helps only underestimators | DuckDB 304,835 → 14.7 | Supported |
| Best guard (both) | p95 288 [96, 648] | Supported |
| Sample as estimate beats every guard | L̂ p95 1.72 [1.37, 4.23]; sound L 16.7 | Supported; two queries where the sample misses all matching tuples (q-error 5,091 and 224) |
| LpBound as the estimate vs guarding with it | median 29.1 vs 9.00 (worse); p95 2,862 vs 4,733 upper guard alone (better) and vs 288 both guards (10× worse) | Corrected 2026-10-16 (the earlier "worse than guarding" was false at p95 against the upper guard alone) |
| τ = 1 vs τ = 2 | identical tails | Supported |
| A tighter sound bound reaches more estimators (146 queries, τ=1, `tightness_all_maxdegfix_t1.json`) | max-degree 1,423× / 8.2%; SafeBound 29.9× / 21.5%; LpBound 30.8× / 22.3%; per query LpBound is a median 24× tighter; max-degree fires almost only on NeuroCard (68 of 72), LpBound also on PostgreSQL 33, FLAT 27, FactorJoin 34 | Supported; max-degree with NULL-free statistics; the pooled "46× → 2.7×" ratio of medians is no longer used |

## In-engine cost (`subset_counts_summary.json`, `sampling_overhead.json`)
| Claim | Value | Status |
|---|---|---|
| Sampled count / exact count | median 0.33 [0.31, 0.36] over 2,509 sub-plans | Supported (shared host; ratio measured back to back) |
| Ratio falls with join cost | 0.48 (<100 ms), 0.28, 0.12, 0.10 (>10 s) | Supported |
| ~~Sampling all sub-plans of a query vs executing it~~ | ~~median 2.82× [2.03, 4.50]; > 1 on 73% of 127 queries (cross-run)~~ | (v2 cross-run) | **Superseded** by the single-run v3 measurement (2.68× [1.87, 3.84], 73% of 127); not in the paper |
| Earlier "0.66×, never cheap" | measured in our Python counter | **Retracted** for an engine |

## Plan certificates — registered study and Python-sampler analysis (140 queries; `plan_certificate.json`, `plan_certificate_joint_r025.json`)
The Python-sampler joint-certificate rows below are **superseded in the paper** by the in-engine rows in the next section; kept for provenance.
| Claim | Value | Status |
|---|---|---|
| Pre-registered study (decoupled G) | 0 / 140 on both registered rules | **Both rules unattainable by construction** (C_out first-two symmetry; G ≳ 1/r with a sampled floor, min G·r = 0.98) — disclosed |
| Joint certificate, LpBound × Python sample (superseded) | 54.3% [46.4, 62.9] ≤ 2×; median 1.51; prunes 33.4% vs 3.6% | Post-hoc; paper uses in-engine 55.0% |
| Without the shared final result | 27.9% | Post-hoc |
| Without a lower bound | 0% on every upper bound | Post-hoc |
| Best plan's true regret (all queries, superseded) | median 1.00, max 1.36 over all queries; ≤ 1.03 over certified queries | Paper reports the certified-only maximum |

## Certificates from PostgreSQL's in-engine samples (primary since 2026-10-15)
| Claim | Value | Status |
|---|---|---|
| Joint, LpBound × in-engine sample r = 0.25 (`plan_certificate_joint_pg_r025.json`) | 55.0% [46.4, 62.9]; median bound 1.47; pruning 33.0% vs 2.7% under G; certified plans' true regret ≤ 1.03; no final 27.9%; SafeBound 56.5%; max-deg 35.7% | Post-hoc; 43 r=0.25 samples timed out (L = 1, sound, conservative) |
| Same at r = 0.10 (`..._pg_r010.json`) | 40.7%; SafeBound 42.8%; max-deg 29.3% | Post-hoc |
| JOB-light, r = 0.10 (`plan_certificate_joint_joblight_pg_r010.json`, 67 queries) | LpBound 52.2% [40.3, 64.2]; SafeBound 37.3%; certified plans optimal; L≡1: 0%; no final 10.4% | Post-hoc; replicates the STATS pattern |
| Sampling rate (`sampling_rate.json`, 548 matched quiet-host sub-plans, 58 queries) | cost 0.35× → 0.56×; >1 s: 0.10× → 0.25×; L median q-err 10.04 → 3.98; L̂ p95 1.72 → 1.56 | r=0.25 timings used only at load ≤ 60 (2,004 rows excluded) |
| Registered study numbers (Python sampler) | 0/140 both rules; min G·r 0.98; commutation 5/140 (3.6% [0.7, 7.1]) | Now audited |

## Plan certificates in the engine (post-hoc)
| Claim | Value | Status |
|---|---|---|
| Selective (certify-then-stop) sampling, r = 0.10, v3 native times (`selective_sampling_v3.json`) | certifies the same 40.7% [32.9, 48.6] (identical set); median 29% of sub-joins sampled; gives up on 55%; cost 0.44× [0.36, 0.62] of native execution vs 0.88× [0.71, 1.32] for all 1,560 sub-joins; 0.32× on certified; certified true regret ≤ 1.01; 136 timed queries | Post-hoc, offline simulation, cross-run timings; denominator check (`selective_denominators.json`, post-hoc): the execution run times native plans 1.24× / 1.33× slower than the forced-order / in-run runs; with their times selective 0.53× / 0.59×, full 1.24× / 1.30× — selective stays < 1, full sampling does not |
| ~~Certified orders executed (`certified_plans_summary.json`)~~ | ~~0.28× native speed~~ | **Withdrawn 2026-10-16** (hint not obeyed). Replaced by the forced-order run: certified orders 1.07× [0.94, 1.23] vs native (see the execution section) |

## Note on SafeBound violations (2026-10-16)
The frozen plan-certificate protocol (`docs/c4_preregistration.md`, item 2) states that SafeBound is below the truth on
7 sub-joins. Every recount from the current files gives **3 sub-joins in 2 queries (42, 142)** beyond the 1e-9 rounding
tolerance — in both sub-join files and over all 2,577 sub-plans of the artifact — and 9 if rounding-level differences
(≤1e-12 relative, removed by the protocol's own rounding rule) are counted. The 7 could not be reproduced; the paper
uses 3 (SafeBound rows over 138 queries).

## Execution in PostgreSQL — v3 single run per benchmark, τ = 1 (`results/pg_injection/v3/`)
v2 (`results/pg_injection/v2/*_inject.jsonl` and summaries) is superseded: its "PostgreSQL (ours)" source used
stand-alone estimates. v3 uses PostgreSQL's in-context estimates for that source and re-measures every source.

| Claim | Value (v3) | Status |
|---|---|---|
| Protocol | one run per benchmark; uncensored plans timed 3×; 25 (STATS) / 11 (JOB-light) plans censored at 300 s; 0 crashes; median start load 4.9 / 2.9; none flagged | Measured (`*_protocol.json`) |
| Injection is neutral | own in-context estimates change 3/146 STATS plans, 0/70 JOB-light; 1.00× [0.98, 1.02] | Measured (was 90/146 and 1.12× with stand-alone estimates) |
| Headroom (STATS) | published PG 4.38×, DuckDB 1.70×, our PG 1.35× [1.23, 1.51] | Supported |
| Upper guard, published PG | 3.08× [2.35, 4.15], 66% faster / 3% slower; vs native 0.95× [0.88, 1.03] (both guards 1.06×) | Supported; guarded plans do not beat native |
| Our PG's own estimates | upper 1.02× (23/146 plans change), floor 1.05×, both 1.07×, upper + L̂ floor 1.18× | Supported: little recovered of a 1.35× headroom. Not because the errors are invisible — 404 of the 475 sub-plans with q-error > 10 lie outside the bracket (`v3/native_bracket.json`) — but because clamping to a wide bracket corrects too little (an earlier draft wrongly said the errors were not where a bound can see them) |
| DuckDB | upper 1.02×, floor 1.39× | Supported |
| Planning on L̂ (128 queries with a complete sample) | 3.57× / 1.69× / 1.30× vs headroom 3.59× / 1.70× / 1.30×; vs truth 0.99× | Supported; 18 excluded queries carry 17.83× headroom |
| Planning on LpBound vs truth (STATS) | 0.81× [0.76, 0.86] (44% slower); vs our own estimates 1.09× | Configuration-specific (artifact: 1.02×) |
| JOB-light | published PG floor 1.38× of 1.94× (vs native 0.79× [0.69, 0.91]); FactorJoin upper 1.18× of 1.34×; BayesCard 1.00×; DuckDB upper 1.10× / floor 1.08×; L̂ vs truth 1.00×; LpBound vs truth 0.87× | Supported |
| Sensitivity | ±5%/±20% band and 120/60 s caps keep every direction (`*_timing_sensitivity.json`) | Supported |
| Sampling all sub-plans vs execution, one run | median 2.68× [1.87, 3.84], > 1 on 73% of 127; 19 excluded (3 native, 16 sample timeouts); timeouts at cap 3.31× (76% of 143) | Supported (`sampling_overhead_inrun_summary.json`) |
| Replication (`v3/artifact_replication.json`) | headroom 1.28× vs 1.35× (STATS), 1.21× vs 1.12× (JOB-light); LpBound vs truth replicates on JOB-light only; published PG estimates 4.7× ours (20× on 2-rel) on STATS, 0.33× on JOB-light | Supported |
| Certified plans, forced orders (`v3/forced_orders_summary.json`) | forcing the native order 0.89× [0.81, 0.97]; C_out-optimal order 1.15× [1.02, 1.31]; certified 1.07× [0.94, 1.23] vs native (certified order = C_out-optimal order on 52/57, so the 1.01× vs C_out-optimal is a repeatability check, not reported); choice matters (21): 1.32× [1.10, 1.63], with sampling 0.34× [0.17, 0.57]; does not matter (36): 0.95× [0.80, 1.11]; forcing changes the physical plan on 50/118 (0.76× there, 1.01× elsewhere), C_out-optimal vs forced native 1.20× [1.08, 1.38]; with sampling 0.56× [0.42, 0.72] | Supported; replaces the withdrawn `Leading`-hint result |
| Sampling seeds 0–4 (`v3/seed_robustness.json`) | L̂ p95 1.35–2.05; best guard p95 250–288; L median 9.88–10.05; certified share 40.0–41.4% | Supported: no conclusion rests on one draw |

| JOB-light cardinality (`cardinality_baselines_pg_joblight.json`, 560 pairs) | under share 1%–97%; L̂ p95 1.22 vs best guard 19.0; NeuroCard 84% under (5% on STATS) | Supported |

## Withdrawn from earlier versions (see `results/SUPERSEDED.md`, `docs/AUDIT_REPAIR_LOG.md`)
Safe-CEG selector / 3×3 configuration results (estimator defect); "certificate is vacuous on cycles"
(misdiagnosed); executed DuckDB/SQLite aggregates of an earlier, unsubmitted draft (duplicate counting); every
max-degree number computed with NULL-counting statistics.

## Independent verification of v3 (2026-10-19)
Separate-lane verifier (`reviews/verifier_v3_2026-10-19/report.md`): ~100 numbers recomputed from raw JSONL, all match; 315/315 forced plans honour their order. Its 3 major findings (1.01× is a self-comparison; selective-sampling cost depends on the native-time reference; certified queries mostly choice-does-not-matter) and 11 minor ones were fixed in the text and are now audited (`summarize_forced_orders`, `selective_denominators`, extended `timing_sensitivity` / `protocol_summary`). One verifier statement corrected on recheck: the 19 high-load STATS-CEB queries are not consecutive (ids 31–48 and 52, one stretch of 22).

## Mechanism analyses (post-hoc; predictions frozen in PREDICTIONS.md)
| Claim | Number | Source | Status |
|---|---|---|---|
| Joint certificate: R(p) <= t iff B(A) - tL(D) <= (t-1)L(Sh) for all p' | 0 disagreements in 10,458 random plans (author) and 59,605 (reviewer) | `tests/unit/test_mechanisms.py`, review | Proposition (proved from the pair formula) |
| Certificate depends only on B/L | B/2.5 and L x 2.5 both certify 55.0% (= r 0.25 share) | `certificate_mechanisms.json` h2 | Identity; unit test |
| rho separates certified queries | AUC 0.93 [0.88, 0.98] STATS, 0.93 [0.87, 0.98] JOB-light; relations alone 0.87 / 0.64; JOB-light accuracy 82% at STATS threshold | `mechanism_intervals.json` h1 | Supported (post-hoc); rho does not explain the size trend on JOB-light |
| Oracle sampling price | online rule optimal on 77% of 57; end to end 0.51x [0.28, 0.78] where choice matters, faster on 14% | `price_of_certification.json`, `mechanism_intervals.json` h5 | Supported (post-hoc) |
| A-posteriori certification of PostgreSQL's plan | 86.1% [80.3, 91.2] / 83.9% [77.4, 89.8] (sample / none) of 137 STATS; JOB-light 100% / 97.0% of 67; without final 21.2% / 0% and 41.8% / 1.5% | `certificate_mechanisms.json` h4, `mechanism_intervals.json` h4 | Supported (post-hoc); driven by the final result |
| Guard funnel (both guards) | ours 91 -> 86 -> 32 changed -> 17 faster (5 less work); published PG 132 -> 131 -> 121 -> 100 faster | `v3/guard_funnel.json` | Supported (post-hoc) |
| Tightness sweep | ours: 69% [51, 87] of the true-size C_out gain at k = 2, ~40% for k = 4-32 (wide); published PG 73% [59, 84] at k = 32 | `mechanism_intervals.json` sweep | Supplement only (fragile for our estimates) |
| Probabilistic floor | 79-81% certified vs 40-41%; violations 8-13% (c from seed 0), 1.5-6.4% from other draws | `certificate_mechanisms.json` h6 | Supplement only; prediction failed |
| Executed tightness points (both synthetic guards, our estimates) | true sizes 1.34x [1.23, 1.50]; k=2 71% [59, 82] of it (1.23x); k=8 36% [20, 49] (1.11x); real bracket 23%; no plan hit the 2 GB temp cap | `v3/tightness_points_summary.json`, `v3/stats_summary.json` | Supported (post-hoc), in the paper |
| Witness floors (LIMIT k) vs sampled counts | median 0.96x [0.80, 1.04], 70 sub-joins | `v3/witness_floor_summary.json` | Supported (post-hoc); H7 prediction failed |
| Held-out JOB-range (pre-registered C1-C7) | C1 86.6%, C2 88.8%, C3 7.3% (inconclusive), C4 35.8%, C5 sound, C6 19.4% -> 5.8%, C7 not testable (1.03x) | `results/heldout_jobrange/confirmatory_results.json` | Confirmatory; 4 of 5 empirical predictions confirmed, 1 inconclusive |
| In-context estimates on a stock server (`joinrel_log`) | 2,577/2,577 STATS-CEB and 696/696 JOB-light equal to the 13.1 patch values; forced prefixes differ on 1,052 | `v3/joinrel_log_*_pg13.summary.json`, `v3/prefix_estimates_stats_pg13.summary.json` | Validated |
| PostgreSQL 16 replication (STATS-CEB execution) | headroom 1.42x; guards 1.07/1.06/1.13x; published upper 2.42x; LpBound vs truth 0.81x; k=2 81%, k=8 51% | `results/pg_injection/pg16/pg16_comparison.json` | Post-hoc replication; 4 of 6 frozen predictions held (R3, R6 failed narrowly) |
