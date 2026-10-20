# Result files that no longer feed any paper number

Kept on disk for provenance. Nothing in the ICDE 2027 paper (`docs/icde_2027/paper.tex`)
reads them: `experiments.scripts.audit_icde_numbers` fails if one of its sources is listed
here. Canonical files are the ones named in `docs/icde_2027/REPRODUCE.md`. (The table below
was first compiled on 2026-09-07 for the earlier EDBT draft; later sections are dated.)

| File | Superseded by | Why |
|---|---|---|
| `plan_regret/plan_regret.json` (106 queries, ≤5 aliases) | `plan_regret/plan_regret_a6.json` (120 queries, ≤6 aliases, 1,080 pairs) | a6 is the canonical §RQ6 guard run; `tab:planregret` built from the 106-query file was removed on 2026-09-03 |
| the previous `bootstrap/bootstrap_ci.json` (overwritten during the panel-2 fixes, 2026-09) | `bootstrap/bootstrap_ci.json` (regenerated) | earlier file bootstrapped the 106-query plan-regret file and counted 448 zero-event trials; both were stale relative to the paper |
| `tau_sweep/tau_5seed.json`, `tau_sweep/tau_sweep_mixed.json`, `tau_sweep_cross_dataset/tau_sweep.json` | `tau_sweep/tau_sweep.json` | earlier prototype sweeps (different workload, seed set, or a mixed-direction variant); the paper quotes only the 5-seed cyclic sweep |
| `ablation_quick/`, `ablation_test/` | `ablation/ablation.json` | smoke runs with fewer seeds/queries used while developing `run_ablation` |
| `real_data/job_light_results.json` | `utility_table/real_joblight*.csv` (real JOB-light from the LpBound artifact) | produced on the **synthetic** `data/job_light` fixture; the name collides with the real benchmark and the paper no longer uses the fixture result |
| `real_job/c3_cost_scale.json` | `job_full/job_full.json` + `censoring_bounds.json` | the C3 pilot on a 2k–50k-row subset; still the source of the `findings.seed_multiplier` note that cost is linear in sampling draws, which the paper now states explicitly |
| `eval/eval_results.json`, `synthetic/*.json`, `synthetic_safe_ceg_*.json` | `shift/shift_results.json`, `ablation/ablation.json` | early synthetic evaluation outputs from before the family-aware workload generator |
| `learned_baseline/learned_baseline.json` | — | computed, never cited in the EDBT paper (D-W4); kept reproducible, not a paper number |
| `checkpoints/` | — | model checkpoints are never committed (`*.pt`, `*.ckpt`); directory is local scratch |
| `utility_table/real_{joblight,stats}_tau{1,4,8}.csv` | — (not superseded) | these **are** consumed by `intervention_tally` for the nested-τ count; listed here only so they are not mistaken for duplicates of the τ=2 tables |
| `config_diversity/config_diversity.json` (+ `figures/config_diversity.*`) | — (still read) | the heatmap figure left the paper on 2026-09-07 for page budget; §Setup still quotes the 67% cycle share from this JSON |

**Naming note (D-W12).** `data/job_light` and `data/job_medium` are synthetic
IMDb-like fixtures; the real benchmarks are `utility_table/real_joblight*.csv`
(JOB-light via the LpBound artifact) and `job_full/` (real IMDb, 113 templates).

## 2026-10-04 — optimistic-CEG edge-weight defect: every CEG_O-derived result is invalid

`src/core/ceg.py::_optimistic_edge_weight` multiplied only the joins crossing from the source subset and
dropped the joins among relations added by one edge (found by internal review panel 4, reviewer s1;
confirmed and fixed 2026-10-04, regression tests in `tests/unit/test_ceg_baseline.py`). With the fix,
**all nine configurations return the independence (AVI) estimate**, so the configuration diversity was
produced entirely by the defect. Superseded (kept for provenance, never quote):

| Result | Why invalid |
|---|---|
| `shift/`, `ablation/`, `ablation_quick/`, `ablation_test/`, `checkpoints/` (GNN/MLP selector) | selector chooses among configurations that are identical once fixed |
| `config_diversity/` (+ `figures/config_diversity.*`) | the diversity is the defect |
| `tau_sweep/`, `tau_sweep_cross_dataset/`, `scalability/`, `synthetic/`, `synthetic_safe_ceg_*.json`, `eval/`, `real_data/`, `learned_baseline/` | built on CEG_O proposals |
| `certificate_upgrade/`, `bound_verification/`, `bracket/` (synthetic or CEG_O proposal arms) | proposal side uses CEG_O; bound side (CEG_M) unaffected but to be re-checked if reused |
| `plan_regret/plan_regret*.json`, the CEG arm of `plan_guard/plan_guard_bidirectional.json`, plan-runtime executions with CEG plans | the nine "CEG proposals" are defect-inflated estimators |

**Unaffected:** everything built only on the LpBound artifact's estimators and bounds plus our sampling
lower bound (`utility_table/real_*`, `utility_table/guard_policy*.json`, `utility_table/tightness.json`
apart from its max-degree arm, `job_full/`, `lower_certificate/`, the external-estimator arms of
`plan_guard/`, `plan_certificate/` brackets, which never used CEG_O). The max-degree bound (CEG_M) is
computed by a different code path and is not affected by this defect (its NULL-key statistics are a
separate open item).

## 2026-10-16 (ICDE panel 6)
- `results/pg_injection/certified_plans.jsonl`, `certified_plans_summary.json` — **withdrawn**. The
  `pg_hint_plan` `Leading` hint was reported as used but not followed by our patched PostgreSQL 13.1
  (disable-cost totals ~1e17, row estimates ~1.5e19; 53 of 57 executed plans did not follow the hinted
  order). The 0.28× / 0.20× / 0.66× "certified plans are slower" numbers are artefacts. Replaced by
  `results/pg_injection/v3/forced_orders.jsonl` (explicit JOINs + `join_collapse_limit = 1`, every
  executed plan checked against its order).
- `results/pg_injection/native_subquery_estimations.csv` (and the JOB-light copy) — superseded as the
  "PostgreSQL (ours)" source: read by a stand-alone `EXPLAIN` of each sub-plan, not inside the full
  query; injecting them reproduced the native plan on 90/146 STATS-CEB queries. Replaced by
  `results/pg_injection/v2/native_incontext_estimations{,_joblight}.csv` (143/146, 70/70). All
  `native_*` policies in `results/pg_injection/v2/` are superseded by `results/pg_injection/v3/`.

## 2026-10-19 (v3 re-measurement replaces v2 execution results)
- `results/pg_injection/v2/stats_inject.jsonl`, `results/pg_injection/v2/joblight_inject.jsonl`, `results/pg_injection/v2/stats_summary.json`, `results/pg_injection/v2/joblight_summary.json`, `results/pg_injection/v2/stats_protocol.json`, `results/pg_injection/v2/joblight_protocol.json` — superseded by the
  same files in `results/pg_injection/v3/`: v2's "PostgreSQL (ours)" source used stand-alone estimates
  (90/146 native plans reproduced); v3 uses in-context estimates (143/146) for all sources in one run.
- `results/pg_injection/v2/guarded_vs_native.json`, `results/pg_injection/v2/artifact_replication.json`, `results/pg_injection/v2/sampling_overhead.json` — superseded by
  `results/pg_injection/v3/guarded_vs_native.json`, `.../v3/artifact_replication.json` and the single-run
  `.../v3/sampling_overhead_inrun_summary.json` (v2's sampling cost divided times from two runs).
- `results/plan_certificate/selective_sampling_v2.json`, `results/plan_certificate/selective_sampling.json` — superseded by
  `results/plan_certificate/selective_sampling_v3.json` (v3 native execution times).
