# AUDIT Repair Log

Tracks remediation of `AUDIT_2026-06-20.md` (the June data-integrity audit; it was named `AUDIT.md` until 2026-10-16, when `AUDIT.md` became the comprehensive ICDE audit). Status: ✅ done · 🟡 partial · ⬜ pending.
Date: 2026-06-22.

## Blockers
- ✅ **B1 — no-op train/eval CLI.** Added real `main()` to `src/training/train.py`
  (synthetic workload → checkpoint + train log) and `src/training/evaluate.py`
  (checkpoint → eval results JSON). Rewrote `experiments/configs/train_default.yaml`
  and `eval_shift.yaml` to a schema the code consumes (removed nonexistent
  `data/imdb…` and unimplemented baselines). Verified end-to-end: both commands
  produce artifacts (`results/checkpoints/safe_ceg.pt`, `results/logs/train_log.json`,
  `results/eval/eval_results.json`).
- ✅ **B2 — estimator-derived synthetic labels (keystone).** Rewrote
  `src/training/workload.py` to GENERATE join-key data per family, DERIVE statistics
  from that data, and compute the true cardinality by EXECUTING the join
  (`src/utils/join_counting.count_join`, moved out of `experiments/`). Label
  generation no longer imports/calls `pessimistic_bound` or `BaselineCEGEstimator`
  (verified statically). Side effect: the max-degree bound is now valid on 100% of
  executed-join labels (the prior ~12% "violations" were a label artifact) — a
  *stronger*, honest result. Config diversity across families is preserved
  empirically.
- ✅ **B3 — identical distribution-shift workloads.** `run_distribution_shift.py`:
  differentiated `cyclic_large`/`complex`/`synthetic_cycle_C` by relation count and
  added a deterministic per-scenario seed offset (`NAME_SEED_OFFSET`). The three
  scenarios now draw distinct query streams. (Cycle generation is costlier under the
  data-grounded generator; cycle domain factor tuned to keep ≤7-relation cycles
  tractable.)
- ✅ **B4 — multiple papers / stale drafts.** Archived all non-canonical drafts to
  `docs/archive/` (CIDR `paper.tex`+`.bak*`, `icde_2027/`, `paper_draft.md`, Korean
  draft, 51 files). Canonical submission = `docs/vldb_2027_vision/paper.tex`.
  (Canonical result manifest: 🟡 pending, see below.)
- ✅ **B5 — geometric "certificate" violates / unverified.** `run_certificate_upgrade.py`
  now measures and reports `molp_bound_violation_rate` and `geo_bound_violation_rate`
  vs the true cardinality. On the corrected (executed-join) labels both are **0%**
  (the audit's 37/240 was an old-label artifact). Removed the false "valid upper
  bound under log-convexity" claims from `src/core/ceg.py`; the geo bound is now
  documented as a heuristic (not a sound certificate), with LpBound as the sound
  upgrade path.

## Critical
- 🟡 **C1 — paper fallback semantics vs code (bidirectional).** Canonical VLDB paper
  already describes the upper gate + a lower-certificate prototype (§5.9). Code
  `SafeCEGEstimator` has both upper and (soft) lower fallback. Remaining: a sentence
  in the canonical Method section making the bidirectional behavior explicit. (ICDE
  draft with the conflicting statement is archived.)
- ✅ **C2 — learned-baseline normalization leakage.** `run_learned_baseline.py` now
  splits FIRST and fits normalization on the train partition only.
- ✅ **C3 — config lists unimplemented external baselines.** Removed
  GNCE/LMKG/FactorJoin from `eval_shift.yaml`; documented they are not implemented.
- 🟡 **C4 — "real data"/JOB overclaim.** JOB-medium is a synthetic IMDb-like fixture;
  the canonical paper already frames it as such and the real-JOB pipeline
  (`prepare_real_job.py`, filter support) is local-run. Remaining: a one-line
  caption/label audit to ensure "synthetic fixture" wording everywhere.
- ✅ **C5 — ICDE robustness theorem indefensible.** The ICDE draft is archived; the
  canonical VLDB paper does not contain that theorem.

## Major
- ⬜ **Ablation fairness** (`run_ablation.py`): per-method objectives/wrappers differ;
  oracle wrapped in SafeCEG. Needs a shared loss/seed/budget harness and a pure-oracle
  block. (Blocks ablation regeneration.)
- ✅ **eval `model.eval()`/`torch.no_grad()`** added to `evaluate_model` (restores
  caller state).
- ⬜ **Table exporter** (`export_paper_tables.py`) incompatible with nested result
  schema. Needs rewrite to regenerate paper tables.
- 🟡 **Feature-claim mismatch** (paper vs `features.py`): paper edit pending.
- ⬜ **Alpha-CEG weak** (`predict_alpha_weights` returns constant 0.5): demote claim
  or implement edge-level alpha.
- ⬜ **Dependency pinning / lockfile / Docker / Python 3.11**: add lockfile + env.
- ⬜ **Runtime protocol** (single first-query): measure many queries + percentiles.

## Moderate
- ✅ **CI on bounded metrics** clipped to [0,1] in `aggregate_multi_seed`.
- ✅ **Percentile** switched to linear-interpolation (matches `numpy.quantile`).
- ✅ **`safety_violation_rate` naming** clarified (threshold/policy metric) and a true
  `bound_validity_violation_rate` added.
- 🟡 **plan-cost surrogate**: rename/label as surrogate in reports (pending).
- 🟡 **seed/determinism docs**: `set_seeds` added + PYTHONHASHSEED note in `train.py`;
  broader determinism doc pending.
- ✅ **lint gate on `experiments/`**: `ruff check src tests experiments` passes.

## Minor
- ⬜ git provenance / result manifests with commit hash.
- 🟡 cache/backup artifacts: paper backups archived; add `.gitignore` for caches.
- 🟡 stale `--help`/docs: `run_real_data` help still JOB-light-centric.
- ⬜ "all tables/figures generated by scripts" manifest + README reproduce section.

## Cascade regeneration (after B2 changed all synthetic labels)
Regenerated with PYTHONHASHSEED=0:
- ✅ `bound_verification/bound_validity.json`: **3,600 q, 100% valid** (0 violations,
  all families), max true/bound ratio 0.38. (Was 88%; the 12% was a label artifact.)
- ✅ `certificate_upgrade/cert_upgrade.json`: MOLP fallback q-err mean 7.7e5 / med
  1.3e5; geo mean 3.1e3 / med 2.2e3; improvement 250×/61×; **geo violates true 0%**.
- ✅ `learned_baseline/learned_baseline.json`: learned regressor q-err mean 1.46 /
  med 1.21; CEG oracle 12.1 / 8.09; pessimistic 6.0e4. (Post leakage fix; the direct
  regressor now *beats* the oracle in-distribution — sharpens the "accuracy is not the
  differentiator, the guarantee is" framing.)
- ✅ `lower_certificate/lower_cert.json`: oracle median q-err 11.4 → floored 1.0
  (rate 0.25); sound lower bound valid 100%.
- ✅ `scalability/runtime.json`: new protocol (n=20/size, warmup, median/mean/p95,
  hardware metadata). 5-rel full-pipeline median **3.9 ms**, ML ~0.8 ms, 8-rel ~50 ms.
- ✅ `real_data/job_medium_5seed.json`: regenerated with the refactored harness.
- 🟡 `ablation/ablation.json`: regenerating with the fairness fix (1-seed smoke; the
  paper used 5 seeds — rerun at 5 seeds before final sync).
- ⬜ `shift/shift_results.json`: rerun with differentiated scenarios (slow: trains a
  model per scenario over data-grounded cycles).

## NARRATIVE SHIFTS the paper sync must absorb (flag to author)
- Bound validity **88% → 100%** on executed-join labels: the "conditional guarantee /
  honest 12% limitation" framing changes to "valid on 100% of tested queries" (the
  conditional theorem still holds; the precondition now empirically always holds).
- The direct learned regressor now **matches/beats the CEG oracle in-distribution**
  (median 1.21 vs 8.09): reinforce that Safe-CEG's contribution is the worst-case
  guarantee under shift, not in-distribution accuracy.
- Geometric bound: reframed from "tighter certificate" to "heuristic tighter bound,
  empirically 0% violation but not proven sound; LpBound is the sound upgrade".

## ⚠️ CRITICAL scientific finding from regenerated ablation (needs author decision)
With honest (executed-join) labels, on the synthetic in-distribution ablation:
`independence (AVI) == Safe-CEG-GNN == Safe-CEG-MLP == oracle == 9.998 mean / 7.68 median`
(exactly equal); only `pessimistic = 20,016`. So **Safe-CEG's synthetic *accuracy*
advantage over AVI (old Table 3: 2.60 vs 148, "50×") was itself a label artifact** —
the old estimator-derived labels were unfavorable to AVI. Config selection still
matters (oracle q-err ~6 vs worst-config ~5.7e6), but on *independent* synthetic data
the trivial AVI baseline already recovers the best estimate. Root cause: the new
generator samples join keys independently, so the independence assumption holds and
the optimistic configs collapse to AVI. Real data (job_medium) has correlations, so
config selection there is non-trivial.

**RESOLVED — implemented (B): correlated generator.** Added `_PROFILE_CORRELATION`
(path/star 0.4, cycle 0.85): within each relation a fraction of rows tie later join
keys to the first, breaking per-edge independence (cycle-closing correlation is the
canonical violation). Labels remain executed-join true cardinalities. Effect (honest,
no circularity): on cyclic queries AVI fails catastrophically (q-err ~1.9e4–2.3e5) while
the oracle CEG config recovers near-true (~1.3), so Safe-CEG ≈ oracle decisively beats
AVI on the mixed workload (in-distribution: Safe-CEG/oracle **6.24** vs AVI **7,294** vs
pessimistic 372). Acyclic path/star keep AVI ≈ oracle — faithful to the CEG literature
(independence holds on acyclic joins; breaks on cycles). The selector now adds genuine
value and GNN vs MLP diverge under shift (10.6 vs 2144 on cross-workload). Bound remains
100% valid; optimistic configs now overestimate, so the upper certificate/fallback has
teeth. **All synthetic results re-regenerated a third time with this generator (final).**

## Further findings from the correlated-generator regen (final)
- **Geo bound is unsound:** on correlated cycle data the geometric-mean bound violates
  the true cardinality **68.8%** of the time (MOLP 0%). The "certificate upgrade"
  headline (Table 7 / abstract) must be demoted to a CAUTIONARY result: a naive
  tightening is unsound and is NOT a certificate; sound tightening (LpBound) is the path.
- **Learned regressor still beats the CEG oracle** (1.21 vs 6.23 median): keep the
  "accuracy is not the differentiator, the sound guarantee is" framing.
- **Feature-claim mismatch (Major) is MOOT** for the canonical paper — it makes no
  detailed feature claims (that was the archived CIDR `docs/paper.tex`).

## ✅ Canonical paper-table re-sync — DONE
`docs/vldb_2027_vision/paper.tex` fully re-synced to the correlated-generator results
(compiles, 7pp, 0 undefined; stale-number sweep clean). Changes absorbed:
- **Table 1** bound validity 88% → **100%** (0 violations, max ratio 0.38); §4 theorems
  + limitations reframed (guarantee unconditional here; conditional form still matters).
- **Tables 3–4** ablation: Safe-CEG/MLP **6.47** = oracle, beats AVI **6,423** &
  pessimistic **383** (driven by cyclic queries); MLP > GNN; dropped redundant BV split.
- **Table 2** shift: relabeled regimes **selector-recovery** (path/star/size, q-err 11–32,
  acc 39–73%) vs **vacuous-certificate** (cyclic, q-err 1e5–1e7, fallback 0%).
- **Table 7** geo demoted to a CAUTIONARY result (tighter but **68.8% unsound**).
- **Table 6** runtime (new median/p95 protocol, 5-rel 3.8 ms); **§6** learned regressor
  1.21 vs oracle 6.23; **§5.6** τ-sweep reframed (vacuous ⇒ τ irrelevant); abstract +
  intro contributions rewritten; **C1** bidirectional fallback noted in method.
- JOB-medium relabeled **synthetic IMDb fixture** (C4); Table 5 numbers unchanged.

## ✅ Remaining (minor) — DONE
- ✅ **Figure 1 (config diversity) regenerated from current data.** Added
  `experiments/scripts/generate_config_diversity.py` (reproducible: same query set as
  Table 1, executed-join labels, `PYTHONHASHSEED=0`). Writes
  `results/config_diversity/config_diversity.json` + `figures/config_diversity.{pdf,png}`.
  Heatmap now matches the caption (path/star→`max_hop-min`, star also `min_hop-min`,
  cycle→`all_hops-avg` at 0.67). Added to REPRODUCE.md.
- ✅ **Table exporter rewritten.** `export_paper_tables.py` now regenerates
  `docs/paper_results_snapshot.md` from the *canonical* `results/*.json` (the same files
  the paper tables use), with an AUTO-GENERATED / non-canonical banner — eliminating the
  Finding-4 drift hazard (old version read stale March JSONs). CLAUDE.md updated to name
  paper.tex + REPRODUCE.md as canonical and the snapshot as an auto-generated mirror.
- ✅ **`tau_sweep` rerun at 5 seeds.** Now `201{,}525 \pm 25{,}972` invariant across τ∈[1,10],
  fallback 0% throughout — a stronger demonstration of the vacuous-certificate invariance.
  Paper §5.6 number updated; parity with the 5-seed Tables 2–4.
Tables 1/2/3/4/5/6/7 + §5.9 + §6 + abstract numbers must be re-synced to the
regenerated JSON; several headline numbers and two narrative threads change
(above). Best done after ablation (5-seed) and shift are regenerated.

## Verification state
`ruff check src tests experiments` ✅ · `mypy src` ✅ (28 files) · `pytest` ✅ (32) ·
official train/eval CLIs produce artifacts ✅.

## 2026-10-04 — optimistic-CEG edge weight dropped intra-edge joins (fixed)

- **Found by:** internal strict review panel 4 (reviewer s1), on the EDBT-frozen submission.
- **Defect:** `_optimistic_edge_weight` used `query.boundary_joins(source, target)` only, ignoring joins
  among the relations an edge adds. Multi-relation edges over-estimated (reviewer's case: 1e7 vs AVI 1e5).
- **Fix:** multiply every join in `induced_joins(target) - induced_joins(source)`.
- **Test:** `test_every_config_equals_independence_on_a_path_query`,
  `..._on_cyclic_queries`, `test_optimistic_estimates_never_drop_joins_above_the_bound`; the old test
  `test_baseline_estimator_distinguishes_path_length_choices` encoded the defect and was removed.
- **Impact:** all nine configurations collapse to AVI; selector, config-diversity and synthetic Safe-CEG
  results are invalid (listed in `results/SUPERSEDED.md`). The ICDE 2027 paper is rescoped to existing
  methods only (decision 2026-10-04).

## 2026-10-04 — NULL join keys counted in max-degree statistics (fixed)

- **Found by:** internal strict review panel 4 (reviewer s1).
- **Defect:** `build_job_query` computed `distinct_count` / `max_frequency` with the empty string (SQL NULL)
  as a value. On mostly-NULL keys (STATS `votes.UserId`, 89% NULL) the NULL group became the max degree:
  the max-degree bound stayed sound but was needlessly loose. `count_join` already dropped NULL keys.
- **Fix:** NULL (`""` or `None`) excluded from the key statistics. Test:
  `test_null_join_keys_do_not_enter_degree_statistics`.
- **Impact:** every *real-data* max-degree number (tightness ablation's max-degree arm, plan-certificate
  and plan-guard max-degree brackets/gates, the max-degree JOB q-error) must be recomputed before reuse.
  LpBound/SafeBound values (from the artifact) and the sampling bound are unaffected.
