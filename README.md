# Sound Cardinality Bounds in Practice: Guards, Estimates, and Plan Certificates — code and data

Artifact for the ICDE 2027 Experiment, Analysis & Benchmark submission of the same title (Suan Lee,
Semyung University). The paper itself will be added after review.

## Start here
| File | What it is |
|---|---|
| `docs/icde_2027/REPRODUCE.md` | every number → command → output file |
| `docs/icde_2027/CLAIMS_EVIDENCE.md` | every claim → value, interval, status (post-hoc items marked) |
| `docs/AUDIT_REPAIR_LOG.md`, `results/SUPERSEDED.md` | defects in our own pipeline and the results they invalidated |
| `docs/c4_preregistration.md` | the pre-registered plan-certificate study whose two rules the paper reports as unattainable |
| `docs/icde_2027/PREDICTIONS.md`, `PREDICTIONS_RESULTS.md` | predictions frozen before the post-hoc mechanism analyses, and how each fared |
| `docs/icde_2027/PREREGISTRATION_JOBRANGE*.md` | the held-out JOB-range pre-registration and its two addenda (each with a `.sha256` file); results in `results/heldout_jobrange/confirmatory_results.json` |

## Checks without any data download
```bash
pip install -r requirements.txt
PYTHONHASHSEED=0 python -m experiments.scripts.audit_icde_numbers   # needs docs/icde_2027/paper.tex (added after review)
pytest tests/
```

## Which results the paper uses
- Execution (both execution tables, the replication table, the per-query figure): `results/pg_injection/v3/` — one run per benchmark covering every
  source and policy. `results/pg_injection/v2/` is superseded (see `results/SUPERSEDED.md`). The older `results/pg_injection/inject_plans*.jsonl` and `injection_summary*.json`
  are superseded (different load, τ = 2) and kept only for provenance.
- Plan certificates: `results/plan_certificate/plan_certificate_joint_pg_r025.json` (PostgreSQL's
  in-engine samples); the registered study used our own sampler (`plan_certificate.json`).
- Tables with many cells are generated, not typed: `python -m experiments.scripts.render_icde_tables`
  prints every row and the audit requires each row verbatim in `paper.tex`.
- The r = 0.25 sample run shared the host with heavy jobs: its counts are used everywhere, its timings
  only where the recorded load was ≤ 60 (`results/pg_injection/v2/sampling_rate.json`).

## Re-running
PostgreSQL 13.1 with the STATS-CEB benchmark's patch, prepared by
`experiments/pg_injection/prepare_build.sh` and built from `experiments/pg_injection/docker/Dockerfile`; STATS and IMDb
data are public; estimates and truth come from the LpBound artifact (commit pinned in `docs/icde_2027/REPRODUCE.md`).
Wall-clock figures were measured on a shared host with a load gate; the per-query JSON records the load.

Provenance logs and the frozen pre-registration refer to earlier internal drafts of this work; none of
them was submitted to any venue.

## Code this paper does not use
The repository grew out of an earlier project (a learned CEG configuration selector). The ICDE paper's
results come only from `experiments/pg_injection/`, the scripts named in `docs/icde_2027/REPRODUCE.md`,
and the `src/` modules they import (`src/core/ceg.py` for the max-degree bound, `src/utils/`). The GNN and
training code (`src/models/`, `src/training/`), the CEG configuration estimators
(`src/estimators/{safe_estimator,alpha_ceg,baseline_ceg}.py`) and most other scripts in
`experiments/scripts/` belong to that earlier work; they are kept, with their tests, for provenance.

## Third-party components and licences
None of the following is redistributed, except the held-out JOB-range sub-plan list
(`results/heldout_jobrange/jobrange_heldout_subqueries.csv`), a 200-query subset of the LpBound
artifact's, under that artifact's MIT licence (`LICENSES/LpBound-MIT.txt`). Each component is
referenced by version or commit and keeps its own licence. Our own code is under the MIT licence
(`LICENSE`).
- **LpBound artifact** (`github.com/fdbresearch/LpBound`, commit `ee01ad5`):
  MIT licence. We use its published estimates, bounds and true cardinalities.
- **STATS-CEB benchmark** (`github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark`, commit `670cb8d`): the repository has no licence
  file. We reference its PostgreSQL patch, workloads and data rather than redistributing them;
  `experiments/pg_injection/docker/Dockerfile` is adapted from its Dockerfile (changes: `-O2`, archived
  Debian sources, pinned `pg_hint_plan` commit).
- **PostgreSQL 13.1**: PostgreSQL Licence; downloaded and checksum-verified by
  `experiments/pg_injection/prepare_build.sh`.
- **pg_hint_plan** (fork `Nathaniel-Han/pg_hint_plan`, commit `a6052904`): upstream copyright NTT,
  BSD-style licence; cloned at build time.
- **Data**: STATS (Stack Exchange data, via relational.fit.cvut.cz and the benchmark's simplified
  version) and IMDb (JOB-light) are obtained from their public sources under their own terms.

## Not included
Model checkpoints, third-party data, the PostgreSQL source tree, and internal planning notes.
