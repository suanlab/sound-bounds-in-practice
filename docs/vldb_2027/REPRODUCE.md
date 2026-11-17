# Reproducing the PVLDB Vol. 20 (EA&B) paper

Paper: `docs/vldb_2027/paper.tex` (*Sound Cardinality Bounds in Practice: Guards, Estimates, and Plan
Certificates [Experiment, Analysis & Benchmark]*).

1. **Map from numbers to commands:** `docs/icde_2027/REPRODUCE.md`. The directory keeps the name of an
   earlier, never-submitted target because hash files record its paths. That file lists every result,
   the command that produces it and the output file, including the revision analyses of 2026-11-09.
2. **Claims and their status:** `docs/icde_2027/CLAIMS_EVIDENCE.md`.
3. **Check without a database** (minutes, on the committed results):
   ```bash
   python -m venv .venv && source .venv/bin/activate && pip install -r requirements-lock.txt
   export PYTHONHASHSEED=0
   python -m pytest tests/ -q                                   # unit tests
   python -m experiments.scripts.audit_icde_numbers             # every headline number vs results/
   python -m experiments.scripts.review_checks --output /tmp/review_checks.json   # byte-identical to results/review/
   for f in docs/icde_2027/*.sha256 docs/vldb_2027/*.sha256; do sha256sum -c "$f"; done   # frozen registrations
   ```
   The audit re-derives each number from `results/`; it rejects sources listed in
   `results/SUPERSEDED.md` and does not re-run any measurement.
4. **Re-measure** (hours to days; needs Docker, the public STATS and IMDb data and the LpBound artifact):
   the PostgreSQL 13.1 harness (`experiments/pg_injection/`, build steps in the command map) and the
   PostgreSQL 16 harness (`experiments/pg_injection/docker16/`, `run_pipeline_pg16_jitoff.sh`).
   Timings depend on the host. Expect the counts, certificates and plan choices to match exactly given
   the same data, statistics and seeds. Expect execution times to match only in direction and rough
   size: the paper's intervals are over queries within one run, not over runs.
   The S3 re-measurement measures that run-to-run variation. It covers 61 STATS-CEB queries in six blocks:
   ```bash
   bash experiments/pg_injection/run_s3_blocks.sh       # PostgreSQL 13.1; about 1 h per block, blocks 3 h apart
   python -m experiments.scripts.s3_analysis            # registered verdicts -> results/pg_injection/s3/s3_summary.json
   python -m experiments.scripts.s3_report              # supplement table -> results/pg_injection/s3/S3_TABLE.md
   ```
5. **Pre-registrations and frozen predictions:** `docs/icde_2027/PREREGISTRATION_JOBRANGE*.md`,
   `docs/icde_2027/PREDICTIONS*.md`, `docs/vldb_2027/PREDICTIONS_PG16*.md` and `docs/vldb_2027/PREDICTIONS_S3.md`
   (with `S3_DESIGN.json`), each with its SHA-256 file. The S3 runner checks its hashes before it starts.
6. **Environment:** CPython 3.10.12, `requirements-lock.txt` (sha256 prefix `9ef8f5ad30c55ded`). PostgreSQL images are
   built by `experiments/pg_injection/docker/Dockerfile` (13.1, benchmark patch, pinned `pg_hint_plan`
   commit) and `docker16/Dockerfile` (16.15, `postgresql-16-pg-hint-plan=1.6.2-1.pgdg12+1`).
7. **Defaults:** scripts run without arguments produce the paper's (current) results. For example,
   `selective_sampling` defaults to the v3 native times (`results/plan_certificate/selective_sampling_v3.json`).
   Superseded files are listed in `results/SUPERSEDED.md`.
