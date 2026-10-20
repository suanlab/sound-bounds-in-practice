# Predictions frozen before the breakthrough analyses (2026-10-19 21:58 KST)

Written before any of the analyses below was run. Development data: STATS-CEB
(146 queries) and JOB-light (70), which we have already analysed in other ways,
so results there are **exploratory** even when they match. Held-out data:
a seeded random subset of JOB-range from the LpBound artifact, frozen in a
separate pre-registration (`PREREGISTRATION_JOBRANGE.md`) before any of it is
measured. This file's SHA-256 is recorded in `PREDICTIONS.sha256`; any later
change to it is visible.

Notation: `R(p)` joint regret bound; `L` sound sampled floor (r = 0.10 unless
stated); `B` upper bound (LpBound unless stated); "certified" = `min_p R(p) <= 2`
under `C_out` with the final result; "choice matters" = worst/best true `C_out`
over left-deep orders > 2.

| H | Prediction | Falsified if |
|---|---|---|
| H1 shared mass | A single per-query statistic rho = L(final) / (L(final) + B(unique part of the best-bound plan)) predicts certification with AUC >= 0.9 on STATS-CEB (r = 0.10) and, without refitting the threshold, AUC >= 0.85 on JOB-light; certification and rho both fall with the number of relations | AUC < 0.8 on STATS-CEB |
| H2 exchange rate | Tightening every upper bound geometrically by 2.5x (B' = |S| (B/|S|)^(1/k) with k chosen so the median whole-query looseness falls 2.5x) at r = 0.10 certifies 55.0% +- 5 pp of STATS-CEB queries, the share at r = 0.25 | differs from 55.0% by more than 10 pp |
| H3 decision sensitivity | With synthetic sound bounds B_k = k |S| injected as the upper guard on our instance's own estimates, the true-C_out gain of the chosen plans is <= 1.05x for k >= 8 and reaches >= 50% of the true-size C_out gain only for k <= 4 (threshold-shaped, not linear in log k) | gain roughly linear in log k, or k* > 10 |
| H4 a-posteriori certification | Using the executed native plan's own exact intermediate sizes, PostgreSQL's native plan is certified within 2x on >= 55% of STATS-CEB queries at r = 0.10 and on >= 25% with no sampling at all (L = 1 off the plan); the no-sampling certificates fall mostly where the choice does not matter | no-sampling share < 10% |
| H5 price of certification | The hindsight-optimal set of sub-joins to sample (minimum total sampling time that still certifies) costs more than the native plan's execution on >= 50% of the choice-matters queries that sampling certifies | oracle cost <= 0.2x the greedy rule's cost on the median query |
| H6 probabilistic floor | A floor L-hat / c, with c calibrated on seed 0 for 95% coverage of the truth, certifies >= 55% of STATS-CEB queries at r = 0.10 on seeds 1-4, with floor violations <= 5% of sub-joins and true regret <= 2 for >= 95% of certified plans | violations > 5%, or gain over the sound floor < 5 pp |
| H7 witness floor | `SELECT 1 ... LIMIT k`, with k the floor the certificate needs (computed from bounds only), costs a median <= 0.2x the sampled count of the same sub-join | median ratio > 0.5 |

Exploratory or not, every result is reported against these numbers, including
failures.
