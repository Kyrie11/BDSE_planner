# V64.3.50.5 result audit and V64.3.50.6 fit-only engineering repair

## Executive verdict

The uploaded V50.5 result is **not scientifically attributable**. The correct status is:

> **ENGINEERING / ANALYSIS-PIPELINE STOP — V50 PIOR has not yet received a valid scientific GO/STOP evaluation.**

This is materially different from the earlier V50.4 failure: the paired closed-loop evidence source is now complete and metric-safe, but the downstream nested fit implementation is defective.

## 1. Paired evidence collection is complete and reusable

The V50.5 output contains an exact 502-token paired outcome table. Both control and treatment have 9 completed batch certificates, 502/502 successful scenarios, zero failures, and one intended PIOR probe per scenario. The full metric-safety audit passes for both arms and reports no metric-engine failure logs or missing V50.5 safety markers.

Observed paired outcome support:

- total paired scenarios: 502;
- beneficial: 121;
- nonbeneficial: 381;
- hard harm: 25;
- closed-loop score delta: 122 positive / 209 zero / 171 negative;
- frozen-fold beneficial counts: 29 / 22 / 15 / 28 / 27.

One score-positive example is still non-beneficial because it violates a hard-safety coordinate; this is consistent with the preregistered PIOR label definition rather than a data inconsistency.

Therefore the expensive V50.5 closed-loop collection does **not** need to be rerun for this repair.

## 2. Why the reported exception occurs

`fit_v64_3_50_eaf_icer_pior.py` imports `_conformal_threshold` from V48. V48 contains a historical fixed guard:

```python
if n < 16:
    raise ValueError("V48 calibration has too few frozen-policy positives")
```

V50's frozen nested protocol uses calibration fold `(k+1) mod 5`. One calibration fold has exactly 15 beneficial paired outcomes, hence the observed exception.

The threshold rank itself is:

```text
r = ceil((n + 1) * (1 - alpha))
```

with frozen `alpha ≈ 0.077918552`. For `n=15`, `r=15`, which is a valid finite empirical rank. The exact condition for a finite empirical threshold is `r <= n`; at this alpha the minimum support is 12. The V48 fixed 16-example guard is therefore an inherited implementation assumption, not a mathematical requirement of the frozen V50 rank rule.

V50.6 removes only this unrelated support constant and fails closed on the exact finite-rank condition instead. There is no alpha or threshold sweep.

## 3. A second hidden bug would have corrupted the aggregate gate

The exception masks a second, more serious analysis bug.

V50 `_join()` returns rows sorted by `scenario_token`. Inside `_nested()`, OOF keep decisions are generated fold-by-fold and appended to `all_keep`. The final line then applies this fold-concatenated mask positionally to token-sorted rows.

On the uploaded 502-token frozen fold assignment, only **3/502 positions** are identical between these two orders. Thus aggregate selected count, beneficial retention, hard-harm count, score sum and negative RMS would be computed using decisions from the wrong scenarios.

V50.6 records every OOF keep decision under its `scenario_token`, checks exact one-to-one identity, and aligns the mask back to row order before aggregate metrics. Per-fold fitting/calibration is unchanged.

## 4. Scientific conclusions that are explicitly *not* allowed from V50.5

Because the nested fit did not complete and the original aggregate gate is defective, this result cannot answer any of the following:

- whether paired interventional outcome supervision beats the frozen OBS/EGO baselines;
- whether `[Q, P-Q, E-P]` is sufficient for true paired selected-outcome discrimination;
- whether the low-capacity PIOR sign-risk functional is sufficient;
- whether PIOR passes the causal retention gate;
- whether V50 should be GO or scientific STOP.

Accordingly, no new algorithm family is promoted or closed, and no V51 mechanism is designed from this run.

## 5. What V50.6 changes—and does not change

Changed only:

1. finite-support validation for the already frozen conformal rank;
2. OOF keep-mask token alignment;
3. fit-only launcher/provenance checks/tests/documentation.

Unchanged: RSMR, Q/P/E, pairwise risk loss, lambda=1, alpha, rank rule, calibration folds, paired labels, hard-safety metrics, runtime retention contract, no-fallback containment and all earlier no-repeat closures.

## 6. Safe acceleration

The stage timing in the uploaded run is dominated by paired collection: 79,541 s versus ~2 s for nested fitting. Reusing the already complete metric-safe paired TRAIN evidence is therefore both scientifically cleaner and orders of magnitude faster than re-running simulation.

The new launcher performs no closed-loop simulation. It rechecks manifests/evidence identity/metric safety and runs only the repaired fit. Do not alter 502 TRAIN size, worker count, replan cadence, candidate bank or metrics merely to accelerate this repair.

## 7. Next command

```bash
cd bdse_v64_3_50_6_eaf_icer_pior_fitrepair
V50_5_ROOT=/home/senzeyu2/code/BDSE_planner/outputs/outputs_v64_3_50_5_eaf_icer_pior_train_2gpu_v1 \
V49_ROOT=/home/senzeyu2/code/BDSE_planner/outputs/outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1 \
bash RUN_V64_3_50_6_EAF_ICER_PIOR_FIT_REPAIR.sh
```

If your default output layout already matches those names, simply run:

```bash
bash RUN_V64_3_50_6_EAF_ICER_PIOR_FIT_REPAIR.sh
```

Do not consume untouched validation until this repaired TRAIN fit says PASS.
