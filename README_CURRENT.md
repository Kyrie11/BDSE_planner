# BDSE current algorithm — V64.3.53 EAF-ICER-POTR

**Paired Operator-Trajectory Retention**

V53 is the preregistered follow-up to the reliable V52 HODR result.

## Scientific status before V53

V52 showed:

- QPE+D structural-null/effect-support identification: **PASS**, AUC `0.6516244589`, 5/5 folds > random;
- conditional HURDLE-SIGN on the same state: **not identified**;
- conditional HURDLE-PARETO on the same state: **not identified**;
- no V52 runtime arm is promoted.

The current dominant bottleneck is **effectful selected-outcome state sufficiency**: scalar operator magnitude tells whether the intervention matters, but not the direction/order of its effect.

## V53 change

Freeze:

- full-set RSMR winner;
- Q/P/E;
- exact V52 QPE+D effect-support hurdle;
- exact V50.5 metric-safe 502/502 paired outcome labels;
- pairwise lambda=1, alpha/calibration and no-fallback containment.

Change only the effectful conditional-outcome state.

### ENDPOINT

`[Q, P-Q, E-P, D, dx_T, dy_T, wrap(dyaw_T), dv_T]`

### TEMPORAL

ENDPOINT plus two fixed DCT-II modes for signed `dx(t),dy(t),dyaw(t),dv(t)`.

No horizon/basis sweep, attention, MLP, safety weight, new offline future observable, rerank, fallback or new threshold.

## State acquisition

The exact historical V50.5 planner remains byte-identical. V53 uses a dedicated process-local nuPlan wrapper to record a treatment-only pre-execution sidecar containing frozen-proposal vs actual-runtime-incumbent trajectory contrast. The old paired outcome labels are reused; they are not recollected.

## Run

```bash
cd bdse_v64_3_53_eaf_icer_potr
bash RUN_V64_3_53_EAF_ICER_POTR_TRAIN_1GPU.sh
```

Set `V49_ROOT`, `V50_5_ROOT`, `V51_ROOT`, `V52_ROOT`, `NUPLAN_ROOT` or `EAF_CKPT` only if your server layout differs from the defaults.

Untouched validation must remain unconsumed unless a preregistered V53 arm passes identification **and** the unchanged deployment gate.
