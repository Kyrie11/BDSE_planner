# BDSE current algorithm — V64.3.54 EAF-ICER-PDRM

**Paired Dynamic Response Mediation**

V54 is the preregistered follow-up to the reliable V53 POTR scientific STOP.

## Scientific status after V53

- V52 QPE+D structural-null/effect-support: **retained**, AUC `0.6516244589`, 5/5 folds > random.
- V53 ENDPOINT conditional state: **identification STOP** (`0.50495`, 2/5 > random).
- V53 TEMPORAL planned state: aggregate signal exists (`0.57438`, 5/5 > random, 4/5 > endpoint) but fails the preregistered frozen-control consistency requirement (3/5 > V51 and 3/5 > V52); **STOP**.
- Neither V53 arm passes all-fold nonharm; no runtime arm is promoted.

Dominant bottleneck: **post-intervention paired dynamic response / outcome-process sufficiency for effectful selected-outcome order**.

## V54 change

Freeze RSMR, Q/P/E, scalar planned D, V52 effect support, V50.5 502 paired labels, lambda=1 and the no-fallback operator.

Re-run both paired arms only through the first scheduled replan and collect realized treatment-control ego-state divergence. Full-horizon outcome labels are reused; metrics are disabled.

### REALIZED-ENDPOINT

`[Q, P-Q, E-P, D, realized_dx_end, realized_dy_end, realized_dyaw_end, realized_dv_end]`

### REALIZED-TEMPORAL

REALIZED-ENDPOINT plus fixed DCT-II `k=1,2` of realized paired dx/dy/dyaw/dv during the frozen one-shot exposure window.

V54 is **mediator identification only**: post-intervention state is not a legal t=0 runtime input, so V54 emits no deployable retention config and must not consume untouched validation.

## Run

```bash
cd bdse_v64_3_54_eaf_icer_pdrm
bash RUN_V64_3_54_EAF_ICER_PDRM_TRAIN.sh
```

Default is treatment GPU 0 / control GPU 1. On a 1-GPU server set `GPU_TREAT=0 GPU_CONTROL=0`.
