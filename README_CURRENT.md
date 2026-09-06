# BDSE current algorithm — V64.3.55 EAF-ICER-DMOR

**Dynamic-Mediator Outcome Retention**

V54.1 is reliable for TRAIN attribution. It identifies the realized one-replan
paired ego endpoint as a genuine effectful selected-outcome mediator, while the
same historical binary sign retention still fails fold-wise deployment nonharm.
V55 therefore freezes the identified mediator and changes the scientific object
in a strict order: first the outcome **functional**, then—only if that oracle
closes—the t0 **mediator bridge**.

## Frozen evidence chain entering V55

- V52 QPE+D effect-support hurdle: retained (`AUC=0.6516244589`, 5/5 > random).
- V54 REALIZED-ENDPOINT: mediator identification GO (`AUC=0.6117518845`,
  5/5 > random, 5/5 > V51, 5/5 > V52, 4/5 > V53 TEMPORAL).
- V54 REALIZED-TEMPORAL: STOP as temporal necessity (only 2/5 > endpoint and
  lower aggregate AUC).
- V54 binary-sign retrospective retention: STOP because all-fold nonharm fails,
  despite aggregate hard-harm / score-sum / NegRMS improvements.

Dominant bottleneck: **operator-aligned structured outcome order over an
identified realized mediator, followed by a t0 mediator bridge**.

## Arm A — REALIZED-DOMINANCE (diagnostic oracle)

State is exactly the frozen V54 realized endpoint:

`[Q, P-Q, E-P, D, realized_dx_end, realized_dy_end, realized_dyaw_end, realized_dv_end]`.

The V52 effect-support hurdle is unchanged. The conditional binary sign ranker
is replaced by an unweighted Pareto pairwise order over the existing paired
closed-loop official-score delta and hard-safety deltas. Trade-off pairs are
omitted rather than scalarized.

Arm A must identify Pareto order and pass the exact historical paired deployment
gate. It is not deployable because the realized mediator is post-intervention.

## Arm B — PREDICTED-DOMINANCE (t0 candidate)

**This arm is not evaluated at all unless Arm A fully passes.**

The V54 realized endpoint is distilled from the already-fixed V53 pre-execution
operator profile (signed terminal contrast + fixed DCT-II k=1,2 contrast) with a
zero-bias, zero-preserving multi-output ridge (`lambda=1`). Predictor fitting is
nested and excluded from outer test/calibration folds; outcome training uses
inner-OOF mediator predictions.

The same Pareto functional and the same deployment gate are then applied. A
V55 TRAIN pass means this t0-available branch passes. At that point algorithm
search freezes immediately; the next work is engineering-only runtime
integration and untouched paired validation.

## Run

```bash
cd bdse_v64_3_55_eaf_icer_dmor
bash RUN_V64_3_55_EAF_ICER_DMOR_TRAIN.sh
```

V55 is **fit-only**. It reuses V50.5 paired outcomes, V53 planned profiles and
V54 dynamic mediator profiles; no GPU simulation or new nuPlan metrics are
required.
