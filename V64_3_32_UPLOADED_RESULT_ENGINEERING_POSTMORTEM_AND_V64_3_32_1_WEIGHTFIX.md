# V64.3.32 uploaded result: engineering-invalid for intended SSIR attribution -> V64.3.32.1 WEIGHTFIX

## 1. Decision

The uploaded V64.3.32 run must **not** be used to claim that SSIR, scene-simultaneous conformal ordering, or post-selection conformal has failed. The run is execution-complete only through the nested TRAIN fitter and it correctly stops before CAL/fresh, but the fitted SCIR/SSIR ridge objective does not match the frozen design contract inherited from V31.

This is an engineering/logic error, not a negative algorithm result. Therefore V64.3.32.1 makes no new algorithmic mechanism change.

## 2. What executed correctly

The uploaded run reproduces the frozen V30.3/V31 prerequisites, passes the targeted regression (`138 passed`), reads the frozen 3000-scene / 75,133-row TRAIN frontier, builds the 9,394 direct admissible/support-positive edges over 782 direct scenes, and executes all five nested fit/calibrate/test folds. The launcher stops at the intended TRAIN gate before selecting any CAL500/A500/B500 tokens.

Observed V32 signature:

- nested SSIR fold pass count: `0/5`;
- mean-rank aggregate: `572 selected / 319 positive / sum -23.842886184829602 / worst -4.041263536178889`;
- V31 common-veto diagnostic: `0 selected`;
- SSIR simultaneous-LCB: `0 selected`;
- fold scene-max conformal q: `1.5803, 1.7583, 1.5938, 1.7442, 2.1202`;
- mean-selected predicted improvement: mean `0.04845`, max `0.32866`;
- mean-selected leverage scale: mean `1.5678`, max `6.8676`.

These values are internally consistent with the implementation, but the implementation is not the frozen objective stated by the V31/V32 design.

## 3. Root cause: scene-equal weights were globally renormalized before a fixed-lambda ridge solve

The V31 design states:

> Every scene receives total training weight 1.

The intended fixed ridge objective is therefore, up to an irrelevant constant convention only if lambda is rescaled consistently,

`sum_scene mean_candidate (y - f(x))^2 + lambda ||w||^2`, with frozen `lambda=1`.

The V31 and V32 fitters first correctly assign each edge weight `1 / n_candidates(scene)`, but then execute a second normalization:

```python
w = w / w.sum()
```

before solving the ridge system while leaving `lambda=1` unchanged.

The implemented objective is consequently

`(1 / N_fit_scenes) * sum_scene mean_candidate loss + lambda ||w||^2`.

Multiplying through by `N_fit_scenes`, this is equivalent to the intended loss with an effective ridge coefficient

`lambda_eff = N_fit_scenes * lambda`.

For the five V32 outer fits, `N_fit_scenes = 483, 500, 461, 450, 452`. Thus the code used an effective regularization roughly **450--500 times larger** than the frozen per-scene-total-mass contract.

This is not a harmless weight normalization: ridge is not invariant to global sample-weight scaling when lambda is held fixed.

## 4. The same scale error also contaminates V32 leverage normalization

V32 constructs

`G = Z^T W Z + lambda I`, `h = z^T G^{-1} z`, `scale = sqrt(1+h)`.

The same globally normalized weights were used in `G`. Under the frozen contract, the Gram matrix should use the unnormalized scene-mass loss weights, while only feature mean/variance estimation should use a normalized probability copy.

Therefore both pieces that determine the V32 lower bound were affected:

1. the conditional-mean predictor `mu` was over-regularized relative to the intended objective;
2. the candidate-specific leverage normalization was computed on the wrong loss scale.

The observed `mu << q*scale` and universal abstention are therefore not admissible evidence for falsifying SSIR.

## 5. Scope of the scientific damage

The valid evidence chain through V30.3 remains unchanged. In particular, the independent capacity closure is not affected by this fitter bug.

However, the V31 SCIR fitter contains the same global normalization. Therefore the *implemented* V31 0/5 result remains a factual result for that historical code, but the stronger intended-mechanism statement -- that a correctly implemented `lambda=1`, per-scene-total-weight-1 same-scene mean ridge is insufficient -- must be treated as unresolved until the repaired objective is rerun.

Accordingly, do not use the uploaded V32 zero-selection result to further tighten the dominant bottleneck or to design V33.

## 6. V64.3.32.1 hotfix

V32.1 changes only the objective scaling:

```python
loss_w = 1 / n_candidates(scene)      # each scene sums to 1
moment_w = loss_w / loss_w.sum()      # moments only

mean, std <- moment_w
ridge solve <- loss_w
leverage Gram <- loss_w
```

Frozen unchanged items:

- B16 / M24 and all upstream evidence/query semantics;
- 19-D same-scene candidate-minus-incumbent representation;
- continuous teacher-improvement target;
- ridge `lambda=1`;
- candidate admissibility/support prerequisites;
- `alpha=0.05`;
- scene-max simultaneous nonconformity definition;
- positive-LCB extremal selection and incumbent default;
- no-fallback / structural delegation;
- nested 3-fit / 1-cal / 1-test TRAIN gate;
- CAL500 + A500 + B500 protocol and all independent promotion criteria.

This is therefore an engineering hotfix, not an algorithmic rescue sweep.

## 7. Fresh-data policy

The uploaded V32 run stopped inside the TRAIN fitter, before the launcher entered `calibration_and_fresh_selection`. The result archive contains no CAL/fresh token manifests.

V32.1 therefore reuses the already frozen hash seed:

`v64.3.32-eaf-icer-ssir-cal500-double-fresh-v1`.

The V32.1 launcher hard-checks the original V32 result signature and refuses to reuse the seed if any original V32 CAL/A/B token manifest exists and is nonempty.

## 8. Validation of the hotfix package

Local code validation:

- Python compile: PASS;
- V32.1 focused + V32 + V31: `13/13 PASS`;
- V13--V32.1 targeted regression: `140/140 PASS`;
- repository regression split into two deterministic halves: `235/235 PASS` + `235/235 PASS` = **470/470 PASS**;
- warnings: `36`, all pre-existing Transformer nested-tensor warnings, no new warning class;
- launcher `bash -n`: PASS.

A new unit test explicitly solves the frozen per-scene-total-weight-1 ridge objective and verifies that the hotfix matches it while differing from the historical globally normalized objective.

## 9. Next action

Run only V64.3.32.1. Do not tune ridge lambda, conformal alpha, scale multipliers, support thresholds, candidate-set definitions, or any downstream algorithm before this repaired run closes.

If the repaired nested TRAIN gate fails, that repaired failure can be used for algorithm attribution. If it passes, the same launcher proceeds to the still-untouched CAL500/A500/B500 protocol.
