# V64.3.35 Result Postmortem and V64.3.36 SGRR Design

## 0. Executive decision

The uploaded V64.3.35 run is **engineering-valid for TRAIN-level nested-cross-fit algorithm attribution**. No runtime, feature-schema, candidate-set, query-accounting, structural-delegation, objective-scale, or calibration-unit implementation blocker was found. The preregistered TRAIN scientific gate failed before any CAL500/A500/B500 population was selected, so no fresh-level claim is permitted and the permanent 10700-token design exclusion remains unchanged.

The scientific result is not “basepoint context works” and not “basepoint context is useless.” The correct conclusion is narrower:

1. FDSR factorization alone fails: it restores/expands opportunity coverage but collapses null-action discrimination, producing 180/208 no-positive-opportunity false interventions and 50 catastrophes.
2. FBCSR absolute incumbent context contains **weak and cross-fold-consistent intervention-existence signal**: no-op false interventions improve 180→159 in all five folds, catastrophes 50→40, and selected sum +23.49→+29.31.
3. The effect is insufficient for the preregistered first-order mediation claim: the no-op reduction is only 11.7% rather than >=20%, capture falls 58.01%→54.01% (>3 pp tolerance), and two FBCSR test folds remain negative.
4. V35 does not cleanly isolate a pure context boundary effect because the delta and context heads are jointly refit. Although context is a common shift *within a fixed FBCSR model*, FDSR and FBCSR learn different delta weights; 59 common-selected scenes change challenger winner across arms.
5. The next causal question is therefore not “add richer context.” Freeze the useful V34 RSMR challenger ordering and learn only a non-negative scene-level **reservation** that may veto the exact frozen winner to incumbent but may never re-rank challengers. Compare a clean incumbent-basepoint reservation with a selection-geometry reservation.

V64.3.36 implements exactly this diagnostic as **SGRR — Selection-Geometry Reservation Recovery**.

---

## 1. Engineering validity of V35

### 1.1 Uploaded run status

The uploaded output contains only the nested TRAIN artifacts:

- `logs/v64_3_35_fbcsr_fit.out`;
- `provenance/v64_3_35_fbcsr_fit.json`;
- `provenance/v64_3_35_fbcsr_train_scene_audit.csv`;
- prerequisite/regression logs and stage timing.

No CAL/fresh token manifest exists. Stage timing ends at:

1. exact V34 failure reproduction and fresh-unspent guard;
2. prerequisites/regression;
3. nested FBCSR TRAIN gate.

The exact stop is:

`V64.3.35 FBCSR nested TRAIN gate failed (factorized_existence_ordering_loss_does_not_resolve_v34_tradeoff_under_delta_only_representation); STOP before CAL/fresh selection`.

### 1.2 Independent code audit

The uploaded V35 package passes:

- Python compile;
- V35 focused tests: 7/7;
- launcher `bash -n`;
- uploaded server targeted regression: 161/161.

The runtime and fitter agree on:

- deployment-admissible/support-positive direct challenger population;
- incumbent pseudo-item score exactly zero;
- FDSR/FBCSR score semantics;
- FBCSR context as a scene-common challenger shift;
- selected-policy residual and veto unit;
- incumbent-default and no-fallback behavior.

No engineering hotfix is justified. The run is scientifically failed, not engineering-invalid.

### 1.3 Attribution scope

The valid scope is **development/TRAIN-level mechanism attribution only**. V35 did not consume CAL500/A500/B500, so no independent fresh generalization statement may be made.

---

## 2. Exact V35 aggregate result

The paired direct-domain TRAIN population contains 782 scenes: 574 positive-opportunity scenes and 208 no-positive-opportunity scenes.

| Mechanism | Selected | Positive | Precision | Useful capture | Sum ΔT | Worst ΔT | NegRMS | Catastrophes | No-op false intervention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corrected MEAN | 476 | 260 | 54.62% | 45.30% | -18.030 | -6.735 | 0.627 | 47 | 110/208 |
| V33 PAIR | 98 | 47 | 47.96% | 8.19% | -4.668 | -3.992 | 0.861 | 18 | 15/208 |
| V34 RSMR | 502 | 221 | 44.02% | 38.50% | **+43.294** | -3.806 | **0.356** | **28** | 107/208 |
| V35 FDSR | 739 | 333 | 45.06% | **58.01%** | +23.494 | -6.735 | 0.496 | 50 | **180/208** |
| V35 FBCSR-RANK | 687 | 310 | 45.12% | 54.01% | **+29.314** | -6.735 | 0.505 | 40 | **159/208** |
| V35 FBCSR-MAIN | 13 | 6 | 46.15% | 1.05% | -4.898 | -1.233 | 0.679 | 4 | 3/208 |

The preregistered TRAIN gate correctly fails.

---

## 3. Did FDSR solve the V34 trade-off?

No.

FDSR explicitly separates a positive scene into an incumbent-existence constraint and a challenger-ordering constraint. That removes V33's all-rival average dilution and V34's single-max masking, but under the frozen delta-only representation it becomes much too intervention-permissive:

- selected 502→739 vs V34 RSMR;
- useful capture 38.50%→58.01%;
- no-op false intervention 107→180;
- catastrophes 28→50;
- selected sum +43.29→+23.49;
- worst returns to -6.735.

The explicit factorization therefore does not resolve `should intervene? + which intervention?` by itself. It mostly removes suppression and drives challenger scores positive in almost every opportunity scene and in 86.5% of no-opportunity scenes.

This is an important falsification: the V34 trade-off is not explained only by “a single scene-max surrogate mixed existence and ordering.”

---

## 4. Did FBCSR solve `should we intervene?`

It shows a real signal, but it does not solve the problem.

### 4.1 Cross-fold consistency

No-opportunity false interventions improve in every outer test fold:

- fold 0: 44→41;
- fold 1: 31→30;
- fold 2: 41→35;
- fold 3: 35→28;
- fold 4: 29→25.

Aggregate:

`180 → 159`, a reduction of 21 scenes or **11.7%**.

Catastrophes fall `50 → 40`, and selected sum improves `+23.49 → +29.31`.

This consistency is evidence that absolute incumbent state contains information relevant to the null-action boundary.

### 4.2 Why this is not a first-order mediator

The preregistered mediation requirement was >=20% no-op false-intervention reduction with <=3 pp capture loss. V35 gets:

- no-op reduction: only 11.7%;
- useful capture: 58.01%→54.01%, a **4.01 pp** loss;
- catastrophes: still 40, worse than V34 RSMR's 28;
- FBCSR fold selected sums: `+3.372, -1.650, +16.638, +13.542, -2.588`.

Therefore the strong hypothesis

> “19-D pure contrast is missing a first-order incumbent base-point variable, and constrained FBCSR solves intervention existence”

is **not supported**.

A more defensible conclusion is:

> absolute incumbent base-point information is a weak secondary signal, but not the dominant missing mediator under the current representation/operator.

---

## 5. Scene-level attribution: what did FBCSR learn?

The full 782-scene audit is retained in `V64_3_35_SCENE_LEVEL_CAUSAL_AUDIT_FOR_V64_3_36.csv`.

### 5.1 FDSR→FBCSR transitions

Across all scenes:

- same winner: 618;
- winner changed: 59;
- FDSR proposal → FBCSR abstain: 62;
- FDSR abstain → FBCSR proposal: 10;
- both abstain: 33.

Net FBCSR-vs-FDSR teacher effect is `+5.820`.

On no-opportunity scenes:

- 25 bad FDSR proposals are removed;
- 4 new bad FBCSR proposals are created;
- net teacher effect is `+3.312`.

On opportunity scenes:

- 37 FDSR proposals are removed;
- 6 proposals are added;
- net teacher effect is only `+2.509` and capture declines.

This is exactly the desired direction qualitatively—move some scenes toward incumbent—but not strongly enough and not selectively enough.

### 5.2 Examples where context helps

- `00bf712a8e9e59bf`: no positive opportunity; FDSR executes a -2.6145 intervention; FBCSR abstains.
- `1088d47284565808`: no-opportunity FDSR harmful proposal is removed.
- `b16054ccc4bd54f9`: opportunity scene; FDSR chooses -3.9925, FBCSR chooses +1.8927.
- `275e9cfb52fd59de`: +0.1273→+3.4955.

These examples show that context can alter the intervention boundary and, because the delta head was jointly refit, sometimes also changes winner identity beneficially.

### 5.3 Examples where context hurts

- `75fd8a14d25350e5`: FDSR +6.0603 is suppressed to incumbent.
- `49f109b9cca4567a`: FDSR +4.0405 is suppressed.
- `2d4f708394ef542e`: FDSR correctly abstains on a no-op scene; FBCSR creates a -3.8961 intervention.
- `2f19f98982615f60`: context changes a near-zero harmful action into a roughly -3.83 harmful action.

Thus the current context map is neither a pure safety prior nor a clean monotone boundary correction.

### 5.4 Critical causal-identifiability limitation of V35

FBCSR's architecture guarantees that, **holding its fitted delta head fixed**, `h_context(c_inc)` is common to all challengers and cannot change challenger pairwise ordering.

However, FDSR and FBCSR are separately fit models. Adding the context head changes the jointly optimized delta weights. In 59 scenes where both arms select a proposal, the challenger winner differs. Their net FBCSR-vs-FDSR teacher effect is +10.404 (33 better, 23 worse, 3 equal).

Therefore V35 cannot cleanly answer whether the observed 180→159 no-op reduction comes from:

1. a genuine base-point boundary shift; or
2. a changed delta/order head induced by joint optimization.

This is a causal-design limitation, not an execution bug. V36 removes it by freezing the ordering head.

---

## 6. What is the dominant bottleneck after V35?

The evidence chain now separates the system into two very different capabilities.

### 6.1 Challenger ordering has useful signal

V34 RSMR is still the strongest clean evidence here:

- all five TRAIN folds selected sum positive;
- aggregate selected sum +43.29;
- opportunity-domain sum +71.40;
- NegRMS 0.356.

This layer should be **frozen**, not replaced again.

### 6.2 Intervention existence remains unstable

The difficult decision is whether the frozen winner's apparent gain is sufficiently trustworthy to leave the incumbent.

V33 obtains safety mostly by global suppression and loses recovery. V34 recovers value but reopens no-op errors. V35 FDSR becomes even more permissive. FBCSR adds some base-point discrimination but not enough.

The current dominant bottleneck is therefore more precisely:

> **scene/set-level intervention reservation under extremal selection-induced overestimation at the incumbent boundary.**

The ranker chooses the maximum among several noisy challenger scores. The selected maximum is systematically exposed to winner's-curse/selection bias. A fixed zero boundary asks a different statistical question from “which challenger has the largest relative score.”

This diagnosis subsumes the current evidence better than “pure contrast lacks basepoint information.”

---

## 7. Model-layer status

### Mature / freeze

- bounded B16/M24 auditable interface;
- EAF complete action-local frontier;
- exact selected-evidence attribution;
- support and deployment admissibility population;
- structural-domain delegation;
- admissible-incumbent default and no-fallback containment;
- V30.3 conclusion that capacity-only transmission is not the first-order solution.

### Useful signal / freeze as control

- V34 RSMR contrastive regret ordering: useful aggregate and opportunity-domain ranking signal.

### Weak / unresolved

- intervention-existence/null-action boundary;
- hard catastrophic selected tail;
- absolute incumbent basepoint: weak secondary signal, not first-order confirmed.

### Stop as main route

- FDSR factorized loss as a standalone solution;
- joint FBCSR richer/nonlinear context iteration without a clean frozen-order causal test;
- selected-policy marginal conformal as a hard zero-catastrophe certificate.

---

## 8. Updated no-repeat constraints

Retain all prior prohibitions: learned acquisition/selector-v2, coreset/beam/swap, B/M sweep/capacity-only, FCR/global reconstruction, DRC K/threshold/downside tuning, KNN/type/family/radius/OOD, PTMC/classifier variants, naive feature concatenation, support/scalar threshold rescue, action blacklist, learned incumbent→anchor veto, ridge/alpha/leverage/score-threshold sweeps, post-hoc candidate-set reduction, and A/B pooling.

New after V35:

- do not tune existence-vs-ordering loss weights to rescue FDSR;
- do not add richer/nonlinear incumbent context merely because FBCSR has weak signal;
- do not claim basepoint sufficiency from the 180→159 result;
- do not use candidate count as a hard intervention gate;
- do not refit challenger ordering when testing a pure intervention-boundary mechanism;
- do not continue selected-policy marginal conformal as the main answer to a hard selected-tail contract;
- do not use an independent binary `intervene/not-intervene` classifier that can create a second unconstrained decision path.

---

# 9. V64.3.36: SGRR — Selection-Geometry Reservation Recovery

V36 deliberately changes **only the intervention-existence layer** while freezing the strongest current ordering layer.

## 9.1 Frozen ordering

Fit the exact V34 RSMR ordering on TRAIN:

`u_b = f_RSMR(delta_b)`.

The frozen proposal is

`b_hat = argmax_{b:u_b>0} u_b`.

If no positive challenger exists, return incumbent.

V36 never refits or re-ranks this ordering when learning the reservation.

## 9.2 Scene-common reservation

Learn a non-negative scalar `rho(x_scene) >= 0` and execute the **same** frozen winner only if

`u_bhat - rho(x_scene) > 0`.

Otherwise return incumbent.

Because the same scalar is subtracted from all challengers and the proposal identity is frozen before reservation:

- accepted V36 interventions are a subset of RSMR interventions;
- accepted challenger identity is exactly the RSMR winner;
- no new proposal can be created;
- no second-best fallback is possible.

This is a deterministic monotone containment property.

## 9.3 Reservation target

On an independent selected-policy calibration fold, for each frozen RSMR proposal record

`r = max(0, u_bhat - Delta_T(b_hat;i))`.

This is the **selected-policy overprediction** caused by the frozen extremal operator.

Why this target is aligned with intervention existence: if a reservation upper-bounds the realized overprediction, then

`u_bhat - rho <= Delta_T(b_hat;i)`.

Therefore a positive adjusted margin would imply positive teacher improvement on that event. V36 does not claim a distribution-free per-scene guarantee for the ridge estimate; it uses independent cross-fit/fresh validation to test whether a low-capacity scene reservation is a reproducible mechanism.

## 9.4 Two strictly separated diagnostic arms

### BPR — Basepoint Reservation

Use only the already-observed absolute incumbent 18-D evidence plus incumbent support to predict the non-negative reservation.

Ordering weights are frozen. This is the clean version of the V35 basepoint hypothesis. If BPR now reduces no-op errors while preserving capture, V35's weak context signal can be attributed to the boundary rather than joint re-ranking.

### SGRR — Selection-Geometry Reservation

Use a low-dimensional, permutation-invariant description of the **frozen RSMR score set**:

1. top score;
2. top gap to runner-up or incumbent zero;
3. score RMS;
4. positive-score fraction;
5. log effective competitor mass.

These variables represent the geometry that creates extremal-selection overestimation. They are not new evidence queries and are not a candidate-count hard gate.

If SGRR succeeds where BPR does not, the dominant missing mediator is selection geometry / winner's-curse reservation rather than absolute incumbent state.

BPR is diagnostic only and may not rescue an SGRR failure.

## 9.5 Nested TRAIN design

For each outer fold:

- three folds: fit frozen RSMR ordering;
- one independent fold: collect the frozen policy's selected proposals and fit BPR/SGRR reservation;
- one fold: evaluate RSMR, BPR and SGRR.

Minimum reservation calibration support: 64 proposals per fold.

SGRR TRAIN promotion requires:

- exact monotone subset and winner identity;
- >=20% reduction in no-positive-opportunity false interventions vs frozen RSMR;
- capture no worse than RSMR by >3 pp;
- >=25% reduction in catastrophes vs RSMR;
- NegRMS no worse than RSMR;
- aggregate and every fold selected sum nonnegative;
- >=64 selected and >=32 positive.

These are fixed before fresh selection.

## 9.6 Fresh protocol if TRAIN passes

V35 consumed no fresh population. Permanent exclusion remains 10700 tokens.

Use new label-free seed:

`v64.3.36-eaf-icer-sgrr-cal500-double-fresh-v1`.

Select independent `CAL500 + A500 + B500` only after nested TRAIN passes.

CAL500 fits the final BPR and SGRR reservation heads on outputs of the full-TRAIN frozen RSMR model.

Each fresh block independently evaluates:

`RAW / V20 / PRESERVE / RSMR / BPR / SGRR`.

No A/B pooling.

Fresh SGRR must preserve the same containment, reproduce meaningful no-op reduction without >3 pp capture loss, have zero catastrophic accepted direct interventions and worst > -0.5, provide useful recovery gain over PRESERVE, and remain endpoint-noninferior to PRESERVE and V20.

---

## 10. Paper-line implication

The high-level paper should not be renamed after SGRR. The CCF-A-oriented mechanism line is now better expressed as:

`bounded auditable interface`

`→ exact EAF action-local attribution`

`→ contrastive regret-aligned challenger ordering`

`→ selected-policy scene reservation for the incumbent boundary`

`→ deterministic monotone incumbent containment`

`→ independent double-fresh/full-validation/closed-loop evidence`.

The candidate conceptual contribution is an **invariance-factorized decision-sufficiency decomposition**:

- `which challenger?` is a relative/contrastive ordering problem;
- `should leave the incumbent?` is a scene-level reservation problem after data-dependent extremal selection;
- the reservation may shrink the frozen intervention set but cannot re-rank or create a new action path.

The low-dimensional reservation features and ridge solver are not themselves the novelty. The novelty hypothesis is the alignment of the statistical unit with the deployment operator and the deterministic separation of ordering from intervention existence.

This is a CCF-A-level research direction only if V36 survives nested TRAIN, double fresh, a single frozen full-validation reproduction, and closed-loop evaluation. No acceptance or SOTA claim is warranted before those results exist.
