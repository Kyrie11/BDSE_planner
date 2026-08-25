# V64.3.37 uploaded result postmortem and V64.3.38 DAVR design

## Executive decision

The uploaded V64.3.37 result is complete and engineering-valid for **nested TRAIN-level algorithm attribution**. It does not require a V37.1 hotfix. The run completed all five cross-fit folds, all five selected-policy value-calibration populations exceeded the preregistered minimum, the frozen-winner/monotone-containment contract held, targeted regression passed, and the launcher stopped at the intended scientific TRAIN gate before selecting CAL500/A500/B500. Fresh data therefore remains untouched.

V37 is a **partial tail success but a failure of the preregistered post-selection value mechanism**. AVR and OPVR both reduce no-opportunity false interventions and catastrophes relative to frozen RSMR, but both destroy too much useful-opportunity capture. OPVR additionally produces a negative selected sum in one outer test fold. No promotion-level claim is valid.

The new dominant bottleneck is not challenger ordering. It is **selected-policy cardinal zero-crossing/value identifiability after a useful ordinal extremal ranker**. V37 does not identify a stable generalizable signed absolute-value mapping from the sparse selected-only population. The existing 19-D orthogonal evidence does contain weak downside/tail structure, but V37 cannot establish that this is a robust absolute sign/value representation.

V64.3.38 therefore tests a different, paper-level mechanism: **rank/value estimand factorization with supervision-unit decoupling**. V34 RSMR remains the sole ordinal challenger ranker. A separate dense cardinal value model is trained on all fit-fold candidate edges using the already repaired V32.1 scene-equal objective, but it is evaluated only after RSMR freezes one winner. An independent selected-policy fold then fits only a one-dimensional affine recalibration. This distinguishes sparse selected-policy identification failure from representation insufficiency without returning the old conditional-mean model to the extremal selector.

---

## 1. V37 result completeness and engineering reliability

Uploaded artifacts audited:

- code package SHA256: `bd152136f003b45cb4c7d8ba4f8adc5c2c738bc18ceadf41c8b805597550b35a`;
- output package SHA256: `5ad14d870506d608d95c128b3208dde4849ab45de278d540749ff3af01b1000a`.

The uploaded code package is byte-identical to the preregistered V37 package delivered before this experiment. There is no code-version drift.

The output contains the intended TRAIN artifacts and no evidence of a truncated launcher:

- `exact_v36_failure_reproduction_and_fresh_unspent_guard`: completed;
- `prerequisites_and_regression`: completed;
- `train_nested_pvr_gate`: completed;
- server targeted regression: **172/172 PASS**;
- scene audit: **782 rows / 782 unique direct-scene tokens**;
- selected-policy value-calibration proposal counts: **97 / 100 / 98 / 86 / 110**, all >=64;
- `monotone_frozen_winner_contract_valid = true`;
- TRAIN gate stopped with the exact registered diagnosis: `proposal_conditioned_value_readout_does_not_resolve_intervention_existence_without_destroying_RSMR_capture`.

No non-empty CAL500/A500/B500 token manifests exist. Hence V37 did **not** spend fresh data.

The implementation was re-audited against the preregistration:

1. RSMR proposal identity is frozen before AVR/OPVR.
2. Post-selection value can only accept that exact winner or return incumbent.
3. No second-best fallback or re-ranking path exists.
4. OPVR constructs the selected evidence residual orthogonal to the frozen RSMR weight direction on both fitter and runtime paths.
5. Fixed `lambda=1`; no threshold/alpha/feature/ridge sweep is used.
6. Fit/calibration/test folds are distinct in each outer fold.

**Decision: V37 is reliable for TRAIN-level scientific attribution.**

---

## 2. V37 against its preregistered gate

| TRAIN cross-fit | Frozen RSMR | AVR | OPVR |
|---|---:|---:|---:|
| selected | 502 | 377 | 341 |
| positive | 221 | 158 | 137 |
| precision | 44.02% | 41.91% | 40.18% |
| useful opportunity capture | **38.50%** | 27.53% | **23.87%** |
| selected teacher sum | **+43.2941** | +33.8665 | +38.8819 |
| catastrophe (`Delta_T <= -0.5`) | 28 | 19 | **15** |
| no-positive-opportunity false intervention | 107 | 72 | **71** |
| NegRMS | 0.3557 | 0.3516 | **0.3325** |

The registered no-op target is at least 20% reduction relative to 107, and both value arms satisfy it. The registered catastrophe target is at least 25% reduction relative to 28, and both also satisfy that tail-count target.

The decisive failure is capture. The minimum permitted capture is

`38.50% - 3 pp = 35.50%`.

AVR reaches only 27.53%; OPVR reaches only 23.87%. OPVR therefore loses **14.63 pp** of useful capture relative to frozen RSMR.

OPVR also violates the all-fold nonnegative-direction contract. Outer-fold selected sums are approximately:

`+15.144 / -4.073 / +6.441 / +10.535 / +10.834`.

Thus the scientific STOP is correct. V37 is not promotable, and fresh validation must not be consumed.

---

## 3. Did learning post-selection intervention value create a large gain?

No. V37 validates the **importance of the layer**, but not the effectiveness of the current estimator.

Compared with frozen RSMR, OPVR changes:

- catastrophes `28 -> 15`;
- no-op false interventions `107 -> 71`;
- NegRMS `0.3557 -> 0.3325`;
- but useful capture `38.50% -> 23.87%`;
- selected sum `+43.29 -> +38.88`;
- one outer fold becomes negative.

So V37 cannot support the claim that “once the model learns post-selection value, gain increases substantially.” The actual evidence is narrower:

> a post-selection value readout can remove some harmful/tail proposals, but the V37 selected-only estimators do not identify the positive zero-crossing well enough to retain the useful RSMR recovery population.

The layer remains the principal immature layer because RSMR already supplies useful ordinal recovery and the value arms can alter only the incumbent-exit decision. But **the current learning method for that layer fails**.

---

## 4. AVR diagnosis: the RSMR scalar is not an absolute-value coordinate

AVR asks whether only a scalar offset/scale correction is missing.

Across folds, its fitted standardized scalar slopes are:

`+0.1866 / -0.0114 / +0.2070 / -0.1641 / +0.0646`.

The sign itself is unstable. Out-of-fold selected-proposal diagnostics are:

- MSE: `0.5182`;
- Pearson: `-0.0375`;
- positive AUC: `0.4394`;
- sign accuracy: `0.4382`.

AVR does lower catastrophes and no-op interventions, but it does so while discarding too much opportunity. Therefore the V36/V37 evidence closes the strong hypothesis that RSMR is already an absolute intervention-value score needing only scalar calibration.

RSMR should continue to be interpreted as an **ordinal/structured regret score**, not a calibrated cardinal value.

---

## 5. OPVR diagnosis and the 19-D orthogonal-information question

The registered scientific question was:

> Does the original 19-D evidence, after removing the exact RSMR ranking-score direction, still contain selected-proposal absolute sign/value information?

The correct answer from V37 is:

> **No robust, generalizable signed absolute-value mapping was identified. There is weak downside/tail information in the orthogonal subspace, but V37 does not establish stable absolute sign/value sufficiency.**

Evidence for weak tail information:

- OPVR improves catastrophe count from AVR `19 -> 15`;
- NegRMS improves `0.3516 -> 0.3325`;
- non-catastrophe AUC improves to `0.6069`.

Evidence against a stable signed/cardinal value mapping:

- OPVR selected-proposal MSE is **0.7428**, worse than AVR's `0.5182`;
- positive AUC is only **0.4428**;
- zero-sign accuracy is **0.4263**;
- Pearson correlation with teacher improvement is `-0.0695`;
- only outer fold 0 improves test MSE; fold 1 explodes from AVR `0.3883` to OPVR `1.6325`;
- OPVR further lowers capture relative to AVR.

Most importantly, the fitted orthogonal residual directions are cross-fold unstable. Pairwise cosine similarities include approximately `-0.81`, `-0.78`, `+0.75`, `+0.68`, while the fifth direction is near-orthogonal to the others. The residual fit improves in-sample on every calibration fold, yet this direction does not transport consistently to the held-out outer fold.

Therefore one must **not** conclude that the 19-D representation definitely contains a usable high-dimensional absolute-value head. But one also cannot yet conclude that the 19-D representation is intrinsically insufficient, because V37 attempts to identify an 18-D residual from only **86-110 selected proposals per fold**.

This leaves a clean causal ambiguity:

1. **selected-only identification/sample inefficiency**, versus
2. **true representation insufficiency for cardinal value**.

V38 is designed to distinguish these two.

---

## 6. Evidence chain V32.1 -> V37

The recent sequence now supports a sharper mechanism decomposition.

### V32.1: dense edge mean has signal but fails as an extremal deployment operator

Corrected scene-equal conditional mean improves edge precision but creates 110 no-op false interventions, 47 catastrophes, and negative total selected value. Dense edge supervision therefore contains signal, but letting that pointwise mean itself choose the extremal challenger is not reliable.

### V33: explicit null action works, but average pair supervision over-suppresses

No-op false interventions fall dramatically, showing that incumbent/no-intervention must be explicit. Capture collapses, revealing candidate-count-dependent supervision imbalance and safety-through-abstention.

### V34: regret-aligned scene argmax learning recovers useful ordinal structure

RSMR restores capture to 38.50%, makes all five fold sums positive, and obtains +43.29 aggregate selected value. It becomes the strongest recovery/ranking sublayer, but no-op false intervention remains 107 and catastrophes 28.

### V35: factorization and basepoint are not first-order solutions

FDSR increases intervention frequency without closing the trade-off. FBCSR shows weak, 5/5-fold-consistent basepoint signal, but not enough to mediate the dominant failure.

### V36: frozen-order reservations suppress tails mainly by abstention

BPR and SGRR reduce catastrophes/no-op errors while destroying capture. This closes scene-common basepoint/selection-geometry reservation as the first-order answer.

### V37: selected-proposal value is the right layer, but sparse selected-only high-dimensional identification is unstable

AVR proves the ranking scalar is not a cardinal value coordinate. OPVR exposes weak orthogonal tail signal but fails to learn a transportable signed value mapping and loses even more capture.

The surviving interpretation is:

`useful ordinal challenger ranking` + `immature selected-action cardinal zero-crossing/value estimation`.

---

## 7. Current model-layer status

| Layer | V37 judgment |
|---|---|
| bounded B16/M24 interface | mature / freeze |
| EAF complete frontier and exact attribution | mature / paper backbone |
| capacity/evidence visibility | closed as first-order bottleneck |
| support/admissibility | stable / freeze |
| 19-D candidate-incumbent contrast for relative ranking | useful signal |
| V34 RSMR structured ranking | strongest current recovery sublayer / freeze |
| explicit incumbent/null action | necessary and conceptually established |
| basepoint reservation | secondary signal only / do not expand |
| selection-geometry reservation | tail signal but broad abstention / do not expand |
| score-affine absolute value | insufficient |
| selected-only 18-D orthogonal residual | unstable/high-variance / stop as main route |
| post-selection cardinal value / zero crossing | **dominant immature layer** |
| hard selected tail | still immature |
| marginal selected-policy conformal | guarantee/operational target mismatch; not current main route |
| structural delegation / incumbent default / no fallback | mature / freeze |

Updated dominant bottleneck:

> **selected-policy cardinal zero-crossing/value identifiability under sparse selected-only supervision after a useful ordinal RSMR extremal ranker.**

---

## 8. No-repeat constraints after V37

Keep all prior bans. In particular, do not:

- tune RSMR score threshold, value threshold, `lambda`, alpha/q, or reservation multipliers;
- refit RSMR while claiming a pure post-selection value experiment;
- add more basepoint/context features or more selection-geometry statistics;
- use candidate count/top-K/action blacklist as a gate;
- add a standalone intervene/not-intervene classifier;
- return to classifier/KNN/DRC/FCR/PTMC/tail-confirmation routes already falsified;
- enlarge OPVR with nonlinear/high-capacity selected-only residuals;
- interpret OPVR's tail improvement as proof of robust 19-D cardinal sufficiency;
- let a dense pointwise value model rank/select challengers, which would simply return to the V32 failure mechanism;
- consume fresh data after a failed TRAIN gate.

---

# V64.3.38 EAF-ICER-DAVR

Full name: **Decoupled All-edge Value Recovery**.

The paper-level mechanism is **rank/value estimand factorization with supervision-unit decoupling**.

## 9. Core structural decomposition

For each scene, RSMR alone freezes the proposal

`b_hat = argmax_{b: u_b > 0} u_b`.

No value model is allowed to participate in this argmax.

Separately learn a candidate cardinal value function

`v_D(x_b) ~= Delta_T(b; incumbent)`

using **all eligible fit-fold candidate edges**, not only selected winners. The fitting objective reuses the repaired V32.1 scene-equal ridge definition:

`sum_scene mean_candidate (Delta_T - v_D)^2 + lambda ||w||^2`, with fixed `lambda=1`.

At runtime the dense value is evaluated only on the already frozen `b_hat`:

`execute b_hat iff v_D(x_bhat) > 0`, otherwise incumbent.

Thus dense supervision is reused without giving the dense estimator extremal selection authority.

This is explicitly **not V32 again**. V32 asked the pointwise conditional mean to choose the winner. V38 asks RSMR to choose the winner and uses the dense model only for cardinal zero-crossing after selection.

## 10. Why dense all-edge supervision is the right next diagnostic

V37 uses only 86-110 selected proposals per calibration fold to fit an 18-D orthogonal residual. The full TRAIN fit folds contain orders of magnitude more eligible candidate edges, each with a well-defined teacher improvement relative to the incumbent.

If cardinal value is a different estimand from ordinal ranking, there is no reason to throw away those labels merely because those candidates were not selected by the ranker. Dense supervision should reduce the identification variance of the cardinal map.

However, all-edge training introduces a distribution shift from candidate population to frozen-RSMR-selected population. V38 isolates that issue with an independent calibration fold and only a **one-dimensional selected-policy affine recalibration**.

This deliberately separates:

1. dense candidate-level cardinal identification; and
2. low-dimensional correction for post-selection distribution shift.

It avoids the high-dimensional selected-only fitting that destabilized OPVR.

## 11. V38 arms

### RSMR

Exact frozen V34 structured ranker. Sole provider of proposal identity.

### AVR

Same scalar-score affine control as V37. Tests whether scalar RSMR calibration alone works.

### DENSE

All-edge scene-equal absolute-value ridge. Trained on fit-fold candidates. It can only accept/veto the frozen RSMR winner.

### DAVR (main)

DENSE followed by a one-dimensional affine map fitted on the independent selected-policy calibration fold:

`v_DAVR = a + c * standardized(v_DENSE)`.

The calibration fold never fits a high-dimensional residual. It only corrects selected-policy scale/offset.

Every post-selection arm satisfies exact containment and same-winner identity.

## 12. Nested TRAIN causal experiment

Reuse the exact historical five-fold hash:

`3 fit folds + 1 independent value-calibration fold + 1 test fold`.

Per outer fold:

1. fit RSMR using only fit folds;
2. fit DENSE using all eligible candidate edges from the same fit folds with scene-equal total loss mass 1;
3. freeze RSMR proposal on calibration scenes;
4. fit AVR from RSMR scalar to selected teacher value;
5. fit DAVR's 1-D map from DENSE value to selected teacher value;
6. evaluate RSMR/AVR/DENSE/DAVR on untouched outer-test scenes.

Each calibration fold must contain >=64 frozen RSMR proposals.

DAVR must satisfy before fresh data:

- >=20% no-op false-intervention reduction vs RSMR;
- capture no worse than RSMR by more than 3 pp;
- >=25% catastrophe reduction;
- NegRMS no worse than RSMR;
- aggregate selected sum >=0;
- all five outer-fold selected sums >=0;
- >=64 selected and >=32 positive;
- exact subset containment / same RSMR winner when accepted.

DENSE is reported with the same gate as a causal diagnostic; it cannot rescue a DAVR failure.

## 13. V38 preregistered mechanistic branches

### Branch A: DENSE/DAVR succeeds

Then the current 19-D evidence **does contain usable cardinal value information**, and the V37 failure is primarily selected-only identification/sample inefficiency. The paper-level result becomes stronger: ordinal ranking and cardinal valuation require different supervision units.

### Branch B: DENSE value prediction improves materially, but zero-crossing/gate still fails

Then candidate-level cardinal signal exists, but selected-policy shift / zero-crossing reliability remains unresolved. The next mechanism should target distributional/selective value calibration, not RSMR ordering or feature expansion.

### Branch C: DENSE does not improve selected-proposal cardinal prediction beyond AVR and DAVR fails

Then there is stronger evidence that the current 19-D contrast representation itself is insufficient for selected absolute value. Stop the current 19-D cardinal route. The next paper-level step must introduce a **new value representation/target justified specifically for absolute incumbent-relative valuation**, not more head capacity or thresholds.

## 14. Fresh protocol if and only if TRAIN passes

V37 spent no fresh population. Permanent design exclusion remains **10700 tokens**.

New label-free seed:

`v64.3.38-eaf-icer-davr-cal500-double-fresh-v1`.

Only after nested TRAIN pass select independent `CAL500 + A500 + B500`.

CAL500 fits only the final scalar selected-policy recalibration on frozen full-TRAIN RSMR outputs. A/B independently evaluate:

`RAW / V20 / PRESERVE / RSMR / AVR / DENSE / DAVR`.

No A/B pooling. Fresh diagnostics use the **actual final selected action** after value veto. DAVR may only accept the exact RSMR proposal or preserve the incumbent.

Fresh promotion additionally requires hard tail (`0` catastrophes, worst > `-0.5`), meaningful useful-capture gain over PRESERVE, and endpoint non-inferiority to both PRESERVE and V20 on each block.

## 15. Paper-line implication

The prospective contribution is not ridge regression or affine calibration. It is:

> **operator-conditioned rank/value factorization with supervision-unit alignment** under a bounded auditable planner interface.

The hypothesis is that autonomous-planning intervention sufficiency is not one scalar property. The deployment operator contains at least two distinct estimands:

- **ordinal extremal choice**: which challenger should win? -- learned with scene-level regret-structured competition;
- **cardinal incumbent-exit value**: is that frozen winner actually better than doing nothing? -- learned from dense absolute intervention supervision and then aligned to the selected-policy distribution.

The two estimands share the same bounded evidence but need not share the same objective or sampling unit. Value cannot influence ranking, and post-selection recovery retains deterministic incumbent containment/no-fallback semantics.

This is the V38 paper-level mechanism to test. It should only survive if the preregistered nested/double-fresh evidence supports it.
