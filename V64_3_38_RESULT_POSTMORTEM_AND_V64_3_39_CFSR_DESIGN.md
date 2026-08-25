# V64.3.38 uploaded result audit and V64.3.39 CFSR design

## 1. Reliability decision

The uploaded V64.3.38 code archive is byte-identical to the preregistered delivery package (SHA256 `2fdd008b010600d83638d28531c9408f41e32f82d6f691654fe714a2b0c01fd2`). The uploaded output is complete for the point where the launcher stopped:

- the exact V37 failure/fresh-unspent prerequisite stage completed;
- the prerequisite EAF re-audit completed;
- server targeted regression is **178/178 PASS**;
- all five nested TRAIN folds completed;
- the TRAIN scene audit contains **782/782 unique direct scenes** with fold counts 168/131/151/170/162;
- nested value-calibration proposal counts are **97/100/98/86/110**, all above the frozen minimum 64;
- DENSE/DAVR are monotone subsets of RSMR and never change the frozen winner identity;
- the run stopped at the preregistered scientific gate with diagnosis `dense_all_edge_value_does_not_improve_selected_absolute_value_identification_or_capture_tradeoff`;
- no CAL500/A500/B500 token manifests were created, therefore V38 did not spend fresh data.

No V38.1 engineering hotfix is justified. V38 is valid for **nested TRAIN-level algorithm attribution only**.

## 2. V38 result against the preregistered contract

Frozen V34 RSMR remains:

- selected 502;
- positive 221;
- precision 44.02%;
- useful-opportunity capture 38.50%;
- selected teacher-improvement sum +43.2941;
- catastrophe count 28;
- no-positive-opportunity false interventions 107;
- NegRMS 0.3557.

Raw DENSE all-edge value gives:

- selected 263;
- positive 138;
- precision 52.47%;
- useful capture 24.04%;
- sum +12.2184;
- catastrophes 23;
- no-op false 50;
- NegRMS 0.4702.

DAVR selected-policy affine recalibration gives:

- selected 398;
- positive 178;
- precision 44.72%;
- useful capture 31.01%;
- sum +34.5835;
- catastrophes 19;
- no-op false 81;
- NegRMS 0.3473.

DAVR passes the registered tail, population, and 5/5 nonnegative-fold-sum sub-gates, but fails the existence/capture gate. The allowed capture floor is `38.50%-3pp = 35.50%`; DAVR reaches only 31.01%. V38 therefore fails promotion before fresh data.

## 3. What DENSE actually learned

The most important V38 result is not its final selected count. Selected-proposal value diagnostics show:

| predictor | MSE | positive AUC | non-catastrophe AUC | zero-sign accuracy |
|---|---:|---:|---:|---:|
| RSMR score | 0.5798 | 0.5928 | 0.4816 | 0.4402 |
| AVR | 0.5182 | 0.4394 | 0.5272 | 0.4382 |
| **DENSE** | 0.5478 | **0.6528** | **0.3261** | **0.5857** |
| DAVR | 0.5294 | 0.4374 | 0.5754 | 0.4761 |

Dense all-edge supervision therefore recovers a real **ordinary cardinal sign signal** from the same 19-D incumbent-contrast representation. Positive AUC improves by about 6.0 pp over RSMR and zero-sign accuracy improves by about 14.5 pp. This rules out the strongest version of the V37 interpretation that the 19-D evidence contains no usable cardinal information.

However, the same DENSE predictor anti-orders the catastrophic tail. Its non-catastrophe AUC is only 0.326. Of the 28 RSMR catastrophes, DENSE accepts **23**, while deleting only 5. The mean DENSE prediction on true-positive RSMR proposals is about 0.057, but on catastrophes it is about **0.269**. The highest DENSE values include several known catastrophes such as `c70954fab4a650c7`, `b3c270be7e145adc`, and `d5b4a2295da55f8c`.

DENSE also deletes 239 RSMR proposals whose total teacher improvement is **+31.076**, including 83 true positives. Thus its improved ordinary sign discrimination does not translate into a deployment-quality zero crossing under extremal selection.

## 4. Why DAVR is only partial success

The five independent selected-policy affine slopes learned from DENSE are approximately:

`+0.1867 / -0.0474 / +0.2032 / -0.1971 / -0.0338`.

Three of five folds learn a **negative slope**. This is why DAVR improves non-catastrophe AUC relative to DENSE yet destroys DENSE's ordinary positive ordering: it literally reverses the DENSE value order in those folds.

DAVR reduces catastrophes 28->19, no-op false interventions 107->81, improves NegRMS 0.3557->0.3473, and obtains positive selected sum in all five test folds. Those are real gains. But it deletes 104 RSMR proposals whose net teacher improvement is still **+8.711**, including 43 positives. Capture falls by 7.49 pp. Therefore the selected-policy affine stage is not a reliable cardinal calibration mechanism; it trades one ordering for another.

## 5. Updated mechanism conclusion

V37 established:

`RSMR ranking score != absolute intervention value`.

V38 adds a second distinction:

`population-edge absolute value != selected-policy absolute value`.

A useful way to state the evidence is now:

1. RSMR learns a useful **ordinal extremal ranking estimand**.
2. Dense all-edge regression learns useful **ordinary edge cardinal sign information** from the same 19-D evidence.
3. The RSMR selection operator induces a selected-output population in which the DENSE conditional residual/tail behavior is strongly distorted.
4. A one-dimensional selected-policy affine map cannot repair this distortion without destroying the useful DENSE ordering.

If the 19-D representation were fully sufficient for the absolute value under the frozen policy and the linear conditional-value model were correctly specified, selection on an X-only RSMR policy should not create this severe conditional tail reversal. The empirical reversal therefore points to **selection-conditioned residual insufficiency**: omitted policy-output information, misspecification, or insufficient selected-policy identification remains after ordinary dense cardinal learning.

The dominant bottleneck is tightened to:

> **selection-induced conditional residual/tail distortion at the frozen RSMR output: stable identification of the selected proposal's absolute incumbent-relative value after a useful ordinal ranker, without sparse selected-only overfit or blanket abstention.**

## 6. Evidence chain from V32.1 to V38

- **V32.1:** repaired dense conditional mean contains intervention signal but is not a selection-stable extremal selector; no-op scenes and heavy selected tail dominate failure.
- **V33:** explicit incumbent/null action strongly suppresses no-op interventions but sacrifices opportunity through abstention.
- **V34:** scene-level regret alignment restores useful challenger ordering and aggregate value (`5/5` fold positive, sum +43.29), but null-action/tail reliability remains poor.
- **V35:** base-point context is a weak secondary mediator; factorized loss alone does not close the trade-off.
- **V36:** frozen-order base-point/selection-geometry reservation controls tail mainly by blanket abstention and cannot retain RSMR capture.
- **V37:** post-selection value is the correct unresolved layer; selected-only 18-D residual contains downside/tail structure but is statistically unstable with only 86-110 selected samples per fold.
- **V38:** dense all-edge supervision recovers ordinary 19-D cardinal sign information, proving V37 was not simply representation-no-signal; however catastrophic selected outputs are anti-ordered, and scalar selected-policy recalibration flips value ordering in 3/5 folds.

The new combined conclusion is stronger than either V37 or V38 alone: **central cardinal sign and selected-tail distortion are complementary statistical objects.**

## 7. New no-repeat constraints after V38

Keep all historical no-repeat constraints. In addition, do not:

- declare 19-D absolute-value sufficiency merely because DENSE positive AUC improves;
- return to DENSE/conditional-mean extremal selection;
- tune the DENSE zero threshold, RSMR threshold, ridge lambda, conformal alpha/q, or an affine-slope constraint after seeing V38;
- add a generic catastrophe classifier/tail head;
- enlarge the V37 selected-only nonlinear residual head;
- expand base-point or selection-geometry features as a first-order rescue;
- use candidate count/top-K/action blacklist gates;
- fit an unconstrained selected-policy affine map that may reverse the learned cardinal ordering;
- refit RSMR while claiming to test a pure selected-value mechanism.

## 8. V64.3.39 EAF-ICER-CFSR

Full name: **Cross-Fitted Selection Residual Recovery**.

V39 tests a new mechanism claim rather than another feature/threshold trick:

> DENSE captures the population-edge cardinal component, while the remaining error after the RSMR extremal operator is a policy-output residual. That residual must be learned on honest selected-policy outputs. V37 had too few such outputs; V39 generates a denser selected-policy residual training population by inner cross-fitting, without touching the outer test fold.

### 8.1 Frozen ordinal and dense cardinal base

RSMR remains the sole challenger selector:

`b_hat = argmax_{b:u_b>0} u_b`.

DENSE remains the corrected V32.1 scene-equal all-edge value estimator, but never participates in ranking.

### 8.2 Honest inner cross-fitted selected residuals

For every outer fold, keep the historical `3 fit + 1 calibration + 1 test` split. Within the three fit folds, perform a three-way inner cross-fit. For each inner held-out fit fold:

1. train inner RSMR and DENSE on the other two fit folds;
2. freeze the inner RSMR winner on the held-out fit fold;
3. record the honest residual

`e = Delta_T(b_hat) - DENSE^{(-h)}(x_{b_hat})`.

Each residual target is therefore generated by models that did not train on that scene. V39 requires at least `3*64 = 192` such OOF selected proposals per outer fit, otherwise it fails closed before scientific attribution.

### 8.3 Orthogonal selection-residual correction

Fit a fixed-lambda linear residual correction on the OOF selected proposal evidence. The residual coefficient is constrained to the orthogonal complement of the **final outer-fit RSMR direction and DENSE direction** expressed in a common selected-residual standardized coordinate system.

This structural constraint prevents the new component from simply relearning either:

- the ordinal RSMR score, or
- the dense pointwise cardinal scalar.

The raw corrected value is:

`v_CFSR_raw = v_DENSE + g_selection_residual(x_b_hat)`.

The RSMR winner identity remains frozen before this value readout.

### 8.4 Translation-only selected-policy zero alignment

V38 showed that an unconstrained affine selected-policy map often learns a negative slope and reverses DENSE's useful ordering. V39 therefore gives the independent calibration fold only one degree of freedom:

`v_main = v_raw + c`,

where `c = mean(Delta_T - v_raw)` on frozen RSMR calibration proposals.

This is **unit-slope translation only**. It can align the zero point but cannot reorder value across scenes or flip signs by reversing the axis.

### 8.5 Causal arms

- `RSMR`: frozen ordinal baseline;
- `DENSE`: V38 all-edge cardinal value;
- `DENSE-SHIFT`: DENSE plus translation-only selected-policy calibration; answers whether V38 mainly failed because unconstrained affine slopes destroyed DENSE ordering;
- `CFSR-RAW`: DENSE plus cross-fitted orthogonal selected residual, no calibration shift;
- `CFSR-MAIN`: CFSR-RAW plus independent translation-only zero alignment.

If DENSE-SHIFT passes and CFSR does not add value, the mechanism must be simplified and the residual model discarded. If DENSE-SHIFT fails but CFSR passes, this supports a genuine feature-dependent **selection residual** beyond ordinary dense cardinal value. If both fail despite a sufficiently large OOF residual population, the current linear 19-D selected-value route should be closed rather than enlarged.

### 8.6 TRAIN gate

CFSR-MAIN keeps the same no-abstention-rescue contract as V38:

- >=20% no-op false-intervention reduction vs RSMR;
- useful capture >= RSMR - 3 pp;
- >=25% catastrophe reduction;
- NegRMS no worse than RSMR;
- aggregate selected sum >=0;
- all five test-fold sums >=0;
- selected >=64, positive >=32;
- every accepted action is exactly the frozen RSMR winner;
- inner OOF selected residual population >=192 in every outer fit;
- residual correction numerical cosine to the removed RSMR/DENSE span <=1e-8.

No threshold/lambda/feature/temperature sweep is permitted.

## 9. Fresh protocol

V38 did not consume fresh data. Permanent design exclusion remains **10700 tokens**. V39 uses a new label-free seed:

`v64.3.39-eaf-icer-cfsr-cal500-double-fresh-v1`.

Only after the complete nested TRAIN gate passes may the launcher select independent `CAL500 + A500 + B500`. CAL500 estimates translation only. A/B remain independent and unpooled, evaluating `RAW / V20 / PRESERVE / RSMR / DENSE / DENSE-SHIFT / CFSR-RAW / CFSR`.

The paper-level candidate contribution is **operator-conditioned ordinal/cardinal/residual factorization with cross-fitted policy-output adaptation** under the bounded auditable planner interface, not the particular ridge solver or a post-hoc threshold.
