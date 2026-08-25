# V64.3.36 uploaded result postmortem and V64.3.37 PVR design

## Executive decision

The uploaded V64.3.36 EAF-ICER-SGRR result is **engineering-valid for nested TRAIN-level algorithm attribution**. No V36.1 engineering hotfix is warranted. The run stopped at the preregistered TRAIN scientific gate and did not generate/consume CAL500/A500/B500.

V36 is a **partial mechanism success but promotion failure**: the frozen-order reservation arms strongly reduce no-opportunity false interventions and catastrophes, but obtain this mainly through broad abstention and destroy the useful-opportunity capture that V34 RSMR had recovered. The dominant bottleneck therefore moves from scene-common reservation to **proposal-conditioned absolute intervention value/sign after extremal ranking**.

The next version is V64.3.37 EAF-ICER-PVR. It freezes the exact V34/V36 RSMR winner before any new estimator acts. A scalar-affine value control (AVR) tests whether the RSMR score merely needs calibration; the main OPVR arm reads the selected proposal's existing 19-D evidence only in directions orthogonal to the frozen RSMR score direction and predicts absolute teacher improvement. Both arms can only accept the exact frozen winner or return the incumbent.

---

## 1. Source/paper scope retained

The current manuscript frames the problem as decision-sufficient evidence under a fixed auditable planner-interface budget, with complete EAF attribution and monotone/no-fallback intervention containment. That backbone remains aligned with the current experimental program. The PTMC-specific estimator section is historically outdated relative to V29-V36, but the paper-level invariants that should remain frozen are:

1. fixed bounded evidence/query interface;
2. exact EAF action-local attribution of already selected evidence;
3. deployment-admissible direct incumbent-replacement population;
4. no new action path from reliability/refinement modules;
5. fail-closed development diagnostics and untouched independent promotion evidence.

V37 changes neither the evidence budget nor candidate acquisition. It changes the statistical object estimated after the extremal winner is frozen.

---

## 2. Engineering validity audit of V36

### 2.1 Uploaded execution reached the intended scientific STOP

Uploaded output contains only prerequisite/regression and nested V36 fitter artifacts. Stage timing is:

- exact V35 failure reproduction + fresh-unspent guard: 0 s;
- prerequisites/regression: 17 s;
- nested SGRR TRAIN gate: 56 s.

There is no non-empty CAL500/fresh token manifest. Therefore only TRAIN-level mechanism claims are allowed; fresh data are still untouched.

### 2.2 Regression/runtime contracts

Server targeted regression is **167/167 PASS**. Independent static/runtime audit found no schema, candidate-population, objective-scale, fold-leakage, stale-model, query-budget, structural-delegation, or no-fallback bug affecting the nested result.

The five selected-policy reservation calibration folds contain:

`97 / 100 / 98 / 86 / 110` proposals,

all above the preregistered minimum 64. Fit/cal/test roles are scene-disjoint by the frozen fold hash.

Every fold reports:

- monotone subset valid;
- frozen winner identity valid.

### 2.3 Exact RSMR replay check

The V36 RSMR baseline was compared against the bundled V34 782-scene audit. For all 782 direct scenes, the following match exactly:

- outer fold;
- candidate count;
- positive-opportunity label;
- selected action;
- selected score;
- selected teacher improvement.

Maximum selected-score absolute difference is **0.0**.

This rules out a hidden V34→V36 ranker drift as an explanation of the result.

### 2.4 One downstream checker ambiguity does not invalidate V36

The V36 fresh checker (never reached in this run) inherited a policy diagnostic that counts raw `proposal_exists` even if the reservation later vetoes that proposal. This would be ambiguous for a future V36 fresh interpretation, but it does **not** affect the uploaded V36 result because execution stopped before fresh selection/checking. V37 fixes the fresh diagnostic to use the actual final selected action.

**Engineering verdict:** V36 is reliable for TRAIN-level algorithm attribution. No hotfix-only branch is needed.

---

## 3. Official V36 nested result

| TRAIN cross-fit | V34/V36 RSMR | V36 BPR | V36 SGRR |
|---|---:|---:|---:|
| selected | 502 | 152 | 204 |
| positive | 221 | 83 | 88 |
| precision | 44.02% | 54.61% | 43.14% |
| useful opportunity capture | 38.50% | 14.46% | 15.33% |
| teacher improvement sum | +43.2941 | +25.5542 | +18.9162 |
| catastrophe <= -0.5 | 28 | 7 | 6 |
| no-positive-opportunity false intervention | 107 | 22 | 39 |
| NegRMS | 0.3557 | 0.3926 | 0.3329 |

SGRR outer-fold selected sums are:

`+4.9521 / -0.4349 / +2.5234 / +8.1516 / +3.7242`.

The preregistered `selection_geometry_tail_gain` is true, but `selection_geometry_existence_gain` is false because capture collapses by about **23.17 pp** relative to RSMR; all-fold nonnegative direction also fails due fold 1.

Therefore the official TRAIN STOP is correct:

`selection_geometry_reservation_does_not_resolve_intervention_existence_without_destroying_v34_capture`.

---

## 4. What V36 learned, and what it did not learn

### 4.1 What it learned

Both clean reservation arms can identify some scenes/proposals where leaving the incumbent is risky enough to veto:

- BPR reduces no-op false interventions `107→22`;
- SGRR reduces them `107→39`;
- BPR reduces catastrophes `28→7`;
- SGRR reduces catastrophes `28→6`.

This is genuine signal. V35's weak basepoint mediation was not imaginary, and V36 selection geometry does contain some selected-tail information.

### 4.2 What it failed to learn

The key question is not whether the arms can become conservative; it is whether they can **selectively** veto bad RSMR interventions while retaining good ones.

Among the 502 frozen RSMR proposals, SGRR acceptance rates are:

- positive proposal: **39.82%**;
- harmful proposal in an opportunity scene: **44.25%**;
- no-opportunity false proposal: **36.45%**;
- catastrophe: **21.43%**.

The harmful-opportunity acceptance rate is actually higher than the positive acceptance rate. So SGRR does not learn a clean intervention-value sign boundary.

SGRR deletes 298 RSMR proposals, including **133 teacher-positive proposals**. The deleted set has net teacher improvement **+24.3778**. Tail improvement is therefore purchased with a large amount of useful-recovery deletion.

---

## 5. Why the V36 reservation hypothesis fails

### 5.1 RSMR score is useful for ranking, not calibrated absolute value

On frozen RSMR proposals:

- AUC(score, positive outcome) = **0.5928**;
- AUC(score, non-catastrophe) = **0.4816**;
- mean score on positive proposals = **0.2089**;
- mean score on catastrophes = **0.4813**.

Thus a high structured score is not a reliable non-catastrophic-confidence signal. This is consistent with the V34 objective: the scene-max structured margin is designed to control argmax regret/order violations, not to make `u_b` an unbiased or calibrated estimate of `Delta_T(b;i)`.

The selected-policy overprediction target further exposes the mismatch:

- mean target on positives = **0.1502**;
- harmful opportunity = **0.2429**;
- no-opportunity false intervention = **0.4487**;
- catastrophe = **1.8622**.

There is real post-selection value error, but it is not sufficiently represented by the raw scalar score.

### 5.2 SGRR collapses toward score-dependent shrinkage

For SGRR:

`corr(reservation, raw RSMR score) = 0.9353`.

By contrast `corr(reservation, actual teacher improvement) = -0.0132`.

Its adjusted margin has AUC only **0.4864** for harmful-vs-nonharmful discrimination. This is exactly the behavioral signature of a broad shrinkage/abstention operator rather than a proposal-value estimator.

### 5.3 Scene-level counterexamples show aliasing

High-value positive RSMR proposals vetoed by SGRR include:

- `75fd8a14d25350e5`: score `0.0451`, teacher improvement `+6.0603`, vetoed;
- `7418da8c04e85efb`: score `0.2122`, teacher improvement `+4.0080`, vetoed;
- `7ca96f8e844f56ca`: score `0.3163`, teacher improvement `+3.2522`, vetoed;
- `d8a9f79a9b2f5af3`: score `1.2070`, teacher improvement `+3.1497`, vetoed.

Catastrophes still accepted include:

- `2b32a9f406845f75`: no positive opportunity, score `0.00342`, teacher improvement `-3.8065`, reservation `0`, accepted;
- `39346eef49ad52a6`: no positive opportunity, score `0.00039`, teacher improvement `-1.8276`, reservation `0`, accepted;
- `2154d495b9b657a5`: no positive opportunity, score `1.4060`, teacher improvement `-1.2203`, accepted.

A single nonnegative scene-common reservation cannot simultaneously keep the low-score +6.06 proposal and reject the low-score -3.81 proposal unless the relevant proposal-specific information enters the decision.

---

## 6. V32.1→V36 evidence chain and mechanism-level verdict

The recent evidence chain now has a coherent progression:

1. **V32.1 corrected MEAN:** same-scene continuous signal exists, but independent edge/mean selection creates many no-opportunity interventions and heavy tail.
2. **V33 SPCR:** explicit incumbent/null action strongly suppresses no-opportunity interventions, but average pair supervision over-suppresses opportunity recovery and worsens winner quality.
3. **V34 RSMR:** scene-level regret-aligned argmax surrogate restores opportunity/value direction; 5/5 fold sums positive and aggregate +43.29, but null-action decision remains weak.
4. **V35 FDSR/FBCSR:** pure factorization is too intervention-permissive; absolute incumbent basepoint has weak 5/5-consistent signal but is secondary, not first-order.
5. **V36 BPR/SGRR:** clean frozen-order basepoint and selection-geometry reservations can suppress tail, but neither preserves V34 capture. Selection geometry mainly becomes score-correlated shrinkage.

This closes another first-order route:

> `good RSMR ordering + scene-common nonnegative reservation` is not sufficient.

The model is no longer best described as missing only `should intervene?` scene context. The more precise failure is an **estimand mismatch**:

> the scalar used to rank challengers is being asked to also serve as the absolute value/sign of the post-selection intervention.

---

## 7. Current model-layer state

| Layer | State after V36 |
|---|---|
| bounded B16/M24 evidence/query interface | mature / freeze |
| EAF complete frontier and exact attribution | mature / paper backbone |
| capacity/evidence visibility | V30.3 closed as first-order bottleneck |
| support/admissibility population | useful and stable / freeze |
| structural delegation | mature / freeze |
| incumbent default / no fallback | mature / freeze |
| 19-D relative challenger evidence | useful for ordering |
| V34 RSMR regret-aligned challenger ranking | strongest current recovery sublayer / freeze for next causal test |
| absolute incumbent basepoint | weak secondary signal; do not expand yet |
| selection-geometry scene reservation | tail signal but not selective value; stop expansion |
| RSMR scalar absolute sign/value | immature / currently misused |
| proposal-conditioned selected value | not yet explicitly learned; current dominant layer |
| hard selected catastrophic tail | unresolved |

Dominant bottleneck:

**proposal-conditioned absolute intervention-value/sign sufficiency after data-dependent extremal challenger selection.**

---

## 8. Why V37 is not another trick/head stack

The proposed change is an operator/estimand decomposition, not a threshold or feature sweep.

The deployed decision is factorized into two distinct statistical objects:

1. **pre-selection rank estimand:** which challenger minimizes decision regret relative to rivals? — retained from V34 RSMR;
2. **post-selection value estimand:** after the policy has selected that winner, is its absolute improvement over the incumbent positive? — newly learned only on policy outputs.

The action topology remains monotone because the value estimator never participates in challenger search.

This directly addresses the empirical fact that ranking and absolute value are not interchangeable under extremal selection.

---

## 9. V64.3.37 PVR algorithm

### 9.1 Frozen proposal

Fit exact V34 RSMR on three fit folds. On calibration/test scenes define

`bhat = argmax_{b:u_b>0} u_b`.

No V37 parameter is allowed to change `bhat`.

### 9.2 AVR: score-affine value control

On independent calibration-fold RSMR proposals fit

`v_A = beta_0 + beta_1 * (u - mean(u))/std(u)`

to the actual teacher improvement `Delta_T(bhat;i)`.

The slope has fixed ridge `lambda=1`; no validation tuning. AVR answers:

> Is the remaining problem merely that RSMR score has the wrong scale/offset?

### 9.3 OPVR: score-orthogonal proposal-value residual

For the selected proposal, use the exact standardized 19-D RSMR input

`z = x_bhat / scale_RSMR`.

With frozen ranking direction `w`, form

`z_perp = z - w (w^T z)/||w||^2`.

Then fit

`e = Delta_T(bhat;i) - v_A`

with fixed-lambda ridge on centered/scaled `z_perp`:

`g(z_perp) = beta_perp^T z_perp_std`.

Final post-selection value:

`v_O = v_A + g(z_perp)`.

Because `w^T z_perp = 0`, the incremental component is structurally prevented from simply reproducing the scalar RSMR score direction.

### 9.4 Runtime operator

The runtime first freezes `bhat`. Then:

- AVR arm executes `bhat` iff `v_A>0`;
- OPVR arm executes `bhat` iff `v_O>0`;
- otherwise return incumbent.

No arm may:

- inspect/choose a second challenger after value rejection;
- create a proposal if RSMR has none;
- re-rank alternatives;
- change evidence queries;
- fall back to second best.

Thus `I_AVR subseteq I_RSMR` and `I_OPVR subseteq I_RSMR` with exact action identity on accepted interventions.

A key conceptual point is that the post-selection correction is allowed to be signed. Deterministic action containment does **not** require every reliability correction to be a nonnegative subtraction; that extra restriction was specific to V36 and is now empirically falsified.

---

## 10. V37 causal branches

The two-arm design isolates the next ambiguity:

### AVR passes and OPVR provides no robust incremental benefit

RSMR scalar ordering contained enough information; the dominant error was calibration/offset. Prefer the simpler scalar value calibration and do not headline OPVR.

### AVR fails and OPVR passes

Strong evidence that RSMR scalarization discarded proposal-specific absolute-value/sign information that remains observable in the existing 19-D evidence. This supports the rank-vs-value estimand-factorization hypothesis.

### Both fail

Stop:

- richer basepoint;
- larger scene geometry;
- linear selected-proposal residual expansion;
- threshold/ridge tuning.

Then reopen selected-proposal representation/target sufficiency rather than modifying RSMR ordering.

---

## 11. V37 nested TRAIN gate

Reuse the exact historical fold hash:

`3 fit + 1 independent selected-policy value calibration + 1 test`.

Each value-calibration fold must produce >=64 RSMR proposals. OPVR must satisfy:

- exact subset containment and winner identity;
- >=20% no-opportunity false-intervention reduction vs RSMR;
- capture >= RSMR - 3 pp;
- >=25% catastrophe reduction;
- NegRMS <= RSMR;
- aggregate sum >=0;
- all five outer-fold sums >=0;
- selected >=64;
- positive >=32.

Failure stops before CAL/fresh selection.

AVR receives the same diagnostic metrics but cannot rescue an OPVR failure; it is the scalar-calibration causal control.

---

## 12. Fresh protocol if and only if nested TRAIN passes

V36 consumed no fresh population. Permanent exclusion remains 10700 design tokens plus the frozen 3000 TRAIN exclusion for token selection.

New label-free seed:

`v64.3.37-eaf-icer-pvr-cal500-double-fresh-v1`.

Select independent:

`CAL500 + A500 + B500`.

CAL500 fits the final AVR/OPVR heads on full-TRAIN frozen RSMR policy outputs. A/B each independently evaluate:

`RAW / V20 / PRESERVE / RSMR / AVR / OPVR`.

No pooling. OPVR fresh promotion requires actual-final-action mechanism gain, zero catastrophes and worst > -0.5, useful-capture gain over PRESERVE, and endpoint non-inferiority to both PRESERVE and V20 on each block.

---

## 13. CCF-A level paper implication

The new paper-level candidate mechanism is not “ridge + orthogonal projection.” The contribution, if independently validated, is:

**post-selection estimand factorization for intervention-conditioned decision sufficiency under a bounded auditable planning interface.**

A possible final mechanism chain is:

`bounded auditable evidence`

`-> exact action-local EAF attribution`

`-> deployment-admissible contrastive regret ranking`

`-> frozen extremal proposal`

`-> proposal-conditioned absolute value recovery`

`-> deterministic incumbent/no-fallback containment`

`-> double-fresh/full-validation/closed-loop evidence`.

This strengthens the paper's original decision-sufficiency claim: sufficiency is not a property of one representation in isolation; it is relative to the downstream operator and estimand. Evidence that is sufficient for relative challenger ordering need not be sufficient after scalarization for the absolute post-selection decision to leave the incumbent.

---

## 14. Implementation validation

V37 code validation completed locally:

- Python compile: PASS;
- launcher `bash -n`: PASS;
- V37 focused: **5/5 PASS**;
- V13-V37 targeted: **172/172 PASS** in fixed three batches (`77 + 55 + 40`);
- full repository: **502/502 PASS** in fixed four file-partition batches (`71 + 109 + 172 + 150`);
- warnings: **36**, all historical warning classes; no new warning class.

The full one-shot pytest process reached 100% test progress but the harness did not return before the outer execution timeout, so the full-repository claim above is based on the deterministic four-part complete file partition, not on treating a timed-out process as a pass.
