# V64.3.42 EAF-ICER-OVDR uploaded TRAIN result postmortem and V64.3.43 CFRV design

## Executive verdict

**Engineering reliability: PASS.** The uploaded V42 code archive has SHA256 `d29e14c0d30833d42ca1b8f70afc6f08a71183cad8b05c6e63a990acfaf216b7`, identical to the preregistered V42 package. The uploaded run completed the frozen 3000-TRAIN observable replay and all five nested outer folds, produced 782/782 unique direct scene audits, reproduced the historical frozen RSMR and V41 EPV paths exactly, passed server targeted regression `198/198`, stopped at the preregistered TRAIN scientific gate, and did not create/consume CAL500/A500/B500 fresh manifests. There is no engineering reason to invalidate V42 or create V42.1.

**Scientific verdict: substantial partial success + promotion failure.** V42 does not pass the preregistered capture floor, so it is not promoted. However `EPV+QUALITY` is a strong new mechanism: at almost exactly the same intervention count as EPV-RAW it rotates the accepted set toward higher teacher value, lowers catastrophes/no-op false interventions, improves NegRMS, preserves 5/5 nonnegative fold sums, and even slightly exceeds frozen RSMR aggregate teacher value. This is not blanket abstention and not a null result.

**Dominant bottleneck after V42:**

> **prospective response-conditioned consequence observability at the frozen RSMR winner's absolute incumbent-exit boundary.**

The strongest direct support is that 17/28 frozen-RSMR catastrophes and 21/50 material positives (`Delta_T>0.2`) have **all six current V42 RISK deltas exactly zero**. A larger current-risk head, threshold, or scalar calibration cannot recover information that is absent from the observable itself.

V43 therefore implements **Counterfactual Future-Response Valuation (CFRV)**: freeze RSMR and the V42 QUALITY core, roll current agents under runtime-only `cv / ca / brake / yield / nonyield` response hypotheses, form a prospective interaction-cost distribution for each already-generated candidate, and compare a future-mean arm with a frozen teacher-style mean+CVaR arm. No logged future, teacher label, new neural query, reranking, second-best fallback, or hyperparameter sweep is introduced.

---

## 1. V42 engineering reliability audit

### 1.1 Code identity and run completeness

- uploaded V42 code SHA256: `d29e14c0d30833d42ca1b8f70afc6f08a71183cad8b05c6e63a990acfaf216b7`;
- exact match to the preregistered V42 package: **PASS**;
- frozen TRAIN scenes replayed: **3000**;
- direct support-positive scene audit: **782/782 unique**;
- nested outer folds: **5/5**;
- independent value-calibration proposal counts: **97 / 100 / 98 / 86 / 110**, each above the fixed minimum 64;
- server targeted regression: **198/198 PASS**, only two historical Transformer warnings;
- frozen-winner monotone containment: **PASS**;
- CAL500/A500/B500 manifests: **not created / not consumed**.

### 1.2 Historical-path replay

The V42 audit was joined scene-by-scene against the packaged V41 causal audit. The following are exact:

- outer test fold identity;
- calibration-fold identity;
- candidate count;
- positive-opportunity identity;
- frozen RSMR winner action;
- EPV-RAW accepted action;
- EPV-RAW predicted value.

The maximum teacher-improvement floating difference is only about `4.44e-16`. Therefore new observable instrumentation did not perturb proposal generation, challenger ordering, or V41 endpoint valuation.

### 1.3 Scientific STOP is not an engineering crash

The fitter writes the complete nested report and then exits at `train_gate_pass=false`; the launcher consequently stops before fresh-data selection. This is the intended fail-closed behavior.

One caution: the fit report's automatic `failure_diagnosis` string is selected by a preregistered diagnostic AUC branch. It is useful metadata, but it is **not** itself the scientific verdict. The final attribution must use the gate outcomes, set rotations, scene slices, and observable-support analysis below.

---

## 2. V42 preregistered-order result

The frozen reference is RSMR. The preregistered capture floor is

`38.5017% - 3pp = 35.5017%`.

| TRAIN cross-fit arm | selected | positive | precision | capture | sum Delta_T | catastrophe | no-op false | NegRMS | 5/5 sum>=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| RSMR | 502 | 221 | 44.02% | 38.50% | +43.2941 | 28 | 107 | 0.3557 | yes |
| EPV-RAW | 203 | 118 | 58.13% | 20.56% | +25.9689 | 15 | 39 | 0.4543 | yes |
| **EPV+QUALITY** | **205** | **129** | **62.93%** | **22.47%** | **+43.9055** | **13** | **30** | **0.3127** | **yes** |
| EPV+RISK | 205 | 122 | 59.51% | 21.25% | +43.3001 | **11** | 40 | 0.3901 | yes |
| EPV+JOINT | 186 | 109 | 58.60% | 18.99% | +40.7775 | **11** | 37 | 0.4244 | yes |
| OVDR-MAIN | 285 | 129 | 45.26% | 22.47% | +41.2464 | 16 | 63 | 0.3894 | yes |

### 2.1 QUALITY branch: strong partial success, but not promotion success

QUALITY passes all of the following relative to RSMR:

- no-op false intervention: `107 -> 30`, reduction **71.96%**;
- catastrophe: `28 -> 13`, reduction **53.57%**;
- NegRMS: `0.3557 -> 0.3127`, improvement about **12.1%**;
- aggregate teacher value: `+43.2941 -> +43.9055`;
- all five outer-fold selected sums nonnegative;
- population minimum.

It fails only the intervention-existence/capture component because capture is `22.47% < 35.50%`.

This is much stronger than “QUALITY has some AUC signal”. It demonstrates that current observable trajectory consequence is a real **cardinal mediator** of selected intervention value. At equal-ish coverage (`205` vs EPV's `203`), it replaces harmful EPV decisions with higher-value decisions rather than merely abstaining.

### 2.2 RISK branch: tail-count signal is real but tail magnitude is not deployment-sufficient

RISK reduces catastrophe count to `11` and keeps aggregate sum essentially equal to RSMR (`+43.3001`). Its selected-proposal non-catastrophe AUC is the strongest among V42 arms. However NegRMS worsens to `0.3901`, so catastrophic **count** reduction is not accompanied by sufficient control of negative **magnitude**. Capture also remains only `21.25%`.

Thus the correct conclusion is not “current physical risk is the solved bottleneck”; it is narrower:

> current physical-risk geometry contains useful tail information, but its support and magnitude calibration are incomplete.

### 2.3 JOINT-RAW: naive linear composition is not additive

JOINT reduces selected count further (`186`) while losing positive capture (`18.99%`) and worsening NegRMS (`0.4244`). QUALITY and RISK are not simply two independent linear coordinates whose benefits add when fitted as one residual block.

The disagreement regions explain why. QUALITY-vs-RISK:

- both accept: 180 scenes, teacher sum `+45.2905`;
- RISK-only: 25 scenes, sum `-1.9904`;
- QUALITY-only: 25 scenes, sum `-1.3849`;
- neither: 272 scenes, sum `+1.3789`.

The shared core is excellent; the disagreement regions are net harmful. A single linear residual cannot automatically turn this pattern into a reliable zero boundary.

### 2.4 OVDR-MAIN: translation moves on the same coverage-tail Pareto front

Selected-policy translation changes JOINT-RAW from `186` to `285` interventions, but positives rise only `109 -> 129`; catastrophes rise `11 -> 16` and no-op false interventions rise `37 -> 63`. It is another coverage recovery by globally moving the zero point, not a new decision-sufficient value representation.

Therefore threshold/translation tuning is explicitly closed.

---

## 3. V42 proves that QUALITY is not blanket abstention

On the frozen 502 RSMR proposal population, compare EPV-RAW and QUALITY:

- both accept: `183`, sum `+34.5517`, 11 catastrophes;
- **QUALITY-only**: `22`, sum **`+9.3538`**, 15 positives, 4 material positives, 2 catastrophes;
- **EPV-only**: `20`, sum **`-8.5828`**, only 4 positives, 4 catastrophes;
- neither: `277`, sum `+7.9713`.

The intervention counts are nearly identical, yet QUALITY swaps a net-harmful EPV-only population for a strongly positive QUALITY-only population. This is direct evidence that the observable block changes the **value geometry**, not just the acceptance rate.

RISK shows an analogous independent rotation relative to EPV:

- RISK-only: 17 scenes, `+12.6479`, 10 positives, 5 material positives, **0 catastrophe**;
- EPV-only: 15 scenes, `-4.6833`, 4 catastrophes.

So both deployment-observable blocks contain real information absent from EPV.

---

## 4. Why capture still fails

The frozen RSMR 502 proposals have a highly asymmetric outcome distribution:

| class | count | total teacher value |
|---|---:|---:|
| catastrophe (`<=-0.5`) | 28 | -38.6649 |
| mild harm (`-0.5<y<=0`) | 253 | -1.2872 |
| near-zero positive (`0<y<=0.01`) | 167 | +0.1419 |
| small positive (`0.01<y<=0.2`) | 4 | +0.3706 |
| material positive (`>0.2`) | 50 | +82.7336 |

QUALITY misses 92 of RSMR's 221 true-positive proposals:

- 71 are near-zero positives, total only `+0.04064`;
- 1 is `0.01..0.2`, value `+0.11936`;
- **20 are material positives**, total **`+23.34784`**.

Therefore two failure sources remain coupled:

1. **count-level zero-boundary ambiguity**, dominated by near-zero positives;
2. **material opportunity generalization failure**, which cannot be dismissed as a metric artifact.

The first explains why aggregate value can look excellent while capture fails. The second prevents us from relaxing the capture gate.

---

## 5. Scene-level mechanism attribution

### 5.1 Severe false positives that current RISK cannot see

Representative QUALITY-accepted catastrophes include:

- `c70954fab4a650c7`: true `Delta_T=-1.2330`, QUALITY value `+1.7213`; all six current RISK deltas are exactly zero.
- `30a3163cf0805a69`: true `-1.2325`, QUALITY `+0.0902`; all current RISK deltas zero.
- `a8e99095a8235549`: true `-1.2319`, QUALITY `+2.4080`; all current RISK deltas zero.
- `95a810911ba359ec`: true `-1.2318`, QUALITY `+0.3068`; all current RISK deltas zero.
- `d5b4a2295da55f8c`: true `-1.2230`, QUALITY `+0.9347`; all current RISK deltas zero.

These are not small threshold errors. The current risk observable literally contains no differentiating signal for many of them.

A counterexample is `c7bff0705f645b30`: true `-1.2311`; QUALITY remains slightly positive (`+0.0215`) but RISK is strongly negative (`-0.5431`) because soft-agent/TTC deltas become nonzero. This shows that when current physical geometry *is observable*, the RISK mechanism can work.

### 5.2 Material positives that current consequence blocks still miss

Examples include:

- `30c657863dc05485`: true `+2.0203`, QUALITY `-0.0282`, current RISK all zero;
- `b95dc2615771588d`: true `+1.9697`, QUALITY `-0.0120`, current RISK all zero;
- `57a59636cde25b02`: true `+1.5958`, QUALITY `-0.0314`, current RISK all zero;
- `faf6056c372e54f2`: true `+1.2462`, QUALITY `-0.0480`, current RISK all zero;
- `842b7f0b68fc5245`: true `+1.2322`, QUALITY `-0.0180`, but RISK becomes `+0.4092`, showing a case where interaction information can rescue a missed opportunity.

The same missing-observability problem therefore appears on both sides of zero.

---

## 6. The decisive V42 support-limit result

Among frozen RSMR proposals:

- **17/28 catastrophes** have `L1(current RISK delta)=0`;
- **21/50 material positives** have `L1(current RISK delta)=0`;
- 159/253 mild harms and 88/167 near-zero positives also have zero current-RISK delta.

This changes the algorithmic diagnosis. We should no longer ask:

> How can we fit the six current risk features better?

We should ask:

> What deployment-available variable changes before the physical consequence manifests in the current snapshot?

The teacher already exposes the natural answer in its causal construction: dynamic interaction costs are evaluated under a distribution of agent-response modes, and robust aggregation includes a CVaR component. V42 RISK only observes a current deterministic geometry. The missing layer is therefore **prospective response-conditioned consequence**, not another static head.

---

## 7. Evidence chain V32.1 -> V42

1. **V32.1**: dense cardinal mean has real signal, but pointwise mean used as extremal selector creates no-op/heavy-tail failure.
2. **V33**: explicit incumbent/null action is necessary; safety obtained by blanket abstention is insufficient.
3. **V34 RSMR**: scene-level argmax-regret learning produces a strong frozen challenger ordering (`502/221`, `+43.294`, 5/5 positive). This layer is mature and stays frozen.
4. **V35/V36**: scene-common basepoint/selection reservation provides secondary signal but not a first-order solution.
5. **V37-V39**: selected-policy residual/tail structure is real; CFSR can remove half the catastrophes almost without aggregate-value loss, but the zero crossing is not learned.
6. **V40**: sign/upside/downside distributional factorization on the same pure 19-D delta still fails; pure-delta selected-value head route is closed.
7. **V41**: endpoint/basepoint potential geometry adds an independent high-quality intervention core, but severe absolute consequence sign flips remain; endpoint algebra alone is insufficient.
8. **V42**: deployment-observable trajectory QUALITY and current RISK add independent cardinal/tail information; QUALITY becomes the strongest post-RSMR value core so far. But current RISK is exactly zero on most catastrophes and many material positives, revealing an **observable support failure**.

The bottleneck has therefore narrowed monotonically from selector learning -> selected cardinal value -> endpoint representation -> **prospective consequence observability**.

---

## 8. Model-layer status after V42

| Layer | Status |
|---|---|
| B16/M24 bounded interface | mature, freeze |
| EAF complete frontier | mature, paper backbone |
| exact action-local attribution | mature, freeze |
| capacity/evidence visibility | first-order bottleneck closed |
| support/admissibility | mature, freeze |
| RSMR ordinal challenger ordering | most mature learned layer, freeze |
| explicit incumbent/null action | necessity established |
| structural delegation/no fallback | mature, freeze |
| ordinary edge cardinal sign | signal exists |
| selected tail residual | signal exists |
| endpoint/basepoint potential geometry | real independent signal, partial mature |
| **current trajectory QUALITY consequence** | **new strong cardinal mediator, partial mature** |
| current static RISK consequence | tail signal but support-sparse, immature |
| absolute selected zero/capture | immature |
| response-conditioned future downside | **unmodeled / dominant missing layer** |
| material-opportunity prospective benefit | **unmodeled / coupled missing layer** |

---

## 9. V64.3.43 EAF-ICER-CFRV

Full name: **Counterfactual Future-Response Valuation**.

### 9.1 Core hypothesis

For a frozen RSMR proposal `b_hat`, current trajectory QUALITY gives a good present-time cardinal core, but current interaction RISK is insufficient because many important scenes have zero current-risk delta. Deployment value depends on how surrounding agents may evolve over the candidate horizon.

V43 therefore constructs a label-free prospective response distribution from **current** agent state only. For each current agent, roll the already existing frozen response family:

`cv, ca, brake, yield, nonyield`.

The `logged` response mode is structurally forbidden because V43 calls `build_response_modes(runtime, label_future=None, cfg)` and fails closed if a mode reports `uses_label_future=true`.

For each candidate trajectory and each response mode, compute continuous box-aware interaction severities using the existing runtime-safety geometry:

- hard-agent proximity severity;
- soft-agent proximity severity;
- TTC severity.

These produce per-mode lower-is-better interaction costs `C_m(b)`. V43 then exposes three distribution functionals:

- `E_m[C_m(b)]`;
- `CVaR_alpha(C_m(b))`;
- `(1-eta) E_m[C_m(b)] + eta CVaR_alpha(C_m(b))`.

`alpha` and `eta` use the same frozen robust-teacher defaults/config; response-mode probabilities also come from the existing frozen response-mode configuration. They are **not tuned** in V43.

As in V42, value-oriented input is always counterfactual improvement

`Delta C = C(i) - C(b_hat)`.

The same-winner containment therefore remains exact.

### 9.2 Why this is an algorithmic mechanism rather than feature stacking

V43 does **not** concatenate another feature bank into a large head. The topology is factorized:

`RSMR selection -> EPV endpoint latent value -> frozen V42 QUALITY residual -> one prospective response functional residual -> incumbent containment`.

Only one scalar prospective distributional functional is used by each causal arm. The model family remains fixed-lambda, zero-bias, scene-equal ridge. This isolates whether the missing information is:

1. **prospective horizon** itself; or
2. **tail-aware response uncertainty**.

This is a falsifiable mechanism statement about decision sufficiency under an extremal deployment operator, not an unconstrained capacity increase.

### 9.3 Preregistered arms

1. **QUALITY-CONTROL**: exact V42 QUALITY mechanism.
2. **QUALITY+FUTURE-MEAN**: add only expected prospective interaction cost. Tests whether looking forward under response hypotheses is sufficient.
3. **QUALITY+FUTURE-ROBUST**: add only the frozen mean+CVaR response functional. Tests whether tail-sensitive response uncertainty is required.
4. **CFRV-MAIN**: FUTURE-ROBUST plus one independent CAL500 unit-slope selected-policy translation. Translation cannot rerank or reverse the value axis.

V43 explicitly does not add a separate `CVaR-only` arm because mean vs frozen mean+CVaR is already the minimal two-arm test of horizon vs tail-sensitive aggregation without proliferating degrees of freedom.

### 9.4 V43 engineering hard gate

Before scientific interpretation, V43's new 12-D instrumentation must exactly reproduce both:

**RSMR**
- 502 selected;
- 221 positive;
- 107 no-op false;
- 28 catastrophes;
- sum `+43.29405361274824`;
- capture `0.38501742160278746`.

**V42 QUALITY**
- 205 selected;
- 129 positive;
- 30 no-op false;
- 13 catastrophes;
- sum `+43.905547394411805`;
- capture `0.22473867595818817`;
- NegRMS `0.3126575113037135`.

Any mismatch is an **engineering STOP** and V43 must not be scientifically interpreted.

### 9.5 Promotion gate

Unchanged from the V42/V41 preregistration:

- no-op false reduction >=20% vs RSMR;
- capture >= RSMR - 3pp;
- catastrophe reduction >=25%;
- NegRMS no worse than RSMR;
- selected teacher sum >=0;
- 5/5 outer fold sums >=0;
- selected >=64 and selected-positive >=32;
- exact frozen-winner containment.

Only CFRV-MAIN passing nested TRAIN permits new CAL500+A500+B500 selection.

### 9.6 Falsification branches

- **FUTURE-MEAN passes but FUTURE-ROBUST fails:** future horizon observability is the missing source; the fixed teacher-style mean/CVaR functional is not deployment-sufficient. Keep the mean branch; do not tune CVaR.
- **FUTURE-ROBUST materially improves non-catastrophe/tail diagnostics but gate still fails:** runtime response distribution adds real downside information, but the handcrafted response family is behaviorally insufficient. Next step must be a plan-conditioned learned behavior/occupancy response observable, not a larger value head.
- **Neither future arm adds independent gain:** close simple kinematic response-mode valuation. Move directly to a learned interactive response distribution; do not add polynomial/current-risk features.
- **CFRV-MAIN passes TRAIN:** only then select independent CAL500+A500+B500. If double fresh also passes, freeze CFRV and proceed to one frozen full-validation reproduction and official paper-facing closed loop.

---

## 10. No-repeat constraints carried into V43

All historical no-repeat constraints remain active. In particular:

- RSMR remains the sole challenger selector;
- no 19-D/endpoint MLP or richer polynomial rescue;
- no current-RISK classifier or binary catastrophe veto;
- no threshold, lambda, CVaR alpha/weight, mode-probability, candidate-count, top-K, temperature, or selected-value cutoff sweep;
- no refit of RSMR while claiming a value-only mechanism;
- no second-best fallback;
- no A/B pooling;
- no logged future or demo label in the deployment observable;
- no official closed-loop-driven tuning before TRAIN and double-fresh promotion.

---

## 11. Paper-level implication if V43 reproduces

The paper should not be framed as “adding a CVaR risk feature”. The stronger contribution is a **selection--valuation observability separation** under a bounded auditable planner interface:

> ordinal extremal challenger selection can be decision-sufficient with bounded local evidence, while absolute deployment valuation of the frozen proposal requires a different invariant and a prospective response-conditioned consequence distribution.

The emerging factorization is:

`bounded auditable evidence -> exact EAF attribution -> ordinal extremal selection -> freeze proposal -> endpoint/current-quality cardinal core -> prospective response-distribution valuation -> deterministic incumbent containment`.

This is falsifiable: V34 supports the ordinal layer; V37-V40 falsify pure-delta selected-value sufficiency; V41 supports endpoint geometry but falsifies endpoint completeness; V42 supports current consequence mediation but exposes the current-risk support hole; V43 directly tests whether prospective response distribution closes that hole.

At present this is a **candidate paper contribution**, not a claimed result. It only becomes a paper-facing mechanism after TRAIN + double-fresh + frozen full validation + official closed-loop reproduction.
