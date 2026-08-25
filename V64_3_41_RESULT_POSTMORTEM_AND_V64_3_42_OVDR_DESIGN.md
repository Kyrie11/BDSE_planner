# V64.3.41 result postmortem and V64.3.42 OVDR design

## Executive decision

V64.3.41 is **engineering-valid and attributable at nested TRAIN level**. No V41.1 hotfix is required. The uploaded code archive exactly matches the preregistered V41 package (SHA256 `3bd2dfb9bf81687adde1f7fec008fdf6a77aee71666d13727802c4c54ef591c2`). All five outer folds completed, the frozen-winner contract held, `195/195` targeted tests passed on the server, the run stopped at the preregistered scientific TRAIN gate, and no CAL500/A500/B500 population was selected or consumed.

The V41 mechanism is **partially successful but not promotable**. Endpoint/basepoint-conditioned geometry is not a null result: EPV-RAW creates a much cleaner high-precision intervention core than the pure-delta controls. However, neither EPV-RAW nor its unit-slope selected translation closes the preregistered capture/tail contract. V41 therefore supports a stronger conclusion than “basepoint does not work”: **endpoint geometry contributes independent selected-value information, but the existing EAF-derived endpoint state still lacks a transferable absolute incumbent-exit scale and catastrophic consequence observable.**

This triggers the preregistered next branch: stop expanding value heads/targets over the existing EAF representation and introduce a genuinely value-specific deployment observable. V42 implements **Observable Value Decomposition Recovery (OVDR)**.

---

## 1. Reliability and completeness audit

### 1.1 Code identity

Uploaded V41 code SHA256:

`3bd2dfb9bf81687adde1f7fec008fdf6a77aee71666d13727802c4c54ef591c2`

This is exactly the preregistered V41 archive delivered after V40 analysis. There is no server-side version drift.

### 1.2 Nested protocol completeness

The run contains all five nested outer folds. The TRAIN scene audit contains exactly `782` unique direct scenes. Independent value-calibration proposal counts are:

`97 / 100 / 98 / 86 / 110`

all above the preregistered minimum of `64`. Hence V41 is not a V33-like case where the intended mechanism was never statistically instantiated.

The V41 runtime enforces the preregistered topology:

1. RSMR selects at most one frozen challenger.
2. ZDELTA/DNL/EPV may only accept that same challenger or return to incumbent.
3. No arm can re-rank challengers, create a new proposal, fall through to second-best, or use fallback to rescue a rejected proposal.

The nested report records `monotone_frozen_winner_contract_valid=true`.

### 1.3 Stop semantics and fresh status

The run stops with:

`endpoint_potential_adds_tail_signal_but_zero_crossing_still_fails`

at the nested TRAIN scientific gate. It does not crash, OOM, fail schema validation, or stop because of insufficient calibration data.

No non-empty CAL500/A500/B500 token manifests exist. Therefore the permanent 10700-token design exclusion remains valid and V41 supports only TRAIN-level scientific attribution.

---

## 2. V41 preregistered branch results

| TRAIN cross-fit | RSMR | ZDELTA | DNL | EPV-RAW | EPV-MAIN |
|---|---:|---:|---:|---:|---:|
| selected | 502 | 227 | 317 | 203 | 397 |
| positive | 221 | 128 | 124 | 118 | 184 |
| precision | 44.0% | 56.4% | 39.1% | **58.1%** | 46.3% |
| useful capture | **38.50%** | 22.30% | 21.60% | 20.56% | **32.06%** |
| teacher sum | **+43.294** | +10.152 | +30.719 | +25.969 | **+40.363** |
| catastrophe | 28 | 22 | 17 | **15** | 22 |
| no-op false intervention | 107 | 43 | 60 | **39** | 77 |
| NegRMS | **0.3557** | 0.5017 | **0.3338** | 0.4543 | 0.3656 |
| all 5 folds sum nonnegative | yes | no | no | **yes** | **yes** |

The preregistered capture floor remains:

`38.50% - 3 pp = 35.50%`.

Neither EPV arm reaches it. Therefore V41 does not pass promotion.

---

## 3. What each branch actually falsifies or supports

### 3.1 ZDELTA: zero-preserving topology alone is insufficient

ZDELTA is the clean control for the possibility that previous dense value heads failed merely because centering/intercept destroyed the semantic zero. It enforces identity-zero and antisymmetry while retaining the same 19-D delta.

It improves point sign accuracy, but selected sum falls to `+10.15`, capture to `22.3%`, catastrophes remain `22`, and NegRMS worsens to `0.502`.

Therefore the dominant failure is not merely an intercept / null-origin implementation issue.

### 3.2 DNL: generic nonlinear delta is not the missing first-order mechanism

DNL uses only delta plus an odd signed-square term. It can improve aggregate magnitude (`+30.72`) and catastrophe count (`17`), but capture remains `21.6%`, no-op false interventions remain `60`, and not all folds have nonnegative selected sum.

Thus “the linear delta is too weak; add generic nonlinear capacity” is not supported as a first-order answer.

### 3.3 EPV-RAW: endpoint/basepoint interaction is a real mediator

EPV-RAW is the strongest conservative arm:

- precision `58.1%`;
- catastrophe `15`;
- no-op false `39`;
- selected sum `+25.97`;
- all five outer folds nonnegative.

Its selected-proposal value diagnostics are also meaningfully different from ZDELTA/DNL:

- positive AUC `0.637`;
- zero-sign accuracy `0.625`;
- non-catastrophe AUC `0.389`.

The preregistered branch did **not** classify this as a clean cardinal-signal win over ZDELTA because EPV positive AUC is not sufficiently above ZDELTA (`0.651`). The supported claim is narrower: endpoint interaction adds selected-tail / selective-set structure.

A particularly strong set-level comparison is EPV-RAW versus ZDELTA on the 502 frozen RSMR proposals:

- accepted by both: 147 proposals, `+9.858`, 15 catastrophes;
- **EPV-only**: 56 proposals, **`+16.111`, 0 catastrophes**;
- ZDELTA-only: 80 proposals, only `+0.294`, 7 catastrophes;
- neither: 219 proposals, `+17.031`, 6 catastrophes.

This is direct evidence that the endpoint interaction is not simply redundant capacity. It selects a qualitatively better additional subset than the pure-delta zero-consistent control.

### 3.4 EPV-MAIN: selected translation exposes the unresolved absolute-scale problem

Independent calibration learns a unit-slope translation. Fold biases are:

- fold 0: `-0.0413`;
- fold 1: `+0.1330`;
- fold 2: `+0.1018`;
- fold 3: `+0.1655`;
- fold 4: `+0.0725`.

Four of five folds shift the zero boundary upward. This restores many interventions:

- selected `203 -> 397`;
- capture `20.56% -> 32.06%`;
- sum `+25.97 -> +40.36`.

But it simultaneously re-admits failures:

- catastrophes `15 -> 22`;
- no-op false `39 -> 77`;
- NegRMS returns near RSMR.

The translation is therefore not “calibration completing a good value model.” It is moving along an unresolved coverage-tail frontier.

---

## 4. Representative scene-level failure modes

### 4.1 Catastrophic selected proposals remain assigned positive EPV value

Examples include:

- `2b32a9f406845f75`: no positive opportunity, true improvement `-3.8065`, EPV-RAW value `+0.1002`;
- `5693aed0af7e548c`: no opportunity, true `-2.0204`, EPV-RAW `+0.0839`;
- `c70954fab4a650c7`: no opportunity, true `-1.2330`, EPV-RAW `+1.7208`;
- `a8e99095a8235549`: no opportunity, true `-1.2319`, EPV-RAW `+2.4031`.

These are not near-zero calibration mistakes. The existing endpoint evidence can be confidently wrong about large physical downside.

### 4.2 EPV-RAW also misses large positive proposals just below zero

Examples:

- `7418da8c04e85efb`: true `+4.0080`, EPV `-0.0102`;
- `054033a129155c88`: true `+3.9941`, EPV `-0.0087`;
- `30c657863dc05485`: true `+2.0203`, EPV `-0.0357`;
- `b95dc2615771588d`: true `+1.9697`, EPV `-0.0114`.

This combination is diagnostically important: the same representation can put a `-1.23` catastrophe at `+2.4` while putting a `+4.0` opportunity at approximately zero. That is not a scalar threshold problem.

---

## 5. Updated evidence chain from V32.1 through V41

The recent evidence chain now supports a layered view rather than a single “selector is bad” diagnosis.

1. **V32.1**: repaired scene-equal conditional mean has real signal, but using pointwise cardinal estimates as the extremal selector produces no-op interventions and a heavy selected tail.
2. **V33**: explicit incumbent/null action is necessary; it can suppress no-op intervention, but the all-rivals objective obtains safety by excessive abstention.
3. **V34 RSMR**: regret-aligned scene-level argmax training restores high-value ordinal challenger ordering (`+43.29`, 5/5 positive folds). This is currently the most mature learned decision layer and should stay frozen.
4. **V35**: pure loss factorization does not solve the tradeoff; absolute basepoint information provides a weak secondary signal.
5. **V36**: clean frozen-order basepoint/selection-geometry reservation reduces tail largely by blanket abstention, so scene-common reservation is not the first-order answer.
6. **V37**: selected-only value residual has tail information but is high variance under ~100 selected samples/fold.
7. **V38**: dense all-edge supervision proves the 19-D representation contains ordinary edge cardinal sign signal, but that population value does not transfer to the selected extremal tail.
8. **V39**: honest cross-fitted selected residual is real and can remove substantial tail with little aggregate-value loss, but signed-mean zero crossing remains poor.
9. **V40**: hurdle/sign/upside/downside factorization fails even with adequate selected positive/non-positive samples. This closes “keep changing the target/head on the same pure-delta 19-D value route.”
10. **V41**: endpoint/basepoint geometry adds independent high-quality subset/tail information beyond pure delta and generic nonlinear delta, but does not provide a stable absolute incumbent-exit scale.

The current hierarchy is therefore:

- relative challenger ordering: relatively mature;
- ordinary edge cardinal value: partially learnable;
- endpoint-conditioned selected value: improved but insufficient;
- deployment-relevant catastrophic consequence / absolute zero: dominant immature layer.

---

## 6. Updated dominant bottleneck

The dominant bottleneck is now best stated as:

**value-specific observable sufficiency for the frozen extremal proposal, especially observable physical downside at the absolute incumbent-exit boundary.**

The key distinction is no longer merely “rank versus value.” V41 says the existing endpoint evidence can improve selection structure but still cannot explain large teacher-value reversals. This points to information that is not represented by a richer algebraic transformation of the same EAF endpoint state.

A candidate replacement can have similar EAF endpoint statistics while differing sharply in actual consequences such as:

- route deviation;
- progress deficit;
- trajectory comfort;
- current-agent proximity / collision severity;
- TTC;
- off-route severity;
- red-light interaction.

These are not additional neural queries. They are direct deployment-time consequences of the already generated candidate trajectories in the current scene.

---

## 7. V42 mechanism: Observable Value Decomposition Recovery (OVDR)

### 7.1 Core principle

V42 does **not** add these observables to RSMR and does not allow them to choose a challenger.

The decision remains:

1. RSMR selects one frozen proposal `b_hat`.
2. EPV estimates the endpoint-state component of its value.
3. A value-specific observable residual estimates the part of teacher improvement explained by explicit deployment consequences.
4. The same frozen proposal is accepted iff the resulting value is positive; otherwise incumbent is preserved.

Thus every V42 value arm remains a subset of the exact same RSMR proposal path.

### 7.2 New label-free deployment-observable blocks

#### QUALITY block

The first block mirrors the deployment-observable portion of the teacher base cost using the same configured weights/scales:

- route deviation cost;
- relative progress-deficit cost;
- global comfort cost.

The demonstration/imitation term is **excluded**, because it requires the label-only logged future.

#### RISK block

The second block uses continuous current-map/current-agent runtime consequences already computable at deployment:

- hard agent risk;
- soft agent risk;
- agent TTC risk;
- hard off-route risk;
- soft off-route risk;
- red-light risk.

These are distinct from a hard candidate-count or binary-safety gate. They measure physical severity continuously.

### 7.3 Counterfactual value semantics

All nine observables are lower-is-better candidate costs. For incumbent `i` and the frozen candidate `b`, V42 uses:

`delta_obs = cost(i) - cost(b)`.

Hence positive means the candidate improves the observable consequence. The representation is exactly zero when candidate equals incumbent and changes sign under endpoint exchange.

### 7.4 Additive residual, not naive concat

V42 does not refit a large monolithic `[EAF, quality, risk]` head. It first fits the preregistered EPV component, then fits only the residual:

`teacher_improvement - EPV_prediction`

from each observable block using scene-equal all-edge ridge with fixed `lambda=1` and zero bias.

The causal arms are:

- `EPV-RAW`: V41 control;
- `EPV+QUALITY`: tests whether trajectory-quality consequences are the missing cardinal mediator;
- `EPV+RISK`: tests whether current physical downside is the missing tail mediator;
- `EPV+JOINT`: tests complementarity;
- `OVDR-MAIN`: JOINT plus independent unit-slope selected-policy translation.

No arm may re-rank RSMR challengers.

---

## 8. V42 preregistered diagnosis branches

### Branch A: QUALITY succeeds, RISK does not

Interpretation: missing ordinary trajectory-quality consequence is first-order. Keep the quality block and discard the risk expansion.

### Branch B: RISK succeeds, QUALITY does not

Interpretation: selected catastrophic failure is principally a physical downside-observability problem. This is the strongest support for a value-specific safety-consequence contribution.

### Branch C: JOINT succeeds while neither single block closes the gate

Interpretation: selected value is jointly mediated by quality and risk consequences; the decomposition, rather than any single feature, is necessary.

### Branch D: observables improve value/tail diagnostics but still miss capture or zero crossing

Interpretation: current observable partition is real but incomplete. The next mechanism must target the remaining **future-sensitive** component (for example future agent-response / robust teacher evidence), not expand another EAF/MLP head.

### Branch E: QUALITY, RISK and JOINT add no independent gain over EPV

Interpretation: close this current deployment-observable partition. Move directly to a genuinely new future-sensitive value observable / uncertainty mechanism.

---

## 9. V42 scientific gate

The promotion gate is intentionally unchanged:

- no-op false-intervention reduction >= 20% versus RSMR;
- useful capture no worse than RSMR by more than 3 percentage points;
- catastrophe reduction >= 25%;
- NegRMS no worse than RSMR;
- aggregate selected teacher improvement >= 0;
- all 5 outer test folds selected sum >= 0;
- selected >= 64;
- positive selected >= 32.

There is no lambda/threshold/alpha/top-K/candidate-count/capacity sweep.

The V42 TRAIN replay also has a hard engineering guard: after adding observable instrumentation, the frozen RSMR aggregate must exactly reproduce the historical V34/V41 signature. If it does not, the fitter raises an **engineering stop** and no scientific attribution is permitted.

---

## 10. Should nuPlan closed-loop be run now?

### Official promotion closed loop: not yet

V41 failed TRAIN and has not passed double fresh. An official paper-facing closed-loop result now would mix an immature value layer with deployment effects and could tempt post-hoc tuning. It must not replace the preregistered TRAIN / double-fresh gate.

### Small diagnostic closed loop: yes, now useful

The situation is different from early versions because several upstream layers are already comparatively mature and frozen:

- bounded B16/M24 interface;
- EAF complete frontier and exact attribution;
- support/admissibility;
- RSMR ordinal challenger selection;
- structural delegation;
- incumbent default/no fallback.

The unresolved layer is now narrow: intervention value at the frozen proposal boundary. A small paired closed-loop diagnostic can therefore answer questions that open-loop teacher labels cannot:

1. Do the many near-zero open-loop capture misses materially reduce closed-loop progress?
2. Do open-loop catastrophes correspond to actual collision/TTC/drivable/progress degradation?
3. Does the high-precision EPV core improve real closed-loop safety but over-abstain?
4. Do new observable-value corrections change physically meaningful outcomes rather than only teacher-score accounting?

V42 includes `RUN_V64_3_42_DIAGNOSTIC_NUPLAN_CL20.sh`. It selects 20 scenarios deterministically and label-free from the **already permanent-design-excluded** validation population and compares V20, PRESERVE, RSMR, EPV-RAW and OVDR-RAW. The script marks the result `diagnostic_only`; it must not alter V42 gates or hyperparameters.

Official closed-loop should follow only after double fresh + one frozen full-validation reproduction.

---

## 11. Directions that remain prohibited

V42 continues to prohibit all previously falsified/rescue directions, including:

- learned acquisition / selector-v2 / coreset / beam / swap;
- broad B/M or capacity sweeps;
- FCR/global reconstruction;
- DRC/KNN/type/family/radius tuning;
- PTMC/classifier resurrection;
- support threshold, action blacklist, candidate-count gate, top-K gate;
- ridge/alpha/q/value-threshold/temperature sweeps;
- larger selected-only OPVR/CFSR/SDFR heads or MLPs;
- further endpoint polynomial expansion;
- naive `[candidate, incumbent, delta, observables]` concat into the ranker;
- retraining RSMR while claiming to test value only;
- second-best fallback;
- fresh A/B pooling;
- using diagnostic closed-loop to tune V42.

---

## 12. Paper-level hypothesis

If supported by V42 and independent validation, the paper-level mechanism is not “nine more features.” It is:

**operator-conditioned value sufficiency requires explicit decomposition between latent endpoint-state value and directly observable trajectory consequences, while ordinal challenger selection remains causally separated and frozen.**

A compact mechanism stack is:

`bounded auditable evidence -> exact action-local attribution -> regret-aligned ordinal challenger selection -> freeze proposal -> endpoint latent value + deployment-observable consequence residual -> incumbent containment -> independent validation/closed loop`.

This provides a falsifiable route to distinguish representation insufficiency from downstream calibration tricks and keeps every additional signal tied to an interpretable physical consequence of the deployed trajectory.
