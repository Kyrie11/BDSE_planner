# V64.3.44 PCOR TRAIN Postmortem and V64.3.45 PIRF Design

## Executive decision

V64.3.44 is **engineering-valid and scientifically attributable at TRAIN level**. No V44.1 engineering repair is required. The uploaded code exactly matches the preregistered V44 package, all frozen historical controls replay, the nested protocol is complete, the learned intervention remains a same-winner accept/veto operator, and no fresh CAL/A/B data were consumed.

Scientifically, V44 is a **strong partial success but promotion failure**. The unchanged capture floor remains 35.50%, and every V44 value arm is below it. The key result is nevertheless sharper than a generic failure:

1. the V44 **scene-global five-mode behavior posterior fails** as a behavior-identification mechanism;
2. the V44 **full-horizon ungated occupancy support strongly succeeds** as a new value-specific observable;
3. occupancy-CVaR gives a modest additional tail rotation but cannot yet be promoted as a core mechanism because the response posterior is not independently identified;
4. selected translation again improves coverage/aggregate metrics by moving the zero boundary rather than solving absolute sign identification.

The dominant bottleneck therefore tightens from “plan-conditioned response distribution” to **agent-local, action-conditioned continuous future-response/consequence representation at the absolute incumbent-exit boundary**.

V45 is designed around that conclusion: **PIRF — Plan-Conditioned Interaction Response Field**. It retains V44's successful ungated occupancy support, closes the failed scene-global categorical response path, and factorizes continuous response into a candidate-independent agent-local term and a zero-at-zero-interaction candidate-conditioned residual.

---

## 1. Engineering reliability audit

### 1.1 Version identity

Uploaded V44 code archive SHA256:

`be8e2da8525f89c369f19027aa48d1bec745261ca0be6ddfe1107e95e5f7c18a`

This exactly matches the V44 package preregistered and delivered before the experiment.

Uploaded V44 output archive SHA256:

`03c7e12916b1bd616a5526826209dbefa3dfaa8410069d2c3d86b4ad1279daea`

There is no code-version drift between preregistration and experiment.

### 1.2 Protocol completeness

- frozen TRAIN population: 3000/3000;
- direct scientific scene audit: 782/782 unique;
- nested outer folds: 5/5;
- historical independent value-calibration proposal counts remain 97/100/98/86/110;
- server V13→V44 targeted regression: 207/207 PASS;
- independent rerun on the uploaded code: 207/207 PASS;
- RSMR frozen-winner identity: preserved;
- value arms are strict subsets of the same frozen winner population;
- no reranking, second-best, or fallback rescue;
- V44 TRAIN gate stopped scientifically;
- no nonempty CAL500/A500/B500 manifests exist.

### 1.3 Historical hard replay

V44 instrumentation exactly reproduces the preregistered V43 signatures:

| arm | selected | positive | no-op false | catastrophe | sum | capture | NegRMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| RSMR | 502 | 221 | 107 | 28 | +43.2940536 | 38.5017% | 0.355688 |
| QUALITY | 205 | 129 | 30 | 13 | +43.9055474 | 22.4739% | 0.312658 |
| V43 FUTURE-MEAN | 231 | 134 | 42 | 13 | +44.8914798 | 23.3449% | 0.294140 |
| V43 FUTURE-ROBUST | 228 | 131 | 42 | 13 | +45.9189131 | 22.8223% | 0.296069 |

The new V44 instrumentation therefore does not perturb challenger ordering or historical post-selection paths.

### 1.4 Logged-future contract

V44 logged agent future is consumed only by TRAIN behavior-supervision construction. Runtime PCOR consumes current runtime state, candidate trajectories and frozen posterior parameters. Teacher improvement is not used as a behavior label. This is valid as an offline conditional-response supervision experiment and does not constitute deployment leakage.

### Reliability verdict

**PASS. V44 is an engineering-valid scientific experiment.**

---

## 2. Preregistered V44 gate result

Frozen RSMR baseline:

- 502 selected;
- 221 selected positives;
- capture 38.5017%;
- teacher sum +43.2941;
- 28 catastrophes;
- 107 no-op false interventions;
- NegRMS 0.3557.

Unchanged capture floor:

`38.5017% - 3pp = 35.5017%`.

V44 results:

| TRAIN cross-fit | RSMR | V43 MEAN | PC-REWEIGHT | PC-OCC-MEAN | PC-OCC-ROBUST | PCOR-MAIN |
|---|---:|---:|---:|---:|---:|---:|
| selected | 502 | 231 | 236 | 218 | 222 | 291 |
| positive | 221 | 134 | 139 | 124 | 125 | 141 |
| capture | 38.50% | 23.34% | 24.22% | 21.60% | 21.78% | 24.56% |
| teacher sum | +43.294 | +44.891 | +45.883 | **+60.375** | **+61.617** | **+62.974** |
| catastrophe | 28 | 13 | 13 | **9** | **8** | 9 |
| no-op false | 107 | 42 | 41 | 41 | 44 | 57 |
| NegRMS | 0.3557 | 0.2941 | 0.2910 | **0.2408** | **0.2291** | **0.2261** |
| 5/5 fold sum >=0 | yes | yes | yes | yes | yes | yes |

No V44 arm meets the 35.50% capture floor. Hence V44 cannot be promoted.

However, the new occupancy arms dominate several tail/value metrics and deserve mechanistic analysis rather than a binary “fail”.

---

## 3. PC-REWEIGHT: the behavior mechanism fails

### 3.1 OOF accuracy is exactly the majority baseline

The TRAIN behavior label distribution is:

- yield: 2557 / 3000 = 85.23%;
- cv: 351;
- brake: 54;
- nonyield: 26;
- ca: 12.

Honest OOF accuracy is exactly 85.2333%, equal to the majority baseline. Every fold reproduces its fold-specific majority accuracy. Reconstruction of the OOF predictions shows that all 3000 held-out scenes are predicted as `yield`.

The behavior head therefore does **not** demonstrate learned candidate-conditioned response identity.

### 3.2 This is not merely a capacity failure

The behavior target itself is structurally mismatched to the intended mechanism.

V44 creates one target mode for a whole scene by averaging mode ADE over all valid tracked agents. The supervision contains 88,318 valid agent instances over 3000 scenes: about 29.4 agents per scene on average, median 32. Thus the behavior label is dominated by agents irrelevant to the frozen proposal's decisive interaction.

By contrast, the model input is an ego-plan interaction-consequence vector. It asks a different question: what response modes matter for this ego plan?

So the supervision maps a **global average scene response** to **plan-specific interaction features**. The target and conditioning object are not aligned.

The discrete target is also ambiguous for a nontrivial subset. The median top-1/top-2 mode ADE gap is ~0.448 m; ~21.3% of scenes have gap <=0.1 m and ~34.5% <=0.25 m. Minority-mode examples are especially tie-heavy. Treating the argmin mode as a hard five-way truth therefore injects label noise exactly where non-yield modes should matter.

### 3.3 Value behavior confirms the posterior contributed little

V43 FUTURE-MEAN vs PC-REWEIGHT:

- both accept: 230 scenes;
- V43-only: 1 scene, teacher sum approximately -0.00024;
- PC-REWEIGHT-only: 6 scenes, teacher sum approximately +0.991;
- the remaining policy is identical.

Thus PC-REWEIGHT is nearly the V43 fixed-prior mean policy. The small value change cannot support a claim that plan-conditioned behavior was learned.

### Judgment

**Close the scene-global five-mode posterior path.** Do not repair it using class weights, focal loss, balanced sampling, temperature scaling, a larger multiclass network or post-hoc probability tuning. Those changes would optimize a supervision object that the experiment itself shows is misaligned.

This does not prove that all discrete response representations are impossible; it proves the specific scene-global five-mode identification used by V44 is not a valid first-order mechanism.

---

## 4. PC-OCC: the full-horizon occupancy mechanism succeeds

This is the strongest V44 result.

### 4.1 Selective rotation relative to V43 future mean

V43 FUTURE-MEAN vs PC-OCC-MEAN on frozen RSMR proposals:

- both accept: 149 scenes, sum +45.2103, 9 catastrophes;
- V43-only: 82 scenes, sum -0.3188, 4 catastrophes;
- **PC-OCC-only: 69 scenes, sum +15.1651, zero catastrophes**.

This is not blanket abstention. Occupancy mean discards a population with slightly negative aggregate value and four catastrophes, while adding a completely different population with +15.17 aggregate value and no catastrophe.

### 4.2 Selective rotation relative to QUALITY

QUALITY vs PC-OCC-MEAN:

- both accept: 139 scenes, sum +46.9936, 9 catastrophes;
- QUALITY-only: 66 scenes, sum -3.0881, 4 catastrophes;
- **PC-OCC-only: 79 scenes, sum +13.3818, zero catastrophes**.

Again, the new support variable creates useful decision discrimination, not merely lower coverage.

### 4.3 Material-positive recovery

Among the frozen RSMR population there are 50 material positives with teacher improvement >0.2, total +82.73.

Accepted material positives:

- QUALITY: 30/50;
- V43 FUTURE-MEAN: 33/50;
- PC-REWEIGHT: 34/50;
- PC-OCC-MEAN: **39/50**;
- PC-OCC-ROBUST: **40/50**;
- PCOR-MAIN: 43/50.

Thus the occupancy observable improves not only near-zero count behavior but recovery of genuinely valuable interventions.

### 4.4 Why this mechanism works

V43 hard/soft/TTC risk is finite-envelope and can be exactly zero before interaction enters the safety envelope. V44's bounded full-horizon potential

`1 / (1 + normalized_separation^2)`

preserves weak interaction support without introducing a new threshold or veto. This directly addresses the V42/V43 support-hole evidence.

### Judgment

**Promote V44 ungated full-horizon occupancy support as a retained mechanism for V45.** It is the clearest V44 algorithmic success and has a plausible paper-level role as pre-gate prospective interaction support.

---

## 5. PC-OCC-ROBUST: modest tail signal, not yet a core mechanism

PC-OCC-MEAN vs PC-OCC-ROBUST:

- both accept: 211 scenes, sum +58.6496, 8 catastrophes;
- mean-only: 7 scenes, sum +1.7258, 1 catastrophe;
- robust-only: 11 scenes, sum +2.9676, zero catastrophes.

This is a favorable but small rotation: robust adds about +1.24 net relative to mean and removes one catastrophe.

However, the mode posterior is behaviorally unidentified and close to a global prior. Therefore the experiment cannot cleanly attribute this increment to correct response-tail uncertainty. CVaR may be useful, but V44 does not prove it is the first-order missing functional.

### Judgment

Keep CVaR only as a secondary diagnostic concept. Do not tune alpha or the mixture weight, and do not make it the V45 headline mechanism. First identify the response representation itself.

---

## 6. PCOR-MAIN: high aggregate value does not solve the zero boundary

PCOR-MAIN improves aggregate sum to +62.974 and NegRMS to 0.226, but capture remains 24.56%.

Relative to PC-OCC-ROBUST:

- common accepted: 207 scenes, sum +59.6429, 8 catastrophes;
- robust-only: 15 scenes, sum +1.9742, no catastrophe;
- MAIN-only: 84 scenes, sum +3.3312, but only 21 positives and 63 non-positive outcomes, with one catastrophe.

The translation therefore mostly expands the intervention set. It can recover material opportunities but also admits many mild harms. This is another coverage-tail/zero-boundary movement rather than a principled absolute-value solution.

### Judgment

Do not tune or retain selected translation as the next main mechanism.

---

## 7. Capture failure decomposition

The unchanged capture requirement still fails by a large margin. Near-zero positives remain a major count source, but material misses are still scientifically important.

PC-OCC-ROBUST captures 40/50 material positives, leaving 10 material opportunities and about +11.1 teacher value unrecovered. Thus even if near-zero positives were ignored for scientific interpretation, the absolute-value layer is not complete.

Several remaining cases show response-mode disagreement: one response mode predicts interaction improvement while another predicts deterioration. These are precisely cases where action-conditioned response representation should matter.

Other accepted catastrophes are more concerning: in some scenes all simple response modes give weak/same-direction occupancy evidence while the teacher says the intervention is catastrophic. Such cases imply that **mode reweighting alone cannot be the final representation**; either the future response geometry is too simple, or another future consequence family is missing.

This motivates a factorized V45 instead of a single larger predictor.

---

## 8. Updated model-layer maturity

| layer | V44 status | next action |
|---|---|---|
| B16/M24 bounded interface | mature | freeze |
| EAF complete frontier | mature / paper backbone | freeze |
| exact evidence attribution | mature | freeze |
| capacity/evidence visibility | first-order closed | no sweeps |
| support/admissibility | mature | freeze |
| RSMR ordinal challenger ordering | most mature learned layer | freeze |
| incumbent/null + no fallback | mature | freeze |
| selected residual/tail existence | proven | not current question |
| EPV endpoint geometry | real but partial | retain as latent-value base |
| current QUALITY consequence | strong partial | retain |
| static/current RISK | partial, sparse support | do not enlarge |
| prospective horizon | proven useful by V43 | retain concept |
| full-horizon ungated occupancy support | **strong V44 success** | retain/upgrade |
| scene-global five-mode posterior | **failed** | close |
| CVaR on current mode distribution | weak/modest independent evidence | defer |
| absolute zero/capture | immature | still a primary symptom |
| material-positive recovery | improved but incomplete | continue |
| agent-local continuous response | missing | V45 target |
| candidate-conditioned agent response | missing | V45 target |

The next model should learn **which agent response is relevant locally and how the ego candidate modulates that response**, while preserving the already mature selector and support layers.

---

## 9. Evidence chain V34 -> V44

The recent experiments now form a coherent narrowing chain:

1. **V34 RSMR** establishes reliable ordinal extremal challenger selection. Challenger ordering is no longer the weakest layer.
2. **V37-V39** establish that selected-policy residual/tail structure is real, while the incumbent-exit zero remains unstable.
3. **V40** falsifies the idea that pure 19-D delta value can be rescued by target factorization or a new selected head.
4. **V41** proves endpoint/basepoint geometry supplies independent high-quality signal but is not sufficient for physical consequence.
5. **V42** proves current deployment QUALITY/RISK consequences are real mediators and exposes current-risk support holes.
6. **V43** proves prospective horizon adds real value/tail signal, while fixed mean/CVaR over candidate-independent kinematic modes is mostly redundant.
7. **V44** proves full-horizon ungated occupancy support is a strong mechanism, while the scene-global plan-conditioned mode posterior is behaviorally unidentifiable.

The scientific target has therefore moved from “find a better value head” to:

**selection is already relatively mature; the unsolved problem is deployment-sufficient future consequence representation for a frozen extremal action.**

The highest-value next experiment is not another scalar calibration. It is a response representation experiment.

---

## 10. V45: Plan-Conditioned Interaction Response Field

### 10.1 Objective

Replace the failed scene-global categorical response target with an agent-local continuous response representation, while keeping the successful V44 occupancy support and preserving the complete no-rerank/no-fallback causal topology.

### 10.2 TRAIN-only continuous target

For each valid tracked TRAIN agent, use logged agent future to fit a physically bounded constant longitudinal acceleration relative to current-state constant velocity.

No teacher improvement, teacher action label or selected-value label enters this target.

The target is clipped to the existing V43 response acceleration envelope:

`[-2.0, +0.5] m/s^2`.

This is inherited from the already frozen brake/CA response basis, not newly tuned.

### 10.3 Factorization

For agent `j`:

`a_local(j) = f_local(current/history agent state, current ego-agent geometry)`.

For candidate ego trajectory `a`:

`a_plan(j,a) = a_local(j) + g_plan(current state, candidate a)`.

Local features include speed, recent longitudinal acceleration, current ego-agent longitudinal/lateral geometry and agent dimensions.

Plan features are continuous full-horizon interaction-exposure statistics against the candidate. Every plan feature is multiplied by the same ungated interaction exposure. The plan residual has zero bias.

Therefore:

`interaction_exposure -> 0  =>  plan_correction -> 0`.

This structural condition prevents a counterfactual candidate that does not interact with an agent from arbitrarily changing the predicted agent response.

### 10.4 Runtime future and value observable

Current agents are rolled forward under the predicted continuous acceleration. The candidate is evaluated with the same V44 full-horizon ungated occupancy potential. Costs are lower-is-better; candidate value input is incumbent cost minus candidate cost.

Runtime uses no logged future.

### 10.5 Honest response identification gate

For outer fold k, response models exclude fold k and calibration fold (k+1)%5.

Report three scene-equal OOF response losses:

- CV baseline MSE;
- LOCAL response MSE;
- PLAN response MSE.

A plan-conditioned behavior claim requires:

- aggregate PLAN MSE < LOCAL MSE; and
- PLAN beats LOCAL on at least 4/5 folds.

This gate is separate from value performance. A value gain without response identification is explicitly fail-closed and cannot be presented as learned behavior response.

### 10.6 V45 causal arms

**CV-OCC**

Uses one CV response and V44 ungated occupancy support. No response learner. This tests whether V44's gain was fundamentally support geometry and whether behavior learning is unnecessary.

**LOCAL-RF**

Uses agent-local continuous response, candidate independent. This tests whether per-agent continuous behavior itself is the missing layer.

**PLAN-RF / PIRF**

Uses agent-local response plus the zero-at-zero-interaction candidate-conditioned residual. This tests whether ego plan conditioning supplies independent response/value signal.

No selected translation and no CVaR are part of the V45 primary mechanism test.

### 10.7 Promotion branch logic

Use the simplest sufficient arm, preregistered before results:

1. CV-OCC value gate passes -> retain support only; no behavior learning claim.
2. CV fails, LOCAL passes + local response identified -> retain agent-local continuous response.
3. CV/LOCAL fail, PLAN passes + plan response identified -> retain plan-conditioned response field.
4. PLAN value passes without response identification -> scientific STOP; value improvement cannot be attributed to response learning.
5. response is identifiable but no arm closes value gate -> move to a richer continuous plan-conditioned occupancy/trajectory response representation and/or an additional future consequence family.
6. response is not identifiable -> close this low-order longitudinal field; do not increase ridge/MLP capacity as a first response.

### 10.8 Statistical/causal limitation

The TRAIN future observes agent behavior only under the logged ego trajectory. Consequently V45 learns a **conditional observational response field**, not a fully identified causal intervention model for arbitrary ego plans. Applying the learned plan residual to counterfactual candidates relies on cross-scene structural generalization.

This is realistic and leak-free for deployment, but a paper must not claim causal agent-response effects solely from this open-loop supervision. If PLAN-RF survives TRAIN and independent fresh validation, frozen closed-loop/interventional evidence is required before strengthening the claim from “plan-conditioned response sufficiency” to a causal interaction-response claim.

---

## 11. Directions explicitly prohibited after V44

Do not:

- repair the V44 global mode classifier via class balancing/focal loss/temperature or a larger multiclass net;
- add more handcrafted modes as the first-order response;
- tune response probabilities;
- tune CVaR alpha or mixture weight;
- tune selected translation/threshold;
- enlarge a pure 19-D/endpoint value head;
- add a binary catastrophe veto;
- add a static/current risk block;
- modify RSMR ordering;
- use second-best fallback;
- sweep B/M, candidate count, top-K agents, occupancy threshold, kernel bandwidth or evidence capacity;
- use logged future/teacher future at deployment;
- relax capture based on high aggregate sum.

---

## 12. Paper-mainline implication

A plausible CCF-A-level mechanism is now broader than the original PTMC-specific implementation:

**Selection–Valuation–Response Sufficiency under a bounded auditable planning interface.**

The evidence so far supports a hierarchy:

`bounded EAF evidence -> ordinal extremal proposal -> freeze proposal -> endpoint/current consequence -> prospective interaction support -> response-conditioned absolute valuation -> incumbent containment`.

The novelty is not “one more future feature.” The emerging claim is that **ordinal selection and absolute deployment valuation require different sufficient statistics, and absolute value becomes response-dependent after extremal selection**. V45 tests whether that response dependence can be represented by an agent-local continuous interaction field without abandoning the bounded/no-fallback interface.

This claim remains provisional until a V45-or-later mechanism passes TRAIN, double-fresh evaluation, frozen full validation and official closed-loop evaluation.
