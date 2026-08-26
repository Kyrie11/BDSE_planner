# V64.3.45 PIRF result postmortem and V64.3.46 DIRP design

## Executive verdict

V64.3.45 is **engineering-valid** and supports TRAIN-level algorithm attribution. No V45.1 engineering repair is required. The uploaded code SHA256 is `d8acc5abb5d5b63c24e93d96884649c0b0a055676a3f684a1861bdffca8752ca`, exactly matching the preregistered package. The uploaded result ZIP SHA256 is `b557eef258718f8381e7373bcbcaa7bb53a6f4cf5a38c48e1cbcb815c9d493bc`.

Scientifically, V45 is **strong partial success + promotion failure**. It solves the V44 behavior-identification failure: agent-local continuous response is honestly identifiable, and plan conditioning is independently identifiable in all five folds. It also gives a genuine selective value improvement. However no V45 arm satisfies the preregistered RSMR-minus-3pp capture floor. The deterministic response mean therefore remains insufficient for absolute incumbent-exit valuation.

The strongest new cross-version evidence is that V44's response-ensemble/robust arm retains seven material opportunities that V45's more accurately learned deterministic PLAN response loses, while V45 additionally accepts one catastrophe that V44 robust rejects. The next bottleneck is consequently not “learn the mean response better.” It is the information discarded by two compressions: **conditional response distribution -> one point response**, and **future interaction trace -> one time-average scalar**.

V46 is therefore preregistered as **Distributional Interaction Response Profile (DIRP)**. It independently tests response-distribution sufficiency and temporal-profile sufficiency while freezing RSMR, QUALITY, the V44 ungated support geometry, and the V45 response-mean field.

---

# 1. V45 reliability audit: PASS

The experiment can be interpreted scientifically.

| Reliability item | Result |
|---|---:|
| uploaded code SHA256 | `d8acc5...52ca` |
| exact preregistered V45 code | **PASS** |
| frozen TRAIN | **3000/3000** |
| direct scientific scenes | **782/782 unique** |
| outer folds | **5/5** |
| V44 historical controls | **exact replay** |
| server targeted regression | **213/213 PASS** |
| independent rerun on uploaded code | **213/213 PASS** |
| RSMR sole challenger selector | **PASS** |
| same-winner containment | **PASS** |
| second-best/fallback | **none** |
| response deployment logged future | **none** |
| response target uses teacher value/improvement | **no** |
| A500/B500 fresh consumed | **no** |
| termination | **preregistered TRAIN scientific STOP** |

The V45 response supervision contains 3000 frozen TRAIN scenes and 88,318 agent-level examples. Logged agent future is used only to construct the offline nuisance response target. Runtime response observables consume current state, candidate ego trajectories and frozen parameters only.

The launcher stops immediately after the nested TRAIN fit returns nonzero. Fresh A/B selection is located later in the launcher and was never reached. The output directory contains no A500/B500 token manifests.

Therefore:

> **V45 is an engineering-valid scientific failure/partial success, not an engineering failure.**

---

# 2. Preregistered V45 GO conditions

The unchanged RSMR reference is:

- selected `502`;
- selected positive `221`;
- capture `38.5017%`;
- teacher sum `+43.2941`;
- catastrophe `28`;
- no-positive-opportunity false interventions `107`;
- NegRMS `0.355688`.

The preregistered capture floor remains

`38.5017% - 3pp = 35.5017%`.

TRAIN nested cross-fit:

| Arm | selected | positive | capture | sum Delta_T | catastrophe | no-op false | NegRMS | 5/5 sum>=0 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| RSMR | 502 | 221 | **38.50%** | +43.294 | 28 | 107 | 0.3557 | yes |
| QUALITY | 205 | 129 | 22.47% | +43.906 | 13 | 30 | 0.3127 | yes |
| CV-OCC | 220 | 120 | 20.91% | +45.208 | 13 | 43 | 0.3030 | yes |
| **LOCAL-RF** | **218** | **122** | 21.25% | **+54.580** | **9** | **37** | **0.2396** | **yes** |
| **PLAN-RF** | **217** | **121** | 21.08% | **+56.551** | **9** | **38** | **0.2402** | **yes** |

Every V45 arm passes the tail, population and five-fold nonnegative-sum sub-gates. Every arm fails only the existence/capture sub-gate.

Hence the formal decision is:

> **V45 does not promote.**

But the response layer itself can still be causally diagnosed because V45 deliberately preregistered an independent behavior-identification gate.

---

# 3. The response learner itself succeeds

V44 failed because its scene-global five-mode classifier predicted the majority yield mode on all OOF scenes. V45 replaces that target with agent-local continuous longitudinal response.

Honest OOF nuisance response MSE:

| Fold | CV | LOCAL | PLAN |
|---|---:|---:|---:|
| 0 | 0.28745 | 0.12031 | **0.11921** |
| 1 | 0.29686 | 0.12042 | **0.11925** |
| 2 | 0.32497 | 0.13409 | **0.13343** |
| 3 | 0.30116 | 0.12709 | **0.12605** |
| 4 | 0.29646 | 0.12238 | **0.12133** |
| aggregate | 0.30138 | 0.12486 | **0.12385** |

Thus:

- LOCAL beats CV in `5/5` folds;
- PLAN beats LOCAL in `5/5` folds.

This is a qualitatively different result from V44. The nuisance response estimator actually learned behaviorally predictive structure.

The ordinary all-agent PLAN-vs-LOCAL improvement is small (~0.8%), but this average is dominated by weakly interacting agents. A post-hoc diagnostic (not a retroactive promotion gate) weights OOF errors by continuous ego-agent interaction exposure. PLAN beats LOCAL in all five folds, with roughly 6--8% improvement in the high-exposure region. This makes the plan-conditioned signal credible but still incremental.

---

# 4. LOCAL-RF is a real selected-value mechanism

CV-OCC -> LOCAL-RF set rotation:

| Frozen RSMR subset | count | sum Delta_T | positive | catastrophe |
|---|---:|---:|---:|---:|
| both accept | 188 | +49.688 | 105 | 9 |
| CV-only | 32 | **-4.480** | 15 | **4** |
| **LOCAL-only** | **30** | **+4.892** | **17** | **0** |

So continuous agent-local response is not merely an abstention mechanism. It deletes a net-harmful CV population and adds a net-beneficial zero-catastrophe population.

The same conclusion holds relative to V42 QUALITY:

| Frozen RSMR subset | count | sum Delta_T | catastrophe |
|---|---:|---:|---:|
| both | 149 | +44.892 | 9 |
| QUALITY-only | 56 | **-0.987** | 4 |
| **LOCAL-only** | **69** | **+9.687** | **0** |

This is strong support for retaining the V44 ungated full-horizon occupancy layer and the V45 agent-local continuous response layer.

---

# 5. PLAN conditioning is real but not first-order sufficient

LOCAL-RF -> PLAN-RF:

| subset | count | sum Delta_T | material positive | catastrophe |
|---|---:|---:|---:|---:|
| both | 212 | +54.574 | 33 | 9 |
| LOCAL-only | 6 | +0.0058 | 0 | 0 |
| **PLAN-only** | **5** | **+1.977** | **2** | **0** |

So the plan-conditioned correction is not noise. It makes a small but favorable selective rotation and is independently supported by OOF nuisance-response MSE.

However it removes none of LOCAL's nine catastrophes, and capture remains 21.08%. It must therefore be retained only as a low-capacity response-mean component, not declared a deployment-sufficient mechanism.

The full-TRAIN fitted plan correction is also small: correction standard deviation is about `0.0254 m/s^2`, and only ~1.69% of agent rows have an absolute plan correction above `0.1 m/s^2`. Meanwhile PLAN residuals retain large tails (`~26.7% >0.25`, `~6.7% >0.5`, `~3.2% >1.0 m/s^2`). A point response is still a lossy summary.

---

# 6. Cross-version evidence: point response loses useful uncertainty

The strongest V45 diagnostic is not LOCAL vs PLAN. It is **V44 PC-OCC-ROBUST vs V45 PLAN-RF**.

| subset | count | sum Delta_T | positive | material positive | catastrophe |
|---|---:|---:|---:|---:|---:|
| both | 188 | +54.698 | 106 | 33 | 8 |
| **V44 robust only** | **34** | **+6.920** | **19** | **7** | **0** |
| V45 plan only | 29 | +1.854 | 15 | 2 | **1** |
| neither | 251 | -20.177 | 81 | 8 | 19 |

The seven material opportunities uniquely retained by V44 robust include examples such as `b95dc2615771588d` (`+1.9697`) and `366e33370f605634` (`+1.2327`). Their different fixed response modes often yield conflicting occupancy directions. Compressing behavior to one learned mean acceleration erases this decision-relevant ambiguity.

The reverse also exists: V45 PLAN uniquely recovers high-value proposals such as `a657d16d34c3519a` (`+1.7842`) that V44 robust rejects. Therefore neither old handcrafted modes nor the deterministic learned mean is individually sufficient. The missing object is the conditional response distribution itself, not a return to V44 mode heuristics.

This supports:

> **response uncertainty is not synonymous with “use CVaR.” It is information that must first be represented before any distributional valuation functional can be meaningfully tested.**

---

# 7. Remaining catastrophes expose consequence aliasing

PLAN-RF accepts nine catastrophes. Representative cases:

| scene | Delta_T | QUALITY | LOCAL | PLAN | V42 current risk |
|---|---:|---:|---:|---:|---|
| `c70954fab4a650c7` | -1.2330 | +1.721 | +1.654 | +1.663 | all zero |
| `30a3163cf0805a69` | -1.2325 | +0.090 | +0.391 | +0.447 | all zero |
| `a8e99095a8235549` | -1.2319 | +2.408 | +2.399 | +2.398 | all zero |
| `2154d495b9b657a5` | -1.2203 | +1.099 | +1.203 | +1.225 | all zero |
| `05ba4ac2db9c50d2` | -0.9894 | +0.0079 | +0.0734 | +0.0781 | all zero |

All nine have the V42 six-dimensional current-risk delta exactly zero.

V44 fixed-mode occupancy is sometimes mixed, e.g. candidate-favorable under one response and unfavorable under another. But some accepted catastrophes look candidate-favorable under nearly every simple mode. Therefore a richer response distribution may help, but it may still not be enough if the underlying response trajectory family is too low-order or if a non-agent future consequence is missing.

This is why V46 is deliberately a stopping experiment rather than an assumption that “variance will solve everything.”

---

# 8. Capture failure still contains a real material-opportunity problem

PLAN-RF misses 100 RSMR positives:

- `69` with `0 < Delta_T <= 0.001`;
- another `16` with `0.001 < Delta_T <= 0.01`;
- **15 material positives with `Delta_T > 0.2`**, total teacher value about `+15.80`.

Thus the count-level capture problem remains dominated by near-zero events, but promotion cannot be rescued by redefining them away. Fifteen material misses remain.

Material-positive acceptance among the fixed 50 material RSMR opportunities:

- QUALITY: `30/50`;
- CV-OCC: `32/50`;
- LOCAL-RF: `33/50`;
- PLAN-RF: `35/50`;
- V44 ROBUST: `40/50`.

V45 improves the mechanism layer yet still loses material recall relative to the V44 response ensemble. This is exactly the pattern expected from point-estimate response collapse.

---

# 9. Model-layer maturity after V45

| Layer | Status after V45 | Action |
|---|---|---|
| B16/M24 bounded interface | mature | freeze |
| EAF complete frontier | mature / paper backbone | freeze |
| exact action-local attribution | mature | freeze |
| acquisition/capacity visibility | first-order closed | no B/M sweep |
| support/admissibility | mature | freeze |
| RSMR ordinal challenger ranking | most mature learned layer | permanently freeze |
| incumbent/null + no fallback | mature | freeze |
| selected residual/tail existence | established | no longer an open question |
| EPV endpoint geometry | genuine partial mediator | retain underneath valuation |
| current QUALITY consequence | genuine partial mediator | retain |
| V44 ungated full-horizon interaction support | strongly supported | retain |
| agent-local continuous response mean | **identified and useful** | retain |
| plan-conditioned mean correction | **identified but incremental** | retain low-capacity form |
| response distribution | immature / missing | V46 first target |
| temporal interaction profile | immature / compressed to mean | V46 second target |
| absolute zero | immature | still deployment bottleneck |
| material opportunity recovery | improving but incomplete | must improve without harming tail |

The model should no longer spend research capacity relearning “which challenger” or “is there any future interaction support.” It should learn the **distribution and temporal shape of the consequence induced by the already-frozen challenger**.

---

# 10. Evidence chain V34 -> V45

The recent progression now forms a coherent falsification chain:

1. **V34 RSMR:** solves the ordinal extremal selection layer; freeze which-challenger ranking.
2. **V37--V39:** selected-policy residual/tail structure is real; simple zero crossing is not enough.
3. **V40:** pure 19-D target/head factorization fails; close the “just change value loss/head” route.
4. **V41:** endpoint/basepoint geometry is a genuine mediator but does not provide physical absolute value.
5. **V42:** current deployment QUALITY/RISK consequences are real mediators; current-risk support is sparse.
6. **V43:** prospective horizon is useful; fixed mean vs mean+CVaR on candidate-independent modes is largely redundant.
7. **V44:** full-horizon ungated interaction support is strongly useful; scene-global five-mode behavior posterior is unidentifiable and closed.
8. **V45:** agent-local continuous response is genuinely identifiable and improves selected value, but a deterministic point response loses ensemble-only material opportunities and does not solve absolute zero.

The bottleneck has therefore moved through **selection -> value target -> endpoint representation -> current consequence -> prospective support -> response representation**, rather than oscillating among arbitrary tricks.

---

# 11. Dominant bottleneck after V45

The narrowest supported definition is:

> **Distributional agent-response and temporal interaction-profile sufficiency for the frozen extremal proposal at the absolute incumbent-exit boundary.**

Two factors are deliberately separated:

### A. Conditional response distribution

V45 estimates only a bounded constant longitudinal response mean. Residual tails are large, and V44 ensemble-only decisions show that uncertainty can change the sign of deployment value.

### B. Temporal interaction structure

V45 rolls the response forward but then reduces the complete future occupancy trace to

`mean_t h_t`.

Two trajectories can have the same mean interaction mass but very different peak conflict, early conflict and concentration. This creates temporal aliasing before the value residual sees the observable.

The next experiment should distinguish A from B before introducing any general trajectory network.

---

# 12. Post-hoc design diagnostic: second response moment is learnable

This is **not V45 preregistered evidence and cannot promote V45**, but it determines whether V46 is a reasonable next falsification experiment.

Using the exact V45 frozen response supervision and the same outer/calibration exclusions, fit the conditional second moment `a^2` with the same low-capacity LOCAL/PLAN factorization.

| Fold | constant MSE | LOCAL m2 MSE | PLAN m2 MSE |
|---|---:|---:|---:|
| 0 | 0.70458 | 0.37545 | **0.36778** |
| 1 | 0.71473 | 0.38249 | **0.37424** |
| 2 | 0.80910 | 0.44137 | **0.43531** |
| 3 | 0.72772 | 0.38155 | **0.37415** |
| 4 | 0.71333 | 0.37838 | **0.37084** |

LOCAL beats the constant baseline in `5/5`; PLAN beats LOCAL in `5/5`.

This means V46 can test response uncertainty without changing the response target family to a high-capacity neural distribution model. If the second moment does not help selected value, that negative result will be meaningful.

---

# 13. Directions now prohibited

Based on the cumulative changelog, do not:

- change RSMR ordering or candidate generation;
- resweep B/M, top-K, candidate count, evidence capacity or acquisition;
- return to pure 19-D/endpoint value-head scaling, MLP residuals or target/loss tricks;
- add standalone catastrophe classifier or binary veto;
- tune selected zero threshold/translation to repair capture;
- restore second-best fallback;
- revive the V44 scene-global five-mode classifier using balancing/focal/temperature/more modes;
- simply combine V44 robust and V45 plan accept sets after seeing their labels;
- tune CVaR alpha/weight before identifying the response distribution;
- add occupancy/TTC distance thresholds, bandwidth sweeps or top-K interactor filters;
- feed logged future or teacher value into deployment;
- claim interventional ego->agent causality from observational nuisance-response fitting alone.

---

# 14. V46 mechanism: Distributional Interaction Response Profile

V46 preserves all mature layers and introduces only two new scientific objects.

## 14.1 Conditional second moment

Let V45's response mean for agent `j` and candidate `a` be

`mu_j(a)`.

V46 learns

`m2_j(a) = E[A_j^2 | x, xi_a]`

with the same LOCAL + zero-bias PLAN factorization and fixed `lambda=1`.

Variance is reconstructed as

`var_j(a) = max(m2_j(a) - mu_j(a)^2, 0)`.

The acceleration distribution is propagated through the fixed sigma rule:

- points `mu - sqrt(3)sigma`, `mu`, `mu + sqrt(3)sigma`;
- weights `1/6`, `2/3`, `1/6`;
- physical clipping only to the already-frozen `[-2,+0.5] m/s^2` response envelope.

This has no tuned uncertainty coefficient, probability temperature, response sample count or mode set.

An independent nuisance gate requires LOCAL second moment to beat a constant baseline and PLAN second moment to beat LOCAL in aggregate and at least `4/5` folds.

## 14.2 Temporal interaction profile

Before V45's final time average, retain the KxT ungated interaction hazard trace `h_a(t)`.

Expose exactly four lower-is-better deterministic functionals:

- `mean(h)` — exact V45 statistic;
- `max(h)` — peak conflict;
- `mean((1-tau) h)` — fixed early-weighted conflict;
- `mean(h^2)` — temporal concentration.

No learned attention or tuneable temporal weight is introduced.

---

# 15. V46 causal arms and interpretation

| Arm | Response object | Temporal object | Question |
|---|---|---|---|
| PLAN-CONTROL | V45 deterministic PLAN mean | time mean | exact V45 replay |
| **DIST-MEAN** | mean + second moment | time mean only | is point-response collapse first-order? |
| **TEMPORAL-PROFILE** | deterministic V45 mean | mean/peak/early/second | is time-average compression first-order? |
| **DIRP-JOINT** | distributional | full 4-functional profile | are both jointly required? |

Promotion uses the simplest sufficient arm, in that order.

### If DIST-MEAN succeeds

Retain response distribution; do **not** retain the larger temporal profile. The first-order missing statistic was response uncertainty.

### If TEMPORAL-PROFILE succeeds

Retain temporal structure and do not require response variance. The main error was temporal aliasing after response rollout.

### If only DIRP-JOINT succeeds

Support the stronger claim that a frozen proposal requires both conditional response uncertainty and temporal consequence structure.

### If second moment is identified but all arms fail

Close the low-order longitudinal acceleration distribution / hand-designed temporal-profile family as deployment-sufficient. The next step should be a richer **2D continuous plan-conditioned trajectory or occupancy response distribution**, or a targeted test of a remaining future non-agent consequence family. Do not enlarge the scalar value head.

### If second moment is not identified

Close acceleration-distribution modeling immediately and move to the general trajectory-response representation.

---

# 16. V46 runtime and information contract

V46 uses no new neural evidence query and does not change B16/M24.

Deployment uses only:

- current ego/agent histories already present in runtime state;
- current map/rule context;
- already generated ego candidates;
- frozen TRAIN parameters.

Logged future is used only as TRAIN nuisance-response supervision for the response moments. Teacher improvement/value is never a response target. Every outer response fit excludes its outer test fold and corresponding value-calibration fold.

The same caveat as V45 remains: logged observational data identifies a predictive conditional response model, not a fully causal intervention model. A causal ego-action/agent-response paper claim still requires frozen closed-loop/interventional confirmation after TRAIN and independent fresh gates pass.

---

# 17. V46 engineering hard gates

Before scientific attribution, the launcher must exactly reproduce:

- V45 RSMR: `502 / 221 / 107 / 28 / +43.29405361274824`;
- V45 QUALITY: `205 / 129 / 30 / 13 / +43.905547394411805`;
- V45 CV-OCC: `220 / 120 / 43 / 13 / +45.20842296723279`;
- V45 LOCAL-RF: `218 / 122 / 37 / 9 / +54.57972428889805`;
- V45 PLAN-RF: `217 / 121 / 38 / 9 / +56.55117310290402`;
- V45 response OOF MSE: `0.30137842796229286 / 0.12486025654085724 / 0.12385468573917016`;
- V45 PLAN occupancy generated by the V46 instrumentation to maximum absolute error <= `1e-10`;
- V45 fresh-unspent state.

Any mismatch is an **engineering STOP**, not a scientific result.

The unchanged TRAIN value gate remains: >=20% no-op reduction, capture within 3pp of RSMR, >=25% catastrophe reduction, non-worse NegRMS, nonnegative aggregate, 5/5 fold sums nonnegative, selected>=64, positive>=32, frozen-winner containment.

Only after TRAIN passes may V46 select independent A500+B500 with seed:

`v64.3.46-eaf-icer-dirp-double-fresh-v1`.

A/B must remain unpooled.

---

# 18. Paper-line implication

The current strongest CCF-A-level mechanism hypothesis is not “we use a variance feature” or “we use four temporal statistics.” Those are falsification instruments.

The deeper line is:

> **Selection–Valuation–Response Sufficiency:** under a bounded auditable planner interface, ordinal extremal challenger selection and absolute execution valuation require different decision-sufficient statistics. Once the extremal proposal is frozen, deployment value depends on prospective action-conditioned response/consequence information; V45 further indicates that a point response and scalar time-average consequence may be insufficient, motivating a distributional temporal interaction sufficient statistic.

The evidence chain is increasingly mechanism-level because each version freezes the previously mature operator and tests the next missing statistical object. The claim remains provisional until an arm clears TRAIN, independent A/B fresh, frozen full validation and official closed-loop evaluation.
