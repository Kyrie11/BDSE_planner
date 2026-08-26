# V64.3.43 TRAIN Result Postmortem and V64.3.44 PCOR Design

## Executive verdict

**V64.3.43 is engineering-valid and supports reliable TRAIN-level algorithm attribution. No V43.1 engineering hotfix is required.**

The uploaded V43 code archive has SHA256

`a9266f271b17c77944deca4bd4c518b6cc7a803f7560ce2eca7208fb773c4e7c`

which exactly matches the preregistered V43 package. The uploaded result archive has SHA256

`3e82e1bf86c86574f8287f2ca74a1bf257dd9838739951c10eeb38731b7a9385`.

The scientific verdict is:

> **V43 is a substantial partial success, but a preregistered promotion failure.** Prospective interaction horizon is a real missing mediator: both FUTURE-MEAN and FUTURE-ROBUST improve selected-value correlation/tail discrimination and reduce NegRMS while preserving 5/5 nonnegative folds. However neither closes the capture/absolute-zero gate. The fixed mean+CVaR functional is not independently validated because mean and robust make almost identical decisions. CFRV-MAIN recovers additional material opportunities and aggregate value, but does so partly through a selected zero translation and still misses the preregistered capture floor.

The dominant bottleneck therefore tightens from V42's generic prospective observability to:

> **plan-conditioned prospective interaction consequence observability and support for the already frozen RSMR proposal at the absolute incumbent-exit boundary.**

This motivates V64.3.44 **EAF-ICER-PCOR — Plan-Conditioned Occupancy Response**. V44 does not enlarge the value head. It factorizes two structural deficiencies exposed by V43: candidate-independent response probabilities and finite-envelope/gated risk support.

---

# 1. Engineering / experimental reliability audit: PASS

The result is usable for causal algorithm attribution.

| Reliability item | V43 result |
|---|---:|
| Uploaded code SHA matches preregistered V43 | **PASS** |
| Frozen TRAIN population | **3000/3000** |
| Direct TRAIN scientific scenes | **782/782 unique** |
| Nested outer folds | **5/5 complete** |
| Independent inner value-calibration proposals | **97 / 100 / 98 / 86 / 110** |
| Server targeted regression | **201/201 PASS** |
| Re-run targeted regression on uploaded code | **201/201 PASS** |
| Frozen RSMR selector identity | **PASS** |
| V42 QUALITY control replay | **PASS** |
| Deployment use of logged/label future | **No** |
| Second-best / reranking / fallback rescue | **No** |
| CAL500/A500/B500 consumed | **No** |
| Termination | **Preregistered TRAIN scientific STOP** |

The launcher exits immediately after the nested TRAIN fitter returns failure and before the fresh-token selection stage. Therefore the result is not contaminated by CAL500/A500/B500.

`build_response_modes(runtime, label_future=None, cfg)` is used for V43 deployment observables. The runtime observable path rejects response modes marked as using logged/label future. Thus the result is not explained by future-label leakage.

The new future-response instrumentation is downstream of the frozen RSMR winner. Every V43 value arm can only retain that same winner or return to incumbent. Hence V43 does not alter the already-mature challenger-ordering layer.

Conclusion:

`engineering_valid = true`; proceed with scientific attribution.

---

# 2. Preregistered V43 ordering and promotion decision

The frozen RSMR reference is unchanged:

- selected: **502**
- positive selected: **221**
- positive capture: **38.5017%**
- teacher-improvement sum: **+43.2941**
- catastrophes: **28**
- no-positive-opportunity false interventions: **107**
- negative RMS: **0.355688**

The preregistered capture floor is:

`38.5017% - 3pp = 35.5017%`.

| TRAIN cross-fit | RSMR | QUALITY-CONTROL | FUTURE-MEAN | FUTURE-ROBUST-RAW | CFRV-MAIN |
|---|---:|---:|---:|---:|---:|
| selected | 502 | 205 | 231 | 228 | 347 |
| positive | 221 | 129 | 134 | 131 | 162 |
| precision | 44.02% | 62.93% | 58.01% | 57.46% | 46.69% |
| capture | **38.50%** | 22.47% | **23.34%** | 22.82% | **28.22%** |
| sum teacher improvement | +43.294 | +43.906 | **+44.891** | **+45.919** | **+54.978** |
| catastrophe | 28 | 13 | **13** | **13** | **13** |
| no-op false | 107 | **30** | 42 | 42 | 69 |
| NegRMS | 0.3557 | 0.3127 | **0.2941** | 0.2961 | **0.2581** |
| 5/5 fold sum >= 0 | yes | yes | **yes** | **yes** | **yes** |

Every new arm satisfies the tail, population, aggregate and 5/5-fold gates. Every arm fails the same preregistered **existence/capture** gate. Therefore V43 is not promotable and fresh data must remain untouched.

This is not a null result. The prospective-response arms improve precisely the quantities V42 predicted should improve if future consequence observability is real.

---

# 3. What FUTURE-MEAN actually proves

Relative to QUALITY, FUTURE-MEAN changes selected count only moderately (`205 -> 231`) but improves:

- aggregate teacher value: `+43.906 -> +44.891`;
- NegRMS: `0.3127 -> 0.2941`;
- selected-value Pearson correlation: `0.2383 -> 0.3450`;
- non-catastrophe AUC: `0.4916 -> 0.5353`.

Its five outer-test fold sums are all positive:

`11.727 / 6.560 / 10.473 / 9.817 / 6.315`.

This supports the V42 preregistered hypothesis:

> **current-time QUALITY/RISK is not the complete deployment valuation statistic; prospective interaction consequences contain independent cardinal signal.**

The set rotation is selective rather than blanket coverage expansion:

- QUALITY and FUTURE-MEAN both accept: **184 scenes**, sum `+39.850`;
- FUTURE-MEAN-only: **47 scenes**, sum `+5.042`;
- QUALITY-only: **21 scenes**, sum `+4.056`.

Thus FUTURE-MEAN adds a positive-net new population, although not a perfectly clean one.

A concrete positive rescue is `b95dc2615771588d`:

- teacher improvement `+1.9697`;
- QUALITY value `-0.0120` (missed);
- future response becomes nonzero and FUTURE-MEAN value rises to `+0.0963`.

This is exactly the kind of opportunity V42 current observables could not see.

Therefore **prospective horizon is an effective mechanism**.

---

# 4. Why FUTURE-ROBUST does not establish an independent CVaR mechanism

The preregistration deliberately compared mean with mean+CVaR. The outcome is highly diagnostic.

FUTURE-MEAN and FUTURE-ROBUST-RAW accept almost the same proposals:

- both: **225**;
- mean-only: **6**;
- robust-only: **3**.

Only **9/502 frozen proposals** differ in the final accept/reject decision.

The raw observable improvements are also strongly collinear:

- corr(mean, CVaR) = **0.7854**;
- corr(mean, robust) = **0.9231**;
- corr(CVaR, robust) = **0.9630**.

Moreover their correlation with teacher improvement is:

- future mean delta: **0.3836**;
- future CVaR delta: **0.2555**;
- frozen mean+CVaR robust delta: **0.3256**.

Prediction-level metrics similarly favor the simpler mean on several dimensions:

- mean Pearson `0.3450` vs robust `0.3177`;
- mean positive AUC `0.6605` vs robust `0.6588`;
- mean RMSE `0.6719` vs robust `0.6836`.

Therefore the supported conclusion is **not** “CVaR tail modeling solved or materially improved V43.” The correct conclusion is:

> **the prospective response family is informative, but within the current five fixed kinematic modes, the mean/CVaR distinction is mostly redundant.**

Do not tune `alpha`, CVaR weight or fixed response probabilities based on this result.

---

# 5. CFRV-MAIN: useful recovery, but still a zero-boundary Pareto move

CFRV-MAIN is numerically the strongest V43 aggregate arm:

- selected `347`;
- positive `162`;
- sum `+54.978`;
- catastrophe `13`;
- NegRMS `0.2581`;
- 5/5 folds positive.

It reduces RSMR material-positive misses from **20** under QUALITY to **9**. That is real progress.

But capture remains only **28.22%**, still approximately **7.28 percentage points** below the preregistered floor.

The fold-specific unit-slope translations are:

`-0.0430, +0.1252, +0.0850, +0.1203, +0.0103`.

Four of five folds shift the decision boundary in the permissive direction.

Set attribution between FUTURE-ROBUST-RAW and CFRV-MAIN shows:

- common: 205 scenes, sum `+46.670`;
- RAW-only: 23 scenes, sum `-0.751`;
- MAIN-only: 142 scenes, sum `+8.308`.

The MAIN-only set contains **97 mild/non-positive harms** and only **43 positives**. Thus translation is not useless—it has positive net value—but much of the coverage recovery still comes from moving a global zero boundary through a dense near-zero population.

Consistently, MAIN improves non-catastrophe AUC to `0.5784` but degrades zero-sign accuracy to `0.5139` and positive AUC to `0.5960`.

So V43 does not justify tuning translation or replacing the fixed zero by a post-hoc threshold.

---

# 6. Capture failure: near-zero count ambiguity remains, but material misses are still real

For the 221 true-positive frozen RSMR proposals:

### QUALITY misses 92

- `0 < Delta <= 0.001`: 64, sum `+0.0161`;
- `0.001 < Delta <= 0.01`: 7, sum `+0.0245`;
- `0.01 < Delta <= 0.2`: 1, sum `+0.1194`;
- `Delta > 0.2`: **20**, sum **`+23.3478`**.

### FUTURE-MEAN misses 87

- near-zero (`<=0.01`): 70;
- material (`>0.2`): **17**, sum **`+22.2767`**.

### CFRV-MAIN misses 59

- near-zero (`<=0.01`): 50;
- material (`>0.2`): **9**, sum **`+10.9388`**.

Thus the capture gate is numerically dominated by near-zero positives, but it is not merely an inconvenient metric. There remain high-value opportunities that the current value layer places on the wrong side of zero. The gate must remain frozen.

---

# 7. The strongest new V43 observation: future support improves, but remains incomplete

V42 found that all six current RISK deltas were exactly zero for:

- `17/28` catastrophes;
- `21/50` material positives.

V43 prospective response observables reduce this support hole, but do not close it:

- future mean/CVaR/robust are all exactly zero for **11/28 catastrophes**;
- all three are zero for **18/50 material positives**;
- all three are zero for **249/502 frozen RSMR proposals**.

The intersection of “current RISK all zero” and “V43 future response all zero” remains:

- **8/28 catastrophes**;
- **14/50 material positives**.

So V43 genuinely adds observability—it converts 6 of the 17 current-risk-invisible catastrophes and 3 of the 21 current-risk-invisible material positives into nonzero prospective evidence—but a large unsupported region remains.

This is a representation/support problem, not a scalar-value-head capacity problem.

---

# 8. Even nonzero V43 future signal can have the wrong semantics

The more important failure is not just zero support. Some V43 nonzero signals point in the wrong direction.

### Catastrophe `c70954fab4a650c7`

- true teacher improvement: `-1.2330`;
- QUALITY value: `+1.7213`;
- future observable improvement `[mean, CVaR, robust] = [+0.9280, +4.0214, +2.1653]`;
- FUTURE-MEAN value: `+1.9030`;
- FUTURE-ROBUST value: `+2.0351`.

The prospective mechanism makes an already severe false positive **more positive**.

### Catastrophe `2154d495b9b657a5`

- true `-1.2203`;
- QUALITY `+1.0988`;
- future deltas `[+0.9040, +3.9173, +2.1093]`;
- robust value `+1.4171`.

Again the sign of the response consequence is wrong.

By contrast, `fefa06851c7f58a7` has true `-1.2323`; future deltas are all negative and correctly make the value more conservative. Therefore the future mechanism is not random: it has correct signal in some scenes and structurally incorrect signal in others.

---

# 9. Why the V43 code explains this failure

The code reveals two structural limitations that match the data.

## 9.1 Response trajectories are candidate-independent

The frozen response bank `cv / ca / brake / yield / nonyield` is generated only from the **current agent state**. The agent response trajectory does not take the candidate ego plan as an input.

The ego candidate changes the collision/TTC cost *against* each mode, but it does not change what response modes are likely.

Hence V43 estimates approximately:

`P(response mode | current scene)`

when interactive planning actually requires something closer to:

`P(response mode | current scene, candidate ego plan)`.

This distinction can reverse the sign of intervention value. A candidate that induces another agent to yield and the same geometric candidate under a non-yielding response can have opposite value.

## 9.2 Mode probabilities are fixed across candidates

The V43 response prior is globally inherited from the frozen runtime/teacher configuration. The same probability vector is used for every candidate in a scene.

Therefore candidate A cannot increase the probability of yield while candidate B increases the probability of non-yield, even if their interaction geometries differ.

This also explains why mean and CVaR are so redundant: they aggregate the **same five candidate-independent response modes with the same prior**, so changing only the tail functional cannot manufacture missing action-response dependence.

## 9.3 Existing risk severity is finite-envelope/gated

V43 reuses hard/soft/TTC runtime safety semantics. Those quantities are intentionally zero before a candidate-agent interaction enters their fixed risk envelope/horizon.

That is appropriate for a safety guard but not necessarily sufficient as an **early value-support observable**. The 249/502 exact-zero future cases are direct evidence of this mismatch.

Therefore simply adding more regression capacity on V43's three scalar outputs cannot solve the support hole.

---

# 10. Mechanism ledger after V43

## Mechanisms with strong evidence and should remain frozen

1. **B16/M24 bounded planner interface** — mature backbone.
2. **EAF complete attributed frontier** — mature backbone.
3. **Exact action-local attribution** — mature.
4. **Support/admissibility and structural delegation** — mature.
5. **Explicit incumbent/null action and no-second-best containment** — necessary and mature.
6. **RSMR ordinal extremal challenger ranking** — strongest learned layer; freeze.
7. **Selected-policy residual/tail structure exists** — established V37-V39.
8. **Endpoint/basepoint interaction is a real cardinal mediator** — V41 EPV-only high-quality subset.
9. **Current trajectory QUALITY consequence is a real value mediator** — V42.
10. **Prospective interaction horizon adds independent value/tail signal** — V43.

## Mechanisms falsified/closed as first-order solutions

1. More acquisition/capacity/beam/swap/B/M search.
2. More pure 19-D selected-value heads or target reformulation.
3. MLP/nonlinear CFSR/OPVR resurrection.
4. Hurdle/sign/upside/downside factorization on pure 19-D as sufficient solution.
5. Generic higher-order delta polynomial.
6. Endpoint polynomial/head enlargement.
7. Basepoint common shift/reservation or selection-geometry reservation.
8. Binary catastrophe classifier/veto.
9. Threshold/lambda/alpha/q/top-K/candidate-count/temperature sweeps.
10. QUALITY/RISK post-hoc weight tuning.
11. Selected zero translation tuning.
12. **V43 fixed-prior mean+CVaR as an independently established tail mechanism.**
13. Simply adding more fixed kinematic response modes or tuning their probabilities before solving action conditioning/support.

---

# 11. Evidence chain V32.1 -> V43

The experimental sequence now supports a progressively narrower story.

- **V32.1**: dense edge conditional mean contains real signal but is unsafe as an extremal selector.
- **V33**: incumbent/null action is structurally necessary; safety cannot come only from reranking.
- **V34 RSMR**: scene-level ordinal regret learning solves the relative challenger-ordering layer well enough to freeze it (`+43.29`, 5/5 positive folds).
- **V35-V36**: scene-common basepoint/geometry reservations mostly trade coverage for safety; not the dominant mediator.
- **V37-V39**: post-selection residual and tail signal are real, but pure-delta selected zero identification remains unstable.
- **V40**: sign/upside/downside factorization still fails on pure 19-D; close the pure-delta value-head family.
- **V41 EPV**: endpoint/basepoint interaction identifies an independent high-quality subset; absolute physical consequence is still missing.
- **V42 QUALITY/RISK**: deployment consequence observables are real mediators; QUALITY is strong, but current RISK has exact-zero support on many decisive scenes.
- **V43**: prospective response horizon improves value correlation and tail magnitude, proving future consequence observability matters; however fixed candidate-independent response priors and gated severity still leave both zero-support holes and wrong-sign response semantics.

The dominant bottleneck has therefore moved through:

`candidate ordering`

-> `selected absolute value`

-> `value representation`

-> `current consequence observability`

-> **`plan-conditioned future consequence observability/support`**.

This is a much stronger paper-level causal chain than a sequence of tuned heads.

---

# 12. Model-layer maturity after V43

| Layer | Status after V43 | Main remaining issue |
|---|---|---|
| bounded B16/M24 interface | **mature / frozen** | none first-order |
| EAF complete frontier | **mature / backbone** | none first-order |
| exact attribution | **mature / frozen** | none first-order |
| capacity/evidence visibility | **closed as first-order bottleneck** | do not reopen |
| support/admissibility | **mature / frozen** | none first-order |
| RSMR ordinal challenger ranking | **most mature learned layer / frozen** | not current bottleneck |
| incumbent/null + no fallback | **mature / frozen** | none |
| EPV endpoint geometry | **real but partial** | not absolute consequence-sufficient |
| current QUALITY consequence | **strong partial maturity** | misses sign/capture |
| current deterministic RISK | **partial signal / sparse support** | many exact zeros |
| prospective fixed response horizon | **real partial mechanism** | candidate-independent response likelihood |
| fixed-prior CVaR tail functional | **not independently validated** | nearly redundant with mean |
| selected absolute zero | **immature** | near-zero + material opportunity misses |
| plan-conditioned response distribution | **missing** | V43 does not model action -> response |
| weak-before-gate interaction support | **missing** | finite-envelope risk zeros |

---

# 13. Updated dominant bottleneck

After V43 the dominant bottleneck should be defined as:

> **plan-conditioned prospective interaction consequence observability and support for the frozen extremal proposal, especially before hard/soft/TTC risk activation and at the incumbent-exit zero boundary.**

This has two coupled but separable sources:

1. **response-likelihood insufficiency**: `P(mode|scene)` is fixed across ego candidates, while interactive planning needs `P(mode|scene, ego plan)`;
2. **support insufficiency**: the current gated interaction cost can be exactly zero before a future conflict enters the risk envelope.

V44 is intentionally designed to separate these two sources rather than mixing them in one larger model.

---

# 14. V64.3.44 EAF-ICER-PCOR — Plan-Conditioned Occupancy Response

## 14.1 High-level topology

V44 freezes:

- B16/M24;
- EAF;
- support/admissibility;
- RSMR proposal identity;
- EPV;
- V42 QUALITY;
- V43 fixed response mode basis;
- all historical promotion thresholds.

The deployed decision remains:

`RSMR -> frozen b_hat -> absolute valuation -> {same b_hat, incumbent}`.

No new candidate can be created and no second-best fallback is allowed.

## 14.2 TRAIN-only plan-conditioned behavior posterior

The fixed runtime-only response basis remains:

`cv / ca / brake / yield / nonyield`.

Logged agent future is used **only on frozen TRAIN**, outside the deployment planner, to assign a behavior label: the runtime response mode with minimum mean displacement error to logged agents over the existing soft-check horizon.

No teacher improvement, teacher candidate value, or action-selection label enters this behavior target.

A low-capacity multiclass ridge-score model with fixed `lambda=1` learns:

`P(mode | current scene, ego plan)`

from ten candidate-dependent interaction features:

- the five V43 per-mode gated interaction costs;
- five new per-mode full-horizon occupancy potentials.

Outer-fold isolation is strict: for each scientific outer test fold, the behavior posterior is fitted excluding both the outer test fold and the selected-value calibration fold.

At deployment, only the fitted posterior and current scene/candidate trajectory are used; logged future is absent.

## 14.3 Support-preserving occupancy potential

For each candidate, response mode and agent, V44 adds a continuous full-horizon interaction support statistic:

`occupancy_potential = 1 / (1 + normalized_separation^2)`.

The spatial normalization uses the already-frozen soft box/radius semantics. There is:

- no new distance threshold;
- no tuned kernel bandwidth;
- no new top-K;
- no hard veto.

Unlike V43 gated risk, this potential remains weakly nonzero before the interaction crosses a hard/soft severity boundary. It is therefore an **observability/support statistic**, not a replacement safety rule.

## 14.4 Candidate-specific response functionals

The learned posterior reweights the same five response modes separately for each ego candidate. V44 forms:

- plan-conditioned gated-response mean cost;
- plan-conditioned gated-response robust cost;
- plan-conditioned occupancy mean cost;
- plan-conditioned occupancy robust cost.

The final scalar residual continues to be fitted with scene-total-one weighting, zero bias and fixed `lambda=1` on top of frozen EPV+QUALITY.

---

# 15. Preregistered V44 causal arms

The core experimental arms are deliberately factorized:

| Arm | Question answered |
|---|---|
| **V43-FUTURE-MEAN CONTROL** | Exact prospective-horizon baseline |
| **PC-REWEIGHT** | Is candidate-conditioned response probability the missing mechanism even with the old gated cost? |
| **PC-OCC-MEAN** | Does removing the finite-envelope support hole add the missing information? |
| **PC-OCC-ROBUST** | After plan conditioning + support are correct, does tail/CVaR add independent value? |
| **PCOR-MAIN** | PC-OCC-ROBUST + independent CAL500 unit-slope translation only |

V43 FUTURE-ROBUST is also replayed as a frozen control but is not promoted to a new mechanism merely because it uses CVaR.

## Interpretation branches

### PC-REWEIGHT succeeds, OCC does not add independent value

Conclusion: the dominant missing mechanism was candidate-conditioned response likelihood. Keep the simpler reweighting model; do not retain occupancy complexity.

### PC-OCC-MEAN succeeds and clearly beats PC-REWEIGHT

Conclusion: finite-envelope/gated support was a first-order bottleneck. A support-preserving prospective interaction representation is necessary.

### PC-OCC-ROBUST adds independent improvement after PC-OCC-MEAN

Only then is there evidence that response-tail functional/CVaR matters after the distribution/support is correctly conditioned.

### Behavior posterior is not better than majority and all PC arms fail

Close the discrete five-mode response basis. Move to a **continuous plan-conditioned occupancy/trajectory response predictor**, not more mode engineering.

### Behavior posterior is predictive but PC arms still fail

The response mode labels are learnable, but the discrete mode trajectories / scalar occupancy functional are not decision-sufficient. Next step should be continuous plan-conditioned occupancy or trajectory distribution, still downstream of frozen RSMR.

### PCOR-MAIN passes TRAIN

Only then select independent CAL500+A500+B500. A/B remain unpooled. Double-fresh pass is required before official full validation / paper-facing closed loop.

---

# 16. V44 promotion gate and engineering hard gate

The scientific promotion gate is unchanged:

- no-op false reduction >=20% vs RSMR;
- positive capture >= RSMR - 3pp;
- catastrophe reduction >=25%;
- NegRMS non-worse than RSMR;
- aggregate selected teacher sum >=0;
- 5/5 outer test fold sums >=0;
- selected >=64;
- selected positive >=32;
- exact frozen-winner containment.

Before any V44 interpretation, the new instrumentation must exactly replay V43 TRAIN signatures for:

- RSMR;
- V42 QUALITY;
- V43 FUTURE-MEAN;
- V43 FUTURE-ROBUST-RAW.

Any mismatch is an **engineering STOP**.

New V44 fresh seed:

`v64.3.44-eaf-icer-pcor-cal500-double-fresh-v1`.

Permanent design exclusion remains 10700 tokens. V43 consumed no fresh evidence.

---

# 17. Explicit no-repeat / prohibited optimization directions after V43

In addition to all historical `ALGORITHM_CHANGELOG.md` closures, V44 explicitly forbids:

1. tuning V43 fixed response probabilities, CVaR alpha or mean/CVaR weight;
2. increasing the number of hand-designed kinematic response modes as the first-order rescue;
3. enlarging the value head/MLP while 249/502 proposals have zero future support;
4. extending current RISK with another thresholded severity head;
5. replacing capture by aggregate value because CFRV-MAIN has a large sum;
6. tuning selected translation / value threshold;
7. using a binary catastrophe classifier/veto;
8. modifying RSMR or allowing a second-best action;
9. using logged/teacher future at deployment;
10. using teacher improvement to supervise the behavior posterior;
11. adding an occupancy distance threshold/kernel bandwidth sweep;
12. tuning response-mode subsets, top-K agents, candidate count or evidence capacity based on V43.

---

# 18. Paper-level interpretation

The strongest paper line after V43 is no longer a PTMC-style “tail classifier” story. The evidence supports a deeper decomposition:

> **Selection–Valuation–Response Sufficiency under a Bounded Planner Interface.**

A bounded EAF representation can be sufficient for **ordinal extremal proposal selection**, while the absolute decision to leave the incumbent requires a different statistic: an **action-conditioned prospective consequence distribution**. In interactive planning, future consequence is not merely uncertain; the response distribution itself depends on the candidate ego plan. Therefore selection evidence, endpoint/cardinal evidence and interactive-response evidence obey different decision sufficiency requirements.

A possible mechanism chain is:

`bounded auditable evidence`

-> `exact EAF attribution`

-> `ordinal RSMR extremal selection`

-> `freeze proposal`

-> `endpoint + current trajectory consequence`

-> **`plan-conditioned prospective response distribution/support`**

-> `absolute incumbent-exit decision`

-> `deterministic no-fallback containment`.

This is an algorithmic hypothesis, not yet a paper claim. It becomes a strong CCF-A-level contribution only if V44/V45-style independent experiments demonstrate that plan conditioning/support closes the selected absolute-value bottleneck and the result survives untouched evaluation and closed loop.

---

# 19. Engineering validation of delivered V44 code

The delivered V44 code has been checked locally:

- Python compile: **PASS**;
- main launcher `bash -n`: **PASS**;
- V43+V44 focused tests: **9/9 PASS**;
- V13->V44 targeted regression: **207/207 PASS**;
- complete repository tests, four mutually exclusive/exhaustive file partitions: **148 + 132 + 126 + 131 = 537/537 PASS**;
- test files covered: **119/119**;
- warnings: historical PyTorch Transformer warnings only; no new warning class observed.

The server run must still execute the hard historical replay and nested TRAIN gate before any fresh token can be selected.
