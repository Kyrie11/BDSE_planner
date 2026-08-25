# V64.3.42 EAF-ICER-OVDR uploaded result postmortem and V64.3.43 CRVR design

## 0. Executive verdict

**Engineering verdict: PASS.** The uploaded V42 result is reliable for TRAIN-level scientific attribution. No V42.1 engineering hotfix is needed.

**Scientific verdict: partial mechanism success, promotion failure.** V42 validates the move from endpoint-only value representation toward deployment-observable trajectory consequence, but the current-time observable partition is not deployment-sufficient at the frozen RSMR incumbent-exit boundary.

The strongest arm is not RISK but **QUALITY** under the preregistered gate. QUALITY is the only new observable arm that passes the complete tail sub-gate while failing only the existence/capture sub-gate. RISK has useful global correlation and lowers catastrophe count further, but its NegRMS remains worse than RSMR because a sparse set of extreme failures is invisible or badly signed. JOINT introduces negative transfer, and the MAIN translation moves along the coverage-tail Pareto frontier instead of learning a better absolute zero.

Accordingly the next bottleneck is:

> **deployment-sufficient counterfactual consequence observability for the frozen extremal proposal, especially future-agent response and other teacher components not identifiable from current trajectory/static-scene observables.**

V43 therefore does **not** add a bigger head, classifier, threshold, or candidate search. It freezes RSMR and tests a counterfactual response-value decomposition with exact normalized analytic cost accounting.

---

## 1. V42 engineering/reliability audit

### 1.1 Code identity

Uploaded V42 code ZIP SHA256:

`d29e14c0d30833d42ca1b8f70afc6f08a71183cad8b05c6e63a990acfaf216b7`

This exactly equals the V42 package hash preregistered in the V41 analysis. There is no code-version drift.

Uploaded V42 output ZIP SHA256:

`020416452bdbf7620790fcc46661fa1c9dc2f2c4758bbdf358303061447d9c2b`

### 1.2 Runtime and nested protocol closure

- server targeted regression: **198/198 PASS**, with only the same two historical Transformer warnings;
- five nested outer folds: complete;
- direct TRAIN scene audit: **782/782 unique**;
- frozen TRAIN population: 3000 scenes;
- newly instrumented V42 replay exactly reproduces historical RSMR:
  - selected 502;
  - selected positive 221;
  - no-op false 107;
  - catastrophe 28;
  - teacher sum +43.29405361274824;
  - capture 38.501742160278746%;
  - NegRMS 0.3556880321206127.

Compared scene-by-scene with the V41 historical causal audit:

- token set: exact;
- outer/calibration fold assignment: exact;
- candidate count: exact;
- RSMR selected action: exact;
- max absolute RSMR score difference: **0**;
- max absolute teacher-improvement difference: **4.44e-16**;
- EPV-RAW selected action/value: exact up to logged numerical precision.

Thus the V42 observable instrumentation did not perturb proposal selection or historical labels.

### 1.3 Leakage/freshness

The uploaded output contains only prerequisite/replay/TRAIN-fit artifacts. There are no non-empty:

- `val_screen_cal500_fresh1500_tokens.txt`;
- `val_screen_calibration_500_tokens.txt`;
- `val_screen_fresh_A_tokens.txt`;
- `val_screen_fresh_B_tokens.txt`.

The launcher stopped at the TRAIN scientific gate before fresh selection. There is no evidence of CAL500/A500/B500 consumption or outer-test leakage.

### 1.4 One reporting issue that does **not** invalidate the experiment

The V42 fitter emitted:

`current_physical_risk_observable_is_primary_missing_tail_mediator`

This string is selected by an AUC-first heuristic. It is **not equivalent to the preregistered promotion/tail gate**. In the actual gate output:

- QUALITY: `tail = true`;
- RISK: `tail = false` because NegRMS is 0.3901 > RSMR 0.3557.

Therefore the correct scientific reading must be derived from the arm metrics and gates, not from that single diagnosis string. This is a post-processing interpretation flaw only; it does not change any V42 action, metric, fold, gate, or fresh-data state. V43 replaces this with gate-first diagnosis logic.

**Reliability conclusion: V42 is engineering-valid and can be scientifically attributed. No V42.1.**

---

## 2. Preregistered V42 decision, in order

The frozen RSMR capture floor is:

`38.5017% - 3 pp = 35.5017%`.

| TRAIN nested cross-fit | RSMR | EPV-RAW | EPV+QUALITY | EPV+RISK | EPV+JOINT | OVDR-MAIN |
|---|---:|---:|---:|---:|---:|---:|
| selected | 502 | 203 | 205 | 205 | 186 | 285 |
| positive | 221 | 118 | 129 | 122 | 109 | 129 |
| precision | 44.02% | 58.13% | **62.93%** | 59.51% | 58.60% | 45.26% |
| capture | **38.50%** | 20.56% | **22.47%** | 21.25% | 18.99% | 22.47% |
| Σ teacher improvement | +43.294 | +25.969 | **+43.906** | +43.300 | +40.777 | +41.246 |
| catastrophe | 28 | 15 | **13** | **11** | **11** | 16 |
| no-op false | 107 | 39 | **30** | 40 | 37 | 63 |
| NegRMS | **0.3557** | 0.4543 | **0.3127** | 0.3901 | 0.4244 | 0.3894 |
| all 5 fold sums >=0 | yes | yes | **yes** | **yes** | **yes** | **yes** |
| existence/capture gate | baseline | FAIL | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| tail gate | baseline | FAIL | **PASS** | FAIL | FAIL | FAIL |
| promotion | baseline | FAIL | **FAIL** | FAIL | FAIL | FAIL |

### 2.1 EPV-RAW control

V42 exactly reproduces V41's endpoint control. This closes the possibility that the V42 gain came from a changed EPV fit or altered RSMR proposal population.

### 2.2 QUALITY branch: **strong partial success, but not promotion**

QUALITY is the most important V42 result.

Relative to RSMR it gives:

- no-op false: 107 -> 30 (**-72.0%**);
- catastrophe: 28 -> 13 (**-53.6%**);
- NegRMS: 0.3557 -> **0.3127**;
- teacher sum: +43.294 -> **+43.906**;
- all five test-fold sums remain nonnegative;
- precision rises to **62.93%**.

So this is not blanket abstention in the V36 sense. It retains essentially all aggregate useful value while removing much of the harmful intervention mass.

But capture is only **22.47%**, far below the 35.50% preregistered floor. Therefore QUALITY cannot be promoted. The correct mechanism statement is:

> **trajectory-quality consequence is a real, high-quality value mediator, but by itself it identifies only a selective intervention core rather than a deployment-sufficient incumbent-exit boundary.**

### 2.3 RISK branch: real signal, but sparse extreme blind spots

RISK reduces catastrophe count even further to **11** and gives the best global value regression among the raw observable arms:

- Pearson ≈ 0.4815;
- RMSE ≈ 0.6218;
- non-catastrophe AUC ≈ 0.5166.

But the selected negative RMS is **0.3901**, worse than RSMR. Its worst retained proposal is still `-3.8065`. Thus the decrease in catastrophe *count* hides a sparse high-magnitude residual tail.

RISK is therefore not the preregistered successful mediator. The correct statement is:

> **current physical-risk observables contain useful ordinary/tail information, but they do not observe the remaining high-leverage catastrophic modes reliably enough.**

### 2.4 JOINT branch: complementarity is not recovered by learned concatenation

QUALITY and RISK individually add signal, but the learned JOINT residual is worse:

- selected falls to 186;
- capture falls to 18.99%;
- sum drops to +40.777;
- NegRMS rises to 0.4244.

Set comparison makes the negative transfer explicit:

- QUALITY∩RISK: 180 proposals, **+45.290**, 10 catastrophes;
- QUALITY-only: 25 proposals, -1.385, 3 catastrophes;
- RISK-only: 25 proposals, -1.990, 1 catastrophe.

The high-value signal lives mainly in a strong consensus core. A single unconstrained ridge over both blocks does not preserve that structure.

This is important because it falsifies the easy next move “just concatenate more physical features.”

### 2.5 OVDR-MAIN translation: coverage-tail Pareto movement, not zero identification

The unit-slope selected translation expands JOINT from 186 to 285 interventions, but:

- catastrophe rises 11 -> 16;
- no-op false rises 37 -> 63;
- NegRMS remains worse than RSMR;
- capture is still only 22.47%.

This repeats the V41 EPVR behavior: a global zero shift re-admits both useful and harmful proposals. It does not resolve the absolute zero.

**Preregistered V42 verdict:** `partial success + TRAIN promotion failure`.

The preregistered branch to take is precisely:

> observable diagnostics improve but the gate does not close -> current observable partition is real but incomplete -> study a future-sensitive consequence observable / robust response uncertainty, not a larger head.

---

## 3. What V42 actually learned

### 3.1 QUALITY substantially repairs EPV's selected core

EPV -> QUALITY set transition:

- accepted by both: 183 proposals, +34.552, 11 catastrophes;
- EPV-only removed by QUALITY: 20 proposals, **-8.583**, 4 catastrophes;
- QUALITY-only newly recovered: 22 proposals, **+9.354**, only 2 catastrophes.

So QUALITY is doing two causal things simultaneously: deleting a strongly net-harmful EPV subset and recovering a net-useful subset. This is qualitatively different from blanket abstention.

### 3.2 RISK also adds a clean EPV-missed core

EPV -> RISK:

- EPV-only: 15 proposals, **-4.683**, 4 catastrophes;
- RISK-only: 17 proposals, **+12.648**, **0 catastrophes**.

This proves current physical consequence is not redundant with endpoint geometry.

### 3.3 V42 rescues several V41 headline false negatives

Examples:

- `7418da8c04e85efb`: true +4.0080; EPV -0.0102; QUALITY +0.4638; RISK +3.4781; JOINT +3.6275.
- `054033a129155c88`: true +3.9941; EPV -0.0087; QUALITY +0.5915; RISK +3.6190; JOINT +3.7401.
- `3da38924ebd45f50`: true +1.7287; EPV -0.0241; QUALITY +0.4846; RISK +0.7887.
- `f9c3f5c430be54aa`: true +1.5646; EPV -0.0124; QUALITY +0.1403; RISK +0.5873.

Thus the V41 hypothesis “the endpoint representation is missing direct trajectory consequence” is empirically supported.

---

## 4. What V42 still does not learn

There are two different unresolved populations and they must not be conflated.

### 4.1 Count-level capture is dominated by near-zero positives

Among the 221 positive RSMR proposals:

- 132 have `0 < ΔT <= 0.001`, total value only **+0.0357**;
- another 35 have `0.001 < ΔT <= 0.01`, total **+0.1062**;
- only 50 have `ΔT > 0.2`, but these carry **+82.734** teacher value.

Thus 167/221 positive labels are extremely close to zero. This is why a value-selective arm can retain aggregate teacher value while failing the count-based capture gate.

However, this does **not** justify relaxing the gate: the next subsection shows real material false negatives remain.

### 4.2 There are still material opportunities invisible to all current raw observable arms

There are **17 material positives (`ΔT>0.2`)**, totaling **+20.902**, that are rejected by QUALITY, RISK, and JOINT simultaneously.

Examples include:

- `30c657863dc05485`: +2.0203, Q/R/J all negative; full current RISK delta = 0;
- `cfdc02dc40405e6f`: +2.0198, Q/R/J all negative;
- `b95dc2615771588d`: +1.9697, Q/R/J all negative; full current RISK delta = 0;
- `57a59636cde25b02`: +1.5958, Q/R/J all negative; full current RISK delta = 0;
- `faf6056c372e54f2`: +1.2462, Q/R/J all negative; full current RISK delta = 0;
- `f7050dbcd3c954f3`: +1.2325, Q/R/J all negative; full current RISK delta = 0.

Therefore the remaining capture failure is not merely a metric artifact caused by tiny positives.

### 4.3 Remaining catastrophe tail is sparse but high leverage

The current blocks often fix the **median** catastrophe sign, but a few catastrophic proposals remain predicted strongly positive.

Representative examples:

- `2b32a9f406845f75`: true -3.8065; EPV +0.1002; QUALITY -0.0820 correctly rejects; RISK +0.0101 and JOINT +0.0547 still accept.
- `5693aed0af7e548c`: true -2.0204; EPV/Q/R/J all around +0.08~+0.09.
- `c70954fab4a650c7`: true -1.2330; EPV +1.7208; Q +1.7213; R +1.7208; J +1.7308.
- `a8e99095a8235549`: true -1.2319; all raw value arms predict about +2.4.

These are not small threshold errors; the value direction is fundamentally wrong.

---

## 5. Why current RISK is not sufficient: observability audit

On the 502 frozen RSMR proposals, candidate-vs-incumbent current-risk differences are nonzero for:

- hard-agent: **0/502**;
- soft-agent: 69/502 (13.7%);
- TTC: 196/502 (39.0%);
- hard off-route: **0/502**;
- soft off-route: 17/502 (3.4%);
- red light: **0/502**.

The entire six-dimensional current RISK block is exactly zero on **286/502** proposals.

Most importantly:

- **17/28 catastrophes (60.7%)** have the entire RISK delta exactly zero; their teacher sum is **-19.308**;
- **21/50 material positives** also have zero RISK delta; their teacher sum is **+25.707**.

So current risk does not merely need better weights. On a majority of catastrophic proposals there is literally no candidate-vs-incumbent signal in this block.

This strongly supports the next question:

> the relevant consequence may depend on how surrounding agents evolve/respond to the candidate trajectory, not just on the current snapshot projected through a single runtime risk calculation.

---

## 6. A second V42 failure mechanism: known physical semantics were handed back to regression

The three QUALITY columns exactly reproduce the teacher base-cost route/progress/comfort terms, excluding only the label-only demonstration term. Yet V42 does not analytically account for them; it asks a ridge residual to relearn their coefficients jointly with EPV.

The full-TRAIN fits show semantic sign reversals:

- QUALITY standardized residual weights: route `+0.2146`, progress **`-0.0164`**, comfort `+0.0476`;
- JOINT standardized residual weights: route **`-0.0645`**, progress `+0.0046`, comfort `+0.0841`;
- RISK/JOINT TTC coefficient is also negative in the learned residual.

These signs should not be interpreted as physical causal weights. The target `teacher_improvement` is normalized by the per-scene pair-margin scale while the observable costs are logged in raw teacher-cost units; ridge can absorb an average scale and correlations but does not preserve the exact known decomposition.

The correct structural treatment is:

`normalized observable improvement = (cost(incumbent)-cost(candidate)) / pair_margin_scale`

with a fixed **+1** coefficient for a teacher-aligned cost term. Only the **unexplained remainder** should be learned.

This creates a second falsifiable hypothesis for V43:

1. if exact normalized current-quality accounting alone closes the gate, V42 was partly a decomposition/units/identifiability failure;
2. if it does not, future-sensitive consequence observability is genuinely required.

---

## 7. V32.1 -> V42 cumulative evidence chain

The progression is now tighter than the original manuscript's DRC/PTMC framing.

1. **V32.1**: dense edge conditional mean contains value signal, but using it as the extremal selector produces no-op/heavy-tail failure.
2. **V33**: explicit incumbent/null action is necessary; safety cannot come only from abstention.
3. **V34 RSMR**: scene-level regret-aligned ranking gives 502 proposals, +43.294 teacher sum and 5/5 nonnegative folds. **Which challenger** is no longer the weakest layer.
4. **V35/V36**: common basepoint shifts / selection geometry can suppress tail but mainly through blanket abstention; not first-order.
5. **V37**: post-selection residual/tail structure exists, but sparse selected-only high-dimensional estimation is unstable.
6. **V38**: dense all-edge 19-D supervision has ordinary cardinal/sign signal but selected-tail mismatch remains.
7. **V39**: honest selected-policy residual is real and can remove catastrophes almost without losing aggregate value, but zero crossing/capture remains poor.
8. **V40**: sign/upside/downside factorization on the same pure 19-D delta still fails. Close the pure-19-D selected-value-head family.
9. **V41**: endpoint/basepoint geometry gives an independent high-quality intervention core, proving local utility geometry matters, but absolute physical downside is still badly mis-signed.
10. **V42**: direct trajectory QUALITY/RISK consequence strongly improves the selected core. QUALITY alone almost preserves the full RSMR aggregate gain while halving catastrophes and sharply reducing no-op false interventions, but current-state observables still fail the absolute zero/capture boundary and leave sparse high-leverage blind spots.

The evidence now supports a sharper separation:

`ordinal proposal sufficiency != endpoint cardinal sufficiency != current-observable consequence sufficiency != counterfactual future-consequence sufficiency`.

---

## 8. Model-layer status after V42

| Layer | Status after V42 |
|---|---|
| B16/M24 bounded interface | mature; freeze |
| EAF complete frontier | mature; paper backbone |
| exact action-local attribution | mature; freeze |
| acquisition/capacity visibility | first-order bottleneck closed |
| support/admissibility | mature; freeze |
| RSMR ordinal challenger ordering | most mature learned layer; freeze |
| incumbent/null + no fallback containment | mature; permanent |
| ordinary edge cardinal sign | real but incomplete |
| selected residual/tail structure | real |
| basepoint-conditioned endpoint geometry | real, partial |
| current trajectory QUALITY consequence | **real and strong; selective core mature, boundary recall immature** |
| current physical RISK consequence | **real but sparse; extreme blind spots remain** |
| learned JOINT fusion | immature / negative transfer |
| selected absolute zero | immature |
| current-state consequence sufficiency | **falsified as complete representation** |
| counterfactual future-response consequence | **not yet identified; next target** |

---

## 9. Dominant bottleneck after V42

The dominant bottleneck should now be defined as:

> **deployment-sufficient counterfactual consequence observability for the already frozen extremal proposal, with particular emphasis on future-agent response and omitted teacher consequence at the absolute incumbent-exit boundary.**

It contains two coupled subproblems:

1. **zero-boundary/coverage:** QUALITY/RISK identify a high-value core but miss too many positives, including 17 material positives worth +20.902;
2. **sparse high-leverage tail observability:** 17/28 catastrophes have zero current RISK delta and several remaining catastrophes are assigned very large positive values.

A third identification issue must be separated experimentally rather than guessed: the teacher target contains not only current route/progress/comfort and selected evidence, but also the label-only demonstration term, logged-future response contribution, and unselected evidence contribution. V43 therefore logs an offline TRAIN-only oracle decomposition to quantify these pieces without ever feeding them to runtime.

---

# V64.3.43 EAF-ICER-CRVR — Counterfactual Response Value Recovery

## 10. Design principle

V43 changes the *observable*, not model capacity.

The frozen deployment chain remains:

`bounded selected evidence -> EAF -> support/admissibility -> frozen RSMR winner b_hat -> value confirmation -> {b_hat, incumbent}`.

No value arm may rerank challengers, choose second best, create a new proposal, or use logged future.

For the frozen RSMR proposal define normalized teacher improvement target `Y` in the same units already used by the EAF/RSMR pipeline.

V43 decomposes value as:

`V = A_known + V_endpoint_remainder`,

where `A_known` is an analytic candidate-vs-incumbent cost improvement expressed in the exact normalized target units.

The endpoint model is trained only on:

`Y - A_known`.

Thus known physical cost components are not relearned by an unconstrained residual regression.

## 11. Counterfactual response observable

V42's current risk block largely evaluates a single current-state projection. V43 instead reuses the **already selected evidence atoms** and evaluates their physical costs under a fixed label-free response envelope generated only from the current runtime state:

- constant velocity (CV);
- constant acceleration (CA);
- braking;
- yielding;
- non-yielding.

No logged agent future is allowed. The implementation explicitly throws if a response mode reports `uses_label_future=true` or `name=logged`.

For each candidate, V43 creates three lower-is-better selected-evidence cost functionals:

1. `C_CV`: selected-evidence cost under the CV response;
2. `C_MEAN`: probability-weighted mean selected-evidence cost over all runtime-only response modes;
3. `C_ROBUST`: the same fixed mean/CVaR functional used by the robust teacher, but evaluated only over runtime-only response modes and selected evidence.

For incumbent `i` and frozen proposal `b`:

`ΔC = C(i)-C(b)`.

This is exactly zero for identical actions and flips sign under endpoint exchange.

## 12. Exact normalized analytic quality accounting

The route/progress/comfort block is already part of the teacher base cost. V43 converts each candidate-vs-incumbent improvement into the learned value target's exact units:

`ΔC_norm = [C(i)-C(b)] / pair_margin_scale`.

The coefficient is then structurally fixed to +1.

This is crucial: raw cost deltas and normalized teacher improvements must never be added directly.

The same normalization is applied to a selected-evidence CV/mean/robust response cost before it is assigned a fixed +1 coefficient.

## 13. Causal arms: simplicity-first

V43 has four preregistered arms, evaluated on the same frozen RSMR winner.

1. **Q-ANCHOR**
   - analytic normalized route+progress+comfort improvement;
   - endpoint potential learns only the remainder.
   - Tests whether V42 was mainly harmed by relearning known cost semantics / target scaling.

2. **CV-ANCHOR**
   - Q-ANCHOR + selected-evidence CV physical cost;
   - endpoint model fits the remainder.
   - Tests whether simply reconstructing selected-evidence physical consequence is sufficient; no multimodal response claim is made if this passes.

3. **MEAN-ANCHOR**
   - Q-ANCHOR + selected-evidence multi-response expected cost.
   - Tests whether interaction uncertainty/response expectation is needed.

4. **ROBUST-ANCHOR / CRVR**
   - Q-ANCHOR + selected-evidence response mean/CVaR cost.
   - Tests whether tail-sensitive response aggregation is specifically needed.

Promotion is **simplicity-first**: choose the first passing arm in the fixed order Q -> CV -> MEAN -> ROBUST. A more complex branch cannot be credited if a simpler branch already passes.

V42 QUALITY is also replayed as a non-promoted causal control.

## 14. TRAIN-only oracle factorization

V43 instrumentation logs, on the frozen TRAIN replay only:

- total teacher base cost;
- total teacher evidence cost;
- teacher evidence cost on the selected atoms;
- teacher evidence cost on unselected atoms;
- exact pair-margin target scale.

For every candidate pair the fitter verifies the normalized identity:

`Y = quality_current + demo_label_only + selected_evidence_teacher + unselected_evidence_teacher`.

The oracle is **never a planner input** and is not available on fresh/runtime decisions. Its role is mechanism attribution if all handcrafted response arms fail:

- large selected-teacher vs runtime-response gap -> response model insufficiency / logged-future interaction;
- large unselected-evidence residual -> bounded interface still misses value-specific evidence despite mature ranking;
- large demo-only residual -> teacher/runtime estimand mismatch becomes dominant;
- small oracle gaps but poor gate -> endpoint remainder/decision functional is still incorrect.

This prevents another ambiguous failure round.

## 15. V43 TRAIN promotion gate

No gate is relaxed. Relative to frozen RSMR:

- no-op false-intervention reduction >=20%;
- capture >= RSMR - 3pp = 35.5017%;
- catastrophe reduction >=25%;
- NegRMS no worse;
- aggregate selected sum >=0;
- all five outer test-fold sums >=0;
- selected >=64;
- selected positive >=32;
- exact monotone same-winner containment.

There is no threshold/lambda/alpha/q/top-K/candidate-count/capacity/temperature sweep and no selected translation.

## 16. V43 falsification branches

- **Q-ANCHOR passes:** exact normalized current-quality accounting is sufficient. Drop response envelope and keep the simplest mechanism.
- **CV-ANCHOR is first to pass:** selected-evidence physical consequence reconstruction is needed, but multimodal response is not.
- **MEAN-ANCHOR is first to pass:** future-response expectation is a necessary missing observable.
- **ROBUST-ANCHOR is first to pass:** tail-sensitive response aggregation is necessary; this is the strongest support for CRVR proper.
- **all fail:** close this hand-designed runtime response-envelope family. Use the TRAIN-only oracle to decide whether the next representation must be a data-conditioned predictive response/world-model observable, bounded-interface augmentation, or a correction to the teacher/runtime value estimand. Do **not** add more ridge/MLP heads to the same observables.

## 17. Fresh/closed-loop protocol

V42 consumed no fresh data. Permanent design exclusion remains 10700 tokens.

V43 has no post-TRAIN calibration parameter, so after TRAIN passes it selects only **A500+B500** under a new label-free seed:

`v64.3.43-eaf-icer-crvr-double-fresh-v1`

There is intentionally no CAL500 split to spend.

A/B remain independent and unpooled. Fresh checks retain exact query parity, structural delegation, same-winner containment, historical zero-catastrophe tail gate, meaningful capture gain over PRESERVE, and endpoint non-inferiority.

Do **not** run official paper-facing closed loop before TRAIN and double-fresh pass. If both fresh blocks pass, freeze the promoted V43 arm, run one independent full-validation reproduction, then run official nuPlan closed-loop evaluation.

---

## 18. Directions explicitly frozen/forbidden after V42

Do not repeat:

- MLP/nonlinear/high-capacity pure-19D selected-value heads;
- more hurdle/sign/upside/downside heads on the same representation;
- threshold, lambda, alpha/q, candidate-count, top-K, temperature sweeps;
- standalone catastrophe classifier or binary safety veto;
- unconstrained selected-policy translation/slope calibration;
- second-best fallback or value-based reranking;
- richer endpoint polynomial / naive `[candidate, incumbent, delta]` concatenation;
- basepoint common shift/reservation;
- selection-geometry reservation;
- naive current QUALITY+RISK feature concatenation with learned coefficients.

RSMR ordering, structural delegation, incumbent/no-fallback containment, B16/M24 interface, EAF, and exact attribution remain frozen unless V43 oracle evidence later identifies the bounded interface itself as the missing value-information source.

---

## 19. Paper-level contribution if V43 is supported

The claim should not be “we use multimodal response hypotheses” or “we add CVaR”; those ideas already exist broadly in uncertainty-aware/contingency planning.

The stronger contribution is:

> **Operator-conditioned counterfactual value sufficiency under a bounded auditable planner interface.** Relative challenger selection and absolute deployment confirmation require different sufficient statistics. After an ordinal extremal proposal is frozen, exact currently observable cost components should be accounted for analytically in the correct normalized units, while only the unobserved counterfactual remainder is learned. Response uncertainty is introduced solely as a proposal-conditioned consequence observable, preserving monotone intervention containment and the original evidence-query budget.

A concise mechanism chain is:

`bounded evidence -> exact EAF attribution -> frozen ordinal RSMR proposal -> analytic current consequence -> selected-evidence counterfactual response consequence -> endpoint residual -> same-proposal accept/abstain`.

This is an algorithmic/statistical decomposition claim rather than a feature-engineering claim, and V43's Q/CV/MEAN/ROBUST branches are designed to falsify progressively stronger versions of it.
