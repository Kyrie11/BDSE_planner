# V64.3.32.1 result postmortem and V64.3.33 SPCR design

## 1. Attribution verdict

The uploaded V64.3.32.1 WEIGHTFIX execution is **engineering-valid for TRAIN-level algorithm attribution of the complete nested SSIR mechanism**. It is not a fresh-validation result because the preregistered nested TRAIN gate failed before CAL500 / A500 / B500 token selection.

The result therefore supports a development-level statement about SSIR, but it must not be reported as an untouched-fresh SSIR failure.

The following contracts were re-audited:

- the V32.1 scene-equal ridge hotfix is active: each fit scene has total squared-loss mass exactly one;
- the five outer-fit loss-weight sums equal their direct-scene counts: 483 / 500 / 461 / 450 / 452;
- feature moments use only a normalized copy of these weights; ridge loss and leverage Gram use the unnormalized scene-total-one weights;
- fixed ridge lambda remains 1; fixed conformal alpha remains 0.05;
- no traceback, CUDA/OOM, shape/schema failure, or evaluator crash occurred;
- the STOP is the frozen nested scientific gate;
- no non-empty CAL500/A500/B500 token manifest exists, so the prior fresh population was not spent;
- runtime SCIR/SSIR feature construction, candidate eligibility, positive-LCB selection, incumbent default, no-fallback behavior, and structural delegation remain consistent with the V32.1 configs.

Thus V32.1 is qualitatively different from the engineering-invalid V32 execution: the intended V32.1 statistical objective really ran.

## 2. V32.1 result

Frozen population:

- TRAIN scenes: 3000;
- direct admissible + support-positive challenger edges: 9394;
- direct scenes: 782;
- positive edge fraction: 34.32%;
- all-edge teacher-improvement sum: -2624.17.

Corrected conditional-mean ranker:

- selected proposals: **476**;
- selected positive: **260**;
- precision: **54.62%**;
- positive-opportunity capture: **45.30%**;
- selected teacher-improvement sum: **-18.0304**;
- worst selected teacher improvement: **-6.7354**;
- negative RMS: **0.6270**;
- selected catastrophes at teacher improvement <= -0.5: **47**.

Per-fold selected sums are -8.660 / -8.764 / +8.277 / +0.982 / -9.866. Two folds have nonnegative aggregate sum, but every fold contains a selected intervention below the historical -0.5 catastrophic threshold. The repaired mean representation therefore still has useful signal but is not a deployable selected-path operator.

The historical V31-style selected-proposal common conformal veto, replayed on the corrected nested splits, retains only **3** proposals. Only one is beneficial, and the surviving worst proposal has teacher improvement **-1.2330**. Therefore simply attaching a common post-selection offset to the corrected mean selector is not an adequate solution.

The V32 SSIR scene-simultaneous mechanism selects **0** interventions in every outer fold. The five scene-max conformal quantiles are approximately:

- 2.995;
- 4.035;
- 3.291;
- 4.845;
- 4.752.

Meanwhile the corrected selected-proposal ridge leverage scale is close to one (mean approximately 1.011). Hence the all-candidate simultaneous certificate is far larger than the predicted intervention benefit on the deployment candidate set and collapses useful coverage to zero.

The correct conclusion is **not** that conformal inference in general failed. It is that this particular decomposition—edge-wise conditional mean plus candidate-leverage normalization plus a 95% scene-max positive bound over every eligible challenger—does not provide a useful selection/coverage tradeoff on this development population.

## 3. The decisive new failure slice: whether to intervene is itself a structured decision

The V32.1 scene audit contains 782 direct scenes and permits exact selected-path slicing.

Of the 476 corrected mean proposals:

- 216 are non-positive;
- 47 are catastrophic at <= -0.5;
- 110 proposals are emitted in scenes in which **no challenger has positive teacher improvement at all**;
- those 110 scenes contribute **-60.1520** total teacher improvement;
- they account for **110 / 216 = 50.9%** of all non-positive selected proposals;
- they account for **31 / 47 = 66.0%** of all selected catastrophes.

In contrast, on selected scenes that actually contain at least one positive challenger:

- selected count: 366;
- selected positive: 260;
- precision: **71.04%**;
- catastrophic count: 16;
- selected teacher-improvement sum: **+42.1216**.

This is the strongest mechanism evidence in V32.1. The dominant error is no longer accurately described only as “wrong winner among many positive-looking candidates.” A large part of the harmful tail comes from scenes where the correct structured decision is **the incumbent / no intervention**, but the edge-wise positive-score operator is forced to make the intervention-existence decision indirectly.

Representative worst corrected-mean selections include:

- `4750b9c4cfbd5d3f`: no positive opportunity, selected teacher improvement -6.7354;
- `a1c1a996e52f5c4e`: no positive opportunity, -4.0413;
- `2b32a9f406845f75`: no positive opportunity, -3.8065.

The candidate-specific leverage scale is not sufficiently discriminative to repair this issue: it remains near one on selected proposals and its relation to overprediction error is weak. This is consistent with the fact that ridge leverage is a geometric normalization, not a semantic model of whether an intervention exists.

## 4. Algorithmic conclusion about SSIR

V32.1 SSIR, as frozen, **fails its complete nested TRAIN mechanism gate and must not be promoted**.

The failure has two coupled causes:

1. **Edge-wise mean semantics remain misaligned with the deployed scene-level null-action/argmax decision.** The corrected mean model enriches positives relative to the 34.3% edge prevalence, but still proposes in many no-opportunity scenes and still admits an extremal heavy tail.
2. **All-candidate simultaneous positive certification is statistically inefficient for this candidate set and scorer.** The scene-max residual is driven by alternatives the deployed policy would rarely choose, causing q≈3–5 and universal abstention.

This result falsifies the stronger V32 hypothesis that candidate-wise simultaneous lower bounds over the complete eligible set are the right first-order solution to the V31/V32.1 selected-path problem. It does **not** justify alpha tuning, leverage tuning, a weaker threshold, or a smaller candidate subset selected post hoc.

## 5. Evidence chain through V32.1

The accumulated falsification chain is now:

1. learned acquisition / coreset / beam / swap: optimizing evidence-subset proxies is not the first-order solution;
2. EAF: complete action-local frontier plus exact selected-evidence attribution remains valuable and is retained;
3. rich attribution / KNN geometry: richer representation does not imply stable reliability geometry;
4. DRC: proposal signal exists but selected tail is unstable;
5. semantic/type-local KNN: local confirmation geometry does not generalize reliably;
6. PTMC: a generic tail classifier can look excellent on TRAIN and have zero harmful-veto specificity on fresh data;
7. FCR: global complete-frontier reconstruction fidelity is not decision sufficiency;
8. FBIC/V30.3: exposing more already-queried decisive evidence does not reproducibly improve direct intervention, so capacity-only is closed as a first-order solution;
9. corrected V32.1 mean: same-scene continuous incumbent contrast contains useful signal but edge-wise positive-score argmax still has a heavy selected tail;
10. V32.1 SSIR: simultaneous all-candidate positive certification is too conservative and does not solve the scene-level null-action/ordering problem.

## 6. Dominant bottleneck after V32.1

The dominant bottleneck should be tightened from

> selection-stable counterfactual lower-bound ordering

into

> **scene-structured intervention-existence and selected-policy ordering reliability at the direct incumbent-replacement boundary.**

The deployment unit is not an independent edge. It is the structured choice

`argmax over {incumbent/no-intervention} union admissible challengers`.

A sufficient mechanism therefore needs to answer jointly:

- should the planner leave the incumbent at all?;
- if yes, which challenger dominates the incumbent and all rival challengers?;
- after this deterministic structured policy has selected one proposal, is that selected policy output reliable enough to execute?

This is more precise than returning to a generic classifier, risk head, or feature geometry.

## 7. V64.3.33 SPCR — Structured Policy-Calibrated Recovery

V33 changes the **training/selection unit**, not the evidence budget, acquisition, EAF, support prerequisite, structural guard, or runtime query path.

### 7.1 Incumbent-augmented structured alternative set

For a direct scene with admissible incumbent i and support-positive admissible challengers C(x), keep the same frozen 19-D challenger-minus-incumbent feature x_b from V32.1.

Add the incumbent as an explicit pseudo-item:

- x_i = 0;
- teacher improvement Delta_i = 0;
- runtime score s_i = 0 exactly.

The teacher-best alternative is

`b_T = argmax over {i} union C(x) of Delta`.

If every challenger has Delta<=0, the incumbent is teacher-best. Therefore a no-opportunity scene directly teaches **no intervention** instead of contributing only unrelated negative edge targets.

### 7.2 Scene-equal teacher-best-vs-rivals pair-gap objective

For each rival r != b_T, train on

- pair feature d = x_{b_T} - x_r;
- target gap g = Delta_{b_T} - Delta_r >= 0.

Every direct scene has total pair-loss mass exactly one. A fixed ridge lambda=1 is retained; there is no validation sweep. Feature scaling is a zero-preserving weighted RMS, no mean centering and no intercept, so the incumbent remains exact score zero.

At runtime:

`s_b = (x_b / scale)^T w`.

The RANK control chooses the highest-scoring challenger only when its score is positive; otherwise it returns the incumbent. The semantic zero boundary is therefore the explicit incumbent score, not a tuned threshold.

This objective aligns fit and deployment:

- no-opportunity scenes teach challengers below incumbent;
- opportunity scenes teach the teacher-best challenger above incumbent;
- best-vs-rival pairs teach winner identity under the same scene-level extremal choice.

### 7.3 Selected-policy conformal certificate

After the structured selector is frozen, calibration no longer takes a max over every candidate. It evaluates exactly the deterministic proposal emitted by the frozen structured policy.

For every calibration scene in which the frozen ranker emits a positive proposal b_hat, use one residual

`R = s_{b_hat} - Delta_T(b_hat;i)`.

With frozen alpha=0.05, compute one one-sided split-conformal q. MAIN executes the **same proposal** iff

`s_{b_hat} - q > 0`.

Otherwise MAIN returns the incumbent. It cannot re-rank, fall through to second best, or create a new intervention path.

Under exchangeability of calibration/future proposal-emitting direct scenes after the deterministic X-only structured selector is frozen, the selected policy outputs remain an exchangeable filtered population. The split-conformal lower bound gives marginal one-sided coverage for the frozen selected-policy output. A harmful accepted proposal is contained in the selected-policy miscoverage event. This is not a per-scene conditional, distribution-shift, or closed-loop absolute-safety guarantee.

The crucial difference from V32 is efficiency/alignment: reliability is calibrated on the action the deployed structured policy actually proposes, rather than requiring simultaneous positive coverage for every candidate that could have been considered.

## 8. V33 causal controls and preregistered gates

V33 keeps four recovery views:

- PRESERVE: admissible incumbent default with historical V20 direct semantics;
- MEAN: exact corrected V32.1 scene-equal edge-mean control;
- SPCR-RANK: only the structured incumbent-augmented pair-gap selector;
- SPCR-MAIN: exact same RANK proposal plus selected-policy conformal accept/incumbent veto.

This gives two causal questions:

1. Does changing the fit/selection unit reduce no-opportunity false interventions and catastrophic proposals versus corrected MEAN?
2. Given that frozen structured proposal, does selected-policy calibration improve its tail without creating another action path?

### Nested TRAIN gate

Reuse the exact V32 fold assignment for paired development attribution. In each outer fold:

- 3 folds fit the structured ranker;
- 1 disjoint fold calibrates selected-policy q;
- 1 fold tests the complete MAIN mechanism.

Before CAL/fresh selection, require:

- structured RANK has fewer no-positive-opportunity false interventions than corrected MEAN;
- RANK catastrophic count is non-worse than MEAN and either selected sum or catastrophic count strictly improves;
- all 5 MAIN test folds have selected teacher-improvement sum >=0 and zero selected catastrophes <=-0.5;
- aggregate MAIN selected count >=64;
- aggregate MAIN positive count >=32.

Failure stops before fresh selection. No lambda/alpha/feature/threshold rescue is allowed.

### Independent CAL500 + double fresh

V32.1 did not spend CAL/fresh. V33 keeps the permanent 10700-token design exclusion but uses a new label-free hash seed:

`v64.3.33-eaf-icer-spcr-cal500-double-fresh-v1`.

Only after TRAIN passes, select new CAL500 + A500 + B500.

CAL500 must provide at least 64 structured-policy proposals and fixes exactly one alpha=0.05 q.

A/B each run six paired arms:

RAW / V20 / PRESERVE / MEAN / SPCR-RANK / SPCR-MAIN.

Each block is judged independently. No pooling.

The fresh screen explicitly requires the structured RANK mechanism to reduce no-opportunity false intervention count versus MEAN before MAIN can be interpreted. MAIN must then provide >=3 pp direct useful-capture gain over PRESERVE, selected count >=8, selected sum >=0, worst >-0.5, NegRMS and precision non-worse than PRESERVE, certificate tail non-worse than RANK, monotone same-proposal containment, and endpoint non-inferiority to both PRESERVE and V20.

## 9. Paper mainline and CCF-A-oriented novelty

Do not headline ridge regression or conformal prediction as the novelty. The CCF-A-oriented contribution stack should remain:

1. **Formulation:** decision sufficiency under a bounded auditable planner interface is an operator-level property, not representation fidelity.
2. **Instrumentation:** EAF provides complete action-local frontier evidence and exact attribution under the fixed query/interface contract.
3. **Structured intervention operator:** the incumbent is an explicit null intervention inside the same extremal decision set; training is aligned to the scene-level best-vs-rivals deployment decision rather than independent edge prediction.
4. **Policy-level calibration:** statistical reliability is attached to the frozen structured policy output, avoiding the complete-candidate simultaneous multiplicity burden while preserving a one-sided marginal finite-sample statement under exchangeability.
5. **Structural containment:** MAIN can only keep the exact structured proposal or return the incumbent; structural-domain delegation and no-fallback behavior remain deterministic guarantees.
6. **Evidence discipline:** nested development gate, independent CAL, double-fresh A/B, later one frozen full-validation reproduction, then closed-loop, strong baselines, budget/latency/tail/CI reporting.

A possible final conceptual headline, only if V33 and later independent validation succeed, is:

> **Intervention-Conditioned Decision Sufficiency via Incumbent-Augmented Structured Recovery and Selected-Policy Calibration under a Bounded Auditable Planner Interface.**

No acceptance guarantee is possible. The mechanism is designed to support a CCF-A-level causal and theoretical story if the independent evidence succeeds.

## 10. Directions that remain frozen / should not be repeated

Retain all historical no-repeat constraints, especially:

- learned acquisition, coreset, beam, swap, selector-v2;
- broad B/M sweep, capacity-only, same-bank rebinding;
- FCR-v2 / global frontier reconstruction fidelity;
- DRC K/threshold/downside tuning;
- attribution/transition/type/family KNN, radius/OOD rescue;
- PTMC/classifier v2/v3 or another generic tail classifier;
- naive feature concatenation;
- action blacklist;
- learned incumbent-to-anchor veto;
- support/scalar threshold rescue;
- pooling A/B or using endpoint gains from another runtime domain to rescue a direct-path failure;
- ridge lambda / conformal alpha / leverage multiplier / positive-score threshold sweep;
- weakening V32 simultaneous q or restricting its candidate set after seeing V32.1 to manufacture coverage;
- another generic edge/listwise score that omits the incumbent as an explicit null action and does not align reliability to the deployed selected policy.

V33 is deliberately a change of **decision unit and calibration unit**, not another feature/head/threshold iteration.
