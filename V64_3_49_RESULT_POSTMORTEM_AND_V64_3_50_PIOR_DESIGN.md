# V64.3.49 Result Postmortem and V64.3.50 PIOR Design

## Executive decision

**V64.3.49 is engineering-valid and scientifically fails its preregistered nested TRAIN gate. No V49 fresh A/B was consumed.** The failure is not a code/data-loader/provenance error and therefore must be attributed to the SIIR mechanism rather than repaired as V49.1.

The decisive failure is the independent selected-risk identification gate. `OBS-SIGN` exactly replays V48 SIGN-NOMULT, while selection-interventional SIIR is worse:

- EGO-REF risk AUC: `0.6298288272`
- OBS-SIGN risk AUC: `0.6139192606`
- SIIR risk AUC: `0.6081222525`
- SIIR > OBS: `3/5` folds
- SIIR > EGO-REF: `1/5` folds
- held-out intervention AUC: OBS `0.6171768707`, SIIR `0.6070578231`

This triggers the prior V49 STOP condition exactly: **close the current offline selected-risk family and move the next evidence source to on-policy/closed-loop/interventional selected-outcome supervision.**

## 1. Reliability audit

| Check | Result |
|---|---:|
| Uploaded V49 code SHA256 | `4bc0044ffb311eaf3db77d221bad58ca8d5de21f6987c337ea098874811d112b` |
| Preregistered V49 code identity | exact match |
| Uploaded result ZIP SHA256 | `62e454ff9305b885c75da6a9d6ce792342d11d04b48cede44b0bb9660e60fbf9` |
| V50-independent check of V49 source manifest | 907/907, zero mismatch |
| Server targeted regression | 242/242 PASS |
| Independent replay of uploaded code | 242/242 PASS |
| V48 science lock | PASS |
| Nested outer folds | 5/5 |
| Frozen TRAIN scene audit | 782/782 unique |
| Fresh A/B selected by V49 | **no** |
| Termination | preregistered TRAIN scientific STOP |

The V49 launcher stops immediately after `fit_v64_3_49_eaf_icer_siir` returns its scientific failure. The `double_fresh_selection` stage is never reached.

## 2. V49 preregistered GO/STOP result

| TRAIN arm | selected | positive | capture | ΣΔT | catastrophe | no-op false | NegRMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| RSMR | 502 | 221 | 38.5017% | +43.2941 | 28 | 107 | 0.35569 |
| EGO-REF | 251 | 136 | 23.6934% | +59.5327 | 9 | 45 | 0.22331 |
| OBS-SIGN | 411 | 187 | 32.5784% | +53.4956 | 18 | 78 | 0.27065 |
| SIIR | 464 | 208 | 36.2369% | +53.4681 | 21 | 93 | 0.27927 |

SIIR crosses the old capture floor (`35.5017%`) but does not satisfy the full `existence_and_capture` contract because false intervention on no-positive-opportunity scenes is `93` rather than the required <=85.6. Tail/population/fold-sum subgates pass, but promotion is impossible because the independent identification gate also fails.

### Formal V49 verdict

`selection_interventional_risk_does_not_outperform_observational_selected_risk_close_current_offline_selected_risk_family`

This is the exact failure string written by the preregistered V49 code, not a post-hoc reinterpretation.

## 3. What SIIR actually changed

OBS-SIGN vs SIIR selected-set rotation:

| subset | scenes | positives | material positive | catastrophe | ΣΔT |
|---|---:|---:|---:|---:|---:|
| both | 405 | 185 | 43 | 18 | +53.4961 |
| OBS-only | 6 | 2 | 0 | 0 | -0.00056 |
| **SIIR-only** | **59** | **23** | **4** | **3** | **-0.02806** |
| neither (within RSMR) | 32 | 11 | 3 | 7 | -10.1735 |

The SIIR-only region is the critical mechanism evidence. Its positive value sum is `+4.2150`, while its nonpositive value sum is `-4.2431`. The intervention-trained law therefore rotates the policy into a region containing both real material opportunities and real catastrophes, almost exactly cancelling in aggregate.

This rules out two weak explanations:

1. **Not blanket abstention.** SIIR accepts 59 proposals that OBS rejects.
2. **Not merely over-conservative calibration.** The newly accepted region itself is sign/tail mixed.

## 4. Threshold postmortem: not a calibration problem

A diagnostic-only oracle sweep is performed on the OOF SIIR risk margin (`risk - fold_threshold`). This is not used to tune V50.

No common threshold offset satisfies the unchanged capture/no-op/tail/population gate.

- Among thresholds that preserve the frozen capture floor, the minimum no-op false count is `91` (required <=85), with 21 catastrophes.
- Among thresholds achieving no-op false <=85, the maximum capture is only `33.798%`, below the `35.5017%` floor.

Thus the V49 failure is again a **risk-ordering / outcome-identification** problem, not a scalar boundary problem.

## 5. Why random-prefix selection intervention was insufficient

V49 intended to break observational selection-regime correlations by changing the TRAIN selection measure. That intervention is scientifically clean as a falsification test, but the realized measure is weakly coupled to the deployed extremal event:

- full-set RSMR proposal population: 502;
- intervention-selected event population: 356;
- same action as full-set winner: 277/356 = **77.81%**;
- actually changed selected action: only 79 events;
- mean prefix fraction: `0.6643`;
- mean prefix size: `7.93` for mean K `12.33`.

The changed minority are not the deployed full-set winner under a different outcome; they are sub-extremal winners under a different candidate exposure. Consequently, V49 changes the sampling measure but still never observes the actual counterfactual outcome of **executing the deployed full-set winner instead of the incumbent**.

Tuning the prefix distribution/seed is prohibited by preregistration and would not answer this structural objection anyway.

## 6. Evidence chain and dominant bottleneck

The evidence chain is now stronger than “another model failed”:

- V44: full-horizon ungated prospective occupancy is a strong value mediator.
- V45: agent-local continuous longitudinal response is identifiable and selectively useful.
- V46: better predictive regression / identifiable variance can fail or hurt the extremal deployment decision.
- V47: strongly identifiable EGO-reference consequence remains insufficient for the signed-mean zero decision.
- V48: in-domain observational selected-risk and multiplicity conditioning do not transport to double fresh.
- V49: changing only the offline selected-event measure with a label-free selection intervention does not recover the outcome law.

Therefore the primary bottleneck is now:

> **deployment-aligned selected-outcome evidence: identify the outcome law of the actual frozen full-set RSMR proposal under an intervention against the incumbent.**

The important distinction is now:

`prediction sufficiency != representation identifiability != post-selection risk fit != selected-outcome causal sufficiency`.

## 7. Layer maturity after V49

| Layer | Status | V50 policy |
|---|---|---|
| B16/M24 bounded planner interface | mature | freeze |
| EAF complete frontier / exact attribution | mature, paper backbone | freeze |
| support/admissibility | mature | freeze |
| RSMR ordinal extremal selection | most mature learned layer | permanently freeze |
| incumbent / no-fallback containment | mature | permanently freeze |
| EPV / QUALITY | real partial mediators | preserve |
| V44 ungated prospective occupancy | strong success | preserve |
| V45 agent-local longitudinal response | successful supporting layer | preserve |
| V47 EGO-REF | strong supporting consequence mediator | preserve |
| V46 variance / handcrafted temporal profile | closed | do not revive |
| V47 constant-drift AGENT-2D | closed | do not revive |
| V48 multiplicity/logK | fresh-falsified | closed |
| V49 random-prefix offline selection intervention | TRAIN-falsified | closed |
| offline selected-risk family | **closed** | no more estimator tricks |
| actual selected-outcome intervention supervision | untested | **V50 primary layer** |
| absolute zero / material opportunity / hard tail | unresolved final gate | evaluate under selected-outcome law |

## 8. V64.3.50 PIOR mechanism

### 8.1 Frozen proposal

Runtime still begins with the full-set frozen RSMR proposal

`b_hat = argmax_b u_RSMR(b)`

under the unchanged support/admissibility interface. No V50 training outcome participates in this argmax.

### 8.2 Paired one-shot intervention

For each of the exact 502 TRAIN scenes with an RSMR proposal, run a deterministic paired nuPlan closed-loop experiment:

**control**

1. remain on incumbent before a proposal exists;
2. at the first frozen RSMR proposal event, execute incumbent;
3. remain on incumbent afterward.

**treatment**

1. remain on incumbent before a proposal exists;
2. at the first frozen RSMR proposal event, execute exactly `b_hat` once;
3. return to incumbent afterward.

Thus the two trajectories differ only by the one selected deployment proposal whose retention law is being identified.

The V50 runner requires exactly one `pior_probe_fired` event in every scenario of both arms. A partial probe population is a data/engineering STOP, not a smaller training set.

### 8.3 Outcome functional

The paired intervention supplies an actual simulation outcome rather than an offline teacher surrogate.

Let `S_T` and `S_C` be official per-scenario closed-loop scores. Define:

`Delta_CL = S_T - S_C`.

A positive selected outcome requires:

- `Delta_CL > 0`; and
- no treatment degradation in:
  - no-ego-at-fault-collision score;
  - TTC-within-bound score;
  - drivable-area-compliance score.

PIOR uses only the **sign** of this outcome for pairwise ranking. It does not add a catastrophe coefficient or tune a safety weight.

### 8.4 Risk state and learner remain frozen

`z = [Q, P-Q, E-P]`.

Same as V49:

- zero-bias pairwise sign-risk;
- fixed `lambda=1`;
- no K/multiplicity/operator coordinate;
- no bigger MLP;
- no new future observable;
- capture-derived conditional false-veto budget `alpha_ret=0.0779185520`.

Only the supervised selected-outcome source is changed.

### 8.5 Independent identification gate

On nested held-out paired outcomes, compare bad-outcome AUC of:

1. `-EGO-REF`;
2. V49 `OFFLINE-OBS` OOF risk;
3. V50 `PIOR` OOF risk.

PIOR is identified only if:

- aggregate `AUC_PIOR > max(AUC_OBS, AUC_EGO)`;
- PIOR > OBS in >=4/5 folds;
- PIOR > EGO in >=4/5 folds.

### 8.6 Paired-outcome deployment gate

At the fold-calibrated retention boundary, PIOR must simultaneously satisfy:

- beneficial RSMR proposal retention recall >= `1-alpha_ret`;
- hard-harm count non-increasing and strictly reduced if baseline hard harm exists;
- negative closed-loop score RMS non-worse;
- strictly fewer retained nonbeneficial interventions;
- retained closed-loop score-delta sum non-worse than retaining all frozen RSMR proposals;
- hard-harm and score-delta non-harm in every outer fold;
- >=64 retained proposals and >=32 retained beneficial proposals.

Only identification + outcome gate yields TRAIN PASS.

## 9. V50 STOP branches

### Engineering/data STOP

If exact 502-token matching, raw DB discovery, treatment/control completion, per-scenario metric identity, required hard-safety metrics, or one-probe-per-scene invariants fail: repair collection only. No scientific conclusion.

### PIOR AUC not identified

If aligned paired outcomes still cannot be ranked from frozen Q/P/E, the problem is no longer the offline label source alone. Close Q/P/E as a sufficient selected-risk state. Next work must use **on-policy outcome state/evidence acquisition** or a structured closed-loop outcome model, while preserving RSMR and no-fallback containment.

### AUC identified but retention gate fails

Then the selected-outcome signal exists but the current sign-risk functional/state is not deployment-sufficient. The next CCF-A-worthy branch is a structured interventional outcome functional, not threshold/class-weight/MLP tuning.

### TRAIN PASS

Freeze the V50 artifact and only then consume untouched paired closed-loop validation. Do not tune from TRAIN causal outcomes.

## 10. Dataset contract implemented in V50

The launcher directly follows the supplied filesystem layout:

NPZ cache root:

`/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2`

with four children:

- `train_boston`
- `train_pittsburgh`
- `train_singapore`
- `train_vegas_2`

and nested `<collection-time>_veh-*/...npz` files.

Raw DB split root:

`/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits`

with direct `.db` files under:

- `train_boston`
- `train_pittsburgh`
- `train_singapore`
- `train_vegas`

The manifest explicitly maps `train_vegas_2 -> train_vegas`.

## 11. CCF-A paper direction

The current TeX is a useful architectural wedge because it already formalizes a bounded auditable interface, extremal proposal construction, separation of proposal and confirmation, monotone no-fallback containment, and fail-closed untouched validation. The accumulated V44–V49 evidence indicates that the eventual paper should not be organized around PTMC/OCRR/SIIR as a sequence of tricks.

A stronger main line is:

> **Selection–Valuation–Outcome Sufficiency under a Bounded Auditable Planner Interface**

The potential algorithmic contribution is the decomposition of extremal planning into operators that need different evidence:

1. bounded evidence is sufficient for **ordinal extremal selection** (RSMR);
2. prospective interaction/reference statistics are required for **absolute valuation**;
3. a post-selection execution operator requires **deployment-aligned selected-outcome evidence**, and observational/offline selected-risk surrogates need not be transport sufficient.

If PIOR succeeds and then reproduces on untouched/closed-loop evidence, the paired selected-outcome intervention is a credible mechanism-level contribution. If it fails, that failure still gives a principled stop to the current Q/P/E retention state and points to genuinely on-policy state/evidence acquisition rather than more offline feature engineering.
