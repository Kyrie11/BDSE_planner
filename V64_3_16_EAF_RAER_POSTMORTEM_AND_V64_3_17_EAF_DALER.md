# V64.3.16 EAF-RAER Result Attribution and V64.3.17 EAF-DALER Design

Date: 2026-08-17

## 0. Executive conclusion

The uploaded V64.3.16 screen gives a substantially cleaner causal diagnosis than the previous iteration:

1. **The frozen EAF representation is still useful.** On the fresh 500-scene screen, raw EAF reduces teacher regret from the DARM/selected-local anchor's `29065.54` to `17435.90` (about **40.0% lower**), despite lower exact teacher match. The complete DARM-anchor frontier remains at `100%` coverage.
2. **Evidence-attributed reliability is real, not an artifact of train fitting.** RAER's train scene-group holdout AUC is `0.7151`, and its fresh all-frontier AUC is `0.7012` over `13,292` challenger edges. There is therefore no causal justification to reopen acquisition, B/M, DARM/DBR, or the EAF checkpoint in the next iteration.
3. **RAER's filtering behavior works; its alternative-challenger re-ranking does not.** Conditional on keeping the legacy challenger, the teacher-better rate is `83.25%`; conditional on choosing any non-anchor challenger it is `80.35%`. But the 20 scenes in which RAER actually selects a different non-anchor challenger have only **50%** teacher-better precision. This is the failed causal link.
4. **The original V64.3.16 checker understates this distinction because its selected-teacher-better statistic counts anchor fallback as a bad challenger.** RAER falls back to anchor in 235/464 raw-proposal scenes (`50.65%`). The unconditional `39.66%` selected-teacher-better number therefore mixes abstention with challenger quality and should not be used alone to diagnose representation quality.
5. **RAER is mostly a more permissive EAIR, not yet a successful extremal recovery operator.** Its final action equals scalar EAIR in `465/500 = 93%` scenes. Compared with raw EAF, RAER cuts harmful interventions by `6.6pp` and retains `71.4%` of beneficial interventions; compared with scalar EAIR it recovers `4.60%` relative regret, but loses `1.4pp` exact match.
6. **There is a concrete train/deployment semantic mismatch in V64.3.16.** RAER eligibility requires only positive raw EAF margin, while the final frozen one-sided guard requires `margin >= 0.015`, non-negative EAF score gain, and the evidence certificate. `30.39%` of raw-proposal scenes have a legacy proposal below the final `0.015` flip margin, and RAER still leaves `3.2%` of scenes to be blocked by the final guard after learned re-ranking.
7. **The right next change is therefore the extremal decision operator, not another threshold.** V64.3.17 is implemented as **EAF-DALER: Evidence-Attributed Deployment-Aligned Listwise Extremal Reliability**. It makes the DARM anchor an explicit abstention item, trains scene-level listwise ordering over only frozen deployment-executable challengers, and keeps the old one-sided/evidence certificate unchanged as an invariant.

The new novelty statement is:

> **evidence-attributed, deployment-aligned listwise reliability for extremal decision selection under a fixed planner-interface evidence budget.**

This is a stronger and more defensible story than “a logistic gate after top-1” or “multiply probability by margin”, because the contribution is the decision operator and its auditable evidence/deployment semantics rather than a generic confidence model.

---

## 1. What the paper is currently trying to establish

The uploaded TeX is the original BDSE paper. Its core motivation is that a real-time planner has a bounded interface and should compress scene uncertainty for the **downstream action**, not for distribution reconstruction. The intended abstraction is:

`dense context -> auditable local evidence atoms -> budgeted evidence selection -> margin-preserving action comparison -> final trajectory`.

The paper defines a finite candidate set, a robust offline teacher cost `J_T`, local evidence atoms, pair-conditioned teacher-margin contributions, a budgeted selector, and a pairwise/risk-aware tournament. Its theoretical object is decisive-margin preservation: if the selected evidence approximately preserves teacher-best-versus-rival margins, the budgeted planner either preserves the teacher action or has bounded teacher regret.

The data protocol is nuPlan-style log replay: runtime features use past/current ego state, tracked agents, map, traffic-light, route, and mission information; logged future information is used only to construct offline teacher labels. The paper plans open-loop log replay plus non-reactive/reactive closed-loop evaluation and reports both standard nuPlan metrics and BDSE-specific evidence/margin diagnostics.

The present code has evolved materially beyond the paper, so **the paper should not yet be edited to describe RAER/DALER as established results**. The final manuscript should be synchronized only after a full-val reproduction, frozen test, and closed-loop evaluation establish the new mechanism.

---

## 2. V64.3.16 fresh-screen results

All three arms use the same fresh 500 validation scenes, selected by frozen scenario-token identity/hash and with zero overlap with the prior 1200 design scenes.

| Method | Teacher match | Teacher regret | Deployed flip | Beneficial intervention | Harmful intervention |
|---|---:|---:|---:|---:|---:|
| DARM / selected-local anchor | **14.40%** | 29065.54 | — | — | — |
| raw EAF | 11.40% | **17435.90** | 57.00% | 5.60% | 8.60% |
| scalar EAIR | **17.80%** | 18574.57 | 35.60% | 4.20% | **0.80%** |
| RAER | 16.40% | 17720.59 | 40.80% | 4.00% | 2.00% |

Additional mechanism diagnostics:

| Diagnostic | Result |
|---|---:|
| RAER train scene-group holdout AUC | **0.7151** |
| scalar EAIR train holdout AUC | **0.7665** |
| RAER fresh all-frontier AUC | **0.7012** |
| Fresh all-frontier edges | 13,292 |
| Raw proposal scenes | 464 / 500 |
| RAER fallback to anchor | 235 / 464 = **50.65%** |
| RAER keeps legacy challenger | 209 / 464 = **45.04%** |
| RAER chooses alternative challenger | 20 / 464 = **4.31%** |
| Keep-legacy teacher-better precision | **83.25%** |
| Selected non-anchor teacher-better precision | **80.35%** |
| Alternative-challenger teacher-better precision | **50.00%** |
| Teacher-better fraction among raw challengers that RAER rejects to anchor | **58.30%** |
| Final RAER action identical to scalar EAIR | **93.0%** |
| RAER post-selection final-guard block rate | **3.2%** |

The current screen therefore passed the broad **capacity**, **preservation**, and **endpoint** gates but failed the intended **extremal recovery mechanism** gate. `full_promotion=false` is the correct decision.

---

## 3. Algorithm attribution: what is effective

### 3.1 Frozen EAF complete-frontier value should be retained

Raw EAF's regret is about `40.0%` lower than the DARM anchor. This is strong evidence that the complete frontier contains useful teacher-cost ordering information even though its extremal action produces poor exact match. The correct diagnosis is not “EAF value learned nothing”; it is “useful value information is converted into an unreliable extremal action”.

The fresh all-frontier reliability AUC of `0.7012` further shows that the already-computed frozen EAF value/attribution statistics carry information about whether a challenger truly beats the DARM anchor. Broadly unfreezing the representation before fixing the decision operator would confound the causal diagnosis.

### 3.2 Evidence attribution should remain part of the main story

The reliability signal survives on fresh validation and the earlier OCFI result already showed that attribution magnitude is not simply an uncertainty radius. The useful semantic is closer to **how strongly the selected B evidence supports a challenger relative to the frontier**. That is a better fit to a reliability/ordering operator than to a global subtraction penalty.

### 3.3 The one-sided/evidence certificate should remain frozen

RAER reduces harmful interventions without changing acquisition or EAF value, and the old final certificate continues to serve as a safety/preservation invariant. There is no evidence in this screen that relaxing the certificate is needed. In fact, the 3.2% post-selection block rate reveals that the learned selector should be brought into alignment with it rather than bypassing it.

### 3.4 Fixed planner-interface budget remains a valid contribution, but the wording must be `B<=16`

The V64.3.16 replay provides a more complete accounting than the previous design note:

- train: `2438/3000 = 81.27%` scenes use exactly 16 atoms;
- fresh validation: `435/500 = 87.0%` use exactly 16;
- validation discovery: `2193/2500 = 87.72%` use exactly 16;
- **every** scene with `proposal_candidate_atom_count >= 16` uses exactly 16 atoms;
- scenes below 16 occur only when fewer than 16 eligible proposal atoms exist.

Therefore the faithful paper statement is: **planner-interface cap `B<=16`, exact B=16 whenever at least 16 eligible proposal atoms exist; report the exact-budget scene rate.** Do not manufacture filler atoms to make every scene exactly 16.

---

## 4. What is not working in V64.3.16

### 4.1 Independent edge reliability is not the same as scene-level extremal ordering

RAER trains a class-balanced independent edge classifier for `P(teacher challenger beats anchor)`, then uses

`u_b = p_b * max(M_EAF(b, anchor), 0)`

and takes the extremum. This still creates a second winner's-curse problem: a probability that is individually calibrated for edges is not trained to make the correct **within-scene top decision among many alternatives plus abstention**.

The data expose this cleanly. RAER is strong when it keeps the existing challenger, but its alternative-challenger precision is only 50%. The failure is not “no probability signal”; it is “the ordering objective does not train the operation used at deployment”.

### 4.2 The anchor is not part of the learned competition

RAER first filters challengers by a fixed `p>=0.5` threshold. If none pass, it returns the anchor. Thus abstention is a post-hoc rule, not a competitor learned in the same scene-level ordering. This encourages a brittle separation between “which challenger wins” and “whether any challenger should intervene”.

### 4.3 RAER's eligible set is looser than the final executable set

The learned re-ranker sees challengers that the subsequent one-sided guard can never deploy. In V64.3.16, `30.39%` of raw-proposal scenes have a legacy EAF proposal below the final `0.015` margin requirement, and the learned RAER choice is still blocked by the final guard in 3.2% of scenes. This wastes model capacity and makes training diagnostics disagree with deployed decisions.

### 4.4 All-flagged scenes need a separate structural safety pathway

`5.6%` of this fresh screen has all valid actions safety-flagged. The planner already has a frozen continuous structural-risk guard for these scenes. A learned pre-structural DALER choice cannot reproduce that risk pool using only tournament statistics, so V64.3.17 explicitly abstains from learned intervention in all-flagged candidate banks and leaves them entirely to the existing structural guard. This closes a potential train/deployment mismatch rather than inventing a second safety model.

---

## 5. Directions that should not be repeated now

The changelog already contains extensive negative evidence. The next iteration should **not** spend experiments on:

- OCFI-v2, alpha/radius sweeps, or another global uncertainty subtraction;
- EAIR/RAER probability-threshold sweeps on validation;
- BTP/RET/CET/AF/HAP branches or a fresh acquisition redesign;
- increasing `B` or `M` to hide the extremal-selection problem;
- relaxing the evidence certificate or safety guard;
- optimizing only average pair-sign accuracy;
- broad EAF representation unfreezing before testing the new decision operator;
- reusing any of the 1200 previous design scenes or the 500 V64.3.16 fresh scenes for V64.3.17 promotion.

These branches would either repeat a failed mechanism or destroy the ability to attribute the next result.

---

## 6. V64.3.17 main algorithm: EAF-DALER

**EAF-DALER = Evidence-Attributed Deployment-Aligned Listwise Extremal Reliability.**

The new complete chain is:

`fixed planner-interface evidence cap B<=16`
`-> auditable evidence atoms`
`-> terminally frozen budgeted acquisition (M=24)`
`-> frozen selected evidence`
`-> frozen EAF complete DARM-anchor frontier value`
`-> exact selected-evidence attribution`
`-> frozen deployment-executable challenger set`
`-> anchor-augmented scene-level listwise evidence-attributed reliability`
`-> extremal action / explicit anchor abstention`
`-> unchanged one-sided + evidence certificate`
`-> unchanged structural risk guard`
`-> final decision preservation`.

### 6.1 Exact learned-intervention candidate set

For scene `x` with DARM anchor `a`, DALER first forms an executable challenger set `E(x)`. A challenger can enter `E(x)` only if all of the following runtime-only frozen conditions hold:

1. valid candidate;
2. an unflagged action exists and the challenger itself is unflagged; all-flagged scenes are handed to the frozen structural-risk guard instead of the learned re-ranker;
3. frozen EAF anchor margin is at least the existing final `flip_margin=0.015`;
4. frozen EAF score gain over the anchor satisfies the existing score margin;
5. the unchanged evidence certificate passes;
6. the challenger belongs to the **exact same utility-equivalence mask** computed by the existing certificate-constrained utility refinement (same score band, safety rule, top-k, pair certificate, and finite deployment-utility condition).

No learned score and no teacher information participates in this set construction.

### 6.2 Evidence-attributed challenger features

Each executable challenger receives a structured runtime feature vector containing:

- frozen EAF challenger-anchor margin;
- exact selected-evidence attribution scale;
- frontier residual and attribution global statistics;
- evidence-certificate fraction and valid-action count;
- margin/attribution ratios;
- within-frontier margin and attribution z-scores/ranks;
- distance below the frontier maximum;
- whether the candidate is the legacy EAF-selected action;
- margin/attribution differences relative to that legacy action;
- EAF score gain vs anchor and score difference vs legacy action;
- deployment utility-cost difference vs legacy action;
- margin excess above the frozen final guard;
- EAF-score rank, deployment-utility rank, and executable-candidate fraction.

The architecture is deliberately a **shared standardized linear readout**. The novelty is not a deeper network; it is how auditable evidence attribution is inserted into the extremal decision operator under a fixed planner interface.

### 6.3 Anchor-augmented listwise training

For each train scene, the DARM anchor is an explicit pseudo-item with fixed logit

` s_a = 0 `.

For challenger `b in E(x)`, DALER computes

` s_b = w^T ((phi_b - mu) / sigma) + beta `.

Let the offline teacher margin be

`Delta_T(b,a) = J_T(a) - J_T(b)`.

The scene target is the executable challenger with the largest positive teacher margin; if no executable challenger has positive teacher margin, the target is the anchor. This is equivalent to selecting the lowest-teacher-cost item among `{a} union E(x)`.

The primary loss is anchor-augmented scene-level listwise cross-entropy:

`L_list = -log exp(s_y*) / (exp(0) + sum_{b in E(x)} exp(s_b))`.

A fixed class-balanced edge BCE term with coefficient `1.0` is added to preserve absolute “teacher-better-than-anchor” semantics, plus small L2 regularization. The coefficient is **pre-registered and the fitter rejects any other value**; it is not validation-tuned.

### 6.4 Runtime decision

At runtime, DALER computes the same `s_b` for the exact executable set and takes

`argmax {0, s_b : b in E(x)}`.

If the fixed anchor logit `0` wins, DALER abstains to the DARM anchor. Otherwise it selects the highest-reliability executable challenger. There is **no validation-chosen probability threshold** and no multiplication by raw EAF margin.

The old final one-sided/evidence certificate remains active. Under the V64.3.17 frozen configuration, DALER's executable mask is designed to make this guard an invariant rather than a hidden cleanup stage; any post-selection guard block above 0.1% is treated as an engineering failure.

---

## 7. Why this novelty is stronger

Generic decision-aware uncertainty is already an active line of work, and listwise decision-focused learning is not new by itself. Therefore a paper story of “we train a confidence model” or “we use a listwise loss” would be weak.

The potentially publishable mechanism is the **joint constraint**:

> under a fixed planner-interface evidence budget, every action intervention is based on a complete frozen anchor frontier whose value is decomposed into auditable selected-evidence support; reliability is trained scene-conditionally over the exact deployment-executable extremal candidate set with the anchor as explicit abstention; the same evidence certificate remains auditable at the final decision.

This differentiates the method from generic confidence gating, generic conformal UQ, generic agent relevance, and generic learning-to-rank. It also creates testable causal predictions: alternative-runner recovery should improve, post-selection guard cleanup should vanish, and beneficial interventions should survive without reopening evidence acquisition.

This is **CCF-A-level motivation/novelty in form**, but not yet CCF-A-level empirical evidence. Acceptance-level strength will require the mechanism screen, independent full-val reproduction, fixed test/closed-loop evaluation, external baselines, latency/interface accounting, and well-chosen ablations.

---

## 8. Design-only diagnostic that motivated DALER

Because the current 500 V64.3.16 scenes have already been inspected, they cannot be used for promotion. Nevertheless, a strictly labeled **design-only** replay was performed to test whether a scene-level listwise operator is a plausible next causal branch.

The old V64.3.16 edge files do not contain the new exact utility/safety executable mask, so the replay approximates executability as `raw_margin >= 0.015` and evidence certificate `=1`, and uses the old RAER feature vector plus a legacy-top indicator. The readout is fitted only on train edges.

Results on the already-seen 500 scenes:

- train approximate executable edges: `38,630`, positive fraction `72.57%`;
- seen-val approximate executable edges: `6,532`;
- seen-val executable-edge AUC: `0.6970`;
- proposal changed: `54.53%`;
- anchor fallback: `47.84%`;
- conditional non-anchor precision: `80.58%`;
- alternative recovery rate: `6.68%`;
- alternative recovery precision: **77.42%**;
- alternative teacher-margin mean: `1.153`.

This is not a paper result and cannot be used to choose thresholds or claim improvement. It only justifies spending the next **fresh** experiment on DALER rather than on a different subsystem.

---

## 9. V64.3.17 next experiment

### Stage A — fresh 500-scene four-arm causal screen

Use the exact same new fresh token set for:

1. frozen raw EAF;
2. repaired scalar EAIR control;
3. frozen V64.3.16 RAER control;
4. V64.3.17 DALER.

All three learned readouts are fitted on TRAIN only. The fresh validation tokens are selected from a 4000-scene discovery replay by fixed SHA256 ranking after excluding **all 1700 already inspected validation tokens** (1200 earlier design + the current 500 fresh screen). Token selection uses identity only, not labels or metrics.

Primary mechanism diagnostics for DALER are:

- train internal-holdout exact-executable-edge AUC;
- fresh exact-executable-edge AUC;
- selected non-anchor teacher-better precision;
- proposal changed rate;
- anchor fallback rate;
- alternative recovery rate;
- **alternative recovery precision**;
- alternative teacher-margin mean;
- post-selection final-guard block rate;
- raw/EAIR/RAER/DALER teacher match and regret;
- harmful/beneficial intervention and flip rate.

Pre-registered screen gates are intentionally strict:

- fresh exact-executable-edge AUC `>= 0.65`;
- proposal changed `>=3%`;
- alternative recovery rate `>=1.5%`;
- alternative recovery precision `>=65%` and at least `+10pp` over frozen RAER's alternative precision when defined;
- alternative teacher-margin mean `>0`;
- DALER post-selection final-guard block `<=0.1%`;
- harmful intervention absolute reduction vs raw `>=5pp`;
- beneficial retention `>=35%`, beneficial > harmful;
- deployed flip `>=3%` and below raw;
- teacher match `>= anchor +0.5pp`;
- regret `<=1.02 x raw EAF`;
- plus a paired gain over RAER: either match `>= RAER +0.5pp` at `<=1.01 x RAER` regret, or regret `<=0.99 x RAER` while match is no worse than `RAER -0.5pp`.

A failure stops the pipeline. No threshold, BCE weight, B, M, or certificate tuning is allowed on this screen.

### Stage B — independent full-validation reproduction

Only if Stage A passes, freeze the exact DALER config and run an independent full-val reproduction. Do not refit/tune on the screen. Reproduce both the mechanism (alternative precision/recovery, alignment) and the endpoints.

### Stage C — test and closed-loop only after reproduction

Only after Stage B reproduces should the method move to frozen test and non-reactive/reactive closed-loop evaluation. The final paper should include nuPlan score/safety/progress/comfort/TTC, runtime latency, evidence query count, exact-B rate, fallback/structural-guard rates, and the decision-sufficiency diagnostics.

---

## 10. Main comparisons and ablations for the eventual paper

The next 500-scene screen should stay causal and compact. After DALER passes and is frozen, the paper-level experimental matrix should include:

### Main internal baselines

- DARM/selected-local anchor;
- raw EAF;
- scalar EAIR (post-top1 gate);
- RAER (independent edge reliability + probability-weighted margin re-ranking);
- DALER (deployment-aligned listwise extremal reliability);
- pair-full/local-pair-full ceilings as diagnostic upper bounds, not deployable methods.

### Mechanism ablations

- **pointwise vs listwise:** frozen RAER-style edge objective vs DALER scene-listwise objective;
- **anchor pseudo-item ablation:** force a challenger whenever executable candidates exist, showing the value of explicit abstention;
- **attribution ablation:** remove attribution-derived features while keeping EAF margins and deployment context;
- **legacy-relative context ablation:** remove `is_legacy_selected`, relative margin/attribution/score/utility features;
- **deployment-alignment diagnostic ablation:** offline-only evaluation with the executable-set restrictions removed; never deploy an ablation that bypasses the frozen safety/evidence certificate;
- **edge-BCE auxiliary ablation:** after the primary method is frozen, compare listwise-only vs listwise+fixed BCE to explain absolute reliability calibration. Do not tune this on the causal screen.

### Fixed-budget evidence studies

After the method is frozen, report `B in {4,8,12,16}` curves with exact interface use, match, regret, alternative-recovery precision, latency, and closed-loop outcomes. This is important because the paper's core claim is not merely a re-ranker; it is reliable extremal decision selection under a **bounded auditable evidence interface**.

### External planning baselines

Use the paper's already planned nuPlan-compatible baselines under matched DB/map/vector inputs and candidate/evaluation assumptions. External baselines are needed for final paper competitiveness, but they should not be mixed into the current causal operator screen.

---

## 11. Engineering and leakage audit

### Changes implemented

- `bdse/planner/tournament.py`
  - extracted the existing utility-refinement candidate logic into a shared `_certificate_utility_refinement_context` so DALER and the frozen legacy planner consume **one implementation** of the utility-equivalence set;
  - added exact DALER executable-mask construction;
  - added 25-feature runtime-only DALER representation;
  - added anchor-augmented DALER extremal operator;
  - DALER and RAER are mutually exclusive causal arms;
  - all-flagged candidate banks are excluded from learned DALER intervention and left to the frozen continuous structural-risk guard.
- `bdse/tools/fit_v64_3_17_eaf_daler.py`
  - train-only scene-group listwise fitter;
  - fixed anchor logit 0;
  - fixed auxiliary balanced BCE weight 1.0;
  - deterministic train internal holdout; no nuPlan validation/test labels.
- `bdse/experiments/evaluate_open_loop.py`, `bdse/experiments/train.py`, `bdse/metrics/bdse_metrics.py`
  - propagate DALER diagnostics and explicit edge-level executable/logit/features;
  - evaluator prefers DALER's own frontier arrays with RAER fallback, avoiding hidden coupling to RAER instrumentation.
- strict DALER contract checker and four-arm screen checker;
- screen checker now fails if any required frozen-interface metric is missing instead of silently omitting it;
- fixed 1700-token V64.3.17 design exclusion;
- next launcher hard-fails on missing tokens, insufficient instrumentation, insufficient executable edges, contract failure, or promotion failure.

### Leakage audit

No runtime teacher/future leakage was found in the DALER path.

Teacher costs appear only in:

- TRAIN-only labels for `teacher challenger better than anchor` and the scene-level teacher-best executable target;
- evaluation diagnostics used after actions are produced.

The fitted runtime config contains only feature normalization, linear weights, bias, fixed anchor logit, and frozen execution constraints. Fresh validation tokens are selected by identity/hash only. The current 500 scenes have been added to the exclusion list and cannot be used for V64.3.17 promotion.

### Behavior-preservation audit

The utility-refinement refactor was checked on **5,000 deterministic random tournament cases** against the V64.3.16 implementation:

- selected-action differences: `0 / 5000`;
- public-diagnostic differences: `0 / 5000`.

This gives direct evidence that extracting the shared utility-equivalence mask did not silently change the frozen planner.

---

## 12. Failure branches after the next result

If V64.3.17 fails, the branch should be determined by the causal gate rather than by manual tuning:

- **exact-executable AUC < 0.65:** the frozen EAF scalar/frontier summaries are insufficient for reliability; move to structured per-atom/query-conditioned evidence reliability while keeping acquisition and B/M frozen;
- **AUC good but alternative precision/recovery poor:** the listwise feature/objective still fails at extremal ordering; diagnose scene-conditional/listwise structure, not thresholds;
- **mechanism and alignment good but regret fails:** add a train-only teacher-improvement magnitude / robust listwise ordering term over the same frozen executable frontier; do not return to selector/acquisition;
- **mechanism good but preservation fails:** audit interaction with the unchanged final certificate and structural post-processing before changing learning;
- **screen passes:** freeze exact config, run independent full-val reproduction, then frozen test/closed-loop only if reproduced.

---

## 13. Bottom line

V64.3.16 should **not** be discarded. It establishes the most important prerequisite: evidence-attributed reliability generalizes to fresh challenger edges and can sharply reduce harmful intervention while preserving most of raw EAF's regret benefit. But RAER's all-challenger story is not yet validated because its actual alternative recovery is weak and its learned eligible set is not identical to what the planner can deploy.

V64.3.17 DALER is therefore a targeted causal upgrade rather than a broad redesign. It preserves the successful fixed-budget/auditable-evidence/EAF-value backbone and changes exactly the failed link:

**independent edge reliability + probability-weighted extremum -> deployment-aligned, anchor-augmented scene-level listwise extremal reliability.**

That is the experiment most likely to tell us whether the current fixed-evidence representation is already sufficient for a strong paper, or whether the next necessary novelty must move one layer deeper into structured per-atom reliability.
