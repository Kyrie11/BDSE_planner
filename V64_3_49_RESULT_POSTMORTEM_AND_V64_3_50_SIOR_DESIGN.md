# V64.3.49 SIIR result postmortem and V64.3.50 SIOR preregistered design

## Executive decision

**Reliability: PASS.** The uploaded V64.3.49 result is engineering-valid for TRAIN-level scientific attribution. No V49.1 repair is needed.

**Scientific verdict: V49 SIIR FAIL under the exact preregistered branch.** The nested TRAIN identification gate fails, the launcher stops before fresh selection, and the current offline selected-risk family must be closed exactly as preregistered.

**Dominant bottleneck after V49:**

> **Selected-outcome causal observability / treatment-effect identification for the actual deployed full-set RSMR winner.**

The remaining problem is not a new future feature, not K/multiplicity, not threshold calibration, not a larger selected-risk head, and not a different offline selector-perturbation distribution. The missing evidence is the outcome of actually executing the frozen deployed winner under an interactive closed-loop environment, relative to preserving the incumbent from the same initial state.

V64.3.50 is therefore **EAF-ICER-SIOR — Selected-Interventional Outcome Retention**. It freezes the mature selector, consequence coordinates, model class, regularization, calibration budget, and monotone same-winner/incumbent runtime operator. It changes only the training evidence source from offline selected-event labels to paired one-shot closed-loop interventions on the exact full-set RSMR winner.

---

## 1. Paper scope and the part of the original manuscript that remains the backbone

The uploaded manuscript studies planning through a bounded, auditable interface rather than unrestricted world-model reconstruction. Its core problem is **decision sufficiency under a fixed evidence budget**: only information that can change the downstream action needs to survive the interface. It explicitly identifies extremal candidate replacement as a tail-amplifying operator and separates proposal generation from confirmation of an already fixed proposal.

The manuscript's durable structural ideas remain aligned with the V34–V50 code line:

1. fixed `B=16 / M=24` bounded planner interface;
2. complete EAF frontier and exact selected-evidence attribution;
3. proposal generation and absolute execution valuation are distinct operators;
4. the extremal proposal is frozen before confirmation/retention;
5. confirmation/retention may return only `{same proposal, incumbent}`;
6. no rerank, no second-best fallback, no new risky action path;
7. logged/simulated future is supervision/evaluation only, not runtime input;
8. TRAIN diagnostics are design gates, not independent paper evidence; untouched evaluation must remain fail-closed.

The original PTMC-specific 24-D Gaussian tail classifier should **not** be treated as the final method headline after the later evidence chain. The stronger evolving paper question is now: **which statistical object is sufficient for each deployed operator, and what outcome evidence is required after extremal selection?**

A suitable current working paper line is:

> **Selection–Valuation–Intervention Sufficiency under a Bounded Auditable Planner Interface.**

The headline contribution should be the operator/sufficiency decomposition and the evidence-source result, not the acronym SIOR itself.

---

## 2. V49 reliability audit: PASS

Uploaded artifact SHA256:

- code ZIP: `81212d1798634b9d6ba1d7f40eebc29d3d060cc8faeffdf527945448a36a2c73`;
- result ZIP: `62e454ff9305b885c75da6a9d6ce792342d11d04b48cede44b0bb9660e60fbf9`.

The V49 result contains a 907-file source manifest. Every manifest entry was independently checked against the uploaded code package:

- manifest entries checked: **907**;
- missing files: **0**;
- SHA mismatches: **0**.

Additional reliability closure:

| Check | Result |
|---|---:|
| V48 OCRR science lock | PASS |
| server targeted regression | **242/242 PASS** |
| local V48/V48.2/V49 focused replay | PASS |
| direct scientific scenes | **782/782 unique** |
| frozen RSMR proposal population | **502** |
| nested outer folds | **5/5** |
| OBS-SIGN exact V48 SIGN-NOMULT replay | PASS |
| same-winner / incumbent containment | PASS |
| rerank / second-best / fallback | none |
| fresh A/B selected or consumed | **no** |
| termination | preregistered TRAIN scientific STOP |

The fit log terminates with the exact registered failure reason:

`selection_interventional_risk_does_not_outperform_observational_selected_risk_close_current_offline_selected_risk_family`

and stops before fresh selection. Therefore V49 is scientifically interpretable and does **not** require an engineering hotfix.

---

## 3. Exact preregistered V49 verdict

V49 was allowed to differ from OBS-SIGN in only one scientific factor: the TRAIN selected-event measure. State `[Q, P-Q, E-P]`, zero-bias pairwise sign-risk, `lambda=1`, split calibration, full-set RSMR deployment selector, and same-winner/incumbent runtime were frozen.

The preregistered identification gate required SIIR risk AUC on the **actual full-set OOF winners** to exceed both OBS-SIGN and `-EGO-REF` in aggregate and in at least `4/5` folds versus each.

Observed risk identification:

| Risk score | Nonpositive-risk AUC |
|---|---:|
| `-EGO-REF` | **0.629829** |
| OBS-SIGN | **0.613919** |
| SIIR | **0.608122** |
| held-out intervention OBS | **0.617177** |
| held-out intervention SIIR | **0.607058** |

SIIR beats OBS in only **3/5** folds and beats EGO-REF in only **1/5** folds. The identification gate is therefore unambiguously false.

Deployment aggregates are:

| Arm | selected | positive | capture | ΣΔT | catastrophe | no-op false | NegRMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| RSMR | 502 | 221 | **38.50%** | +43.294 | 28 | 107 | .3557 |
| EGO-REF | 251 | 136 | 23.69% | **+59.533** | **9** | 45 | **.2233** |
| OBS-SIGN | 411 | 187 | 32.58% | +53.496 | 18 | 78 | .2707 |
| SIIR | 464 | 208 | 36.24% | +53.468 | 21 | 93 | .2793 |

SIIR's nominal capture is above the historical 35.5017% floor, but the complete `existence_and_capture` gate still fails because no-positive-opportunity false interventions rise to **93** rather than meeting the frozen reduction target. More importantly, deployment performance cannot rescue a failed identification mechanism under the preregistration.

Formal verdict:

> **V64.3.49 SIIR = preregistered scientific falsification; close the current offline selected-risk family.**

No V49 fresh block may be selected and no V49 threshold/seed/prefix/loss variant is allowed.

---

## 4. What SIIR actually changed in the selected set

OBS-SIGN versus SIIR on the 502 frozen RSMR proposals:

| Partition | scenes | ΣΔT | positive | material positive (`ΔT>0.2`) | catastrophe |
|---|---:|---:|---:|---:|---:|
| both | 405 | +53.496 | 185 | 43 | 18 |
| OBS-only | 6 | -0.00056 | 2 | 0 | 0 |
| **SIIR-only** | **59** | **-0.0281** | 23 | 4 | **3** |
| neither | 32 | -10.173 | 11 | 3 | 7 |

This is not a beneficial selected-set rotation. SIIR adds 59 interventions but their total value is essentially zero/slightly negative and it opens **three new catastrophes** relative to OBS-SIGN.

Risk movement confirms that SIIR is broadly more permissive rather than selectively better ordered:

- among 28 catastrophes, SIIR assigns lower risk than OBS to **18** and the mean risk change is `-0.044`;
- among 253 non-catastrophic nonpositive proposals, SIIR assigns lower risk to **184**, mean `-0.064`;
- among 50 material positives, SIIR assigns lower risk to only 26, mean `-0.050`.

Thus the learned change is not “lower risk specifically for recoverable opportunities.” It is a broad risk attenuation that also lowers risk on the harmful tail.

Representative SIIR-only catastrophe includes `1c3f933be722564d`, with teacher improvement about `-2.0206`: OBS rejects it while SIIR accepts it after the interventional fit changes the risk ordering.

---

## 5. Structural reason the V49 selection intervention fails

The important failure is not the hash seed. It is the intervention target itself.

The V49 candidate-level audit contains 782 scenes. Full-set RSMR proposes in 502. The hash-prefix intervention produces only 356 selected events:

- 277 intervention winners equal the full-set winner;
- **79** intervention winners switch to a different lower-ranked action;
- **146** full-set winner events disappear because the prefix contains no selected proposal.

For the **79 switched events**:

- teacher sum: **-1.959**;
- positive/nonpositive: **39 / 40**;
- catastrophe: **10**;
- material positives: 8.

By RSMR-score rank, intervention winners include 52 rank-2, 15 rank-3, 7 rank-4, and several rank-5/8/9 events. Rank-3 intervention events alone contribute about `-3.42` with four catastrophes.

This exposes the key conceptual mismatch:

> V49 perturbs **which proposal becomes selected under an artificial exposed-candidate set**, but it does not intervene on the **outcome of executing the actual deployed full-set winner**.

The selected-event training population therefore mixes two changes:

1. changed selection regime / proposal identity;
2. the same offline teacher outcome source.

It does **not** produce a causal contrast of `execute full-set winner` versus `keep incumbent` under interactive closed-loop dynamics. No seed or prefix distribution can repair this identity problem without changing the evidence source.

This is why the next version must not sweep prefix laws or devise “better offline interventions.”

---

## 6. What succeeds in V49 and should be retained

V49 does **not** imply that all post-selection reasoning is wrong. It preserves several scientifically useful decisions:

### 6.1 Correct problem location

The target remains the outcome law of an **already frozen extremal winner**, not a new reranking problem. That high-level migration from “add future representation” to “identify the deployed selected outcome” remains correct.

### 6.2 Q/P/E as controlled consequence coordinates

`Q`, `P-Q`, and `E-P` remain a useful low-capacity controlled state because their upstream components already have independent mechanism evidence:

- QUALITY: physical/current consequence mediator;
- PLAN: V44/V45 prospective interaction + agent-local response support;
- EGO-REF: strong runtime-predictable future reference mediator.

V50 deliberately does not add another observable before testing whether **outcome evidence**, rather than state capacity, is the missing factor.

### 6.3 Monotone same-winner/incumbent operator

The no-rerank/no-second-best/no-fallback structure remains one of the most mature pieces of the method and continues to give a structural intervention containment guarantee independent of learned risk accuracy.

---

## 7. What is formally closed after V49

The following directions are closed and must not be used as V50 rescue knobs:

- V48 `K/logK`, K bins, K interactions, or another multiplicity transform;
- V49 intervention seed or prefix-size law sweep;
- alternative offline subset-exposure intervention distributions;
- pairwise-loss redesign;
- `lambda`, class weights, focal weights, catastrophe weights;
- retention threshold/calibration sweep;
- selected translation/temperature;
- larger selected-risk MLP;
- new generic future-state feature block before changing outcome evidence;
- V46 response variance / hand temporal profile;
- V47 constant lateral-drift AGENT-2D;
- CVaR tuning;
- standalone catastrophe classifier/veto;
- RSMR modification;
- B/M/candidate-count/top-K sweep;
- reranking, second-best, fallback;
- post-hoc union of old policies;
- A/B pooling;
- deployment use of logged future.

This also does **not** resurrect V38–V40's old 19-D selected-value/distribution heads: V50 keeps the later Q/P/E state and tests a new interventional outcome source, so the causal factor is different.

---

## 8. Updated dominant bottleneck

The bottleneck should now be written more narrowly than V48's “selection-regime transportability”:

> **Selected-outcome causal observability and treatment-effect identification under the actual deployed full-set extremal selector.**

A useful estimand is not merely

`P(Y_selected <= 0 | Z, S=selected)`

under an offline selected population. The deployment question is closer to the conditional effect

`tau(x,b*) = E[Y(do(b*)) - Y(do(i)) | x, b*=S_full(x)]`,

where `b*` is the exact full-set RSMR winner and `i` is the incumbent.

The important scientific distinction is:

`offline selector perturbation != intervention on selected action outcome`.

V49 now gives empirical evidence for this distinction.

---

## 9. Layer maturity after V49

| Layer | Status after V49 | V50 policy |
|---|---|---|
| B16/M24 bounded interface | mature | freeze |
| EAF complete frontier / exact attribution | mature, paper backbone | freeze |
| support/admissibility | mature | freeze |
| RSMR ordinal extremal selector | most mature learned layer | permanently freeze |
| incumbent/no-fallback containment | mature | permanently freeze |
| EPV | real partial mediator | retain |
| QUALITY | real partial mediator | retain |
| V44 ungated full-horizon prospective occupancy | strong success | retain |
| V45 agent-local longitudinal response | successful supporting layer | retain |
| plan-conditioned response mean | real incremental signal | retain low-capacity form |
| V47 EGO-REF | strong supporting consequence mediator | retain |
| V46 variance / hand temporal profile | closed | do not reopen |
| V47 constant-drift AGENT-2D | closed | do not reopen |
| V48 multiplicity/logK | fresh falsified | closed |
| V49 offline prefix selection intervention | TRAIN falsified | closed |
| current offline selected-risk family | preregistered closed | do not tune |
| **paired selected-outcome intervention evidence** | untested | **V50 dominant layer** |
| final absolute zero/material opportunity/tail retention | unresolved | V50 gate |

The next model should learn **whether executing the exact selected proposal helps under interactive dynamics**, not another proxy for “what kind of proposal it looks like.”

---

# 10. V64.3.50 EAF-ICER-SIOR

Full name: **Selected-Interventional Outcome Retention**.

## 10.1 Scientific factor changed

V50 freezes:

- full-set RSMR selector and winner identity;
- Q/P/E state `[Q, P-Q, E-P]`;
- zero-bias pairwise sign-risk model class;
- `lambda=1`;
- capture-derived retention budget `alpha_ret = 0.0779185520361991`;
- same-winner-or-incumbent deployment operator;
- no rerank / second best / fallback;
- runtime evidence budget and candidate bank.

The **only first-order change** is the training outcome evidence source.

V49 label source: offline teacher outcomes after selector-measure perturbation.

V50 label source: paired one-shot **closed-loop selected-action intervention** on the actual full-set RSMR winner.

## 10.2 Paired causal probe

For each of the exact 502 frozen TRAIN scenes where full-set RSMR proposes an action, run two simulations from the identical nuPlan scenario start.

### CONTROL

Whenever the direct frozen RSMR proposal exists, preserve the incumbent.

### TREATMENT

At the **first direct RSMR proposal**, execute that exact frozen RSMR proposal **once**. For every later direct proposal in the scenario, preserve the incumbent.

This is a deliberate one-shot treatment. It isolates the causal effect of the selected intervention instead of mixing in a repeatedly changing recovery policy.

The implementation hard-checks:

- both arms expose exactly the same first **live** pre-post-selection RSMR proposal/baseline identity;
- the first proposal may occur after planner iteration `0`; CONTROL and TREATMENT must reach it at the exact same iteration/time with identical pre-intervention action traces;
- the live proposal action must equal the byte-locked V49 full-set RSMR winner for that scenario;
- the same frozen Q/P/E coordinate definitions are evaluated at that exact live pre-intervention state in both arms and must match numerically before treatment;
- CONTROL executes zero interventions;
- TREATMENT executes exactly one intervention;
- TREATMENT action equals the actual full-set RSMR proposal;
- after the intervention, the probe never chooses a runner-up;
- fallback is disabled;
- any mismatch is an engineering STOP.

### V50 event-state alignment engineering repair (pre-outcome protocol correction)

The first attempted paired run exposed an invalid implementation assumption before any paired row was committed: token `03dac455f9ec5792` reached the first synchronized live RSMR proposal at planner iteration `7/7`, not `0/0`.  The old collector therefore stopped before writing a causal label.  No treatment/control score, hard metric, or SIOR fit result was used to design this repair.

Simply deleting the `iteration==0` assertion would be scientifically invalid, because the old fit would then pair a later closed-loop treatment outcome with stale V49 offline Q/P/E values from a different state.  The repaired protocol instead aligns **treatment, proposal identity, and covariates at one live pre-intervention event**:

1. CONTROL and TREATMENT start from the identical nuPlan scenario start and must have identical planner/deployed action traces before the first proposal.
2. The first proposal iteration/time must match exactly across the two arms; it need not be absolute iteration zero.
3. The proposal action must equal the frozen V49 full-set RSMR winner; otherwise the scene is an ENGINEERING STOP rather than a silently changed treatment.
4. Q/P/E keep the previously frozen coordinate definitions and are recorded at that same live event in both arms; any C/T state mismatch is an ENGINEERING STOP.
5. CONTROL vetoes to the incumbent; TREATMENT executes that exact proposal once.  No runner-up/fallback is introduced.

This is an **outcome-blind engineering amendment** needed to make the selected-action intervention well-defined under the actual nuPlan runtime.  It does not constitute V50 evidence and does not relax any promotion gate.  The incomplete first token outputs must be discarded/re-run; the resumable collector only skips tokens after a complete paired row is committed.

## 10.3 Outcome label

Let `S_T` and `S_C` be the official nuPlan aggregate scores for treatment and control on the same scenario.

`paired_score_delta = S_T - S_C`.

A `safe_benefit` positive requires:

1. `paired_score_delta > 0`, and
2. treatment is non-inferior to control on all frozen hard metrics:
   - `no_ego_at_fault_collisions`;
   - `time_to_collision_within_bound`;
   - `drivable_area_compliance`;
   - `driving_direction_compliance`.

This avoids teaching the retention model to trade a small aggregate gain for a hard safety regression.

The default challenge is `closed_loop_reactive_agents`, because the central question is precisely whether the selected action's outcome changes once the environment responds interactively. This is simulator-interventional evidence, **not** a claim of real-world causal identification.

## 10.4 Risk model

Use the exact same low-capacity state:

`z = [Q, P-Q, E-P]`.

Reuse the same zero-bias pairwise sign-risk family and fixed `lambda=1`. A bad event is `not safe_benefit`.

This design is intentional: if V50 improves identification, the attribution is to the **selected-outcome evidence source**, not to capacity, representation, or a new loss.

## 10.5 Nested identification gate

Five fixed outer folds remain. For outer test fold `k`, calibration uses `(k+1) mod 5`; risk fitting excludes both.

On the exact same paired closed-loop labels, compare:

- V50 closed-loop SIOR risk;
- frozen V49 OBS-SIGN OOF risk;
- `-EGO-REF`.

Identification requires:

- `AUC_SIOR > AUC_OBS` in aggregate;
- `AUC_SIOR > AUC_-EGOREF` in aggregate;
- SIOR beats OBS in at least `4/5` folds;
- SIOR beats `-EGO-REF` in at least `4/5` folds.

Thus merely obtaining a different retained set is insufficient.

## 10.6 Preregistered paired-outcome deployment gate

Using fold-specific split-calibrated thresholds from the frozen retention budget, V50 must additionally satisfy all of:

1. retain at least `1-alpha_ret ≈ 92.208%` of `safe_benefit` events;
2. retain **zero** hard-regression events;
3. reduce retained nonbenefit events by at least 20% relative to the full RSMR selected-outcome population;
4. retained paired score sum is nonnegative and no worse than accepting all one-shot RSMR interventions;
5. paired negative RMS is non-worse than accepting all one-shot RSMR interventions;
6. 5/5 test folds have nonnegative retained paired score sum;
7. 5/5 folds retain zero hard-regression events.

TRAIN passes only if **identification + deployment** both pass.

These are preregistered before observing any V50 paired outcomes.

## 10.7 V50 failure branch

If V50 TRAIN fails:

- do not tune Q/P/E weights/features;
- do not change pairwise loss, `lambda`, threshold, calibration, or safety weights;
- do not return to V49 prefix interventions;
- do not add another generic future observable block immediately;
- consume no fresh paired block.

The next diagnostic question becomes whether the **causal selected outcome itself is predictable from the frozen runtime state**. Only if paired outcomes are demonstrably unidentifiable from the current Q/P/E state should V51 consider a new **causal-state representation** explicitly justified by residual paired-outcome structure. That is different from generic feature expansion.

If V50 TRAIN passes:

1. freeze all V50 artifacts;
2. select untouched A500 and B500 independently;
3. collect the same paired one-shot closed-loop contrast on A and B;
4. judge A and B independently, with no pooling/tuning;
5. only after both pass run the official repeated-policy closed-loop evaluation.

---

## 11. Why V50 is a paper-level branch rather than a trick

V50 changes the estimand/evidence relationship, not a hyperparameter.

The evidence chain is now:

1. **V34:** bounded EAF evidence can support ordinal extremal selection;
2. **V37–V40:** selected residual/distribution exists, but generic signed-value heads do not close the deployment boundary;
3. **V41–V45:** endpoint/current/prospective interaction and agent-local response are real mediators;
4. **V46:** better ordinary prediction can worsen deployment decisions — prediction sufficiency is not decision sufficiency;
5. **V47:** identifiable future-state nuisance can still be deployment-insufficient — representation identifiability is not decision sufficiency;
6. **V48:** in-domain selected-risk signal can fail double-fresh transport — post-selection identification is not transport sufficiency;
7. **V49:** offline selector-measure intervention can fail to recover transportable risk — selection-measure intervention is not selected-outcome intervention;
8. **V50:** directly test whether paired interventional selected-outcome supervision is the missing sufficient evidence.

This supports a stronger CCF-A-standard thesis than a list of planner tricks:

> **Different deployment operators require different sufficient statistics and, after extremal selection, potentially different outcome evidence. A bounded auditable selector can be stable while absolute execution retention remains statistically unidentified from observational/offline selected outcomes.**

The original monotone containment theorem remains valid for V50 deployment because SIOR can only retain the frozen RSMR winner or the incumbent. A new paper-level theoretical section should distinguish **structural containment** from **statistical/interventional identification**: containment prevents new action paths; it does not make a risk law identifiable.

---

## 12. Engineering implementation delivered

V50 adds:

- `bdse/planner/selected_outcome_probe.py` — one-shot paired control/treatment operator;
- `bdse/tools/prepare_v64_3_50_eaf_icer_sior_probe_configs.py`;
- `bdse/tools/select_v64_3_50_sior_train_tokens.py`;
- `bdse/tools/run_v64_3_50_paired_selected_outcome_collection.py`;
- `bdse/tools/fit_v64_3_50_eaf_icer_sior.py`;
- `bdse/tests/test_v64_3_50_eaf_icer_sior.py`;
- `RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh`.

The normal planner path is unchanged when the probe is disabled. Probe state is reset at every nuPlan scenario initialization so one-shot treatment cannot leak across scenarios.

The V50 launcher locks the exact uploaded V49 failure artifacts before any new science, reruns code regression, freezes the exact 502 full-set-RSMR TRAIN scenes, performs paired reactive closed-loop collection, and then runs the nested V50 gate. It **never selects fresh scenes** itself. A failed TRAIN fit therefore cannot consume future validation evidence.

---

## 13. Next command

Default server layout:

```bash
cd bdse_v64_3_50_eaf_icer_sior
bash RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh
```

If nuPlan roots differ, override them explicitly:

```bash
cd bdse_v64_3_50_eaf_icer_sior

V47_ROOT=/path/to/outputs_v64_3_47_eaf_icer_fsfr_screen_2gpu_v1 \
V49_ROOT=/path/to/outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1 \
EAF_V64_3_13_ROOT=/path/to/outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1 \
NUPLAN_ROOT=/path/to/nuplan \
NUPLAN_MAP_ROOT=/path/to/nuplan/maps \
NUPLAN_EXP_ROOT=/path/to/nuplan/exp \
NUPLAN_DB_ROOT=/path/to/nuplan-v1.1/splits/train \
GPU0=0 GPU1=1 \
bash RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh
```

The collection is resumable. If a server interruption occurs, rerunning the exact command continues from `paired_train/paired_selected_outcomes.csv`; it must not alter configs, tokens, challenge, or checkpoint between resumes.

Do **not** run a V49 fresh screen and do **not** run V50 fresh/official repeated-policy closed-loop before the V50 TRAIN result is analyzed.
