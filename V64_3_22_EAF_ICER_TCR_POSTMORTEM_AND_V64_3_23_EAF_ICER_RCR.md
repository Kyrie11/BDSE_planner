# V64.3.22 EAF-ICER-TCR Postmortem and V64.3.23 EAF-ICER-RCR Design

## Executive conclusion

The uploaded V64.3.22 run did **not** evaluate V22 on fresh validation. It stopped after the 3000-scene TRAIN transition-frontier replay and before fresh-token selection. Re-running the official fitter on the exact uploaded TRAIN artifact reproduces the direct cause:

`TRAIN internal holdout too small for evidence_only: retention=281 replacement=228`

This is an experiment-protocol/engineering failure, not a fresh algorithm result. Therefore V22 does **not** prove—or disprove—that its key causal paths stably convert into preservation and endpoint improvement.

However, removing only that brittle row-count abort is not sufficient. A corrected TRAIN-only reconstruction shows that the intended V22 transition-conditioned risk head improves the selected replacement path from about `-7.86` to `-3.06` teacher-improvement sum, but the path remains net harmful. The residual bottleneck therefore tightens to **selection-conditioned local regret coherence after extremal selection**.

V64.3.23 EAF-ICER-RCR keeps the established headline novelty:

> **evidence-attributed incumbent-contrastive reliability for deployment-admissible extremal recovery under a fixed planner-interface evidence budget**.

The main mechanism is now **evidence-local Regret-Coherent Reliability (RCR)**. Planner-transition conditioning is retained only as a controlled ablation because its fixed TRAIN cross-fold path sign is not uniformly stable.

“Counterfactual” in this report always means an operational same-scene candidate-versus-incumbent comparison under the same frozen evidence interface; it is not a causal-identification claim.

## 1. What actually ran in V64.3.22

The uploaded result contains:

- prerequisite and regression outputs;
- a complete 3000-scene TRAIN transition-frontier replay;
- a ~662 MB TRAIN frontier edge artifact;
- no fitted V22 risk config/report;
- no fresh-1000 token file;
- no A/B five-arm replay;
- no split checker or double-fresh screen report.

The TRAIN replay took `1786 s` (~29.8 min). The fitter log is empty because the old launcher piped stdout to `tee` but did not capture stderr, so the actual error only appears when the fitter is reproduced locally.

The deterministic V22 holdout contains 228 replacement edges from 51 unique replacement scenes. Requiring `>=256 replacement rows` is poorly aligned with a scene-level extremal operator and is the direct reason fresh validation never started.

## 2. Corrected V22 TRAIN-only attribution

Keeping the exact same uploaded TRAIN artifact and deterministic split, and changing only the arbitrary row-count abort for diagnosis, yields:

| TRAIN-only diagnostic | Evidence-only V22 | Transition-conditioned V22 |
|---|---:|---:|
| replacement holdout edges | 228 | 228 |
| replacement holdout scenes | 51 | 51 |
| replacement edge AUC | 0.6029 | 0.5923 |
| selected replacements, dual | 38 | 30 |
| selected precision, dual | 50.0% | 56.7% |
| selected teacher-improvement sum, dual | **-7.8579** | **-3.0571** |
| selected precision, scalar | 55.3% | 63.3% |
| selected teacher-improvement sum, scalar | -7.8540 | -3.0558 |
| predicted incumbent-fallback teacher-improvement sum | -11.5948 | -7.5369 |

Thus transition semantics contain useful information, but the V22 global additive risk head still fails the object that matters: **the actually selected replacement path**. It would be invalid to simply lower the sample-count gate and spend fresh GPU.

## 3. Per-scene structural attribution

The largest selected V22 transition-dual TRAIN-holdout failure is scenario `2b32a9f406845f75`:

- incumbent action: `1`;
- selected candidate: `2`;
- candidate-minus-incumbent teacher improvement: **-3.8065**;
- scalar dominance logit: **+0.1633**;
- signed-profile equal-mean dominance: **-0.2092**;
- transition-risk logit: **+5.0237**.

The second-largest selected loss is `bb77e9686029538d`, with teacher improvement **-0.9894**. These two scenes alone contribute `-4.7958`, whose magnitude is larger than the full selected-path net loss `-3.0571`; positive replacements partially offset them. V22 is therefore **tail dominated**, not uniformly bad.

The worst scene also exposes an operator inconsistency. V22's signed-profile arm ranks alternatives by the equal-mean scalar/profile dominance score, yet eligibility requires only `scalar_dominance>0` and `risk>0`. It can therefore execute a candidate whose **actual ranking view is negative**.

V23 fixes this without returning to V21's failed hard consensus. The signed-profile main requires the equal-mean score used by the operator itself to be positive, but it does **not** require the profile view to be independently positive.

## 4. Bottleneck after V22

The current evidence does not support a larger network as the next step.

The V22 replacement population contains `1455` support-positive + scalar-dominance-positive admissible alternatives from `310` unique TRAIN scenes. The global transition head reduces tail loss but misses a small number of large negatives. This is the same winner's-curse pattern at a later stage: an average edge model is evaluated after an extremal selection operator.

TRAIN diagnostics also show that directly concatenating all 41 transition dimensions into a local Euclidean metric is fold-sensitive. Transition geometry can dominate the 18 evidence dimensions by feature count rather than by causal usefulness.

The bottleneck is therefore:

> **selection-conditioned local regret coherence under extremal replacement**.

The relevant question is no longer “can the edge classifier obtain a higher AUC?” but “does the evidence-conditioned selected action-change path remain non-harmful across independent scene partitions?”

## 5. V64.3.23 EAF-ICER-RCR

RCR = **Regret-Coherent Local Reliability**.

### 5.1 Frozen paper mainline

`fixed B<=16 planner-interface evidence -> auditable evidence atoms -> frozen M=24 acquisition -> selected B<=16 evidence -> frozen EAF complete DARM-anchor frontier -> exact selected-evidence attribution -> complete deployment-admissible frontier -> frozen support/incumbent dominance -> TRAIN-only evidence-local regret coherence -> self-consistent extremal replacement with incumbent-default preservation -> unchanged final certificate -> unchanged structural-risk guard -> decision preservation`

Frozen components remain unchanged: EAF checkpoint/value arithmetic, acquisition, selector, B/M, evidence certificate, safety mask, one-sided guard, and structural-risk guard.

### 5.2 Asymmetric intervention principle

A raw-EAF incumbent that already passes final-guard admissibility is preserved by default. V23 removes learned admissible-incumbent->anchor veto from the main mechanism.

Learning only needs to justify an alternative replacement. This directly absorbs the V19-V22 observation that already-deployable incumbents should carry a higher evidence burden before being changed.

### 5.3 Evidence-local multiscale regret lower bound — main risk mechanism

The local memory is built only from TRAIN alternatives that are already:

- final-guard admissible;
- positive under the frozen anchor-support head;
- positive under the frozen scalar incumbent-dominance head.

For a runtime alternative, V23 uses two fixed TRAIN neighborhoods (`K=32` and `K=64`) in the standardized **18-dimensional frozen evidence-reliability space**. At each scale it computes an inverse-distance weighted local candidate-vs-incumbent teacher-improvement mean minus one weighted standard error. The risk score is the minimum of the two lower bounds.

Replacement requires this local lower bound to be `>0`.

This is not a validation-tuned threshold. `K={32,64}`, one-standard-error subtraction, the zero boundary, and distance definition are frozen before fresh validation.

The memory contains TRAIN feature vectors and TRAIN teacher-improvement targets. This is an offline nonparametric reliability readout; it does not access teacher/future information for the current runtime scene and it does not consume extra planner-interface evidence.

### 5.4 Signed selected-evidence self-consistency

Evidence-local scalar arm:

`support>0 AND scalar_dominance>0 AND local_regret_lower_bound>0`, ranked by scalar dominance.

Evidence-local signed RCR main adds:

`equal_mean(scalar_dominance, profile_dominance)>0`,

and ranks by that same equal-mean score.

This is deliberately weaker than V21 `scalar>0 AND profile>0`. It only enforces semantic consistency between the score used for final ranking and the decision to replace.

### 5.5 Transition conditioning is a controlled ablation, not the main

A transition-local memory is still produced using three group-balanced distance blocks:

1. 18 evidence features;
2. 21 maneuver/transition semantic features;
3. 20 trajectory-transition geometry features.

Each group contributes an average squared standardized distance, preventing raw dimensionality from determining the metric.

But TRAIN cross-fitting shows:

- evidence-local signed RCR: **5/5** fixed scene folds non-harmful, total selected teacher improvement **+9.8463**;
- transition-local signed RCR: **3/5** folds non-harmful, aggregate total **+14.4132**.

The transition view has higher aggregate TRAIN gain but worse fold stability. It is therefore **not** allowed to define V23 promotion. It remains a causal ablation and can be absorbed only if it gives incremental value in both independent fresh blocks.

These TRAIN values are design diagnostics only and must not be reported as fresh/paper endpoint results.

## 6. TRAIN gate: scene/operator aligned

V23 removes the brittle V22 `replacement holdout rows>=256` rule.

The evidence-local main must pass deterministic 5-fold scene-level out-of-fold operator auditing:

- at least 64 selected replacements in aggregate;
- aggregate selected candidate-vs-incumbent teacher improvement >=0;
- **all 5 fixed folds** individually have a non-harmful selected replacement path;
- instrumentation/population support remains sufficient.

This gate evaluates the same object used at deployment: the extremally selected action-change path. Local-risk edge AUC is reported but is not a promotion condition.

## 7. V23 double-fresh causal experiment

V22 never selected fresh validation identities, so the permanent design exclusion remains exactly **4700** tokens.

V23 selects 1000 untouched validation tokens using scenario identity + fixed SHA256 only, then freezes two independent 500-scene blocks A and B. No pooled rescue is allowed.

Each block runs five arms:

1. **raw EAF** — frozen endpoint reference;
2. **frozen V20 ICER-DC dual** — previously reproduced incumbent-contrastive control;
3. **evidence-local scalar RCR** — local regret coherence without signed extremal ranking;
4. **evidence-local signed RCR** — **V23 main**;
5. **transition-local signed RCR** — controlled transition-conditioning ablation.

Causal comparisons:

- `V20 -> evidence-local scalar`: does local selected-path regret coherence improve the reproduced mechanism?
- `evidence-local scalar -> evidence-local signed`: does exact signed selected-evidence attribution add to extremal ordering under identical local risk?
- `evidence-local signed -> transition-local signed`: does planner-transition conditioning add independent value? This is diagnostic and cannot rescue a failed evidence-local main.
- `raw -> evidence-local signed`: does the main mechanism convert to preservation + endpoint?

For each A/B block, the **main** must independently satisfy:

- frozen interface and structural-domain identity;
- healthy multi-admissible frontier;
- support AUC >=0.65 and direct incumbent-dominance AUC >=0.70;
- zero learned admissible-incumbent->anchor events by construction;
- at least 8 direct incumbent->alternative replacements and selected replacement path regret delta sum <=0;
- alternative recovery >=3%, precision >=80%;
- direct replacement rate >=2%, direct precision >=60%, opportunity capture >=8%;
- harmful intervention reduced by >=5pp vs raw;
- beneficial retention >=35%, beneficial>harmful;
- match >= DARM anchor +0.5pp;
- regret <=1.02x raw and <=1.02x frozen V20;
- signed evidence ranking must add over the otherwise identical evidence-local scalar arm.

Transition conditioning is reported independently; it is absorbed only if it also gives incremental benefit on **both** A and B.

Both A and B must pass. Passing authorizes only one frozen independent full-validation reproduction. Test and closed-loop remain forbidden until that reproduction passes.

## 8. Engineering and data audit

V23 also fixes the V22 experiment infrastructure:

- automatically reuses V22's completed 3000-scene TRAIN frontier when present, avoiding another ~29.8-minute replay;
- safely recreates it only if absent;
- captures fitter stderr with `2>&1 | tee`, so a TRAIN STOP cannot leave an empty diagnostic log;
- keeps validation scenario-token filtering before NPZ deserialization;
- SHA256-locks local memory and caches it once per process;
- uses matrix-product distance evaluation rather than allocating candidate x memory x feature tensors;
- keeps all 4700 previously inspected validation tokens excluded;
- writes a 3000-token TRAIN frontier manifest and hard-stops if any newly selected fresh validation token overlaps the TRAIN local-memory population.

Leakage audit: current-scene runtime inputs contain only frozen EAF/frontier statistics, selected-evidence attribution, planner runtime state, and the TRAIN-fitted/local memory artifact. Teacher labels are present only as TRAIN targets in the offline memory/fitter and in evaluation diagnostics; no validation/test teacher value enters config selection or runtime lookup. The uploaded 3000 TRAIN frontier tokens have **0 overlap** with the 4700 already-inspected validation design tokens, and the V23 launcher repeats an explicit TRAIN-vs-fresh identity check for the newly selected A/B tokens before any fresh replay.

Backward compatibility was checked in independent processes on 5000 randomized tournament cases using an old frozen ICER config: uploaded V22 and V23 produced identical action/score/public-diagnostic hashes, with 0 errors.

## 9. Changelog constraints that remain terminal

Do not retry or tune:

- BTP / RET / CET / AF / HAP;
- selector or acquisition redesign;
- larger B/M;
- V17 utility-equivalence hard mask;
- OCFI radius/alpha;
- EAIR/RAER/DACER/ICER/TCR/RCR threshold sweeps;
- evidence/safety/structural certificate relaxation;
- broad EAF unfreezing before the local-regret hypothesis is exhausted;
- V21 both-positive consensus as main;
- scalar/profile view-weight tuning;
- raw action-slot/transition blacklists;
- learned admissible-incumbent->anchor veto;
- pooled A/B evaluation;
- promotion by edge AUC when the selected path is harmful.

If evidence-local scalar succeeds but signed RCR fails, keep scalar RCR and do not tune view weights. If evidence-local main succeeds but transition conditioning does not, do not tune transition weights; keep transition out of the main. If evidence-local selected-path regret is still harmful, audit local neighborhood/tail support before any representation expansion.

## 10. Current scientific status

V22 cannot establish the final mechanism claim because fresh validation was never run. What V22 does establish from TRAIN is narrower but useful:

1. the old experiment protocol contained a real fitter gate bug;
2. transition-conditioned global risk carries signal but remains selected-path harmful;
3. the harmful result is tail dominated;
4. ranking-view/eligibility semantics were inconsistent;
5. the next object to test is evidence-conditioned local selected-path regret coherence, not a larger average-edge network.

V23 therefore remains on the same CCF-A-oriented paper line. The headline novelty is not yet “proven”; it becomes materially stronger only if evidence-local RCR reproduces the path-level, preservation, and endpoint chain independently on both untouched A/B blocks, followed by one independent frozen full-validation reproduction.
