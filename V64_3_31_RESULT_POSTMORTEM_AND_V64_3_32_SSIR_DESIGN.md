# V64.3.31 uploaded-result postmortem and V64.3.32 SSIR design

## Executive conclusion

The uploaded V64.3.31 run is **engineering-valid and protocol-complete for the stage it was allowed to reach**. It was not blocked by a Python/CUDA/runtime failure. The launcher intentionally stopped at the frozen TRAIN-only scientific gate because the V31 SCIR mean-ranking selected path was harmful in **all five** scene-disjoint folds. Therefore the result is reliable for a **development-level causal attribution of the SCIR mean-ranking proposal component**, but it contains no CAL500 or untouched A/B evidence and cannot be promoted as fresh/paper-level evidence for the complete SCIR-MAIN mechanism.

The V31 result falsifies the sub-hypothesis that a same-scene continuous incumbent-contrastive **mean-improvement argmax** is already selection-stable and only needs a final post-selection certificate. It does **not** falsify the unexecuted V31 conformal certificate itself. More importantly, it reveals why the V31 decomposition is structurally incomplete for the newly localized bottleneck: its post-selection conformal offset is common to every candidate in a scene, so it can abstain after a winner is chosen but cannot correct a wrong within-scene winner.

The next mechanism is V64.3.32 **SSIR — Selection-Stable Intervention Recovery**. SSIR keeps the V31 same-scene continuous intervention mean as a low-capacity signal, but moves reliability **before extremal selection**. It calibrates a scene-level simultaneous lower bound over all direct candidates and selects the action by a positive, candidate-specific lower bound rather than by the conditional mean. The calibration score is scene-uniform over the direct-eligible intervention population, and the candidate-specific scale is a frozen ridge-leverage normalization; conformal calibration, not a Gaussian variance assumption, supplies the finite-sample coverage statement.

---

# 1. Engineering and protocol audit of the uploaded V31 result

## 1.1 What actually executed

The result archive contains the prerequisite re-audit, targeted-regression log, V31 SCIR fit report/log, and stage timing. It does **not** contain CAL500, A/B manifests, fresh evaluator rows/edges, or a V31 double-fresh screen.

This is expected behavior under the frozen launcher. The V31 launcher executes:

1. V30.3 capacity-closure prerequisite;
2. repository/config regression;
3. frozen TRAIN SCIR-RANK 5-fold gate;
4. **only if the TRAIN gate passes**, label-free selection of CAL500/A500/B500;
5. independent calibration and untouched A/B.

The run stopped at step 3 because the fit tool exited fail-closed on the scientific gate. There is no traceback, CUDA OOM, missing tensor/schema exception, or partial evaluator output masquerading as a scientific failure.

## 1.2 Code audit

The uploaded V31 code was re-audited locally:

- `python -m compileall -q bdse`: PASS;
- V31 focused tests: **5/5 PASS**;
- full repository regression: **462/462 PASS**;
- warnings: **36**, all pre-existing PyTorch Transformer warnings;
- launcher `bash -n`: PASS.

The runtime SCIR feature and target semantics match the fitter:

- incumbent is the frozen raw/pre-recovery incumbent;
- the direct domain requires an admissible incumbent;
- candidate prerequisites are frozen deployment admissibility plus support `>0`;
- input is the frozen 18-D EAF evidence view, candidate minus same-scene incumbent, plus support-logit difference (19-D total);
- target is continuous `teacher_margin(candidate)-teacher_margin(incumbent)`;
- scene-equal ridge weighting and fixed `lambda=1` are used;
- teacher/future values are not runtime features;
- the historical binary dominance head remains diagnostic only in SCIR.

No code-level discrepancy was found that would turn the observed 0/5 gate into an engineering artifact.

## 1.3 Reliable attribution scope

The reliable conclusion is therefore:

> **V31's frozen mean-ranking proposal component failed its preregistered TRAIN selected-path gate.**

The result does **not** support conclusions about:

- independent CAL500 behavior;
- V31 post-selection conformal effectiveness;
- fresh A/B SCIR-MAIN generalization;
- paper-level promotion of SCIR.

Those stages were never executed.

---

# 2. V31 quantitative result

The training population is the frozen 3000-scene frontier used throughout the V25–V31 mechanism sequence:

- frozen scenes: **3000**;
- direct admissible/support-positive intervention edges: **9,394**;
- scenes containing such edges: **782**;
- edge-level positive fraction: **34.3198%**;
- total teacher-improvement sum over this broad edge population: **-2624.1702**.

The five-fold SCIR mean-argmax selected path is:

| Fold | Selected | Positive | Precision | Positive opp. | Capture | Sum ΔT | Worst ΔT | NegRMS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 150 | 83 | 55.33% | 130 | 63.85% | -5.9277 | -2.6145 | 0.4333 |
| 1 | 96 | 48 | 50.00% | 97 | 49.48% | -9.2158 | -3.8279 | 0.7088 |
| 2 | 128 | 71 | 55.47% | 122 | 58.20% | -3.1009 | **-4.0413** | 0.4590 |
| 3 | 111 | 70 | 63.06% | 118 | 59.32% | -0.9357 | -2.0207 | 0.2165 |
| 4 | 127 | 70 | 55.12% | 107 | 65.42% | -2.8931 | -2.0199 | 0.4191 |
| **All** | **612** | **342** | **55.88%** | **574** | **59.58%** | **-22.0733** | **-4.0413** | — |

Every fold violates the selected-path non-harm gate. Every fold also contains a replacement far below the historical catastrophic threshold `-0.5`.

The result is not “the mean model has no signal.” It increases selected-positive enrichment from the broad edge prevalence of 34.3% to 55.9%, and it reaches 59.6% positive-opportunity capture. The failure is that this signal is **not selection-stable**: 270 selected interventions are non-positive, and a small number of large negative winners dominate the aggregate sum in every fold.

This is exactly the failure mode expected from an extremal operator acting on a noisy conditional-mean score: selecting the maximum score preferentially selects positive prediction error as well as true benefit (winner's curse / selection-induced overestimation).

---

# 3. Scene-level attribution: what can and cannot be recovered from this upload

The V31 result archive does not contain a token-level cross-fit proposal audit. The V31 fitter serialized fold aggregates only, and the run stopped before any CAL/fresh evaluator output. The exact frozen TRAIN edge file is referenced from the external V30.2 output root, but it is not bundled in the uploaded V31 result archive.

Therefore exact failure tokens, candidate IDs, prediction errors, and per-scene winner alternatives cannot be reconstructed from this upload alone without inventing data. This is a **reviewability/instrumentation gap**, not a reason to invalidate the 0/5 fold result.

What is already scene-level causal evidence despite that limitation:

- all five scene-disjoint folds fail with the same sign;
- failure is not concentrated in one split;
- fold 3 still fails even with 63.1% selected precision, demonstrating that average precision is not enough when the negative tail is large;
- fold 2 reaches a `-4.0413` selected intervention despite positive-opportunity capture of 58.2%;
- the mechanism selects 612 interventions across 782 direct-support-positive scenes, showing that removing the old dominance gate makes the mean operator aggressive rather than merely unable to find opportunities.

V32 fixes the missing forensic artifact: its TRAIN nested tool **always writes a per-scene CSV before any scientific STOP**, including token, fit/cal/test fold, candidate count, mean winner/outcome, replayed V31 common-offset-veto outcome, SSIR lower-bound winner/outcome, selection scale, quantile, and whether the risk-aware winner differs from the mean winner.

---

# 4. What exactly failed in the V31 idea

## 4.1 Falsified sub-hypothesis

V31 assumed that after replacing the unstable independent binary dominance semantics with a same-scene continuous candidate-vs-incumbent target, the remaining problem would primarily be a post-selection acceptance certificate.

The TRAIN result falsifies the first half:

> **Conditional-mean intervention scoring + argmax is not a selection-stable proposal operator.**

The same-scene target is better aligned semantically than the historical binary head, but semantic alignment alone does not eliminate selection-induced overestimation or catastrophic winner errors.

## 4.2 What did not fail

The V31 independent conformal stage did not run. It is incorrect to report “conformal failed” from this result.

However, the V31 conformal form has a structural limitation now made important by the failure:

`LCB_b = mean_b - q`, with one common `q` after the mean winner is fixed.

For any two candidates `b1,b2`, subtracting the same `q` preserves their ordering. Therefore V31 calibration can only:

- accept the already chosen mean winner; or
- reject it and return the incumbent.

It cannot choose a lower-mean but more reliable challenger. If the dominant bottleneck is within-scene ordering under extremal selection, a common post-selection offset is not an ordering solution.

This does not mathematically prove that V31's veto would be useless; it could make the path safer by abstaining. It does show that it cannot simultaneously repair winner identity and retain high recovery coverage when those objectives require distinguishing candidates by reliability.

## 4.3 Experimental-design lesson

The V31 fail-closed stop was correct because it was preregistered. But the gate was imposed on `SCIR-RANK`, whereas the intended final mechanism was `SCIR-RANK + conformal veto`. Consequently the experiment did not test the full scientific object before spending fresh data.

The next gate should evaluate the **complete intended operator** under nested TRAIN fit/calibrate/test separation. This is a protocol correction for the next mechanism, not a post-hoc reinterpretation of V31.

---

# 5. Evidence chain from earlier versions through V31

The historical evidence now supports a narrower chain:

1. **Acquisition complexity exhausted.** Learned/teacher-shaped acquisition, coreset/beam/swap and related subset optimizers can improve proxy objectives without stable endpoint recovery.
2. **EAF instrumentation survives.** Complete anchor frontier plus exact selected-evidence attribution remains the most useful causal instrumentation.
3. **Richer representation is not reliable geometry.** Attribution-spectrum/KNN variants show that more coordinates do not create a stable risk neighborhood.
4. **DRC finds signal but not stable selected-tail reliability.** Aggregate downside logic can produce useful proposals but fresh catastrophe persists.
5. **Local semantic/type confirmation does not close the tail.** KNN geometry variants are exhausted.
6. **PTMC/classifier rescue is falsified.** TRAIN tail discrimination does not transfer to fresh harmful-proposal veto behavior.
7. **Global frontier reconstruction is not decision sufficiency.** FCR improves reconstruction fidelity while recovery does not improve and can become less stable.
8. **Capacity-only transmission is not the first-order missing mediator.** V30.3 independently shows decisive-evidence visibility rising while direct capture, precision and incumbent-relative dominance degrade on both fresh blocks.
9. **Same-scene conditional mean is still not selection-stable.** V31 has high positive-opportunity capture but 0/5 non-harm folds and catastrophic selected winners.

The new conclusion is stronger than “downstream consumer mismatch.” The downstream problem has now been localized to the **operator that turns an uncertain same-scene counterfactual score into an extremal winner**.

---

# 6. Dominant bottleneck after V31

Previous wording:

> selection-conditioned incumbent-contrastive ordering reliability at the direct replacement boundary, together with post-selection calibration.

V31 supports tightening it to:

> **candidate-specific lower-tail ordering under extremal same-scene incumbent replacement: control selection-induced overestimation before the winner is chosen, rather than attempting to repair a mean winner only after selection.**

Equivalently:

> **selection-stable counterfactual lower-bound ordering at the direct incumbent-replacement boundary.**

The weak link is no longer “can the model find a candidate with positive mean signal?” V31 finds many. The weak link is whether the planner can rank candidates by a conservative, candidate-specific decision certificate that remains meaningful under `argmax`.

---

# 7. Paper-level mainline

The paper-level thesis should **not** pivot to “SCIR” or “conformal” based on TRAIN. The previous mainline remains the correct abstraction:

> **Intervention-Conditioned Decision Sufficiency with Selection-Stable Extremal Recovery under a Bounded Auditable Planner Interface.**

V31 strengthens the need for the phrase **Selection-Stable** rather than weakening the thesis.

The mechanism chain should be refined to:

`bounded auditable evidence interface`

`-> exact EAF action-local attribution`

`-> deployment-admissible same-scene incumbent/challenger intervention set`

`-> intervention-conditioned mean evidence signal`

`-> candidate-specific reliability normalization / lower-tail evidence`

`-> scene-simultaneous lower-bound calibration before extremal selection`

`-> extremal selection on positive lower bounds`

`-> incumbent-default / no arbitrary fallback`

`-> structural + catastrophic-tail safety contracts`

`-> double-fresh + one frozen full-validation reproduction + closed-loop validation`.

If V32 succeeds independently, the paper can support several genuinely paper-level contributions rather than a sequence of engineering heads:

1. **Problem contribution:** decision sufficiency is defined at a concrete intervention boundary under a bounded/auditable planner interface, not by reconstruction fidelity or evidence count.
2. **Instrumentation contribution:** EAF exposes a complete action-local frontier with exact additive evidence attribution while preserving query budget.
3. **Operator/theory contribution:** selection is performed only after simultaneous candidate-wise lower bounds are calibrated at the scene level, so the finite-sample guarantee survives data-dependent extremal candidate choice.
4. **Safety-structure contribution:** incumbent default, domain separation and fail-closed/no-arbitrary-fallback semantics constrain learned intervention paths independently of statistical accuracy.
5. **Methodological evidence contribution:** a long falsification chain isolates acquisition, capacity, reconstruction, generic classifier and conditional-mean ordering from the actual selected-path bottleneck.

None of this guarantees a CCF-A acceptance or closed-loop SOTA. It gives a coherent route that can meet that standard **if** double-fresh, independent full validation, nuPlan closed-loop, strong baselines, budget/latency ablations and confidence intervals support the claims.

---

# 8. Directions that remain frozen / must not be repeated

Retain all historical no-repeat constraints. In particular do not reopen:

- learned/teacher-shaped acquisition, coreset, beam, swap, or generic selector-v2 optimization;
- broad B/M sweeps, capacity-only B24, same-bank rebinding or treating B=24 as novelty;
- FCR-v2, frontier L-infinity/RMS objective tuning, or global reconstruction fidelity as decision sufficiency;
- DRC `K`, threshold, downside multiplier, support/scalar boundary rescue;
- attribution-spectrum/transition/signed-profile KNN, semantic-family/type KNN, radius/OOD tuning;
- PTMC/classifier-v2/v3 or another tail-classifier head;
- naive feature concatenation or adding dimensions merely because V31 mean ordering failed;
- action blacklists or scene-ID rules;
- learned admissible-incumbent-to-anchor veto;
- pooling A/B or endpoint-path cancellation to rescue a failed direct mechanism.

New no-repeat constraints after V31:

- do not sweep ridge `lambda`, mean threshold, conformal `alpha`, leverage scale multiplier, or support threshold to rescue the 0/5 result;
- do not rerun the original V31 CAL/fresh path after relaxing its TRAIN gate;
- do not treat 59.6% capture as success while selected teacher sum and tail fail all folds;
- do not use one candidate-independent post-selection offset as if it solved within-scene ordering;
- do not revive a generic listwise/edge scorer unless it directly changes the selection-stability question and is evaluated on the selected path;
- do not claim a probabilistic variance model from the V32 leverage scale; it is only a deterministic normalization for conformal efficiency;
- do not spend new fresh data if the complete V32 nested TRAIN operator fails.

Historical V59-style scene-uniform conformal calibration is not to be naively repeated: that earlier residual-potential path produced a very large global epsilon on a different broad task. V32 is deliberately narrower: exact direct incumbent-replacement domain, same-scene counterfactual target, candidate-specific frozen normalization, simultaneous all-candidate scene score, and bound-based selection itself. If this still collapses coverage at the nested TRAIN gate, stop rather than tune it.

---

# 9. V64.3.32 SSIR — Selection-Stable Intervention Recovery

## 9.1 Frozen parts

V32 freezes:

- upstream B=16 / M=24 acquisition and query accounting;
- value model/checkpoint;
- EAF complete frontier and exact attribution;
- deployment admissibility;
- anchor support head and zero support boundary;
- structural-domain delegation;
- final execution guards;
- admissible-incumbent default established by the PRESERVE control;
- V31 19-D same-scene intervention target and fixed ridge `lambda=1`.

No FCR, FBIC, DRC, PTMC, KNN, classifier, new evidence query, new candidate bank, or validation sweep is added.

## 9.2 Mean signal

For candidate `b` and incumbent `i`, V32 retains the V31 mean model

`mu_b ~= Delta_T(b;i) = J_T(i)-J_T(b)`.

The point is not to declare the mean model safe. It is retained because V31 shows meaningful opportunity signal and because keeping it frozen isolates the new mechanism to selection-stability.

## 9.3 Candidate-specific normalization

From TRAIN-only standardized intervention features `z_b`, the fixed ridge system produces

`h_b = z_b^T G^{-1} z_b`,

`scale_b = sqrt(1 + h_b)`.

This is an auditable extrapolation/leverage normalization. It is **not** interpreted as a calibrated Gaussian standard deviation. Conformal validity does not require that interpretation; any positive frozen scale can be used, while the nested TRAIN gate tests whether this scale gives useful efficiency.

## 9.4 Scene-simultaneous conformal score

For every calibration scene in the deployment-relevant direct domain (admissible incumbent and at least one admissible support-positive alternative), define exactly one score

`R(scene) = max_b (mu_b - Delta_T(b;i)) / scale_b`,

where the max ranges over **all** direct candidates that the runtime operator could consider, not just the mean winner.

With fixed `alpha=0.05`, take the finite-sample one-sided order statistic `q`.

Candidate lower bounds are

`LCB_b = mu_b - q * scale_b`.

The runtime operator then chooses

`argmax_b LCB_b` among candidates with `LCB_b > 0`.

If no positive lower bound exists, the incumbent is retained. There is no second-best fallback after a rejected mean proposal because there is no separate mean proposal in the main operator: reliability is part of the ordering itself.

## 9.5 Finite-sample guarantee scope

Assume the mean model, scale function, direct-domain candidate-set rule and alpha are frozen before calibration, and calibration/future direct-eligible scenes are exchangeable. Split conformal gives marginal scene-level coverage

`P(R_new <= q) >= 1-alpha`.

On the event `R_new <= q`, every candidate simultaneously satisfies

`Delta_T(b;i) >= LCB_b`.

Therefore any data-dependent extremal choice made **after** these simultaneous bounds are available, restricted to `LCB_b>0`, has positive teacher improvement on that event. A harmful accepted SSIR intervention can occur only on a scene-level simultaneous miscoverage event.

This is stronger for extremal selection than V31's selected-winner-only common offset because the candidate set is covered simultaneously before the winner is chosen.

The guarantee is still only marginal over the direct-domain scene population. It is not a deterministic per-scene guarantee, not distribution-shift robustness, and not a closed-loop safety certificate.

---

# 10. V32 TRAIN gate: test the complete mechanism before fresh data

V32 replaces the V31 intermediate-ranker gate with a nested full-mechanism gate.

For each of five outer folds:

- 3 folds fit the frozen mean + leverage normalization;
- 1 separate fold calibrates the scene-simultaneous `q`;
- 1 outer fold tests the final positive-LCB extremal operator.

The rotating split never uses the outer test fold for fitting or calibration.

Hard TRAIN gate:

- all 5 outer test folds must have nonnegative SSIR selected-path teacher-improvement sum;
- aggregate SSIR selected count >= 64;
- aggregate SSIR selected-positive count >= 32;
- no alpha/ridge/feature/threshold sweep.

The tool additionally replays the original V31 **mean-winner + common post-selection conformal veto** on the same nested splits as a diagnostic. This directly answers whether simple abstention would have repaired V31 or whether candidate-specific risk-aware reordering is actually needed. That counterfactual is diagnostic only and cannot become a threshold-rescue branch.

The per-scene TRAIN audit is written even when the hard gate fails.

---

# 11. Independent CAL500 / A500 / B500 protocol

V31 stopped before selecting any CAL/fresh token, so no new validation scenes were spent. The permanent design exclusion remains the V31/V30.3 **10700-token** manifest with SHA256

`041ec824756777576391756ef3721617459bb0c0a45f7f43226b52254d951473`.

V32 uses a new label-free seed:

`v64.3.32-eaf-icer-ssir-cal500-double-fresh-v1`.

Only after the complete nested TRAIN gate passes, 1500 new scenes are selected and frozen as:

- CAL500;
- fresh A500;
- fresh B500.

CAL500 fixes exactly one `alpha=0.05` scene-simultaneous quantile. Calibration uses one score per **direct-eligible** scene, not one score per candidate and not zero-filled unrelated scenes. At least 64 direct-eligible calibration scenes are required.

A and B each run five paired arms:

1. RAW;
2. frozen V20;
3. PRESERVE incumbent-default control;
4. MEAN control (V31 same-scene mean ordering);
5. SSIR main (candidate-specific simultaneous lower-bound ordering).

A/B are judged separately.

Pre-registered promotion requirements per block include:

- exact paired 500-scene identity;
- exact query parity / no extra evidence query;
- all-flagged structural identity and delegation;
- exact positive-LCB winner runtime contract;
- main selected count >= 8;
- main direct capture >= PRESERVE + 3 pp;
- selected teacher-improvement sum >= 0;
- worst selected improvement > -0.5;
- selected NegRMS <= PRESERVE;
- direct precision >= PRESERVE;
- SSIR tail non-worse than MEAN and strictly better on at least one tail/precision axis;
- endpoint non-inferiority to both PRESERVE and V20 (`match` tolerance 0.002, regret tolerance 0.5%);
- no pooled A/B rescue;
- no alpha/scale/ridge/threshold tuning.

Empirical simultaneous coverage and harmful-selected-as-miscoverage containment are reported as theorem diagnostics. A literal observed coverage >=95% is not used as a brittle hard gate because split-conformal coverage is marginal, not a deterministic sample-frequency constraint; the actual selected path has the stronger hard safety gates above.

---

# 12. Expected result branches

## Branch A: nested TRAIN fails

Stop before CAL/fresh. Use the per-scene audit to determine whether failure is caused by:

- leverage normalization not separating overestimated winners;
- scene-simultaneous `q` becoming so large that useful coverage collapses;
- lower-bound ordering still selecting catastrophic overestimated candidates;
- intervention mean itself being too poorly aligned even before risk ordering.

Do **not** tune alpha/ridge/scale. A failure would falsify this low-capacity conformal lower-bound realization and require a new intervention representation/objective, not another threshold.

## Branch B: nested TRAIN passes, but V31 common-offset veto diagnostic also passes similarly

The simplest conclusion is that post-selection abstention may already be sufficient on TRAIN; fresh MEAN/SSIR and the diagnostic should determine whether candidate-specific reordering is actually needed. Do not promote the diagnostic based on TRAIN alone.

## Branch C: nested TRAIN passes because SSIR changes winner identity / tail while V31 common-offset veto remains poor

This is the strongest mechanism evidence for the new hypothesis: **selection stability requires candidate-specific reliability to enter ordering before extremal selection**.

Proceed to CAL500/A/B exactly once.

## Branch D: A/B both pass

Freeze the mechanism. Run exactly one independent full-validation reproduction, then closed-loop/strong-baseline/budget-latency analysis. No more algorithm tuning before that reproduction.

## Branch E: either fresh block fails

Do not pool. Use the failed block's selected-path and scene-simultaneous miscoverage audit to decide whether the remaining bottleneck is representation of intervention-specific tail state or non-exchangeable cross-population reliability. Do not return to acquisition/capacity/classifier/KNN/FCR/threshold sweeps.

---

# 13. Local engineering validation of V32 implementation

Before packaging:

- Python compile: PASS;
- V32 focused tests: **6/6 PASS**;
- V13–V32 targeted regression: **138/138 PASS**;
- full repository regression: **467/467 PASS**;
- warnings: **36**, all pre-existing PyTorch Transformer warnings;
- `RUN_V64_3_32_EAF_ICER_SSIR_SCREEN_2GPU.sh` shell syntax: PASS;
- V31 legacy artifacts without a leverage matrix remain backward-compatible with unit scale;
- malformed scale schema fails closed;
- synthetic test confirms a candidate with higher mean but larger leverage penalty can be demoted by SSIR, proving that risk enters ordering rather than merely post-selection veto;
- synthetic test confirms no positive LCB returns the incumbent with no fallback;
- structural all-flagged delegation remains exact.

No nuPlan GPU result is fabricated locally. The first real V32 evidence must come from the server launcher.
