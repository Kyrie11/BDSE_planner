# V64.3.14 EAF-OCFI causal postmortem and V64.3.15 EAF-EAIR design

## 1. Executive conclusion

V64.3.14 confirms the V64.3.13 diagnosis only at a coarse level: the EAF complete-frontier value was over-intervening on the frozen DARM anchor. OCFI does remove those harmful interventions, but it does so by suppressing essentially every EAF intervention. It therefore does **not** solve the real preservation problem, which is selective discrimination between reliable and unreliable top-challenger edges.

On the held-out 300-scene OCFI evaluation split, raw EAF deployed flips occur on 70.33% of scenes. Attribution-scaled OCFI and the constant-radius control both reduce deployed flips to 0%. Harmful interventions fall from 22.0% to 0%, but beneficial interventions also fall from 13.33% to 0%. The final action exactly reverts to the selected-local/DARM anchor: teacher match rises from raw EAF 17.67% to anchor 26.33%, while teacher regret worsens from 11356.17 to anchor 13061.87. Attribution and constant OCFI produce identical final decisions, so there is no attribution-specific gain.

The next permitted branch should therefore not be another OCFI alpha/radius/threshold sweep and should not reopen acquisition. It should test whether the already-computed selected-evidence/EAF statistics contain enough information to predict whether the *actual top challenger* is teacher-better than the DARM anchor.

## 2. What V64.3.14 actually improved

OCFI is useful as a causal experiment because it establishes that the uncontrolled EAF intervention rate is a genuine failure mode. The one-sided radius can stop harmful flips. However, the mechanism is non-discriminative:

- raw EAF deployed flip: 70.33%;
- attribution OCFI deployed flip: 0%;
- constant OCFI deployed flip: 0%;
- raw beneficial intervention: 13.33%;
- attribution OCFI beneficial intervention: 0%;
- raw harmful intervention: 22.0%;
- attribution OCFI harmful intervention: 0%;
- attribution and constant final decisions: identical.

Therefore the correct conclusion is not “calibration solved preservation.” The correct conclusion is “a sufficiently conservative global lower bound can recover the anchor by abstaining from EAF entirely.”

## 3. More precise bottleneck: extremal/top-1 challenger reliability

The V64.3.13 complete-frontier pair-sign gain was an average-over-frontier mechanism metric. Deployment does something harder: it selects an extremal challenger and asks whether that single edge is reliable enough to change the action. This distinction is now decisive.

On the 300 held-out V64.3.14 evaluation scenes:

- raw EAF proposes a non-anchor challenger on 287 / 300 scenes (95.67%);
- 54.70% of those proposed challengers are actually teacher-better than the DARM anchor;
- the sign of the raw predicted proposed-vs-anchor margin agrees with the teacher on only 48.08% of those proposal edges;
- raw-margin AUC for teacher-better-vs-worse proposal edges is only 0.622;
- proposed attribution scale AUC is 0.715;
- frontier attribution RMS AUC is 0.696.

Thus the failure is localized to

`informative average complete-frontier value -> extremal/top-1 challenger selection -> unreliable intervention edge -> preservation failure`.

This is compatible with a winner's-curse/selection-bias interpretation: even if many frontier pair estimates improve on average, selecting the strongest estimated challenger concentrates the residual tail error. A single global over-prediction quantile then becomes large enough to reject almost everything.

## 4. Attribution energy was used in the wrong semantic role by OCFI

V64.3.14 deliberately avoided claiming attribution energy as epistemic uncertainty, but its main branch still used attribution magnitude to scale the conservative radius. The screen shows that this is poorly aligned with the observed mechanism.

Among exact beneficial EAF interventions, mean proposed attribution scale is 0.0505; among harmful interventions it is only 0.0198. Beneficial raw margins are also larger (0.2135 versus 0.0754). Attribution magnitude is therefore behaving more like **support strength / evidence concentration** than “more attribution means more uncertainty.” Penalizing it multiplicatively can suppress the very interventions that have the strongest selected-evidence support.

This does not prove attribution is a calibrated confidence measure. It only proves that V64.3.14's use of it as a global uncertainty-radius normalizer is not supported by this result.

## 5. Why we do not immediately unfreeze the representation

A diagnostic readout probe was run only to decide the next experiment. It is **not** a V64.3.15 result and must not be used in a paper table or promotion decision because it uses the same V64.3.14 design scenes.

Using only runtime-visible frozen EAF statistics, a small standardized logistic readout trained on the old 200 calibration scenes and evaluated on the old disjoint 300 evaluation scenes reaches teacher-better edge AUC 0.752. With a fixed 0.5 gate, a counterfactual filtering calculation gives:

- anchor teacher match: 26.33%;
- raw EAF teacher match: 17.67%;
- diagnostic gated match: 33.67%;
- anchor regret: 13061.87;
- raw EAF regret: 11356.17;
- diagnostic gated regret: 9561.34;
- diagnostic harmful intervention: 2.33%;
- diagnostic beneficial intervention: 9.67%.

Again, these are design diagnostics, not formal results. They establish only that scalar statistics already computed from the frozen EAF pathway plausibly contain nontrivial reliability information. Therefore jumping directly to a broad backbone/action/evidence unfreeze would be premature and would weaken causal attribution.

## 6. V64.3.15 algorithm: EAF-EAIR

**EAF-EAIR = Evidence-Attributed Intervention Reliability on the frozen EAF decisive frontier.**

The main chain remains:

`fixed planner-interface budget -> auditable evidence atoms -> terminally frozen acquisition -> exact selected B=16 evidence -> frozen EAF complete decisive-frontier value -> evidence-attributed one-sided intervention reliability -> unchanged evidence certificate -> final decision preservation`.

Everything before the preservation interface remains frozen:

- candidate bank unchanged;
- proposal/HAB/family gate unchanged;
- M=24 unchanged;
- exact B=16 selector unchanged;
- DARM unchanged;
- DBR unchanged;
- V64.3.13 EAF checkpoint/value unchanged;
- pair-full/local-pair-full remain EAF/EAIR-free ceilings;
- legacy evidence-certificate requirement is not relaxed.

Only a tiny external standardized logistic readout is fitted. It predicts

`P[J_T(proposed) < J_T(DARM anchor) | runtime EAF diagnostics]`.

Equivalently, the label is positive when the teacher proposed-vs-anchor margin is positive. This is deliberately a one-sided *teacher-improvement* target, rather than exact teacher-winner classification. That aligns the intervention decision with both action preservation and bounded teacher regret: a non-teacher-winner challenger may still be preferable to the current wrong anchor if it lowers teacher cost.

### Runtime features

EAIR uses only quantities already available after the selected-B EAF frontier has been computed:

1. raw proposed-vs-anchor EAF margin;
2. proposed attribution scale;
3. frontier residual RMS;
4. frontier residual absolute mean;
5. frontier attribution-scale RMS;
6. frontier attribution-scale mean;
7. existing evidence-certificate fraction;
8. normalized valid-action count;
9. raw-margin / proposed-attribution ratio;
10. proposed-attribution / frontier-attribution ratio.

No logged future or teacher quantity is available at runtime. No new evidence atom is queried.

### Training/readout protocol

- fit on **train split only**;
- class-balanced BCE + small L2 regularization;
- deterministic internal train-only 20% holdout reports capacity AUC;
- after that diagnostic, refit on all train proposal edges;
- probability threshold fixed at 0.5;
- no validation threshold/alpha/radius sweep;
- EAIR can only *block* an EAF intervention; it cannot manufacture a new challenger or relax the legacy guard/certificate.

This is intentionally a small selective representation/readout-capacity test, not a broad new backbone. If it succeeds, the next paper-facing mechanism can replace the scalar readout with a structured query-conditioned evidence/action reliability representation while retaining the same causal object. If it fails, we have stronger evidence that summary-level frozen representations are insufficient.

## 7. Fresh validation design: no reuse of V64.3.14 design scenes

Because the uploaded V64.3.14 500 scenes were used to diagnose and design EAIR, they are explicitly excluded from the V64.3.15 promotion screen.

The launcher:

1. re-audits the causally valid V64.3.13 EAF checkpoint;
2. runs raw EAF on train to collect EAIR features/labels;
3. runs a larger val discovery replay only to enumerate candidate scenario tokens;
4. excludes all 500 V64.3.14 design tokens;
5. freezes a fresh 500-scene validation token set;
6. fits EAIR from train only with threshold 0.5;
7. concurrently replays raw EAF and EAIR on the exact same fresh validation tokens;
8. checks frozen B/M/acquisition/pair-full/evidence-certificate interfaces;
9. stops before full/test/closed-loop unless the paired screen passes.

### Promotion conditions

Instrumentation/capacity:

- EAIR active >=95%;
- train internal-holdout teacher-better AUC >=0.65;
- fresh-val teacher-better AUC >=0.65;
- fresh-val proposal edges >=64;
- complete-star coverage >=99%;
- B=16, M=24, selected-local anchor, pair-full/local-pair-full and evidence certificate unchanged.

Preservation:

- harmful intervention absolute reduction >=5pp;
- retain >=35% of raw beneficial interventions;
- beneficial intervention rate > harmful intervention rate;
- retain at least 3% deployed intervention so the method cannot pass by reverting to OCFI-style total abstention;
- deployed flips must be lower than raw EAF.

Endpoint:

- teacher match >= anchor +0.5pp;
- teacher regret <= 1.02 * min(raw-EAF regret, anchor regret).

Only all three together promote a separate full-val reproduction. Test and closed-loop are still forbidden from the screen launcher.

## 8. Pre-registered failure interpretations

1. **Low train/val reliability AUC with valid instrumentation:** summary-level frozen EAF representation is insufficient. Next step is a small query-conditioned action/evidence reliability adapter; acquisition stays frozen.
2. **AUC is good but preservation fails:** do not tune the scalar threshold. The remaining issue is structured per-atom/pair reliability representation, not calibration hyperparameters.
3. **Preservation passes but endpoint fails:** reliability filtering works, but EAF top-challenger ranking/value is still misaligned. Audit extremal challenger ranking and teacher-regret ordering; do not reopen acquisition.
4. **Mechanism + endpoint pass:** full-val reproduction first. Only after independent reproduction may test and closed-loop be considered.

## 9. No-repeat constraints

All V64.3.14 constraints remain. Additionally:

- do not make another constant/conformal OCFI radius variant;
- do not tune alpha or the EAIR probability threshold on validation;
- do not retrain EAF-DMVR merely to change its average pair-sign loss;
- do not reopen BTP/RET/CET/acquisition;
- do not change B=16 or M=24;
- do not relax the legacy evidence certificate;
- do not apply EAF/EAIR to pair-full/local-pair-full;
- do not interpret a train-only readout AUC as an endpoint result;
- do not use the 500 V64.3.14 design scenes for V64.3.15 promotion;
- after a valid EAIR failure, move to query-conditioned action/evidence reliability representation rather than another scalar gate.

## 10. Engineering audit

The implementation adds EAIR only inside the existing anchor guard. The EAF residual/margins are bitwise unaffected by the gate; the gate changes only whether the already-proposed EAF challenger may replace the DARM anchor. A specific regression test verifies that EAIR is an exact no-op when EAF is intentionally absent from pair-full/local-pair-full diagnostics.

The fitter writes an explicit fixed feature schema, train statistics and weights into a generated config. The contract checker verifies B=16 both in the true `evidence.budget` field and metadata, M=24, `min_selected_atoms=16`, OCFI disabled, fixed 0.5 threshold, exact feature schema and positive standard deviations.

Final local regression results are recorded in `V64_3_15_ENGINEERING_VALIDATION.txt`.
