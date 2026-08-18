# V64.3.20 EAF-ICER-DC postmortem and V64.3.21 EAF-ICER-MCR design

## Executive conclusion

The uploaded V64.3.20 run must remain a **STOP screen**, but its printed `NEXT_ACTION=V19_mechanism_failed_to_reproduce...` is not the correct causal diagnosis. The checker mixed all-actions-safety-flagged scenes—where V20 deliberately disables learned intervention and delegates to the structural-risk guard—into learned-safe-domain recovery and global guard-cleanup gates.

After correcting that accounting, V64.3.20 is **Case E** from the V20 pre-registration:

- candidate semantics: PASS;
- multi-challenger support: PASS;
- support/dominance signal: PASS;
- V19 safe-domain operator identity: PASS;
- deployment-complete all-flagged semantics: PASS;
- direct incumbent recovery: PASS;
- preservation: PASS;
- endpoint regret: **FAIL**.

Therefore the previous rule—*only enter teacher-improvement magnitude / robust extremal ordering in Case E*—was correct. The next algorithm should not reopen acquisition, B/M, certificates, or thresholds.

## 1. Corrected V20 screen attribution

### Endpoint

| arm | match | regret | beneficial | harmful | flip |
|---|---:|---:|---:|---:|---:|
| DARM anchor | 14.2% | 25247.70 | - | - | - |
| raw EAF | 15.6% | **15496.66** | 8.6% | 7.2% | 55.0% |
| frozen V19 scalar | 19.4% | 16482.96 | 6.8% | **1.6%** | 35.6% |
| V20 scalar | 19.0% | 16420.18 | 6.8% | 2.0% | 37.4% |
| V20 dual | **19.6%** | 16387.74 | **7.4%** | 2.0% | 37.4% |

V20 dual preserves the key preservation gain but misses the pre-registered regret tolerance (`<=1.02 x raw`).

### The official mechanism failure is a checker artifact

There are 28 all-flagged scenes. V20 delegates all 28 to the structural-risk path and final action identity versus raw is 100%. V20-scalar is also 100% identical to frozen V19-scalar in the safe-available domain.

The old checker nevertheless counted delegated scenes in learned recovery. Domain-aware re-audit gives for V20 dual safe-domain only:

- support AUC: **0.7351**;
- direct dominance AUC: **0.7584**;
- direct replacement precision: **60.98%**;
- direct opportunity capture: **32.26%**;
- selected non-anchor teacher-better: **80.90%**;
- alternative precision: **87.80%**;
- safe-domain post-selection guard block: **0%**.

The global 1% guard-block rate is exactly five delegated all-flagged scenes. It is not learned safe-domain cleanup.

Hence the correct next-action classification is endpoint-only Case E.

## 2. The real remaining bottleneck

Per-scene V20-dual versus raw decomposition:

| path | scenes | total regret delta |
|---|---:|---:|
| direct admissible incumbent -> alternative | 82 | **-41,211.95** |
| admissible incumbent -> anchor | 88 | **+486,752.07** |
| raw incumbent inadmissible / anchor-relative | 169 | 0 |
| keep legacy | 133 | 0 |
| all-flagged delegated | 28 | 0 |

So direct alternative recovery is not the endpoint bottleneck. It is net beneficial.

The unstable operation is using the generic all-edge anchor-support classifier as a hard veto over an already deployment-admissible selected incumbent. On the previous V19 fresh split this branch was beneficial (about -96k regret); on V20 fresh it reverses to about +487k. This is a cross-split extremal-calibration problem concentrated in incumbent retention.

The new causal statement is:

> reliability has generalized well enough to recover alternatives, but an already admissible selected incumbent needs a **selection-conditioned, regret-sensitive evidence burden** before it can be demoted to the anchor.

## 3. Mainline and novelty decision

Keep the paper mainline:

`fixed planner-interface evidence budget -> auditable selected evidence -> frozen complete EAF frontier -> exact selected-evidence attribution -> deployment-admissible incumbent-contrastive reliability -> evidence-supported extremal recovery -> unchanged certificate/structural guard -> decision preservation`.

Keep the headline novelty:

> **evidence-attributed incumbent-contrastive reliability for deployment-admissible extremal recovery under a fixed planner-interface evidence budget**.

V20 does not falsify this mechanism. On the contrary, direct incumbent replacement still reproduces at roughly 61% precision and 32% opportunity capture. What has not yet reproduced is the complete conversion from mechanism to endpoint because incumbent demotion has an asymmetric tail cost.

V21 therefore refines the mechanism rather than changing the story: **asymmetric evidence burden** for preserve/replace/anchor decisions.

## 4. V64.3.21 EAF-ICER-MCR

Full name:

**Evidence-Attributed Incumbent-Contrastive Extremal Recovery with Magnitude-aware Corroborated Reliability**.

### 4.1 Selection-conditioned incumbent retention

For a raw-EAF incumbent that already satisfies the final-guard-admissible contract, generic edge support is no longer allowed to veto it directly.

A TRAIN-only linear MSE readout is fit only on raw selected incumbents that are final-guard-admissible. Target:

`J_T(anchor) - J_T(incumbent)`.

The target is divided by a positive TRAIN RMS scale only, preserving zero exactly. Runtime:

- predicted margin >= 0 -> incumbent remains baseline;
- predicted margin < 0 -> anchor becomes baseline.

No validation threshold is used. L2 is fixed at `1e-3`.

Two representations are emitted:

- scalar retention: 18 registered non-atom incumbent-relative evidence statistics;
- profile retention: those 18 plus 12 exact signed selected-atom contribution statistics.

The real V18 TRAIN frontier contains 1674 eligible selected incumbents. Fixed internal holdout diagnostics:

- scalar retention AUC: **0.7930**, sign accuracy **0.7841**;
- profile retention AUC: **0.7676**, sign accuracy **0.7674**;
- profile normalized MSE is slightly lower than scalar.

These are TRAIN-only design diagnostics. They justify testing both representations; they do not establish a signed-profile retention claim.

### 4.2 Corroborated incumbent dominance

V20 dual uses an equal mean of scalar and signed-profile dominance logits. V21 main changes eligibility, not weights:

- alternative support logit > 0;
- scalar direct-dominance logit > 0;
- signed-profile direct-dominance logit > 0;
- only then rank by their equal mean.

This makes the signed attribution view corroborating evidence rather than a compensating term. All boundaries remain zero; no threshold or mixture-weight sweep is permitted.

### 4.3 Structural domain remains frozen

All-flagged scenes continue to preserve the raw proposal and delegate to the unchanged structural-risk guard. V21 learns nothing in that domain.

## 5. Why this is the appropriate Case-E branch

V20 shows:

- semantic candidate/frontier problems are already fixed;
- direct alternative replacement is net beneficial;
- all-flagged deployment semantics are now correct;
- preservation is strong;
- endpoint error is dominated by the incumbent-to-anchor path.

Therefore it would be causally wrong to change selector, acquisition, budget, safety/evidence certificates, or direct-dominance thresholds. The only justified new learning is a selection-conditioned improvement-magnitude readout for the incumbent, plus a structural corroboration rule for high-score alternative false positives.

## 6. V21 experiment: two independent fresh replications

V21 permanently excludes all 3700 already inspected validation tokens, selects 1000 new tokens by identity + fixed SHA256 only, and deterministically splits them into two disjoint 500-scene blocks A/B.

Each block runs five arms:

1. raw EAF;
2. frozen V20 dual;
3. scalar-retention + old dual-mean dominance;
4. profile-retention + old dual-mean dominance;
5. profile-retention + positive-consensus dominance.

The experiment separately tests:

- magnitude objective: V20 -> scalar retention;
- signed selected-evidence retention contribution: scalar retention -> profile retention;
- robust evidence corroboration: profile mean -> profile consensus;
- final mechanism: raw -> consensus.

**Both blocks must pass independently. Pooled 1000-scene metrics cannot rescue a failed block.**

Primary per-block gates include:

- frozen interface and complete frontier instrumentation;
- all-flagged raw identity and safe-domain zero hidden guard cleanup;
- support AUC >=0.65, direct dominance AUC >=0.70;
- profile retention AUC/sign accuracy >=0.65;
- total regret delta of admissible-incumbent->anchor branch <=0;
- direct replacement precision >=60%, opportunity capture >=8%;
- safe selected-nonanchor teacher-better >=80%;
- consensus direct precision >= profile-mean +1pp, capture loss <=6pp;
- harmful reduction >=5pp and beneficial retention >=35%;
- match >= anchor +0.5pp;
- regret <=1.02x raw.

Passing the double-fresh screen permits only one frozen independent full-validation reproduction. It does not permit test or closed-loop.

## 7. Designs retained vs. discarded

Retain in the main mechanism:

- V18 final-guard-admissible complete frontier;
- V19 support/dominance decomposition;
- V20 structural-domain delegation;
- direct incumbent precision/capture as mechanism metrics;
- exact signed selected-evidence attribution as a structured evidence view;
- V21 selection-conditioned magnitude and corroborated dominance if fresh replication supports them.

Keep as mandatory ablations:

- scalar versus profile selected-incumbent retention;
- profile mean versus profile positive-consensus dominance;
- frozen V20 control.

Do not retry:

- OCFI radius/alpha or probability-threshold sweeps;
- BTP/RET/CET/AF/HAP;
- reopening selector/acquisition or increasing B/M;
- relaxing evidence/safety/structural guards;
- restoring V17 utility-equivalence hard mask;
- generic all-edge support as an unqualified veto over an already admissible incumbent;
- broad EAF unfreezing before this double-fresh causal screen.

## 8. Engineering audit

Implemented and checked:

- historical V20 checker is domain-aware; replaying the uploaded V20 output now returns Case E and the correct magnitude-ordering next action;
- retention teacher target is only consumed in TRAIN-only fitter; runtime config contains only fixed feature normalization/weights/bias;
- scalar/profile dominance and anchor-support heads remain byte-for-byte semantically frozen from V20/V19;
- all-flagged delegation remains unchanged;
- retention diagnostics are initialized on every branch and exported per edge;
- all scenario token filtering remains before NPZ deserialization;
- V21 design exclusion is 3700 unique validation tokens;
- launcher uses two disjoint 500-token files and hard paired-identity checks;
- old V20 dual behavior through the modified V21 tournament was replayed on 5000 randomized cases: zero action/score/diagnostic differences, identical SHA256.

Regression:

- targeted V64.3.6--V64.3.21: **101/101 PASS**;
- full repository: **389/389 PASS**;
- warnings: **36**, all pre-existing PyTorch Transformer `nested_tensor/norm_first` warnings;
- real TRAIN fit + all three V21 config contracts: PASS;
- launcher shell syntax: PASS.

## 9. Decision rule after the next run

If both fresh blocks pass, freeze V21 and run exactly one independent full-val reproduction.

If selected-incumbent magnitude fails, do not change alternative dominance or B/M; audit incumbent-conditioning/representation using TRAIN-only data.

If magnitude/recovery pass but corroboration fails, keep profile-mean as candidate and do not tune view weights.

If mechanism/preservation pass on both blocks but endpoint still fails, only then extend the same Case-E branch to explicit regret-tail / teacher-improvement magnitude ordering on the same frozen frontier.
