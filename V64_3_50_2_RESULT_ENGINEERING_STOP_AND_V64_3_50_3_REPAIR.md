# V64.3.50.2 uploaded result: ENGINEERING/DATA STOP and V64.3.50.3 repair

## Verdict

The uploaded V64.3.50 result is **not scientifically attributable**. The preregistered V50 rule says that incomplete paired selected-outcome collection is an **ENGINEERING/DATA STOP** and only the data/identity pipeline may be repaired. Therefore:

- V50 PIOR algorithm success/failure: **NOT EVALUATED**;
- V50 preregistered identification/deployment GO gates: **NOT REACHED**;
- V51 algorithm design: **NOT PERMITTED from this result**;
- untouched validation: **NOT PERMITTED**.

This is not a weak or borderline PIOR result. The required paired outcome table does not exist.

## Reliability audit of the uploaded run

- source provenance: 913/913 manifest entries validate against the uploaded code;
- server targeted regression: 266/266 PASS;
- TRAIN manifest: 502/502 unique frozen V49 full-set RSMR proposal tokens;
- only the 4-scene paired preflight ran;
- control: 3 success / 1 failure / 3 probe events;
- treatment: 3 success / 1 failure / 3 probe events;
- both arms fail the same token `00f4aedf9b3c5f65` before any treatment/control causal difference can be evaluated;
- failure: `frozen manifest proposal is invalid at anchor`, frozen action `21`, runtime raw bank `K=32`;
- no complete 502-pair outcome table and no V50 PIOR nested fit/gate.

The repeated `[PIOR-TICK] ... Simulation failed!` line is a heartbeat printing the last relevant child-log line. It does **not** mean the same arm is creating a new failure every 30 seconds. The batch contains one failed simulation in each arm.

## Root cause

### 1. V50.2 fixed time, but not physical proposal identity

V50.2 correctly recognized that nuPlan tagged scenarios can start before the trigger anchor. However it still represented the frozen V49 proposal primarily by an integer action slot. The candidate generator is state dependent, so the trajectory currently occupying slot `a` is not a stable identity for the cached V49 proposal.

### 2. The uploaded successful events falsify online re-selection as an identity requirement

All three successful probe events have `online_proposal_matches_target=false` in **both** arms:

| token | anchor offset | frozen action | online proposal | online exists |
|---|---:|---:|---:|---|
| `0395a156348a5041` | 0 us | 24 | 13 | True |
| `034da27cd2f75292` | 0 us | 5 | 2 | False |
| `02fca515559f520d` | 2999474 us | 0 | 7 | True |

Two of these three already have `anchor_offset_us=0`. Therefore merely starting at the correct timestamp does not guarantee that an online full-model replay will select the same OOF V49 proposal. This is expected to be treated as a diagnostic, not as the treatment definition.

### 3. Do not compare V49 `candidate_count` with raw CandidateBank.K

The V49 candidate audit's `candidate_count` is the size of the post-admissibility RSMR candidate population. It is **not** the raw candidate bank size. The failed token has V49 `candidate_count=16` while the runtime raw bank is `K=32`; this is not itself a mismatch. V50.3 persists raw cached `candidate_trajectories.shape[0]` separately.

## V64.3.50.3 engineering repair

V50.3 changes **no scientific algorithm**. It repairs the paired intervention identity.

1. **Manifest physical identity.** For every one of the exact 502 V49 proposals, persist the cached selected local trajectory, SHA256, maneuver id, raw candidate-bank size, anchor timestamp and action id.
2. **Exact anchor scenario start.** Read the installed nuPlan scenario mapping and preserve each configured scenario duration/subsample ratio while changing only extraction offset to `0`. Unknown scenario types already use offset 0. This removes planner-controlled pre-roll before the causal event.
3. **Cached proposal execution.** At iteration 0 / exact V49 anchor, treatment executes the cached frozen local trajectory directly once. It never substitutes whatever trajectory currently occupies the same slot. Control executes the incumbent. Both arms use incumbent-only operation afterwards.
4. **Online replay is diagnostic only.** Current-slot validity, current-slot geometry and online full-set RSMR proposal are logged, but none can redefine or veto the treatment target.
5. **Fail-closed identity certificate.** A batch is valid only with exact token identity, exact anchor at iteration 0, exact manifest action, frozen-trajectory SHA256, one fired event, same-proposal/incumbent containment, no fallback, zero failed simulations and exact per-scenario metrics. CandidateBank.K drift still fails closed.
6. **Fresh output root.** Default output root is `outputs_v64_3_50_3_eaf_icer_pior_train_2gpu_v1`, so the invalid V50.2 batch cannot be resumed accidentally.

## What remains frozen

RSMR selection, EAF/ICER interface, Q/P/E coordinates, sign-risk model, fixed lambda, paired-outcome definition, hard-safety coordinates, retention calibration, same-winner/incumbent containment, no rerank, no second best and no fallback remain unchanged. This repair therefore cannot be interpreted as a new V51 algorithm.

## Validation of the repaired code

- V50.3 focused: **27/27 PASS**;
- V13→V50 targeted: **269/269 PASS**;
- full repository: **606/606 PASS** over all 126 test files, run as four mutually exclusive shards `158 + 152 + 155 + 141`;
- Python compile: PASS;
- launcher `bash -n`: PASS.

## Next scientific step

Rerun V50.3 TRAIN paired collection. The first 4-scene paired preflight must pass **4/4 control and 4/4 treatment**, with one exact iteration-0 event per token and treatment `frozen_proposal_trajectory_override_used=true`. Only after the complete 502/502 paired table exists may the original V50 identification gate (`PIOR > OBS/EGO`, aggregate and >=4/5 folds) and causal-retention gate be evaluated. If those gates pass, freeze V50 and proceed to untouched paired validation. If they fail, only then is a V51 scientific branch justified under the pre-registration.
