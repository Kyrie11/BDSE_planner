# V52 Result Diagnosis and V53 WC-BFAR-DBAP

## Executive conclusion

The uploaded V52 experiment does **not** show that BFAR-DBAP failed to learn winner evidence. It shows that the pipeline stopped before BFAR training because the reused-foundation gate evaluated a runtime-dependent budgeted-action metric instead of the immutable base+dense-local anchor.

There is nevertheless a real algorithmic defect in V52: its large `full_action` and `full_margin` weights act on a frozen branch and therefore provide no gradient to the trainable residual/selector modules. V53 fixes both the causal attribution error and the missing winner-learning signal while retaining the fixed-budget decisive-evidence main line.

## 1. What actually ran in V52

The executed stages were:

1. reuse V51 foundation checkpoint;
2. replay 1000 validation samples under the V52 runtime/configuration;
3. run the V52 anchor gate;
4. stop on gate failure.

The following stages did not run:

- BFAR residual/selector training;
- periodic exact AOCC distillation during BFAR training;
- candidate/local/foundation three-way open-loop replay;
- minimum or competitive candidate gate;
- CL20;
- CL100.

Therefore V52 gives no empirical answer to whether the boundary sampler, decisive-evidence selector or certified residual improves action matching or closed loop.

## 2. Why the V52 anchor gate failed

The only hard failure was:

```text
teacher_regret = 12761.731177 > 12000
```

This metric belongs to the final budgeted runtime action. It is not the regret of the immutable anchor being approved for reuse.

A paired comparison of V51 and V52 replay for the same checkpoint and the same 1000 scenario/timestamp keys shows:

| Action/interface | Same rows | Changed rows |
|---|---:|---:|
| teacher action | 1000 | 0 |
| dense full-interface action | 1000 | 0 |
| local pair-full action | 1000 | 0 |
| direct pair-full action | 168 | 832 |
| final budgeted action | 167 | 833 |

The true immutable metrics are identical between the two runs:

| Metric | Value |
|---|---:|
| full-interface action match | 0.359 |
| base winner-rival sign | 0.671 |
| dense winner-rival sign | 0.803 |
| dense near-tie sign | 0.706 |
| dense all-pair sign | 0.718 |

Thus the changed `teacher_regret` comes from runtime-dependent direct-pair/selector behavior. Using it to reject the unchanged base+dense-local anchor is a gate-semantic error.

## 3. Did V52 improve the three previously identified shortcomings?

### 3.1 Pairwise sign is good but action match is low

**Not evaluated.** V52 never trained the residual tournament. The unchanged foundation still has dense winner-rival sign 0.803 and action match 0.359, which confirms the original gap but says nothing about V52's proposed repair.

A new issue was found in the V52 training configuration: `full_action` and `full_margin` had large weights, but their logits are computed only from frozen `base+local` outputs. Their gradients with respect to all trainable V52 residual/selector parameters are zero. The logged loss looked action-aware without actually teaching the residual tournament to select the teacher winner.

### 3.2 Selector does not lock decisive evidence

**Not learned or evaluated.** Runtime-only changes on the untrained checkpoint improved some selector diagnostics, for example selected decisive recall and AOCC certification, but final actions changed in more than 80% of rows. These values are configuration effects, not evidence of learned V52 improvement.

The valid conclusion is that the proposal pool already contains useful decisive atoms, while the learned mapping from boundary pairs to a stable B=16 winner still needs training and causal evaluation.

### 3.3 Computation is spent on irrelevant pairs

**The code path exists but was not executed.** V52's quota sampler and sparse exact cadence are structurally appropriate:

- ordinary steps use at most 64 pairs;
- winner-rival, hard-crossing and near-tie pairs have independent quotas;
- full graph and exact B=16 supervision appear periodically;
- B=8/B=24 are sparse robustness objectives.

No wall-clock speedup can be claimed from the uploaded run because BFAR training did not begin.

## 4. Is repeated gate failure caused by too many/high thresholds?

There are three distinct cases:

1. **Wrong metric/object**: the V52 anchor gate failed for this reason. Lowering the threshold would hide the bug rather than fix it.
2. **Correct safety/causal checks**: unique paired keys, independent calibration, exact B=16 target, fixed budget and no catastrophic regret regression must remain hard requirements.
3. **Paper-grade improvement checks**: action gain, strong decisive recall and strict regret non-regression should not prevent the first paired CL20. They should control CL100 and paper-result escalation.

V53 therefore introduces two tiers rather than globally weakening standards.

## 5. V53 algorithm

### 5.1 Correct immutable anchor gate

The gate now checks only the frozen base+dense-local interface and records the exact scenario-key fingerprint. Runtime-dependent direct-pair, selector and budgeted-action metrics are excluded.

### 5.2 Interface-specific regrets

Future replay summaries expose separate regrets for base, dense full interface, sparse full interface, direct pair-full and local pair-full actions. This prevents another interface attribution error.

### 5.3 Exact no-op residual start

The residual head's final layer is zero after warm start. At step zero:

\[
\Delta m_R(a,b)=0,
\]

so the candidate exactly reproduces the reusable anchor. The variance head starts conservatively. Any later action change is learned rather than caused by random head initialization.

### 5.4 Winner-consistent residual objectives

Let the frozen decision margin be

\[
m_F(a,b)=\Delta J_0(a,b)+\sum_i \Delta g_i(a,b),
\]

and the residual correction be \(r_\theta(a,b)\). V53 trains the full residual tournament

\[
m_{FR}(a,b)=m_F(a,b)+r_\theta(a,b)
\]

with three direct objectives:

1. teacher-action cross entropy on the all-evidence residual tournament;
2. a teacher-winner versus strongest-rival margin;
3. B=16 preservation of a teacher-correct pair-full winner.

This makes the gradient path match the paper statement: evidence matters because it preserves or correctly changes the action winner.

### 5.5 Retained boundary-focused evidence learning

V53 keeps:

- quota-constrained winner/hard-cross/near-tie pair curriculum;
- decisive evidence counterfactual targets;
- exact B=16 AOCC deployment targets;
- fixed-budget fill and certificate objectives;
- do-no-harm on far-correct pairs;
- correction emphasis on near-tie or anchor-wrong pairs;
- uncertainty-certified full-margin flip authorization.

The complete causal line remains:

```text
complete anchor decision boundary
-> flip-critical action pairs
-> decisive evidence coreset under B=16
-> residual correction with uncertainty certificate
-> action winner change/preservation
-> paired closed-loop effect
```

## 6. Training-speed changes

V53 does not replace the deployed exact selector with a surrogate. It reduces unnecessary supervision frequency:

- ordinary pair graph capped at 64 boundary-critical pairs;
- exact selector: one scene per rank every four steps;
- B=16 exact at every exact event;
- B=8/B=24 only every fourth exact event;
- cycle/transitivity mining every four steps;
- consistency triangles reduced from 64 to 48;
- final full-exact alignment tail reduced from 128 to 64 steps;
- frozen full-interface objective forward work skipped;
- batch size target 8 per GPU;
- six residual-training epochs with validation/early stopping.

Server logs, not theoretical call counts, must determine the actual end-to-end speedup.

## 7. Two target levels

### 7.1 Minimum submission completeness

This level is reached when all of the following exist:

- corrected immutable-anchor gate PASS;
- fresh V53 residual/selector checkpoint;
- candidate, same-checkpoint local control and matched foundation control on exactly paired 1000 open-loop samples;
- independent calibration for all three arms;
- minimum-completeness gate PASS;
- paired CL20 for all three arms;
- no new hard collision/TTC/drivable-area failure in candidate;
- at least one CL100 diagnostic run;
- B=8/B=16/B=24 curve;
- random, Top-K/score-only, greedy/exact, no-residual, no-certificate and full-information ablations;
- scenario-key provenance and confidence intervals.

Initial algorithm targets for allowing CL20:

| Metric | Minimum target |
|---|---:|
| candidate teacher match | no worse than best control by >0.01 |
| pair-full match | no worse than local anchor by >0.01 |
| harmful residual rate | <=0.05 |
| selected decisive recall | >=0.50 |
| effective decisive recall | >=0.62 |
| interaction decisive recall | >=0.40 |
| AOCC certified fraction | >=0.40 |
| fallback rate | <=0.60 |
| paired regret | no catastrophic median/p90 regression |

These are diagnostic viability targets, not paper-grade claims.

### 7.2 Competitive fixed-budget CCF-A target

There is no official numeric acceptance threshold. A credible fixed-budget claim should aim for:

| Metric | Competitive target |
|---|---:|
| candidate match gain vs foundation | >=+0.015 |
| residual gain vs local control | >=+0.005 |
| pair-full residual gain | >=+0.005 |
| harmful residual rate | <=0.03 and below beneficial rate |
| proposal decisive recall | >=0.80 |
| selected decisive recall | >=0.55, desirable >=0.62 |
| effective decisive recall | >=0.70, desirable >=0.78 |
| interaction decisive recall | >=0.50, desirable >=0.60 |
| AOCC certified fraction | >=0.55 |
| fallback rate | <=0.40, desirable <=0.25 |
| paired regret | median and p90 non-regression; desirable median improvement >=5% |

For closed loop, the first defensible paper objective is not unrestricted universal SOTA. It is:

> Under the same strict evidence budget B=16, the proposed method achieves the best paired reactive closed-loop result among equal-budget baselines while remaining close to the corresponding full-information planner.

Recommended outcome targets:

- Val14 reactive CLS at least around 90 for a competitive narrative;
- stronger target 91–92 under the exact same protocol;
- within roughly 1–2 CLS of the matched full-information planner;
- positive paired gain over random/score-only/greedy equal-budget baselines;
- Test14-Hard PDMScore around 0.80 or better when using that protocol;
- safety multipliers and collision/TTC components must not be traded for progress;
- confidence intervals or repeated runs must support the claimed gain.

These values are planning targets, not guaranteed acceptance lines, and metrics from different protocols must not be directly mixed.

## 8. What to inspect after the next run

### If the anchor gate fails

Do not tune V53. Compare the replay-key SHA-256 and interface-specific regrets first. A changed key set or foundation checkpoint is an engineering/provenance issue.

### If training is still slow

Inspect:

- `train_forward_ms_per_step`;
- `train_loss_ms_per_step`;
- `train_backward_step_ms_per_step`;
- `selector_exact_wall_time_s`;
- `train_pair_sample_ms_per_step`;
- `training_pair_fraction`;
- `selector_exact_fraction`.

The intended normal-stage signatures are a pair fraction well below one and exact selector fraction near the sparse schedule, followed by a higher final-tail exact fraction.

### If minimum gate fails

Use the failure category:

- action mismatch with good decisive recall: residual tournament/calibration problem;
- poor selected recall with good proposal recall: AOCC objective/selection problem;
- proposal and selected recall both poor: decisive-target construction or feature problem;
- beneficial < harmful: residual authorization or uncertainty calibration problem;
- regret tail failure despite match gain: cost-sensitive winner loss underweights rare high-regret scenes.

## 9. Run command

Use `NEXT_COMMANDS_V53_WC_BFAR.txt` or execute `V53_WC_BFAR_DBAP_NEXT_COMMANDS.sh` with the environment shown there.

## 10. Claim boundary

The code has been validated locally, but fresh V53 training and nuPlan simulation have not been run in this environment. Passing the corrected anchor gate is expected for the uploaded checkpoint; passing the candidate gate or improving closed loop is not assumed.
