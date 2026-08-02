# V55 result diagnosis and V56 DCIP-BFAR-DBAP

## 1. What the uploaded run actually is

The uploaded output is V55 PC-BFAR, although the request calls it V54. The run completed:

- foundation/anchor validation;
- four epochs of residual/selector training;
- independent candidate, local-control, and foundation-control calibration;
- three paired 1000-scene open-loop replays;
- protocol/minimum/competitive gates;
- three paired diagnostic CL20 runs.

CL100 and official Val14/Test14 evaluation were not run.

## 2. Gate result

- protocol gate: **PASS**;
- minimum-completeness gate: **FAIL**;
- competitive gate: **FAIL**.

### Minimum failure

Only two conditions failed:

- certified fraction: `0.2040 < 0.40`;
- fallback rate: `0.801 > 0.60`.

The same-checkpoint local control, with residual uncertainty disabled, had:

- AOCC certified pair fraction: `0.8880`;
- fully certified scene rate: `0.805`;
- fallback rate: `0.11`.

Candidate and local deployed actions were identical on all 1000 replay rows. Therefore V55's minimum failure is predominantly an engineering/definition problem: residual uncertainty was injected into the same certificate used to judge evidence sufficiency, even though the residual did not change the deployed action.

### Competitive failure

The competitive failure is real:

- candidate/local/foundation teacher match: `0.141/0.141/0.141`;
- pair-full/local-pair-full match: `0.141/0.141`;
- total teacher-match gain: `0`;
- residual gain: `0`;
- beneficial/harmful deployed residual rate: `0/0`;
- paired regret deltas: exactly `0`.

Selector recall was close to the competitive thresholds, but no selected evidence was converted into a better winner.

## 3. Why CL20 was slow

The three CL20 branches consumed approximately:

- candidate: `22,858 s` (`6.35 h`);
- local control: `15,924 s` (`4.42 h`);
- foundation control: `14,914 s` (`4.14 h`);
- sequential three-way total: `53,696 s` (`14.92 h`).

Every 10-scenario shard constructed ten independent `BDSEnuPlanPlanner` objects, and every object loaded its own CUDA model. Thus each GPU shard held ten identical model copies and let them contend for the same device. Open-loop profiling also shows prediction dominates one planner call:

- prediction: about `685 ms`;
- selector: about `89 ms`;
- tournament: about `6 ms`;
- planner p95: about `1.25 s`.

The principal engineering fix is model reuse, not further tournament micro-optimization.

## 4. V55 closed-loop signal

Candidate diagnostic CL20:

- score: `0.0917`;
- no-at-fault-collision: `0.35`;
- TTC: `0.30`;
- drivable-area compliance: `0.85`;
- progress: `0.45`.

Local and foundation controls were identical:

- score: `0.0460`;
- collision: `0.30`;
- TTC: `0.25`;
- drivable-area: `0.80`;
- progress: `0.50`.

This is a weak positive diagnostic signal, but it is not causal evidence for the residual module because the 1000-scene paired open-loop replay found zero candidate-local deployed action differences. With only 20 scenarios and very low absolute scores, it must not be presented as a performance result.

## 5. Status of the three core problems

### 5.1 Pairwise quality does not produce the correct global winner

Not solved. The dense local interface still has strong pair signs, while the deployed B=16 action match is only `0.141`. V55's Hodge projection guarantees integrability after projection, but it begins from an arbitrary learned pair field. A single scene-level teacher-potential target is underdetermined: many per-edge or per-evidence assignments can yield the same global correction, so the model need not learn which evidence changes the winner.

### 5.2 Selector does not retain decisive evidence

Partially solved. Proposal/selected/effective decisive recall reached approximately `0.798/0.538/0.704`; local-control selected/effective recall was `0.607/0.773`. Exact AOCC and fixed B=16 are functioning. The main remaining problem is downstream target alignment, not search itself.

### 5.3 Training spends compute on irrelevant pairs

Solved enough to retain. Four V55 epochs took about `18.9–21.7 minutes`, with pair fraction about `0.589` and exact coverage `0.0156–0.0257`. Boundary-pair sampling and sparse exact distillation should remain.

## 6. Effective and ineffective designs

### Effective and retained

- factorized base+dense-local anchor;
- winner/hard/near-tie pair curriculum;
- exact B=16 AOCC and decisive-evidence proposal targets;
- sparse exact supervision and short full-exact tail;
- independent calibration and three-way paired replay;
- fixed-budget accounting and 16/16 fill;
- diagnostic CL20 on open-loop gate failure;
- finite-value and interface-drift checks.

### Ineffective or incorrectly coupled

- one mixed evidence-plus-residual certificate;
- Hodge projection of an arbitrary residual pair field as the deployed correction interface;
- scene-level potential distillation without per-evidence identifiability;
- loading one CUDA model per simulation planner;
- the combined CL summary's weighted `num_scenarios=10` despite 20 actual scenarios.

## 7. V56 algorithm

### 7.1 Direct evidence-attributable potential

For each queried atom `i` and action `a`, the model predicts a normalized residual action potential `h_i(a)`. For selected set `S_B`:

`J_B^DCIP(a) = J0(a) + sum_{i in S_B} g_i(a) + s * sum_{i in S_B} h_i(a)`.

Every pair correction is a difference of one global action potential, so antisymmetry and cycle consistency are exact by construction. No post-hoc graph projection is needed.

### 7.2 Atomwise causal-potential distillation

The cache contains teacher per-atom action costs. V56 directly supervises:

`h_i^T(a) = [g_i^teacher(a) - g_i^local(a)] / s`.

Prediction and target are gauge-centered over valid actions. The loss upweights:

- the teacher winner;
- the selected-local action when it is wrong;
- interaction evidence;
- scenes where the selected-local anchor is wrong;
- large teacher-minus-local corrections.

A lower-weight scene-level potential loss remains as a consistency objective.

### 7.3 Dual certificate

- evidence certificate: exact AOCC is evaluated only on selected-local evidence margins;
- residual certificate: a separate global guard evaluates whether the uncertainty-shrunk residual potential can safely replace the selected-local winner.

Residual uncertainty can no longer make a good evidence set appear uncertified.

### 7.4 Exact no-op and causal controls

At zero residual potential, the candidate action is exactly the direct selected-local argmin. Local/foundation controls disable residual mean and variance. Candidate-local action differences can therefore be attributed to the residual-potential module.

## 8. Closed-loop acceleration

V56 implements:

1. one shared eval CUDA model per checkpoint/model/device in each process;
2. construction protected by a cache lock, preventing concurrent duplicate allocations;
3. one per-device inference lock so four CPU simulation workers can overlap simulation work without concurrent model mutation/contention;
4. `CL_WORKERS_PER_GPU=4` by default;
5. BLAS/OpenMP threads limited to one per worker;
6. expensive summary PDF/histogram rendering disabled by default;
7. per-stage closed-loop timing JSON;
8. corrected combined scenario count and token-hash protocol report.

The actual speedup must be measured on the server. The expected benefit is large because V55 created ten CUDA copies per shard, but no fixed speedup factor is claimed.

## 9. Dataset/test-set assessment

The partial test diagnostics contain `67,042` unique samples and no internal duplicate identities. Candidate/evidence/teacher/preprocess settings match validation. It is substantially harder than validation:

| Metric | Validation | Partial test |
|---|---:|---:|
| full-interface action match | 0.9657 | 0.9343 |
| B=16 oracle decision sufficiency | 0.9120 | 0.8399 |
| runtime decision sufficiency | 0.7490 | 0.6407 |
| safe-candidate-exists | 0.7173 | 0.5779 |
| teacher ADE p90 | 12.92 | 17.14 |
| route-distance p90 | 3.26 | 17.55 |
| quality keep rate | 0.9175 | 0.6791 |

It can be used as a **frozen preliminary stress test**, provided:

- training, hyperparameters, checkpoint selection, calibration, and gates are frozen first;
- it is evaluated once and never used for model selection;
- checkpoint/config hashes and split identity are recorded.

It is not yet a final paper test set because the archive contains diagnostics but no train/val/test log manifest, no cross-split overlap audit, no failed-preprocessing manifest, and only about 28% of the intended cache.

## 10. Experimental targets

### Minimum completeness

- protocol PASS;
- minimum gate PASS under dual certificates;
- three-way paired CL20 and CL100;
- no safety regression against both controls;
- B=8/16/24 closed-loop curve;
- random, score-only, greedy, no-residual, and full-budget baselines;
- paired bootstrap confidence intervals;
- official Val14 plus a frozen held-out split.

### Competitive fixed-budget target

The defensible claim is not unrestricted planner SOTA. It is:

> under the same strict evidence budget B=16, the method gives the best paired closed-loop result among equal-budget methods and approaches its matched full-information planner.

A competitive result should show a statistically significant equal-budget gain, safety non-regression, and a gap to the matched full-information planner of roughly 1–2 score points on the same protocol. Official benchmark values must be reported separately from the custom CL20/CL100 diagnostic split.

## 11. Next run

Use `NEXT_COMMANDS_V56_DCIP_BFAR.txt`. The main diagnostics to inspect are:

- `L_residual_action_atom` decreasing across epochs;
- candidate-local deployed flip rate becoming nonzero but small;
- beneficial certified flips exceeding harmful flips;
- evidence certificate staying near the local-control certificate;
- residual flip certificate rate reported separately;
- candidate action match and paired regret improving over local/foundation;
- shared-model reuse messages and closed-loop profile JSON;
- three-way CL20 token protocol PASS.

## Closed-loop protocol clarification

The uploaded V55 CL20 was run with `closed_loop_nonreactive_agents`.  Its score is useful as a diagnostic but is not a reactive nuPlan result.  V56 exposes `CL_CHALLENGE` (`closed_loop_nonreactive_agents` or `closed_loop_reactive_agents`) and automatically selects the corresponding metric aggregator.  Reactive evaluation must use a separate `OUT_ROOT`, the same frozen checkpoint/calibration, and the same three-way token manifest so results from the two protocols cannot be confused.

## Exact-selector training alignment audit

Before packaging, an additional audit found a critical integration issue: V56 disables the legacy pair head, but the exact-selector training path still expected `pair_atom_delta`.  This would have removed periodic exact AOCC supervision from the direct evidence-potential model.  The released code derives certificate deltas as `g_i(b)-g_i(a)` and sets the exact full-TopM target to the selected-local anchor action, exactly matching the runtime dual-certificate path.  Residual potential is not allowed to contaminate this evidence target.  The full test suite now includes a no-legacy-pair-head exact-selector regression test.

A standalone `RUN_V56_REACTIVE_CL20.sh` is included for the publication-stage reactive protocol.  It requires a completed frozen `SOURCE_OUT_ROOT`, reuses the candidate/local/foundation calibrated configs, writes to a different root, and checks the three-way token hash.  The default pipeline CL20 remains non-reactive for faster diagnostics.
