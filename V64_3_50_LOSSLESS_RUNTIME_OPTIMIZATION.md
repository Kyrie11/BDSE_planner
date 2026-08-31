# V64.3.50 SIOR lossless runtime optimization

Date: 2026-08-30

## 1. Scope

This package does **not** propose V51 and does **not** change the V50 scientific mechanism. It is an engineering-only optimization of the native paired selected-outcome evidence collection path. The purpose is to reduce execution cost while keeping the V50 estimand and all evidence required to argue it unchanged.

The unchanged launcher remains:

```bash
bash RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh
```

Timing heartbeat/telemetry is OFF by default in this optimized package.

## 2. Paper/backbone interpretation used as the optimization constraint

The uploaded manuscript frames the method around a bounded auditable evidence interface, decision sufficiency rather than reconstruction fidelity, extremal selection/tail amplification, proposal freezing, and monotone no-fallback containment. The later V44--V49 experiments progressively move the unresolved question away from feature capacity and toward the selected proposal's absolute deployment outcome law.

The historical mechanism chain used for this patch is:

- V44: full-horizon ungated prospective interaction support succeeds; scene-global behavior classification is falsified.
- V45: agent-local continuous longitudinal response is identifiable and is a real value mediator; deterministic mean response remains incomplete.
- V46: response second moment is identifiable but not decision-sufficient; handcrafted temporal features improve ordinary prediction while worsening deployed catastrophes. This supplies direct evidence that prediction sufficiency is not decision sufficiency.
- V47: simple AGENT-2D drift is closed; EGO-REF is retained as a supporting consequence coordinate; representation expansion is stopped.
- V48: multiplicity-conditioned observational selected-risk fails double-fresh; threshold repair is ruled out.
- V49: hash-prefix offline selection intervention fails because it changes the identity/rank of the supervised action rather than observing the intervention outcome of the actual deployed full-set winner. The offline selected-risk family is closed.
- V50 SIOR therefore changes the **evidence source**: it acquires paired reactive closed-loop outcomes for the actual live full-set RSMR winner while freezing the selector and the Q/P/E consequence state.

This history makes three execution properties non-negotiable for any “lossless” optimization:

1. both CONTROL and TREATMENT native reactive simulations remain complete;
2. the planner replans and re-runs the frozen selector every tick;
3. TREATMENT executes exactly the first live full-set RSMR winner once, while CONTROL preserves incumbent, and all later direct proposals preserve incumbent.

No state fork, no skipped ticks, no early no-proposal termination, no dropping CONTROL, no non-reactive substitute, no rerank, no second-best/fallback, and no model/threshold/feature retuning is introduced.

## 3. What the uploaded timing shows

The final uploaded telemetry contains 668 CONTROL planner ticks and 678 TREATMENT ticks. Mean values near the end of the recorded run are:

| Stage | CONTROL mean | TREATMENT mean |
|---|---:|---:|
| total planner | 2.9616 s | 2.9193 s |
| core planner | 2.9524 s | 2.9097 s |
| candidate generation | 0.4013 s | 0.3947 s |
| candidate-aware agent resort | 0.2969 s | 0.2917 s |
| evidence enumeration | 0.2952 s | 0.2856 s |
| certificate stages | 1.9576 s | 1.9365 s |
| model `_make_batch` | 0.4606 s | 0.4625 s |
| model context encoding | 0.0146 s | 0.0132 s |
| model sparse action scoring | 0.2208 s | 0.2160 s |
| model pair scoring | 0.2222 s | 0.2171 s |
| V50 probe instrumentation | 0.000135 s | 0.000133 s |

Therefore the heartbeat/probe is not the dominant cost. The optimization targets repeated device-transfer overhead and a dead historical value-observable branch inside the certificate path. Timing output is disabled mainly to remove observation/I/O noise and make normal runs quieter, not because the heartbeat itself accounts for the multi-second planner latency.

## 4. Optimization A: packed runtime batch transfer

File: `bdse/model/bdse_model.py`

Historical `_make_batch` normalizes every runtime NumPy array to one of the frozen target dtypes and then calls `.to(device)` separately for each tensor. `runtime_to_model_numpy` normally emits 23 arrays. On CUDA this produces about 23 host-to-device transfer calls per planner tick.

The optimized path preserves the same normalization contract:

- bool -> `torch.bool`;
- integer -> `torch.int64`;
- all other numeric -> `torch.float32`.

Arrays are packed by target dtype into at most three contiguous host buffers, transferred once per dtype, then exposed to the model as non-overlapping tensor views with the legacy key order, shape, dtype and stride. Zero-sized arrays retain the legacy individual path to preserve their special stride metadata.

Switch:

```bash
V50_PACKED_RUNTIME_BATCH_TRANSFER=1   # default
```

Legacy replay:

```bash
V50_PACKED_RUNTIME_BATCH_TRANSFER=0
```

Regression test compares legacy vs packed tensors on mixed float/int/bool/scalar/empty inputs using exact `torch.equal` plus exact dtype/shape/stride/key-order checks. The local environment has no CUDA device, so this package does **not** claim a measured GPU speedup for this change; it targets the measured ~0.46 s/tick `model_make_batch` stage and reduces the normal transfer call count from ~23 to <=3.

## 5. Optimization B: do not evaluate scientifically closed, unconsumed V47 PLAN-2D at V50 runtime

Files:

- `bdse/planner/future_state_factorization.py`
- `bdse/planner/value_observables.py`

The V48/V49/V50 frozen post-selection state consumes:

- `Q`: the three QUALITY coordinates;
- `P`: `fsfr_plan_1d_occupancy_cost`;
- `E`: PLAN-1D plus `fsfr_predicted_demo_cost`.

The historical schema still includes `fsfr_plan_2d_occupancy_cost`, but V47's AGENT-2D branch was scientifically closed and the OCRR/SIIR/SIOR state never indexes this column. Computing it every tick therefore spends runtime on a mechanism that is neither selected nor used by V50.

The optimized V50-only path:

1. leaves `bdse/planner/tournament.py` **byte-identical** to the V48 science lock;
2. preserves the complete historical 12-column `post_selection_observable_names` schema seen by the tournament;
3. computes every consumed Q/P/E coordinate exactly as before;
4. computes the six historical runtime-risk columns exactly as before;
5. skips only the unused `fsfr_plan_2d_occupancy_cost` rollout and fills that non-consumed, non-persisted in-memory slot with zero;
6. fails closed if any future config attempts to use PLAN-2D as Q/P/E.

Why this does not remove V50 evidence: paired collection runs with `BDSE_SELECTED_OUTCOME_DIAG_ONLY=1`. The persisted per-tick evidence contains the selected-outcome probe: live proposal identity/fingerprint, incumbent/proposal slots, treatment state, and live Q/P/E values. The complete historical value-observable matrix is not persisted by the V50 collector. The dead PLAN-2D slot therefore cannot change the treatment assignment, RSMR selector, Q/P/E evidence, SIOR labels, official metrics, or any V50 GO/STOP gate.

Switch:

```bash
V50_ELIDE_UNUSED_FSFR_2D=1   # default
```

Legacy replay:

```bash
V50_ELIDE_UNUSED_FSFR_2D=0
```

Representative local CPU benchmark for this block (`K=64`, `N=32`, `T=81`):

- historical value-observable block median: `0.2826 s`;
- optimized block median: `0.1392 s`;
- block speedup: `2.03x`;
- every column except the intentionally dead PLAN-2D slot: `np.array_equal`;
- all V50-consumed Q/P/E columns: `np.array_equal`.

This is a synthetic block-level benchmark, **not** a claim of 2x end-to-end server speedup. The end-to-end gain must be measured on the real two-GPU run.

## 6. Optimization C: timing tick/heartbeat off by default

Files:

- `RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh`
- `bdse/tools/run_v64_3_50_paired_selected_outcome_collection.py`

Default:

```bash
V50_TIMING_TELEMETRY=0
```

With timing disabled:

- `BDSE_PROFILE_CLOSED_LOOP=0`;
- no `v50_timing` payload is added to every diagnostic row;
- parent process does not poll the children every 30 s;
- no parent `nvidia-smi` sampling;
- no `timing_telemetry.jsonl` heartbeat stream;
- strict `selected_outcome_probe` JSONL is still written every planner tick and remains fail-closed.

To temporarily reproduce the old timing view:

```bash
V50_TIMING_TELEMETRY=1 bash RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh
```

## 7. Explicitly unchanged V50 scientific contract

The optimized package keeps all of the following unchanged:

- `collection_protocol_version=v50-live-selected-event-cohort-v1`;
- frozen 502-scene discovery cohort / live selected-event semantics;
- native `closed_loop_reactive_agents` simulation;
- same initial nuPlan state for paired arms;
- exactly two concurrently launched arms on two GPUs;
- forced `BDSE_REPLAN_INTERVAL_TICKS=1` and `BDSE_FORCE_REPLAN_EVERY_TICK=1`;
- candidate generation, EAF, RSMR and full-set live winner;
- live proposal trajectory/semantic fingerprint checks across paired arms;
- CONTROL always preserves incumbent at proposal events;
- TREATMENT executes the first live RSMR winner once, then preserves incumbent;
- no rerank / no second-best / no fallback;
- Q/P/E coordinate definitions and live event-state alignment;
- official nuPlan score and four hard safety metrics;
- no-live-proposal whole-trajectory equivalence checks;
- SIOR pairwise model, `lambda=1`, calibration budget, identification and deployment gates;
- V48 `tournament.py` science-lock bytes.

Only execution provenance changes:

```text
collection_engine_version=v50-batched-nuplan-lossless-opt-v4
```

## 8. Validation performed

- V48 OCRR science lock: `5/5 PASS`.
- `tournament.py` SHA256 remains `291b3b77202974b74fe42431ee7954de8c401d927591c19a12a5837f18374044`.
- V50 + V48.2 lock-focused check: `36/36 PASS`.
- V13->V50 targeted regression: `275/275 PASS`; two historical Transformer warnings.
- Full repository: 126 test files, four exhaustive shards:
  - shard 0: `158/158 PASS`;
  - shard 1: `152/152 PASS`;
  - shard 2: `161/161 PASS`;
  - shard 3: `141/141 PASS`;
  - total: `612/612 PASS`.
- `python -m compileall -q bdse`: PASS.
- `bash -n RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh`: PASS.
- packed-batch legacy equivalence: exact tensor value/dtype/shape/stride/key order.
- FSFR requested PLAN-1D/EGO-REF values: exact equality with historical full FSFR calculation.
- full vs PLAN-2D-elided OCRR certificate/features: bit-exact for the frozen consumed state.

A single monolithic `pytest -q bdse/tests` invocation exceeded the local execution window after passing >80% of the suite, so the final full-repository result above is from the same 126 test files split into four mutually exclusive and exhaustive shards.

## 9. Resume behavior

The optimized engine deliberately refuses to mix committed paired rows with an older execution engine or different optimization flags.

If the old timed run has **not** produced a committed `paired_selected_outcomes.csv`, simply run the unchanged launcher in the optimized directory; stale per-batch arm directories are deleted and rebuilt.

If the old run has already committed rows, delete **only** the V50 paired collection directory before rerunning:

```bash
rm -rf outputs_v64_3_50_eaf_icer_sior_screen_2gpu_v1/paired_train
bash RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh
```

Do not delete or rerun frozen V49 outputs for this execution-only change.

## 10. Recommended server run

Normal optimized run, timing tick off:

```bash
cd bdse_v64_3_50_eaf_icer_sior_lossless_opt
bash RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh
```

Optional one-batch profiling run if end-to-end timing needs to be re-measured:

```bash
V50_TIMING_TELEMETRY=1 bash RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh
```

For a strict legacy execution comparison under the same V50 scientific configuration:

```bash
V50_PACKED_RUNTIME_BATCH_TRANSFER=0 \
V50_ELIDE_UNUSED_FSFR_2D=0 \
V50_TIMING_TELEMETRY=0 \
bash RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh
```

Do not compare or pool scientific paired rows across these execution engines; use the engine-lock/resume policy already encoded in the collector.
