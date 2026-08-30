# V64.3.50 Timing Telemetry and Science-Preserving Performance Repair

## Scope

This is an execution/observability repair only. The V50 scientific protocol remains `v50-live-selected-event-cohort-v1`.

Unchanged:

- native `closed_loop_reactive_agents` simulation;
- separate CONTROL and TREATMENT rollouts;
- every-tick replanning and every-tick selector decision;
- frozen RSMR/candidate/EAF/Q/P/E definitions;
- live-selection eligibility and no-live-proposal stratum;
- one-shot treatment semantics;
- official nuPlan aggregate score and four hard metrics;
- SIOR state, loss, calibration, and GO/STOP gates.

Only execution provenance changes from `v50-batched-nuplan-v1` to `v50-batched-nuplan-timing-v2`.

## Why the previous batch looked frozen

The previous batched collector printed before starting a 16-scenario batch and then blocked until BOTH CONTROL and TREATMENT child processes completed all 16 scenarios. Therefore the terminal could remain silent for a long period even while native simulation was progressing normally.

After:

```text
[V50 batch 1/32] ...
[V50 batch DB restriction] exact_db_files=13
```

the silence could have meant any of:

1. Python/Hydra/ScenarioBuilder startup;
2. DB/scenario construction;
3. CUDA checkpoint/model initialization;
4. normal scenario 1..16 reactive simulation;
5. expensive per-tick BDSE planning;
6. post-simulation official metric aggregation.

The old parent log could not distinguish these phases.

## New heartbeat

`V50_HEARTBEAT_SECONDS` defaults to 30 seconds. A heartbeat is emitted independently for CONTROL and TREATMENT, for example:

```text
[V50 heartbeat] role=control elapsed=90s pid=... scenarios_seen=1 ticks=37
latest=03dac...@36 planner_total(last/mean)=0.842/0.791s
core=0.744s cand=0.031s evidence=0.046s cert=0.612s
model_ctx=0.084s model_pair=0.351s planner_ready=1
cpu=72.1s rss=4350MB gpu=86% gpu_mem=5210MB ...
```

Each arm also writes:

```text
paired_train/batches/<batch>/control/timing_telemetry.jsonl
paired_train/batches/<batch>/treatment/timing_telemetry.jsonl
```

The existing strict `probe_diag.jsonl` gains a compact `v50_timing` block per tick. Full selector/tournament diagnostics remain suppressed during this evidence collection.

The heartbeat additionally samples child CPU time/RSS through `/proc` and, when available, GPU utilization/memory through `nvidia-smi`. These are parent-side observations only.

## Timing fields

Important fields include:

- `timing.compute_planner_trajectory_total_s`
- `timing.runtime_from_planner_input_s`
- `timing.core_plan_s`
- `timing.to_nuplan_trajectory_s`
- `core.candidate_generation_s`
- `core.candidate_aware_agent_resort_s`
- `core.evidence_enumeration_s`
- `core.certificate_stages_s`
- `core.final_safety_flags_s`
- `core.v50_probe_instrumentation_s`
- `model.model_make_batch_s`
- `model.model_encode_context_s`
- `model.model_action_sparse_s`
- `model.model_pair_scoring_s`

The detailed model timings are host-wall timings around the existing operations, not a CUDA kernel profiler; they are intended to localize the dominant stage, not to claim exact GPU kernel time.

## How to interpret the next run

### A. Startup / DB / Hydra bottleneck

If heartbeats show:

- `planner_ready=0`
- `ticks=0`
- GPU utilization near zero

for a long time, the bottleneck is before the first planner tick: Hydra import/config composition, ScenarioBuilder, DB construction/filtering, map initialization, or model construction.

If this dominates, the next safe optimization candidate is DB-locality-aware batching or further process initialization reuse. It should be validated without changing the 502-token cohort or per-scene simulation semantics.

### B. Scenario initialization bottleneck

If `planner_ready>0` but `ticks=0` for a long time, planner/model construction completed, but nuPlan has not reached the first planning callback. Scenario initialization / observations / map / reactive simulation setup is the likely bottleneck.

### C. Per-tick planner bottleneck

If `ticks` and `latest_iteration` keep advancing, the process is alive. Compare timing components:

- large `cand` -> candidate generation;
- large `core.candidate_aware_agent_resort_s` -> candidate-aware agent reranking and possible second candidate generation;
- large `evidence` -> evidence atom enumeration;
- large `cert` -> selector/tournament/certificate stack;
- large `model_ctx` / `model_pair` -> neural context / pair scoring;
- `planner_total >> core` -> runtime feature extraction and/or nuPlan trajectory conversion.

The V49 runtime configuration has candidate-aware agent selection enabled. In the frozen implementation, this can generate a candidate bank, resort agents using candidates, then regenerate the candidate bank. This is a plausible expensive path, but it is part of the frozen V49 semantics and is NOT removed in this repair. It should only be optimized later via an exact-equivalence implementation change if telemetry proves it dominant.

### D. Simulator / metric bottleneck

If per-tick planner time is modest but scene wall time is much larger, native simulation/observation/metric work dominates.

If ticks stop after the final scenario while the child remains alive, metric computation/aggregation is likely dominant.

At child completion the collector prints:

```text
[V50 arm finalize] role=... simulation_wall=... metric_join=... metric_file=...
```

`metric_join` is only the parent-side read/join of nuPlan's existing per-scenario aggregator parquet; official metric computation itself occurs inside `simulation_wall`.

## Science-preserving micro-optimizations applied

### Persistent line-buffered diagnostic writer

Previously each planner tick reopened and closed `probe_diag.jsonl`. The current process reuses one line-buffered append handle. Completed lines remain immediately visible to the parent heartbeat. Diagnostic writing occurs after the planner action/trajectory is already computed and cannot affect decision semantics.

### Same-tick candidate identity memoization

V50 fingerprints pre-probe, proposal, and post-probe candidate trajectories for causal-pair consistency. These often refer to the same local candidate slot in one tick. The previous implementation re-quantized and re-hashed identical trajectories multiple times. The current implementation memoizes identity by local action slot within that tick only.

The candidate fingerprint is instrumentation only and never enters RSMR, Q/P/E, treatment assignment, metrics, or SIOR.

`core.v50_probe_instrumentation_s` explicitly measures this overhead.

## Deferred optimizations

The following are deliberately NOT applied before timing evidence because they have higher scientific risk:

- replan interval > 1 / skipping selector ticks;
- early termination of no-live-proposal scenes;
- dropping CONTROL;
- replacing CONTROL with offline output;
- dropping or pruning official metrics;
- reactive -> nonreactive closed-loop;
- multi-scenario concurrent inference on one GPU;
- reordering the frozen cohort by DB locality;
- changing batch size based on observed outcomes;
- shared-prefix CONTROL/TREATMENT simulator-state fork.

The last item could potentially save substantial work because both arms are identical before the first live proposal, but an exact state fork must include ego/reactive-agent state, history buffer, traffic-light state, controller/planner state, and metric state. It is not safe to introduce without an independent equivalence experiment.

## What to upload next

If the run is still slow, upload at minimum:

1. `outputs_v64_3_50_eaf_icer_sior_screen_2gpu_v1/logs/v64_3_50_paired_closed_loop_collection.out`
2. `paired_train/batches/batch_0000_*/control/timing_telemetry.jsonl`
3. `paired_train/batches/batch_0000_*/treatment/timing_telemetry.jsonl`
4. corresponding `control/run.log` and `treatment/run.log`
5. optionally corresponding `probe_diag.jsonl`

With those files, the next optimization can target the measured dominant phase without weakening the V49-to-V50 causal comparison.
