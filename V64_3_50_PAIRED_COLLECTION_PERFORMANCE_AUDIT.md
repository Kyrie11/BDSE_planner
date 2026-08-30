# V64.3.50 SIOR paired-collection performance audit and execution-engine repair

## Scope

This change is **performance-only**.  It does not modify the V50 scientific
protocol `v50-live-selected-event-cohort-v1`, the frozen 502-scene discovery
cohort, RSMR/candidate/tournament logic, live Q/P/E coordinates, CONTROL or
TREATMENT assignment, the one-shot intervention, official nuPlan metrics,
`safe_benefit` labels, SIOR model/loss/calibration, or any TRAIN/fresh gate.

The slow component is evidence acquisition, not SIOR fitting.  The previous
collector invoked nuPlan separately for every scenario and arm.  With 502
frozen discovery scenes this means 1004 fresh Python/Hydra/nuPlan processes,
arranged as 502 sequential pair waves (CONTROL and TREATMENT concurrent inside
each wave).

## Why the old collector is slow

For every single-scenario child process the old collector repeats all of the
following even though the scientific model/checkpoint/config are unchanged:

1. Python + Hydra composition and nuPlan runner startup.
2. ScenarioBuilder construction and native DB discovery/filter setup.
3. planner construction and checkpoint/model loading on the GPU.
4. map/metric callback/aggregator initialization.
5. one full reactive closed-loop simulation.
6. full per-tick diagnostic serialization.
7. process teardown.

`BDSE_SHARE_MODEL_PER_PROCESS=1` cannot materially amortize model loading when
there is only one scenario in each process.  Passing all four TRAIN-city DB
directories to every one of ~1004 invocations similarly repeats irrelevant DB
discovery.

The irreducible part is the actual paired reactive closed-loop rollout.  V50
must still execute CONTROL and TREATMENT separately because their trajectories
can diverge after the first live RSMR intervention.  It must also replan every
tick because the treatment opportunity is the first *live* RSMR proposal event.

## Safe optimization 1: deterministic batched nuPlan invocation

The collector now defaults to `V50_BATCH_SIZE=16`.

A batch of up to 16 frozen tokens is passed to a single nuPlan process for the
CONTROL arm and to one separate process for the TREATMENT arm.  The two arm
processes continue to run concurrently on GPU0/GPU1.  Within each process,
`worker.max_workers=1` keeps scenario execution sequential and deterministic.

The number of Python/Hydra/nuPlan process launches therefore changes from:

    502 scenes x 2 arms = 1004 processes

to approximately:

    ceil(502 / 16) x 2 = 64 processes.

This makes the existing process-global read-only model cache useful: planners in
one batch reuse one loaded checkpoint/model on that GPU.

This does **not** aggregate scientific samples.  nuPlan's metric aggregator
already produces an individual per-scenario score.  `NuPlanScenario.scenario_name`
is the initial lidar token, so the collector joins official metric rows back to
the exact requested token.  The paired validator is then run independently for
every token exactly as before.

## Safe optimization 2: one outcome-blind token -> native DB index

At collector startup, the parent process scans the configured original nuPlan
SQLite files once and maps each of the 502 frozen lidar tokens to its exact DB
file using only `lidar_pc.token`.

For each batch, ScenarioBuilder receives only the DB files containing that
batch's tokens instead of all four city split directories.  The index uses no
teacher value, paired score, hard metric, SIOR label, or model output.  If the
index cannot be built exactly, the collector fails back to the original DB
input directories rather than changing the scenario population.

The mapping is cached under `paired_train/token_db_index.json` with a signature
of the frozen token set and DB path/size/mtime provenance.

## Safe optimization 3: V50-only minimal per-tick diagnostics

The paired collector consumes only `diagnostics.selected_outcome_probe`.
Previously the planner serialized the complete selector/tournament/runtime
safety diagnostic dictionary to JSON on every closed-loop tick.  During this
collector only, `BDSE_SELECTED_OUTCOME_DIAG_ONLY=1` writes the selected-outcome
certificate plus scenario token/time/action metadata.

This does not change any planner computation or decision.  It only avoids
serializing unused diagnostic payloads.

`BDSE_REQUIRE_SCENARIO_FOR_DIAG=1` makes nuPlan pass the native scenario into the
planner so batched diagnostic rows carry an exact `scenario_token`.  The
collector rejects missing/mixed token tags.

## Execution-engine provenance

The scientific collection protocol remains:

    v50-live-selected-event-cohort-v1

The execution implementation is separately identified as:

    v50-batched-nuplan-v1

Every committed row records both the engine and the requested batch size.
Resume refuses legacy single-scenario rows, a different engine, or a different
requested batch size.  Therefore do not mix the first seven single-scenario rows
with the batched collector.  Remove **only** V50 `paired_train` before the first
batched run; frozen V49 outputs are unchanged.

## What is intentionally NOT optimized

The following apparent speedups are forbidden for the current V50 validation:

- increasing the replanning interval / disabling `BDSE_FORCE_REPLAN_EVERY_TICK`;
- replacing native reactive closed-loop with cached/offline rollouts;
- running only one arm and synthesizing the other arm's official metrics;
- skipping official nuPlan metrics or the four hard-safety metrics;
- treating `no_live_proposal` as a negative label or early-exiting before the
  whole-run no-treatment equivalence certificate;
- changing the selector, candidate bank, challenge, treatment duration, or
  intervention rule;
- sharing a mutable simulator state between CONTROL and TREATMENT without a
  separately validated nuPlan state-cloning protocol.

A future simulator-state fork at the first live proposal could theoretically
avoid the duplicated CONTROL/TREATMENT prefix and provide another large speedup,
but it is deliberately not introduced in V50 because correct cloning of the
reactive observation/controller/metric state is a new scientific-engineering
contract and is much higher risk than process batching.

## Expected speedup

Let `S` be per-process startup/model/DB initialization time and `T` the actual
per-scenario reactive simulation time.  The old pair wall time is approximately
`S + T` because the two arms run concurrently.  With batch size `B`, amortized
wall time is approximately `T + S/B`, so:

    speedup ~= (S + T) / (T + S/B).

For B=16 this is about 1.9x when startup and simulation cost are equal, 3.4x when
startup is three times one scenario simulation, and 4.6x when startup is five
times the simulation.  When startup dominates, the theoretical limit approaches
16x.  Exact server speedup must be measured from the new batch wall-time logs;
no empirical speedup is claimed before that run.

## What this V50 collection validates

This stage is not training a new planner and is not yet a V50 success claim.  It
constructs the evidence needed to test the V50 SIOR hypothesis.

For each V49-frozen discovery scene, native replay first measures whether the
frozen live RSMR policy actually exposes a treatment opportunity.  Symmetric
`no_live_proposal` scenes remain label-free offline->live selection-transport
observations.  When a synchronized live proposal exists, CONTROL keeps the
incumbent while TREATMENT executes that exact live RSMR winner once and then
returns to incumbent.  The paired official-score delta and four hard-safety
metrics define whether the selected intervention is a `safe_benefit`, and live
Q/P/E are recorded at the same pre-intervention state.

Only after this evidence collection completes does `fit_v64_3_50_eaf_icer_sior`
ask whether a risk law learned from real paired selected outcomes is identifiable
and deployment-useful under the already-frozen gates.  Thus this expensive step
is the empirical bridge from V49's failed **offline selected-risk inference** to
V50's proposed **native selected-outcome interventional supervision**.
