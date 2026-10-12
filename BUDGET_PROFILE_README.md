# BDSE compute and deadline profiling

This profiler measures the frozen planner; it does not prove that a deadline is
required and does not simulate latency by sleeping. Establishing the deployment
premise requires measurements on the target hardware and an independently stated
stack deadline.

## Fixed inputs

- `bdse/configs/deployment.yaml` is the generic deployment configuration supplied
  with this package. It is architecture-compatible with the supplied SAQA/BCC
  checkpoint family, including the HCBE-trained checkpoint named in the user
  report.
- `EAF_CKPT` must point to the real checkpoint. Relative paths are resolved from
  the repository root.
- Output directories must be new, so traces from different runs cannot mix.
- `MODE=native` is the latency measurement. `MODE=attribution` inserts internal
  GPU synchronization and is only for stage attribution.

The previous suggestion to use `v64_3_47_rsmr.yaml` was incorrect: that file is
not present in the uploaded repository. The cleaned package no longer requires
any version-named YAML.

## Reproduce the reported failing path after the fix

From the repository root:

```bash
export EAF_CKPT=outputs/outputs_v64_2_saqa_bcc_hcbe_fast_2gpu_v1/train/bdse_v64_saqa_bcc.best.pt
export OWN_CONFIG=bdse/configs/deployment.yaml

LIMIT=10 WARMUP=2 BUDGETS="16" PROPOSALS="24" MODE=native \
  bash RUN_BUDGET_PROFILE.sh
```

Do not write `export EAF_CKPT=export EAF_CKPT=...`; the value must be only the
checkpoint path. `OWN_CONFIG` is optional because the runner now defaults to the
generic deployment config.

The semantic-attribution crash is fixed. Selected atom family/type metadata is
always available, but contribution rows exist only when decisive-frontier
attribution is active. An absent contribution matrix now produces a schema-stable
zero diagnostic with `decisive_frontier_icer_semantic_attribution_available=0`.
A nonempty contribution matrix still requires exact one-to-one metadata alignment
and fails loudly otherwise.

## Cached-runtime core profile

```bash
export EAF_CKPT=outputs/outputs_v64_2_saqa_bcc_hcbe_fast_2gpu_v1/train/bdse_v64_saqa_bcc.best.pt
LIMIT=1000 WARMUP=20 MODE=native PROFILE_GPU=0 \
  BUDGETS="8 16 24" PROPOSALS="24 24 24" \
  bash RUN_BUDGET_PROFILE.sh
```

For a descriptive per-call threshold, add `DEADLINE_MS=100`. The combined
summary always reports sensitivity at `DEADLINES_MS` (default `50 100 200`).

This scope begins with runtime fields read from NPZ, regenerates candidates and
evidence, and executes the frozen planning core. NPZ I/O is measured separately
and excluded from service time. It is useful for controlled scaling experiments,
but it is not the end-to-end planner service.

Use these sweeps independently:

```bash
# Fixed proposal pool, vary retained interface.
BUDGETS="8 16 24" PROPOSALS="24 24 24" bash RUN_BUDGET_PROFILE.sh

# Fixed retained interface, vary acquisition/proposal work.
BUDGETS="16 16 16" PROPOSALS="24 48 128" bash RUN_BUDGET_PROFILE.sh
```

If only `BUDGETS` changes while executed query counts and latency do not, the
retained-interface budget is not a compute-budget mechanism. Report actual
forward, sparse-query, rollout, and effective-query counts alongside latency.

## Actual nuPlan planner-call profile

NPZ caches cannot measure runtime feature construction or nuPlan output conversion.
For that scope provide raw DB access and an audited token manifest:

```bash
export EAF_CKPT=outputs/outputs_v64_2_saqa_bcc_hcbe_fast_2gpu_v1/train/bdse_v64_saqa_bcc.best.pt
export NUPLAN_DB_ROOT=/path/to/raw/nuplan/db
export NUPLAN_MAP_ROOT=/data0/senzeyu2/dataset/nuplan/maps
export SCENARIO_TOKENS_FILE=/path/to/audited_tokens.txt

PROFILE_GPU=0 BUDGET=16 PROPOSAL_TOP_M=24 MODE=native CL_LIMIT=100 \
  bash RUN_RUNTIME_BUDGET_PROFILE.sh
```

Run both the frozen replanning interval and `REPLAN_INTERVAL_TICKS=1`. The latter
is a frequency ablation, not a reproduction of prior closed-loop quality. Cached
frames and true replans are summarized separately.

## Minimum reporting standard

Report target GPU/CPU, software stack, batch size, worker count, thread settings,
warmup protocol, scene-density strata, successful calls, error calls, p50/p90/p95/
p99/max, peak memory, and all executed-work counters. Keep native and attribution
timing separate. A useful deadline is the stack allocation from input-ready to
trajectory-ready after reserving publication/control/monitor time; it is not
inferred from trajectory sample frequency.

The profiler validates only the deployment premise and cost scaling. It cannot
validate evidence quality or the scientific claim. Those require matched-cost
adaptive-acquisition, stopping, intervention, and closed-loop experiments.

## Tests

```bash
python -m unittest bdse.tests.test_semantic_attribution_contract -v
python -m unittest bdse.tests.test_planning_budget_unittest -v
bash -n RUN_BUDGET_PROFILE.sh
bash -n RUN_RUNTIME_BUDGET_PROFILE.sh
```

The full learned-core and nuPlan tests must be run in the target PyTorch/nuPlan
environment.
