# A30 双卡 external baseline 训练 / 测试优化说明

## 1. 固定的数据与闭环 DB 约定

- Train NPZ: `/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/{train_boston,train_pittsburgh,train_singapore,train_vegas_2}`
- Validation NPZ: `/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/val`
- Test NPZ / paired scenario manifest: `/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2/public_set_test`
- Raw nuPlan test DB (flat directory, direct `*.db` children):
  `/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/test`

`public_set_test` remains the source of the exact ordered `scenario_token` manifest. nuPlan closed-loop simulation reads the raw SQLite DBs. The runner now supports the flat DB directory directly.

The runner also performs a conservative DB-narrowing optimization: it tries to match each selected NPZ's parent folder (acquisition/log name) to a direct `<name>.db` file. It passes only the matched DB files **only when every selected token has a valid match**. On any mismatch, it automatically falls back to the whole raw test DB root, so no requested scenario can be silently lost.

## 2. Source-level bottleneck audit and optimizations

### Shared training data path

**Old bottleneck:** every epoch the generic BDSE tensorizer constructed tensors that the external adapters never read: dense evidence-query placeholders, full padded map/route tensors, full candidate trajectories on GPU, full teacher `g_evid` supervision tensors, extra family/decisive labels and pair tensors. The NPZ loader also deserialized logged-agent future and metadata that are irrelevant to expert-imitation training.

**Optimization:** `bdse/external_baselines/data.py` implements a lean loader/tensorizer specifically for external adapters. It preserves the runtime/candidate/evidence semantics and expert-imitation/proposal targets, but transfers only pooled scene summaries, candidate numeric descriptors, evidence descriptors and the necessary targets.

A local synthetic microbenchmark using the repository's current default external dimensions measured:

- generic input tensor bytes/sample: ~0.675 MiB
- compact external tensor bytes/sample: ~0.028 MiB
- tensor bytes reduction: ~24.4x
- generic CPU tensorization: ~16.6 ms/sample
- compact CPU tensorization: ~2.36 ms/sample
- local tensorization speedup: ~7.0x

These are local microbenchmarks, not promised A30 end-to-end speedups. Actual wall time depends on CPU, storage and GPU utilization.

### GameFormer-inspired adapter

**Primary training bottleneck:** 3 reasoning levels; each level performs candidate↔selected-evidence cross-attention and another full scene Transformer pass. It is GPU-compute dominated after the data-path optimization.

**Changes:** AMP, TF32, PyTorch scaled-dot-product/flash/memory-efficient attention enablement when supported, fused AdamW, and optional `torch.compile(mode=reduce-overhead)`. Effective optimizer batch remains 32 to avoid silently changing this reproduction's optimization protocol.

**Closed-loop bottleneck:** batch-1 neural inference is small compared with nuPlan simulation/metrics/DB work. Compact runtime tensorization removes unnecessary full map/candidate H2D traffic; simulation parallelism is increased at the worker level.

### DTPP-inspired adapter

**Primary training bottleneck:** tree depth 2; every depth does cross-attention + scene Transformer + branch cost head. Like GameFormer, GPU compute becomes dominant once CPU tensorization is reduced.

**Changes:** same AMP/TF32/SDPA/fused-optimizer/compile path. Effective optimizer batch remains 32.

**Closed-loop:** same compact inference path and nuPlan worker parallelism as GameFormer.

### PlanTF-inspired adapter

**Primary training bottleneck before optimization:** the model is lighter than GameFormer/DTPP, so repeated NPZ parsing/tensorization and 32x4 gradient accumulation were a larger fraction of wall time.

**Changes:** lean tensorizer plus A30 default `batch=128, accumulation=1`, preserving effective optimizer batch 128 from the prior 32x4 setup. If memory is unexpectedly tight, use `PLAN_BATCH_SIZE=64 PLAN_GRAD_ACCUM=2` or the conservative `32/4`. No BatchNorm is used; the optimization keeps the intended effective batch size.

### PLUTO-inspired adapter

**Primary bottleneck:** similar to PlanTF, with additional longitudinal/lateral auxiliary heads. Data loading and micro-batch overhead were significant relative to the ~5M parameter model.

**Changes:** same compact data path and default 128x1 effective-batch-preserving schedule. AMP/TF32/SDPA/compile are enabled as above.

### PDM-Closed-style scorer

**Old bottleneck/buglet:** it is deterministic NumPy/rule scoring at runtime, but the old class still allocated an unused ~4.72M-parameter Transformer stack.

**Optimization:** PDM-style now constructs zero trainable neural parameters and bypasses GPU model allocation entirely. Open-loop runs it on CPU. Closed-loop runner also forces this task to CPU even if the suite's neural systems use CUDA.

PDM-style remains a repository adapter, not the official PDM-Closed implementation.

### Open-loop evaluation

For external adapters, `predict_dense_numpy` simply repeated the same certificate model forward while `g=0`. The evaluator now automatically disables that redundant dense diagnostic for external baselines, avoiding a second neural forward per sample.

### Closed-loop nuPlan simulation

**Old bottlenecks:** one model-budget job at a time per task, `worker.max_workers=1`, full DB-root initialization, and unnecessary GPU allocation for PDM-style.

**Changes:**

- `worker.max_workers` is configurable; wrappers default to `CL_WORKERS_PER_JOB=4`.
- `model_pairs` scheduler pins one model to one GPU for B=8→16→24, with two models running concurrently.
- External-only order: GPU0 GameFormer vs GPU1 DTPP; then GPU0 PlanTF vs GPU1 PLUTO; PDM-style is the unpaired CPU-only remainder.
- Safe DB restriction uses only the required flat raw DB files when the NPZ parent-folder mapping is complete; otherwise whole-root fallback.
- Per-task wall time and scenarios/hour remain recorded.

For latency numbers intended for a paper, do not compare concurrent wall-clock latency directly against old serial runs: concurrency intentionally creates CPU/DB contention. Closed-loop planning metrics remain the target of the parallel run; if you need clean latency characterization, rerun the selected model serially.

## 3. Recommended commands

Run from the repository root.

### External baselines: train B=8/16/24 on two A30s

```bash
GPUS=0,1 bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

Scheduling is model-pinned:

- phase 1: GPU0 `gameformer` B8→B16→B24; GPU1 `dtpp` B8→B16→B24
- phase 2: GPU0 `plantf` B8→B16→B24; GPU1 `pluto` B8→B16→B24

Useful knobs:

```bash
# conservative PlanTF/PLUTO memory setting
PLAN_BATCH_SIZE=64 PLAN_GRAD_ACCUM=2 GPUS=0,1 \
  bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh

# disable torch.compile if your PyTorch/driver stack has an Inductor issue
TORCH_COMPILE=0 GPUS=0,1 \
  bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh

# tune I/O concurrency when the cache is on slower storage
NUM_WORKERS_PER_JOB=4 GPUS=0,1 \
  bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

The script preserves full train and full validation behavior of the uploaded reproduction unless you explicitly change its environment/CLI.

### External baselines: open-loop test

```bash
GPUS=0,1 bash RUN_FAIR_EXTERNAL_OPEN_LOOP_TEST_B8_B16_B24_2GPU.sh
```

It uses `bdse_test_2/public_set_test`, not validation data.

### External baselines: closed-loop test on the supplied flat DB directory

The raw DB root is now the wrapper default, so this is sufficient:

```bash
GPUS=0,1 CL_WORKERS_PER_JOB=4 \
  bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
```

Explicit equivalent:

```bash
NUPLAN_TEST_DB_ROOT=/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/test \
GPUS=0,1 CL_WORKERS_PER_JOB=4 \
  bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
```

Run one budget only:

```bash
BUDGETS=8  GPUS=0,1 CL_WORKERS_PER_JOB=4 bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
BUDGETS=16 GPUS=0,1 CL_WORKERS_PER_JOB=4 bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
BUDGETS=24 GPUS=0,1 CL_WORKERS_PER_JOB=4 bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
```

### Your V64.3.48 model: closed loop B=8/16/24

This wrapper resolves the promoted V48 config/checkpoint from provenance as before. The raw DB root now defaults to the supplied flat test directory.

```bash
GPUS=0,1 CL_WORKERS_PER_JOB=4 \
  bash RUN_V64_3_48_OWN_FIXED_BUDGET_CLOSED_LOOP_TEST_2GPU.sh
```

There is only one own model, so `model_pairs` intentionally keeps its B8→B16→B24 sequence on one fixed GPU rather than running different budgets concurrently. This avoids mixing model identity/budget scheduling when producing paired results.

### Combined own + external suite

If `OWN_CONFIG` and `OWN_CKPT` are already resolved:

```bash
OWN_CONFIG=/path/to/resolved_v48.yaml \
OWN_CKPT=/path/to/frozen_v48.pt \
GPUS=0,1 CL_WORKERS_PER_JOB=4 \
  bash RUN_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
```

## 4. DB / map path separation

The test DB path is independent of the map root. Defaults are:

```bash
NUPLAN_ROOT=/data0/senzeyu2/dataset/nuplan
NUPLAN_MAP_ROOT=$NUPLAN_ROOT/maps
NUPLAN_EXP_ROOT=$NUPLAN_ROOT/exp
NUPLAN_TEST_DB_ROOT=/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/test
```

If your maps/exp are also under CapPlan, override `NUPLAN_MAP_ROOT` and/or `NUPLAN_EXP_ROOT`; no code change is required.

## 5. Validation performed in this environment

- external baseline focused tests: 8 passed
- budget/closed-loop focused tests: 20 passed
- total focused regression: 28 passed
- flat direct-DB-root resolver smoke test: passed
- all-or-nothing DB restriction smoke test: passed
- Python compilation for modified modules: passed
- shell syntax checks for all five train/open-loop/closed-loop wrappers: passed

Actual `/data0/...` data, two A30 GPUs and nuPlan DB execution are not accessible in this environment, so real end-to-end speedup and closed-loop metrics must be measured on your server.
