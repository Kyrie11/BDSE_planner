# External baseline progress + closed-loop GPU visibility fix

## Diagnosis

`RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh` is an **evaluation-only** command. It never trains GameFormer/DTPP/PlanTF/PLUTO. The expected trained checkpoints are:

- `outputs/external_fixed_budget/B8/{gameformer,dtpp,plantf,pluto}_budgeted.best.pt`
- `outputs/external_fixed_budget/B16/{gameformer,dtpp,plantf,pluto}_budgeted.best.pt`
- `outputs/external_fixed_budget/B24/{gameformer,dtpp,plantf,pluto}_budgeted.best.pt`

The old closed-loop runner had two observability problems:

1. It scanned the entire `bdse_test_2/public_set_test` NPZ cache before checking whether those checkpoints existed. During that scan no CUDA model is loaded, so GPU memory can remain unchanged for a long time.
2. The nuPlan child process redirected all stdout/stderr to `run.log`, so DB/scenario initialization and planner loading were invisible in the terminal.

The CUDA routing itself was correct: each neural task receives `CUDA_VISIBLE_DEVICES=<physical GPU>` and `BDSE_DEVICE=cuda`, and `load_model_for_config()` calls `model.to(device)`. This revision keeps that route and adds a hard assertion that a CUDA closed-loop run cannot silently keep trainable model parameters on CPU.

## New progress output

### Training

The 2-GPU training wrapper now streams progress to the terminal **and** keeps the same `*.train.out` log via `tee`.

Default dual-GPU mode uses readable line progress:

```text
[train-init] variant=gameformer B=8 device=cuda:0 ...
[train-epoch-start] variant=gameformer B=8 epoch=1/30 batches=...
[train-progress] variant=gameformer B=8 epoch=1/30 step=100/... loss=... rate=...
[val-gameformer-B8-e1-progress] step=.../... loss=...
[train-epoch-done] ...
[train-complete] ... checkpoint=...
```

Actual tqdm bars remain available with `TRAIN_PROGRESS_STYLE=tqdm`. For two independent processes sharing one terminal, `lines` is usually easier to read.

### Closed loop

The closed-loop wrapper now reports:

- whether all required checkpoints exist **before** scanning the test cache;
- NPZ manifest indexing and token-scan percentage/rate;
- which system/B is assigned to which physical GPU;
- nuPlan child PID and per-task log path;
- a heartbeat every 15 s (configurable) with phase, elapsed time and physical GPU memory;
- the exact planner-ready CUDA line once the checkpoint is on-device;
- completion throughput and output directory;
- the last 20 nuPlan log lines immediately on failure.

Example planner confirmation:

```text
[task-gpu-ready] system=gameformer B=8 physical_gpu=1 [planner-ready] variant=gameformer logical_device=cuda:0 CUDA_VISIBLE_DEVICES=1 params=... param_devices=['cuda:0'] cuda_alloc=...MB ...
```

`logical_device=cuda:0` is expected when the outer process sets `CUDA_VISIBLE_DEVICES=1`: inside that process the one visible physical GPU 1 is renumbered to logical CUDA device 0.

## Commands

### 1. Train the 12 budget-specific neural checkpoints

```bash
cd <repo-root>

GPUS=0,1 \
TRAIN_PROGRESS_STYLE=lines \
LOG_EVERY_N_STEPS=50 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

If you specifically want tqdm:

```bash
GPUS=0,1 \
TRAIN_PROGRESS_STYLE=tqdm \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

### 2. Recommended 20-scenario closed-loop smoke test

Do this before a full test-set run:

```bash
GPUS=0,1 \
CL_LIMIT=20 \
CL_WORKERS_PER_JOB=4 \
CL_HEARTBEAT_SECONDS=10 \
CL_TOKEN_SCAN_WORKERS=8 \
bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
```

The raw DB root remains:

```text
/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/test
```

### 3. Full paired test set

```bash
GPUS=0,1 \
CL_LIMIT=0 \
CL_WORKERS_PER_JOB=4 \
CL_HEARTBEAT_SECONDS=15 \
CL_TOKEN_SCAN_WORKERS=8 \
bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
```

### 4. One budget only, for diagnosis

```bash
BUDGETS=8 \
GPUS=0,1 \
CL_LIMIT=20 \
CL_HEARTBEAT_SECONDS=10 \
bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
```

## How to interpret GPU memory

It is normal for GPU memory to stay unchanged during:

1. checkpoint preflight;
2. NPZ scenario-token manifest scan;
3. early nuPlan DB/scenario construction.

For a neural baseline, GPU memory should change by the time `[task-gpu-ready]` is printed. That line also reports `param_devices`, `cuda_alloc` and `cuda_reserved`. If the requested device is CUDA but model parameters are not on CUDA, this version raises a `RuntimeError` instead of silently continuing.

`pdm_closed_style` is intentionally CPU-only and therefore should not allocate GPU memory.

## Validation performed in the delivery environment

- Python compilation passed for the modified training, planner and closed-loop runner files.
- Shell syntax checks passed for the modified wrappers.
- Focused regression tests: `27 passed`.
- Synthetic threaded NPZ manifest scan preserved ordered scenario tokens and log hints.
- Missing-checkpoint smoke test now fails **before** touching the NPZ test cache and prints the required training command.

Actual `/data0/...` DB access and A30 CUDA execution can only be verified on the target server.
