# External baseline training failure fix

This revision fixes and hardens `RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh`.

## Confirmed bug

`bdse/external_baselines/train.py::main()` contained `import torch._dynamo` inside the function. In Python, that import binds the leading name `torch` as a local variable for the entire function, so earlier statements such as `hasattr(torch, ...)` raised:

`UnboundLocalError: local variable 'torch' referenced before assignment`.

The import now uses `importlib.import_module("torch._dynamo")`, which never shadows the module-level `torch` binding.

## Why an older revision could still print `FAILED: gameformer` / `FAILED: dtpp`

The old wrapper collapsed every non-zero Python exit into the same generic line. Without the corresponding `*.train.out`, the historical root cause cannot be proven. Two shared environment-sensitive defaults were nevertheless unsafe for a nuPlan-era environment and have been removed from the default path:

- `TORCH_COMPILE=1`: compile is now opt-in (`TORCH_COMPILE=0` by default). This does not change the architecture/loss/data/optimizer; it only disables graph compilation unless explicitly requested.
- forced `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: the wrapper no longer injects this option. Some older PyTorch builds reject it during CUDA allocator initialization.

## New fail-fast diagnostics

Before the long training loop each model now runs a small eager startup preflight (default 2 samples) that checks:

- cache fields required by expert imitation and proposal supervision;
- batch tensor shape compatibility;
- GameFormer/DTPP/PlanTF/PLUTO eager forward;
- finite loss;
- backward/autograd and finite gradients.

The preflight does not step an optimizer and restores CPU/CUDA RNG state, so it does not perturb the seeded training trajectory.

Each CUDA process prints PyTorch/CUDA/cuDNN/device information. The wrapper also validates that two GPUs are visible before launching the first pair.

On failure the wrapper now prints the exact model, budget, Python exit code, log path, and the final 80 log lines. It also verifies that a non-empty `.best.pt` checkpoint was actually produced before reporting success.

## Recommended full training command

```bash
GPUS=0,1 \
TRAIN_PROGRESS_STYLE=lines \
LOG_EVERY_N_STEPS=50 \
TORCH_COMPILE=0 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

Do not set `PYTORCH_CUDA_ALLOC_CONF` initially. After eager training is known to work, graph compilation can be tested separately with `TORCH_COMPILE=1` if the installed PyTorch/Triton stack supports it.

## Fast two-GPU smoke test before a full run

This revision also adds non-semantic wrapper overrides so the real server/cache path can be validated quickly before starting all 30 epochs. They are disabled by default.

```bash
unset PYTORCH_CUDA_ALLOC_CONF
BUDGETS=8 \
ONLY_FIRST_PAIR=1 \
TRAIN_MAX_SCENARIOS_PER_SPLIT=64 \
VAL_MAX_SCENARIOS=64 \
EPOCHS_OVERRIDE=1 \
GPUS=0,1 \
TORCH_COMPILE=0 \
TRAIN_PROGRESS_STYLE=lines \
LOG_EVERY_N_STEPS=10 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

This runs only GameFormer B8 on GPU0 and DTPP B8 on GPU1 for one epoch, using at most 64 training samples per city split and 64 validation samples. Once this passes, run the full command with the smoke-only overrides removed.
