# V47 D3CE training optimization and resume guide

## Run

From the code root:

```bash
bash V47_D3CE_NEXT_COMMANDS.sh
```

The complete pipeline now starts in a detached session by default, so an SSH or
terminal disconnect does not terminate training, calibration, or evaluation.
The command prints the PID and pipeline log path before returning.

Run in the foreground when desired:

```bash
PIPELINE_DETACH=0 bash V47_D3CE_NEXT_COMMANDS.sh
```

Re-running the same command is safe. It automatically:

1. reuses the existing group-disjoint validation split;
2. selects the furthest valid resumable checkpoint from
   `bdse_v47_d3ce.latest.pt` and `train/checkpoints/*.pt`;
3. resumes the saved epoch and next batch index with optimizer and AMP state;
4. skips completed downstream stages when their outputs are newer than inputs.

To force a specific checkpoint:

```bash
RESUME_FROM=/path/to/checkpoint.pt bash V47_D3CE_NEXT_COMMANDS.sh
```

To force all pipeline stages to run again:

```bash
PIPELINE_FORCE=1 bash V47_D3CE_NEXT_COMMANDS.sh
```

All original output directories and filenames are unchanged.

## Confirmed bottlenecks and changes

### 1. Repeated CUDA synchronization in the exact selector

The training configuration intentionally uses the exact NumPy/CPU deployment
selector for every local scene and every optimizer step. Previously, its inputs
were copied from CUDA one tensor at a time, causing more than ten possible
stream synchronizations per batch.

`bdse/model/losses.py` now packs inputs by destination dtype and performs at
most one float, one integer, and one Boolean CPU snapshot. Shapes, dtypes,
values, exact selector calls, budgets, and selected masks are unchanged.

### 2. Re-decoding completed batches after a mid-epoch restart

The old loop restored `next_batch_index` but still asked DataLoader workers to
read, decode, tensorize, and collate every earlier batch before `continue`.

The DDP loader now uses `ResumableBatchSampler`, which reconstructs the same
deterministic distributed sample order and begins yielding directly at the next
unfinished batch. This removes duplicated cache I/O and CPU tensorization after
a restart without changing the remaining sample sequence.

### 3. Duplicate open-loop evaluation

The top-level script previously called `RUN_MODE=train_open_loop` before
calibration and then repeated the same 1,000-scene open-loop evaluation after
calibration. The pre-calibration result was overwritten and was not used by the
gate.

Stage 1 now uses `RUN_MODE=train`. The required calibrated open-loop evaluation
still runs once in stage 3.

### 4. Incorrect detach scope and stage race

Previously only `run_v47_d3ce.sh` detached. Its parent immediately continued to
calibration even though training was still running. The whole ordered pipeline
is now detached once, while the inner training process remains foreground
relative to that pipeline.

### 5. Robust latest-checkpoint discovery

`run_v47_d3ce.sh` no longer checks only one hard-coded latest filename. It
validates candidate checkpoint dictionaries, ignores unreadable or
inference-only files, and ranks resumable checkpoints by `next_epoch`,
`next_batch_index`, and modification time. Warm start from v30 occurs only when
no valid v47 resume checkpoint exists.

## Intentionally unchanged

- model architecture and trainable parameters;
- losses, weights, budgets, selector implementation, and selector cadence;
- training sample cap, batch size, validation frequency, and dense diagnostics;
- optimizer, learning rate, AMP behavior, and gradient clipping;
- checkpoint and result output directories.

The exact CPU selector remains a major part of step time because the supplied
configuration requires full exact supervision (`scenes_per_rank=0`,
`every_n_steps=1`, minimum exact fraction 0.99). Reducing that work would change
the training objective/schedule, so this revision does not do so.

## Validation

The optimized Python files compile successfully and both shell entrypoints pass
`bash -n`. Added regression tests cover packed snapshot value/dtype equivalence
and direct batch-sampler resume. The delivery environment did not include
PyTorch or pytest, so GPU/DDP execution must be exercised in the original
training environment.
