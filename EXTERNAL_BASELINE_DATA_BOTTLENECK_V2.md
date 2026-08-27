# External baseline training: data-bottleneck V2 fix

## Diagnosis from the A30 profiler

Observed on the target host for GameFormer, B=8, batch=32:

- raw NPZ DataLoader: `0.550 batch/s` = about `17.6 sample/s`
- GPU forward/loss/backward/optimizer replay: `16.041 batch/s` = about `513 sample/s`
- serial estimate: `0.532 batch/s`

So a step spends about 1.82 s waiting for raw NPZ/JSON/tensorization and only
about 0.062 s in model compute. The shared two-GPU DataLoader correctly removes
duplicate reads, but it cannot make the remaining single NPZ decode stream faster.

The nested-tensor warning caused by `norm_first=True` is not the limiting factor at
these measurements. It can only affect the already-fast model side.

## V2 change: persistent compact mmap cache

`bdse.external_baselines.compact_cache` materializes the exact tensor contract used
by the external adapters once, then stores it in three row-major memory-mapped
arrays:

- `float32.npy`
- `int64.npy`
- `bool.npy`

GameFormer, DTPP, PlanTF and PLUTO have the same compact data contract. B=8, B=16
and B=24 also share all scene/candidate/evidence tensors. Only
`oracle_selected_mask` is budget dependent, so the cache stores three small oracle
masks while reusing every other feature.

For the current K=32 / Emax=128 contract the train cache is expected to be roughly
10 GiB for 364,177 samples. The builder prints the exact estimate and checks free
disk space before allocation.

The first build still has to read the raw NPZ files once. It is sequential/locality
friendly and resumable. Every subsequent epoch and every subsequent model/budget
avoids NPZ open/decompression, JSON parsing, EvidenceAtom construction, map pooling,
candidate feature construction and oracle recomputation.

## Recommended command

Checkpoint locations are unchanged. The existing launcher now enables the compact
cache by default and auto-builds it on the first invocation:

```bash
GPUS=0,1 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

The checkpoints remain exactly:

```text
outputs/external_fixed_budget/B8/gameformer_budgeted.best.pt
outputs/external_fixed_budget/B8/dtpp_budgeted.best.pt
outputs/external_fixed_budget/B8/plantf_budgeted.best.pt
outputs/external_fixed_budget/B8/pluto_budgeted.best.pt
... same layout for B16 and B24
```

Default compact-cache directories are derived from the source cache and do not
change the experiment output root:

```text
${BDSE_TRAIN_CACHE}_external_compact_v2_B8_16_24
${BDSE_VAL_CACHE}_external_compact_v2_B8_16_24
```

To put the compact cache on a faster local NVMe while keeping checkpoints in the
same place:

```bash
TRAIN_COMPACT_CACHE=/local_nvme/bdse_external_train_v2 \
VAL_COMPACT_CACHE=/local_nvme/bdse_external_val_v2 \
GPUS=0,1 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

## Profile the compact cache before a long run

After the first build:

```bash
python -m bdse.external_baselines.compact_cache profile \
  --cache-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2_external_compact_v2_B8_16_24 \
  --budget 8 \
  --batch-size 32 \
  --batches 500 \
  --warmup 50 \
  --pin-memory
```

For GameFormer, a compact-cache rate above about 16 batch/s means the A30 model
compute has become the main limiter. If global random mmap access is still below
that value, compare a locality-preserving shuffle:

```bash
python -m bdse.external_baselines.compact_cache profile \
  --cache-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2_external_compact_v2_B8_16_24 \
  --budget 8 --batch-size 32 --batches 500 --warmup 50 \
  --shuffle-mode block --block-size 4096 --pin-memory
```

If `block` is materially faster and the storage device is high-latency, launch with:

```bash
COMPACT_SHUFFLE_MODE=block \
COMPACT_BLOCK_SIZE=4096 \
GPUS=0,1 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

`global` remains the default to minimize changes to the original random-shuffle
training protocol.

## Why not two-GPU DDP for one model?

At the measured raw-cache rates, DDP cannot solve the bottleneck because both GPUs
would still wait on the same ~17.6 sample/s input stream. The models are also small
(about 5-6M parameters), so a two-GPU DDP step adds gradient all-reduce overhead to
an already short ~62 ms compute step.

For a sweep that must train GameFormer + DTPP + PlanTF + PLUTO, one model per GPU is
also throughput-optimal: GPU0 and GPU1 make progress on two independent checkpoints
with no cross-GPU all-reduce. DDP is only worth reconsidering if, after the compact
cache is active, a *single model's* compute time is the dominant cost and minimizing
that one checkpoint's latency matters more than total four-model throughput.

## Validation performed in this package

- 12 external-baseline tests pass.
- Compact mmap tensors match direct compact tensorization on synthetic saved NPZs.
- B-specific oracle-mask selection is preserved.
- Single-model compact-cache preflight + forward/backward + optimizer + checkpoint
  path was exercised successfully.
- `bash -n RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh` passes.
- Python compilation checks pass for `compact_cache.py`, `train.py`, and
  `train_pair.py`.
