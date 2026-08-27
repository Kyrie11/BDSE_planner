# External baseline training bottleneck V4: resume + cache-tier fix

## 1. What the current log means

The supplied GameFormer+DTPP pair log is not stalled. At epoch 1 it progresses from
`step=5000/11381` to `step=11100/11381`, with cumulative shared throughput rising
from about 4.5 to 6.1 batch/s. Batch size is 32 and the full train set has 364177
rows, so one training epoch alone is about 31 minutes at 6.1 batch/s, before full
validation.

The important diagnostic is CUDA allocation: both models report only about
0.08 GiB allocated and about 0.2 GiB peak. A successful float32 device-resident
compact cache would add about 9.92 GiB **per resident GPU**. Therefore the logged
run is not using the device-resident compact cache; it is on the host/mmap path.
The monotonic throughput warm-up is also consistent with OS page-cache warming.

The package's earlier profiler already separated the two stages on this host:

- raw NPZ/input pipeline: 0.550 batch/s at batch 32;
- GPU forward/loss/backward/step replay: 16.041 batch/s;
- compact train cache size: about 9.92 GiB for 364177 rows.

So after raw NPZ preprocessing has been compacted, the remaining performance risk
is random access to the large compact mmap when GPU residency falls back.

A second source of the impression that "nothing was trained" is checkpoint timing.
The old pair trainer writes `*_budgeted.pt` / `*.best.pt` only after the **entire
training epoch and its validation** complete. There is no mid-epoch checkpoint.
Thus a run killed during epoch 1 has no reusable checkpoint even if it already
processed >95% of the epoch.

## 2. Why the previous device-cache fallback is fragile

The old `train_pair.py` attempted to load one full compact cache on each GPU. If
only one A30 could fit it, the code deliberately discarded that valid one-sided
cache and forced **both** models back to the pinned host mmap loader.

With global shuffle, every batch then performs random fancy-index gathers over a
~10 GiB file. On slow local storage, NFS, or a host whose page cache is cold, this
causes random page faults. The model tensors themselves are small, so the GPUs wait
for input instead of consuming their compute budget.

The host loader also performed a redundant second `.copy()` after NumPy fancy
indexing. Fancy indexing already returns an owning writable ndarray, so that copy
was unnecessary.

## 3. V4 cache hierarchy

The tensor contract, float32 values, fixed-budget masks, batch order, losses and
output paths are unchanged. Only data placement changes.

For both training and validation V4 uses this order:

1. **Replicated GPU cache**: if both GPUs can hold the exact float32 compact cache
   while keeping `COMPACT_DEVICE_RESERVE_GIB` free, each GPU gathers its batch
   locally. This is the old best-case path.
2. **Single-GPU cache + peer batch copy**: if only one GPU fits the cache, keep it.
   Gather each ~MiB batch once on that GPU and copy only that batch to the other
   GPU. The old code discarded this useful cache.
3. **Host-RAM cache**: if neither GPU fits and the host has enough `MemAvailable`,
   sequentially materialize the mmap arrays once into ordinary RAM. Global random
   shuffle is preserved exactly, but random gathers no longer fault storage pages.
4. **mmap fallback**: if host RAM is also insufficient, keep the original mmap
   behavior. `COMPACT_SHUFFLE_MODE=block` remains an optional locality-oriented
   override; V4 does not enable it automatically because that would change the
   original global-shuffle order.

Default device float dtype remains `float32`; no input quantization is introduced.
`float16` is still an explicit opt-in only.

## 4. Resume behavior

The public launcher now sets `AUTO_RESUME=1` by default.

For every model/B arm:

- prefer `outputs/external_fixed_budget/B<B>/<model>_budgeted.pt` (latest completed
  epoch);
- if latest is absent, fall back to `<model>_budgeted.best.pt`;
- restore model, optimizer and scheduler from old checkpoints;
- new checkpoints also save AMP scaler and RNG state when available;
- compact loaders use `seed + absolute_epoch`, so a resumed compact-cache run uses
  the same epoch permutation as an uninterrupted run;
- strict pair-resume checks reject a checkpoint with a different generated config
  hash, variant, budget, or refinement mode;
- if one side of a two-GPU pair is complete and the other is not, only the
  unfinished model performs forward/backward/validation/checkpoint updates;
- if both are complete, the pair exits immediately and reuses the existing files.

Old checkpoints have no scaler/RNG payload, but remain loadable. In that case the
optimizer/scheduler/model epoch is recovered; exact stochastic continuation is
best-effort. All checkpoints produced by V4 contain the additional state.

The launcher's old `best.pt -nt run_started_marker` postcondition was removed,
because it forced an already-complete valid checkpoint to look like a failure.
The postcondition now requires the historical latest and best filenames to exist,
without requiring them to be freshly rewritten.

## 5. Command and output directories are unchanged

Run exactly the same command:

```bash
GPUS=0,1 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

Outputs remain, for example:

```text
outputs/external_fixed_budget/B8/gameformer_budgeted.pt
outputs/external_fixed_budget/B8/gameformer_budgeted.best.pt
outputs/external_fixed_budget/B8/dtpp_budgeted.pt
outputs/external_fixed_budget/B8/dtpp_budgeted.best.pt
```

and the same layout for B16/B24 and PlanTF/PLUTO.

To intentionally discard resume behavior and retrain from epoch 1:

```bash
AUTO_RESUME=0 GPUS=0,1 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

To force the one-copy GPU layout even when both GPUs have room (useful for testing):

```bash
COMPACT_DEVICE_CACHE_LAYOUT=peer GPUS=0,1 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

To disable both GPU and RAM residency and reproduce the old mmap behavior:

```bash
COMPACT_DEVICE_CACHE=off COMPACT_HOST_CACHE=off GPUS=0,1 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

## 6. What to look for in the new log

Best case (both GPUs fit):

```text
[compact-device-pair-check] ... A_fits=True ... B_fits=True
[compact-device-pair] layout=replicate
[pair-data] device_resident=1 layout=replicate ...
```

One GPU fits:

```text
[compact-device-pair-check] ... A_fits=True ... B_fits=False
[compact-device-pair] layout=peer source=cuda:0 target=cuda:1
[pair-data] device_resident=1 layout=peer ...
```

Neither GPU fits but RAM does:

```text
[compact-host-cache-check] ... fits=True
[compact-host-cache-ready] resident_gib=... load_s=...
[pair-data] host_storage=ram ...
```

Resume:

```text
[pair-resume] A=gameformer checkpoint=... next_epoch=.../30
[pair-resume] B=dtpp checkpoint=... next_epoch=.../30
```

If a side is already complete, later epoch lines show `activeA=0` or `activeB=0`;
that side is not retrained.

## 7. Validation performed here

- Python compile checks pass for `compact_cache.py`, `train.py`, and `train_pair.py`.
- `bash -n RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh` passes.
- External-baseline + training resume/stability tests: 20 passed.
- A synthetic old-style checkpoint without AMP/RNG fields was successfully loaded
  by the new resume code with the expected next epoch.

A real A30 throughput number cannot be produced in this sandbox because the user's
nuPlan cache and GPUs are not mounted here. The new startup diagnostics make the
actual cache tier explicit, so the next server run will immediately show whether
it is on replicated GPU, peer-GPU, host RAM, or mmap fallback.
