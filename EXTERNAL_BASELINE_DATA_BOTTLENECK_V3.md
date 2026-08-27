# External baseline data bottleneck V3

## Diagnosis from the A30 host

Observed raw profiler:

- `data_rate=0.550 batch/s` at batch 32 => ~17.6 samples/s with 6 workers.
- `compute_rate=16.041 batch/s` => ~0.062 s/model batch.
- compact BUILD with 10 workers reaches ~26-30 samples/s.

The 6->10 worker scaling is close to linear (`17.6 * 10/6 = 29.3 samples/s`). Therefore the cache builder has not yet saturated all useful CPU/storage parallelism. During `compact-cache BUILD` no model forward/backward runs; flat GPU memory is expected.

The current compact schema is ~9.92 GiB for 364177 training rows (`float32 width=6856`, `int64 width=161`, `bool width=544`). The output write rate implied by 30 samples/s is under 1 MiB/s, so mmap output writes are not the limiter. Raw NPZ member access / JSON parse / feature+oracle construction is the limiter.

## V3 changes

1. CUDA is verified and printed before cache construction. BUILD explicitly reports that it is CPU/storage only.
2. BUILD auto-benchmarks NPZ read strategy:
   - `direct`: NumPy seeks individual NPZ members.
   - `bytes`: whole archive is read once, then members are served from RAM. This can be much faster on NFS/high-latency storage.
3. BUILD auto-benchmarks 8/12/16/24/32 workers (bounded by host CPU/RAM), then selects the best measured throughput.
4. Optional `orjson` is used for JSON decode when installed; stdlib JSON remains a correctness fallback.
5. Cache workers emit three packed dtype rows instead of many small tensors, reducing DataLoader collation/IPC overhead and avoidable torch<->NumPy conversions.
6. Under the fixed-budget unit-cost contract, B=8/16/24 oracle masks are exact prefixes of one max-budget greedy sequence. V3 asserts actual active costs are unit before using this optimization; otherwise it falls back to independent exact masks.
7. After the cache is complete, paired training uses `COMPACT_DEVICE_CACHE=auto` by default. If each A30 has enough free memory for the compact tensor cache plus an 8 GiB reserve, a full numeric copy is loaded to each GPU once. Batches are then random-gathered on-device; there is no per-step mmap read and no per-step host->device batch transfer.
8. GPU-resident mode defaults to `float32`, preserving the existing compact tensor values exactly. `float16` is available only as an explicit memory-saving opt-in.
9. Training logs now report CUDA allocated/peak memory so GPU execution is visible directly in the training output.

## Existing partial cache can resume

The on-disk compact schema/version is unchanged. An incomplete V2 cache such as

`/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2_external_compact_v2_B8_16_24`

should NOT be deleted. V3 reads its `build_state.json` and resumes at the last committed `next_index`.

## Main command (checkpoint paths unchanged)

```bash
GPUS=0,1 bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

Checkpoint locations remain:

- `outputs/external_fixed_budget/B8/<model>_budgeted.best.pt`
- `outputs/external_fixed_budget/B16/<model>_budgeted.best.pt`
- `outputs/external_fixed_budget/B24/<model>_budgeted.best.pt`

## Useful overrides

Disable auto worker/read-mode benchmark:

```bash
COMPACT_BUILD_AUTOTUNE=0 COMPACT_NPZ_READ_MODE=direct COMPACT_BUILD_WORKERS=16 \
GPUS=0,1 bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

Force device-resident cache (fails instead of falling back if it does not fit):

```bash
COMPACT_DEVICE_CACHE=on COMPACT_DEVICE_FLOAT_DTYPE=float32 \
GPUS=0,1 bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

Disable device residency and retain pinned mmap behavior:

```bash
COMPACT_DEVICE_CACHE=off GPUS=0,1 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

If exact float32 residency does not fit, an opt-in half-storage copy is available:

```bash
COMPACT_DEVICE_CACHE=on COMPACT_DEVICE_FLOAT_DTYPE=float16 \
GPUS=0,1 bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

This changes input quantization slightly; use float32 for the paper-grade default.

## Precise raw-stage diagnosis

```bash
python -m bdse.external_baselines.compact_cache diagnose \
  --config outputs/external_fixed_budget/configs/B8/external_gameformer_budgeted.yaml \
  --preprocessed-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2 \
  --split train_boston train_pittsburgh train_singapore train_vegas_2 \
  --budgets 8 16 24 \
  --samples 16 \
  --read-modes direct bytes
```

It prints median time split into `npz_json_ms`, `feature_target_ms`, and `oracle_Bs_ms`.

## Expected log sequence

During cache build:

```
[train-launch-env] ... cuda_available=True visible_count=2
[train-launch-gpu] ...
[compact-cache-note] compact BUILD is intentionally CPU/storage-only ...
[compact-autotune] ...
[compact-cache] RESUME ...
[compact-cache] BUILD ... workers=<selected> npz_read_mode=<selected> ...
```

After cache completion and at actual model training:

```
[compact-device-cache-check] ... fits=True
[compact-device-cache-ready] ...
[pair-data] device_resident=1 ...
[pair-train-init] ... cuda:0 ... cuda:1 ...
[pair-cuda-mem] ...
[pair-train-progress] ... cuda_memA=... cuda_memB=...
```
