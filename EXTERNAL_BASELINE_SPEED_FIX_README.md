# External fixed-budget baseline training speed fix

This patch targets the `RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh`
training path without changing the BDSE dataset roots or the fixed evidence budget.

## What was wrong

1. **The two GPU jobs duplicated the same input work.** GameFormer and DTPP were
   launched as independent Python processes with the same seed and dataset order.
   Each process therefore opened/decompressed/JSON-decoded/tensorized the same NPZ
   samples using its own DataLoader workers. Their nearly identical observed
   `0.44 batch/s` is consistent with shared storage/CPU contention.
2. **Proposal-oracle labels had an E x P Python hot loop.** The greedy evidence
   cover evaluated every active atom using a separate Python reduction on every
   sample, every epoch.
3. **GameFormer/DTPP adapters repeated the full scene encoder.** The legacy
   GameFormer graph ran a 4-layer scene Transformer once per 3 reasoning levels
   (12 encoder-layer applications). DTPP did the same over 2 tree depths (8
   applications). The fast graph encodes static scene/candidate context once and
   uses light per-stage refiners.
4. **Progress logs hid convergence details.** They printed only an instantaneous
   total loss. The patch also prints action CE, proposal BCE, etc.

## What changed

- `bdse/external_baselines/train_pair.py`: one shared DataLoader broadcasts each
  compact batch to two GPUs while keeping independent model/optimizer/scaler/state.
- `bdse/external_baselines/data.py`: vectorized greedy-oracle gain evaluation with
  the original ratio/gain/index tie-breaking semantics.
- `bdse/external_baselines/models.py`: optional
  `shared_encoder_light_refine` graph for GameFormer/DTPP; legacy graph remains.
- `bdse/external_baselines/profile_training.py`: measures DataLoader-only vs
  compute-only throughput to identify the real server bottleneck.
- `RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh`: shared DataLoader is
  enabled by default; `FAST_REFINEMENT` explicitly controls the architecture.

## Recommended training command

```bash
GPUS=0,1 \
SHARED_DATALOADER=1 \
FAST_REFINEMENT=1 \
NUM_WORKERS_SHARED=10 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

`FAST_REFINEMENT=1` changes the GameFormer/DTPP adapter architecture and therefore
**requires training those checkpoints from scratch**. This is the recommended v2
adapter because its encoder/decoder organization is closer to the reference
papers and substantially cheaper than the repeated-full-encoder graph.

For exact legacy architecture/checkpoint compatibility while keeping the shared
input optimization:

```bash
GPUS=0,1 \
SHARED_DATALOADER=1 \
FAST_REFINEMENT=0 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

To compare against the original two independent DataLoaders:

```bash
SHARED_DATALOADER=0 FAST_REFINEMENT=0 GPUS=0,1 \
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

## Profile before a long sweep

First generate the budget configs (the training wrapper also does this):

```bash
python -m bdse.tools.prepare_external_fixed_budget_configs \
  --output-root outputs/external_fixed_budget/configs \
  --budgets 8 16 24 --proposal-top-m 24
```

Then measure one model on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m bdse.external_baselines.profile_training \
  --config outputs/external_fixed_budget/configs/B8/external_gameformer_budgeted.yaml \
  --preprocessed-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2 \
  --split train_boston train_pittsburgh train_singapore train_vegas_2 \
  --batch-size 32 --num-workers 6 --prefetch-factor 4 --amp --batches 100
```

Interpretation:

- `data_rate << compute_rate`: NPZ/storage/CPU is the limiter; shared DataLoader is
  the highest-priority fix. Tune `NUM_WORKERS_SHARED` (e.g. 6, 10, 14) rather than
  blindly adding per-process workers.
- `compute_rate << data_rate`: model graph is the limiter; use fast refinement,
  AMP, and optionally `TORCH_COMPILE=1` after a short smoke run.
- if single-job `data_rate` is high but the old two-process jobs both collapse to
  the same low rate, the shared filesystem/CPU duplication is confirmed.

## Validation performed in the supplied environment

- `bdse/tests/test_external_baselines.py`: **11 passed**.
- Randomized vectorized-oracle regression: selected atom sequence matched the
  reference greedy implementation across 50 random problems.
- Shell syntax check passed.
- Generated B=8/16/24 GameFormer+DTPP and PlanTF+PLUTO configs satisfy the paired
  trainer's identical data-contract check.
- Local CPU microbenchmark (B=8, K=32, E=128; relative only): GameFormer legacy
  0.577 s/step vs fast 0.134 s/step (~4.3x); DTPP 0.159 vs 0.125 (~1.28x).
  Greedy oracle 5.28 ms/sample vs 0.77 ms/sample (~6.8x).

These are not A30 end-to-end speed claims. The real server result must be read from
`profile_training` and the new `[pair-train-progress] rate=... shared_batch/s`.
