# v50 Full-Retrain and Training-Speed Analysis

## 1. Decision

The deleted `outputs_v30/train/bdse_v30_pmvrbsr.best.pt` can be reconstructed with the current code. The recommended paper-grade protocol is **not** to train v50 directly from random initialization. It is:

1. rebuild a v30-compatible foundation checkpoint from random initialization with the current code and the original v30 objective/configuration;
2. use the rebuilt best checkpoint to warm-start v50;
3. use that exact same rebuilt checkpoint as the frozen matched control;
4. rebuild all control/open-loop results under the same validation split and calibration provenance.

This preserves within-run attribution. It does not reproduce the deleted checkpoint bit-for-bit, and it invalidates direct numerical comparisons against historical v49/v50 runs initialized from the deleted file.

## 2. Effect on the interpretation of v50

### Still valid after a matched rebuild

- Whether v50 passes the open-loop gate against a control using the same rebuilt foundation.
- Candidate/interface/compression error decomposition within the fresh run.
- Whether residual intervention improves local-only pair-full to residual-combined pair-full behavior.
- Whether AOCC and fixed-budget compression preserve the pair-full action.
- Closed-loop comparison after the fresh gate passes.

### No longer directly valid

- Absolute comparison of the new v50 metrics with old v49/v50 metrics initialized from the deleted historical v30 checkpoint.
- Claims that an observed difference is caused only by v50 when the compared run uses another foundation checkpoint.
- Reusing the old control JSON or old calibration result.

### Required for a component-level v49 versus v50 claim

Use the same rebuilt foundation, seed, train rows, val_tune rows, val_calib rows, candidate bank and teacher for every variant. Historical v49 results can remain diagnostic background, but not the paired baseline for the rebuilt run.

## 3. Rebuilt foundation stage

New configuration:

- `bdse/configs/v50_rebuild_v30_from_scratch_2gpu.yaml`

It retains the original v30 model/loss schedule and explicitly marks the stage as oracle-selector foundation training:

- four epochs;
- seed defaults to 17;
- `allow_oracle_only_selector_training: true` because `predicted_selector_start_epoch=6` is later than the four foundation epochs;
- two-GPU DDP with batch size four per rank;
- AMP, pinned memory, persistent workers, fused AdamW and foreach gradient clipping;
- validation on `val_tune` every epoch;
- `teacher_action_match` selects the rebuilt best checkpoint.

Output:

- `OUT_ROOT/foundation_v30/train/bdse_v30_pmvrbsr_rebuilt.best.pt`

The pipeline writes:

- `OUT_ROOT/provenance/foundation_checkpoint.json`

The provenance records checkpoint/config SHA-256, seed, manifest hashes and environment information.

## 4. Why v50 training was slow

The dominant bottleneck is the exact CPU selector inside `compute_bdse_losses`, not the A30 forward pass.

For every optimizer step and every DDP rank, the paper-grade v50 path performs:

1. CUDA-to-CPU synchronization and a packed NumPy snapshot;
2. exact HAB/Top-M construction;
3. exact AOCC/certificate selector execution for all four local scenes;
4. exact masks for B=16 plus an auxiliary B=8 or B=24 budget;
5. CPU-to-GPU transfer of the discrete masks;
6. only then can backward and DDP gradient synchronization proceed.

With 50,000 scenarios, global batch eight and two A30 cards, there are about 6,250 optimizer steps per epoch. DDP cannot hide selector imbalance because both ranks wait before gradient synchronization.

Secondary costs are:

- per-sample NPZ decoding and tensorization;
- 1,000-scenario dense validation every two epochs;
- checkpoint writes;
- model forward/backward and pair materialization.

The package already records per-stage epoch timing:

- `train_data_wait_ms_per_step`
- `train_h2d_ms_per_step`
- `train_forward_ms_per_step`
- `train_loss_ms_per_step`
- `train_backward_step_ms_per_step`
- `selector_exact_wall_time_s`

These metrics should be used to verify the bottleneck on the user's server.

## 5. Exact, logic-preserving acceleration

### 5.1 Spawn-process scene parallelism

Independent scenes are now executed by a persistent `spawn` process pool per DDP rank.

Configuration:

```yaml
training:
  deployment_selector_cpu_backend: process
  deployment_selector_cpu_workers: 4
```

This does not replace the selector with a surrogate and does not change:

- the selector objective;
- Top-M;
- budgets or budget schedule;
- selected masks;
- losses;
- calibration or gate logic.

`spawn` is used instead of `fork` because each training rank already owns a CUDA context. Worker processes receive only NumPy snapshots and never initialize CUDA.

Representative synthetic benchmark at B=4, K=32, E=128, P=56, budgets B=16+B=8:

| Exact selector backend | Steady-state time |
|---|---:|
| Sequential | 1.679 s |
| 4 spawn workers | 0.428 s |
| Speedup | 3.92x |

The process result masks were exactly equal element-by-element to sequential execution. The first process call took about eight seconds because it includes one-time Python/PyTorch worker startup; this cost is amortized over thousands of steps. A two-rank nested process-pool smoke test also passed.

This benchmark is representative, not a promise of identical speed on another CPU. The included benchmark tool measures the actual server.

### 5.2 Other execution-only optimizations

- one packed CUDA-to-CPU snapshot for all exact-selector inputs;
- one snapshot shared by both budgets;
- persistent DataLoader workers;
- prebuilt preprocessed path index;
- unused NPZ metadata excluded from training loads;
- pinned memory and prefetch factor two;
- fused CUDA AdamW with safe fallback;
- foreach gradient clipping with safe fallback;
- DDP `broadcast_buffers=false` and gradient bucket views;
- checkpoint interval changed to 2,000 steps;
- resumable batch sampler skips already completed batches without decoding them again.

Fused kernels can introduce normal floating-point-order differences, but do not change the optimizer equations or training objective.

## 6. What was intentionally not changed

The main paper run still uses:

- exact CPU selection on every local scene;
- exact selector cadence of every step;
- B=16 primary plus auxiliary budget training;
- 1,000-scene `val_tune` checkpoint selection every two epochs;
- the same v50 losses and best-checkpoint score;
- independent `val_calib` calibration;
- the same strict gate before CL20.

A hybrid GPU selector, sampled exact supervision, reduced validation set or altered budget schedule could be faster, but would change the training protocol. They were not enabled.

## 7. New pipeline behavior

`V50_DBAP_RI_NEXT_COMMANDS.sh` now:

1. builds/reuses log-disjoint `val_tune` and `val_calib`;
2. detects the missing historical v30 checkpoint;
3. rebuilds a v30-compatible foundation from random initialization;
4. writes foundation provenance;
5. warm-starts v50 from the rebuilt best checkpoint;
6. evaluates a frozen control from the same checkpoint;
7. calibrates only on `val_calib`;
8. runs paired open-loop gate;
9. starts CL20 only after the gate passes.

The foundation stage and v50 stage both support resumable mid-epoch checkpoints.

## 8. Recommended command

Do not export the deleted `V30_CKPT_IN` path. Run:

```bash
cd /path/to/bdse_v50_fulltrain_fast

export NUPLAN_ROOT=/path/to/nuplan
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2
export BDSE_VAL_CACHE_ORIGINAL=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2
export OUT_ROOT=outputs_v50_dbap_ri_fulltrain_fast_2gpu_v1

PIPELINE_DETACH=1 \
RUN_CLOSED_LOOP_AFTER_GATE=1 \
RUN_CL100_AFTER_CL20=0 \
REBUILD_FOUNDATION_IF_MISSING=1 \
EXACT_SELECTOR_CPU_BACKEND=process \
EXACT_SELECTOR_WORKERS_PER_RANK=4 \
GPUS=0,1 \
bash V50_DBAP_RI_NEXT_COMMANDS.sh
```

## 9. Validation performed

- Python compilation: passed.
- YAML configuration loading: passed.
- Shell syntax: passed.
- Unit suite: 165 passed, 5 warnings.
- Exact sequential/process mask equality: passed.
- Two-rank nested spawn-pool smoke test: passed.
- Representative hot-path benchmark: approximately 3.92x steady-state exact-selector speedup.

Actual end-to-end speedup must be read from the generated per-stage training metrics because data storage, CPU contention and validation time vary by server.
