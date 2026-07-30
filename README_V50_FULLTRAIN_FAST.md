> **v50.1 update:** use `README_V50_CHECKPOINT_INDEPENDENT.md` and the current `V50_DBAP_RI_NEXT_COMMANDS.sh`. The launcher no longer assigns the deleted `outputs_v30/...` path when variables are unset.

# v50 DBAP-RI Full Retrain + Exact Training Acceleration

Use this package when the historical v30 warm-start checkpoint is missing.

## Main behavior

`V50_DBAP_RI_NEXT_COMMANDS.sh` automatically:

1. rebuilds a v30-compatible foundation checkpoint from random initialization;
2. records checkpoint/config/cache provenance;
3. warm-starts v50 from that rebuilt checkpoint;
4. evaluates the frozen control from the same checkpoint;
5. performs independent calibration and the strict paired open-loop gate;
6. runs CL20 only after gate PASS.

Do not compare the resulting absolute numbers directly with old v49/v50 runs that used the deleted historical checkpoint.

## Run

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

Do not export the deleted `V30_CKPT_IN` path. If a valid checkpoint is supplied explicitly, the pipeline uses it instead of rebuilding.

## Key outputs

```text
$OUT_ROOT/foundation_v30/train/bdse_v30_pmvrbsr_rebuilt.best.pt
$OUT_ROOT/provenance/foundation_checkpoint.json
$OUT_ROOT/train/bdse_v50_dbap_ri.best.pt
$OUT_ROOT/open_loop/open_loop_v50_dbap_ri.json
$OUT_ROOT/open_loop/.v50_dbap_ri_gate_passed
```

## Training speed

The exact selector remains exact. Four local scenes are dispatched to four persistent spawn workers per DDP rank. A representative benchmark produced exactly equal masks and about 3.9x steady-state selector speedup.

Run the benchmark on the target server:

```bash
python tools/benchmark_v50_training_hotpath.py \
  --batch-size 4 --atoms 128 --actions 32 --pairs 56 \
  --process-workers 4 --repeats 2
```

The first process measurement includes one-time worker startup. Use the final repeat as the steady-state value.

## Monitoring

```bash
tail -f "$OUT_ROOT"/logs/pipeline_*.log
tail -f "$OUT_ROOT"/foundation_v30/logs/train_foundation_2gpu.out
tail -f "$OUT_ROOT"/logs/train_2gpu.out
```

After the first v50 epoch, inspect the JSONL training log for:

- `train_loss_ms_per_step`
- `selector_exact_wall_time_s`
- `train_forward_ms_per_step`
- `train_backward_step_ms_per_step`
- `train_data_wait_ms_per_step`

## Fallback

If the operating system disallows spawn workers or CPU memory is constrained:

```bash
EXACT_SELECTOR_CPU_BACKEND=sequential \
EXACT_SELECTOR_WORKERS_PER_RANK=1 \
... \
bash V50_DBAP_RI_NEXT_COMMANDS.sh
```

This is slower but produces the same exact selector masks.
