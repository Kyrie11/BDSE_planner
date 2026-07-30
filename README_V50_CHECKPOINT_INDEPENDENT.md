# v50.1 checkpoint-independent launcher

This revision removes the hard dependency on `outputs_v30`.

## Main command

```bash
unset V30_CKPT_IN FOUNDATION_CKPT CONTROL_CKPT

export NUPLAN_ROOT=/data0/senzeyu2/dataset/nuplan
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2
export BDSE_VAL_CACHE_ORIGINAL=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2
export OUT_ROOT=outputs_v50_dbap_ri_checkpoint_independent_2gpu_v1

PIPELINE_DETACH=1 \
RUN_CLOSED_LOOP_AFTER_GATE=1 \
RUN_CL100_AFTER_CL20=0 \
FOUNDATION_POLICY=auto \
REBUILD_FOUNDATION_IF_MISSING=1 \
RECOVER_SAFE_FOUNDATION_COPIES=1 \
ALLOW_ALGORITHM_CHECKPOINT_INIT=0 \
EXACT_SELECTOR_CPU_BACKEND=process \
EXACT_SELECTOR_WORKERS_PER_RANK=4 \
GPUS=0,1 \
bash V50_DBAP_RI_NEXT_COMMANDS.sh
```

`FOUNDATION_POLICY=auto` conservatively recovers only a true v30-compatible copy. If none exists, it rebuilds a matched foundation from scratch. v50 and control then use exactly the same foundation.

## Check retained checkpoints

```bash
bash INSPECT_RETAINED_CHECKPOINTS.sh retained_checkpoint_inventory.json
```

## Monitor

```bash
tail -f "$OUT_ROOT"/logs/pipeline_*.log
```

## Verify provenance

```bash
cat "$OUT_ROOT"/provenance/foundation_checkpoint.json
cat "$OUT_ROOT"/provenance/foundation_checkpoint_inventory.json
```

## Important experiment rule

Do not substitute a v47-v49 trained checkpoint into the paper main run. Such a run is a transfer-initialization ablation and cannot isolate v50 algorithm gains.
