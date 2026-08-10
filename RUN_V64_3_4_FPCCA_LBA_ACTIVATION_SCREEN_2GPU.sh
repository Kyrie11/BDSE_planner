#!/usr/bin/env bash
set -euo pipefail

export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to the immutable V62/V53 foundation checkpoint}"

export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_4_fpcca_lba_activation_screen_2gpu_v1}"
export GPUS="${GPUS:-0,1}"
export NPROC_PER_NODE=2
export TRAIN_CONFIG="${TRAIN_CONFIG:-bdse/configs/v64_3_4_cc_aocc_fpcca_lba_daepc_screen_2gpu.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-bdse/configs/v64_3_4_cc_aocc_fpcca_cl.yaml}"
export MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-12000}"
export VAL_SCENARIOS="${VAL_SCENARIOS:-500}"
export VAL_EVERY_N_EPOCHS=1
export VAL_BEFORE_TRAINING=1
export BEST_MIN_EPOCH=3
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
export NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}"
export VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export AUTO_RESUME="${AUTO_RESUME:-0}"
export INIT_MODE=warm_start
export RUN_MODE=train

mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"
python -m bdse.tools.validate_v64_pipeline_config \
  --train-config "$TRAIN_CONFIG" --eval-config "$EVAL_CONFIG" \
  --expected-family v64.3.4 \
  --output "$OUT_ROOT/provenance/config_contract.json"
python - "$OUT_ROOT/provenance/screen_code_sha256.json" <<'PY_SHA'
import hashlib, json, sys
from pathlib import Path
files=[
  'bdse/model/bdse_model.py',
  'bdse/model/losses.py',
  'bdse/experiments/train.py',
  'RUN_V64_3_4_FPCCA_LBA_ACTIVATION_SCREEN_2GPU.sh',
]
out={}
for name in files:
    data=Path(name).read_bytes()
    out[name]=hashlib.sha256(data).hexdigest()
Path(sys.argv[1]).write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
PY_SHA

# Screening intentionally stops after training/validation.  It does not consume
# val_calib, does not run formal gate, and must never be reported as final result.
bash run_v64_saqa_bcc.sh

python -m bdse.tools.check_v64_3_3_acquisition_screen \
  --train-log "$OUT_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl" \
  --output "$OUT_ROOT/provenance/critical_acquisition_screen.json" \
  --variant FPCCA+LBA+ACRA-full-support
