#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
: "${DARM_DBR_CKPT:?Set DARM_DBR_CKPT to the same promoted V64.3.7 DARM+DBR-LITERAL full checkpoint}"

: "${DARM_DBR_FULL_REPORT:?Set DARM_DBR_FULL_REPORT to the passed V64.3.7 full audit}"
: "${BDMU_MAIN_SCREEN_REPORT:?Set BDMU_MAIN_SCREEN_REPORT to the passed main BDMU screen report}"
: "${BDMU_FULL_REPORT:?Set BDMU_FULL_REPORT to the passed main BDMU full-pipeline audit}"
python - "$DARM_DBR_FULL_REPORT" "$BDMU_MAIN_SCREEN_REPORT" "$BDMU_FULL_REPORT" <<'PY_GATE'
import json,sys
for name,path in [('DARM+DBR full',sys.argv[1]),('BDMU main screen',sys.argv[2]),('BDMU full',sys.argv[3])]:
    r=json.load(open(path))
    if not r.get('full_promotion', False):
        raise SystemExit(f'{name} is not promoted; theory ablations are blocked to avoid wasting GPU time.')
PY_GATE
export FOUNDATION_CKPT="$DARM_DBR_CKPT"
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export GPUS="${GPUS:-0,1}" NPROC_PER_NODE=2 BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}" NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}" VAL_NUM_WORKERS_PER_GPU="${VAL_NUM_WORKERS_PER_GPU:-4}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-12000}" VAL_SCENARIOS="${VAL_SCENARIOS:-500}" VAL_EVERY_N_EPOCHS=1 VAL_BEFORE_TRAINING=1 BEST_MIN_EPOCH=1 AUTO_RESUME=0 INIT_MODE=warm_start RUN_MODE=train
ROOT="${ABLATION_ROOT:-outputs_v64_3_8_bdmu_theory_ablations_2gpu_v1}"; mkdir -p "$ROOT"
for item in \
  "r1:bdse/configs/v64_3_8_cc_aocc_bdmu_r1_daepc_screen_2gpu.yaml" \
  "nocost:bdse/configs/v64_3_8_cc_aocc_bdmu_nocost_daepc_screen_2gpu.yaml"; do
  tag="${item%%:*}"; cfg="${item#*:}"
  export TRAIN_CONFIG="$cfg" EVAL_CONFIG="bdse/configs/v64_3_8_cc_aocc_bdmu_cl.yaml" OUT_ROOT="$ROOT/$tag"
  mkdir -p "$OUT_ROOT/provenance"
  python -m bdse.tools.validate_v64_pipeline_config --train-config "$TRAIN_CONFIG" --eval-config "$EVAL_CONFIG" --expected-family v64.3.8 --output "$OUT_ROOT/provenance/config_contract.json"
  bash run_v64_saqa_bcc.sh
  python -m bdse.tools.check_v64_3_8_bdmu_screen --train-log "$OUT_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl" --output "$OUT_ROOT/provenance/v64_3_8_bdmu_screen.json"
done
