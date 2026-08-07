#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${SOURCE_OUT:?Set SOURCE_OUT to the completed V64.2 output root}"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to the matched frozen V62/V53 foundation checkpoint}"
export GPUS="${GPUS:-0,1}"
export BDSE_SPLIT_CACHE="${BDSE_SPLIT_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v53_split}"
export REPLAY_OUT="${REPLAY_OUT:-outputs_v64_2_calibration_consistent_replay}"
export OPEN_LOOP_WORKERS_PER_GPU="${OPEN_LOOP_WORKERS_PER_GPU:-2}"

CANDIDATE_CKPT="${V64_CANDIDATE_CHECKPOINT:-$SOURCE_OUT/train/bdse_v64_saqa_bcc.best.pt}"
DUAL_CAL_JSON="${DUAL_CAL_JSON:-$SOURCE_OUT/calibration/v64_dual_certificate.json}"
[[ -s "$CANDIDATE_CKPT" ]] || { echo "missing candidate checkpoint: $CANDIDATE_CKPT" >&2; exit 2; }
[[ -s "$FOUNDATION_CKPT" ]] || { echo "missing foundation checkpoint: $FOUNDATION_CKPT" >&2; exit 2; }
[[ -s "$DUAL_CAL_JSON" ]] || { echo "missing V64.2 calibration json: $DUAL_CAL_JSON" >&2; exit 2; }

mkdir -p "$REPLAY_OUT/calibration" "$REPLAY_OUT/open_loop" "$REPLAY_OUT/logs" "$REPLAY_OUT/train"
cp -f "$SOURCE_OUT/train/bdse_v64_saqa_bcc.train_log.jsonl" "$REPLAY_OUT/train/bdse_v64_saqa_bcc.train_log.jsonl"
cp -f "$DUAL_CAL_JSON" "$REPLAY_OUT/calibration/v64_dual_certificate.json"

CAND_CFG="$REPLAY_OUT/calibration/v64_2_candidate_calibration_consistent.yaml"
LOCAL_CFG="$REPLAY_OUT/calibration/v64_2_local_calibration_consistent.yaml"
FOUND_CFG="$REPLAY_OUT/calibration/v64_2_foundation_calibration_consistent.yaml"
python -m bdse.tools.apply_v64_3_dual_calibration \
  --config bdse/configs/v64_saqa_bcc_cl.yaml --calibration-json "$DUAL_CAL_JSON" --output "$CAND_CFG"
python -m bdse.tools.apply_v64_3_dual_calibration \
  --config bdse/configs/v64_saqa_bcc_local_control_cl.yaml --calibration-json "$DUAL_CAL_JSON" --output "$LOCAL_CFG" --control
python -m bdse.tools.apply_v64_3_dual_calibration \
  --config bdse/configs/v64_saqa_bcc_anchor_control_cl.yaml --calibration-json "$DUAL_CAL_JSON" --output "$FOUND_CFG" --control

SUITE_ROOT="$REPLAY_OUT/open_loop/parallel_suite"
rm -rf "$SUITE_ROOT"
python -m bdse.tools.run_parallel_open_loop_suite \
  --system "candidate::$CAND_CFG::$CANDIDATE_CKPT" \
  --system "local::$LOCAL_CFG::$CANDIDATE_CKPT" \
  --system "foundation::$FOUND_CFG::$FOUNDATION_CKPT" \
  --preprocessed-dir "$BDSE_SPLIT_CACHE" --split val_tune --max-scenarios 1000 \
  --output-root "$SUITE_ROOT" --gpus "$GPUS" \
  --workers-per-gpu "$OPEN_LOOP_WORKERS_PER_GPU" --device cuda

python -m bdse.tools.check_v64_saqa_bcc_gate \
  "$SUITE_ROOT/candidate/metrics.json" "$SUITE_ROOT/local/metrics.json" "$SUITE_ROOT/foundation/metrics.json" \
  --candidate-jsonl "$SUITE_ROOT/candidate/metrics.jsonl" \
  --local-control-jsonl "$SUITE_ROOT/local/metrics.jsonl" \
  --foundation-control-jsonl "$SUITE_ROOT/foundation/metrics.jsonl" \
  --train-log "$REPLAY_OUT/train/bdse_v64_saqa_bcc.train_log.jsonl" \
  --train-config bdse/configs/v64_2_saqa_bcc_hcbe_train_2gpu.yaml \
  --candidate-config "$CAND_CFG" --local-control-config "$LOCAL_CFG" --foundation-control-config "$FOUND_CFG" \
  --dual-calibration-json "$DUAL_CAL_JSON" \
  --report-json "$REPLAY_OUT/open_loop/v64_2_calibration_consistent_gate_report.json" \
  2>&1 | tee "$REPLAY_OUT/logs/v64_2_calibration_consistent_gate.out"
