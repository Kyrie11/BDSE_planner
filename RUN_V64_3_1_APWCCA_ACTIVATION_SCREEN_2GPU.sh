#!/usr/bin/env bash
set -euo pipefail

export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to the immutable V62/V53 foundation checkpoint}"

export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_1_apwcca_activation_screen_2gpu_v1}"
export GPUS="${GPUS:-0,1}"
export NPROC_PER_NODE=2
export TRAIN_CONFIG="${TRAIN_CONFIG:-bdse/configs/v64_3_1_cc_aocc_apwcca_daepc_screen_2gpu.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-bdse/configs/v64_3_cc_aocc_apwcca_cl.yaml}"
export MAX_TRAIN_SCENARIOS="${MAX_TRAIN_SCENARIOS:-12000}"
export VAL_SCENARIOS="${VAL_SCENARIOS:-500}"
export VAL_EVERY_N_EPOCHS=1
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
  --expected-family v64.3.1 \
  --output "$OUT_ROOT/provenance/config_contract.json"

# Screening intentionally stops after training/validation.  It does not consume
# val_calib, does not run formal gate, and must never be reported as final result.
bash run_v64_saqa_bcc.sh

python - "$OUT_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl" "$OUT_ROOT/provenance/apwcca_activation_screen.json" <<'PY'
import json, math, sys
from pathlib import Path
rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
def vals(k):
    return [float(r[k]) for r in rows if k in r and isinstance(r[k],(int,float)) and math.isfinite(float(r[k]))]
def last(k):
    v=vals(k); return v[-1] if v else float('nan')
def mx(k):
    v=vals(k); return max(v) if v else float('nan')
report={
  'screening_only': True,
  'epochs_seen': sorted({int(r.get('epoch',-1)) for r in rows}),
  'critical_adapter_abs_max': mx('critical_proposal_residual_abs_mean'),
  'critical_adapter_rms_max': mx('critical_proposal_residual_rms'),
  'val_proposal_decisive_recall_last': last('val_proposal_decisive_atom_recall'),
  'val_critical_topm_recall_last': last('val_teacher_exact_winner_flip_critical_recall_topm'),
  'val_critical_selected_recall_last': last('val_teacher_exact_winner_flip_critical_recall_selected'),
  'val_teacher_action_match_last': last('val_teacher_action_match'),
}
report['apwcca_activated']=bool(report['critical_adapter_rms_max'] > 1e-8)
# This is a screening recommendation, not a formal gate.
report['continue_to_full_run']=bool(
    report['apwcca_activated']
    and math.isfinite(report['val_critical_topm_recall_last'])
    and report['val_critical_topm_recall_last'] > 0.355
    and math.isfinite(report['val_proposal_decisive_recall_last'])
    and report['val_proposal_decisive_recall_last'] >= 0.78
)
Path(sys.argv[2]).write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
print(json.dumps(report,indent=2,sort_keys=True))
PY
