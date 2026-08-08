#!/usr/bin/env bash
set -euo pipefail

export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to the immutable V62/V53 foundation checkpoint}"

export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_2_apwrcca_activation_screen_2gpu_v1}"
export GPUS="${GPUS:-0,1}"
export NPROC_PER_NODE=2
export TRAIN_CONFIG="${TRAIN_CONFIG:-bdse/configs/v64_3_2_cc_aocc_apwrcca_daepc_screen_2gpu.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-bdse/configs/v64_3_cc_aocc_apwcca_cl.yaml}"
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
  --expected-family v64.3.2 \
  --output "$OUT_ROOT/provenance/config_contract.json"
python - "$OUT_ROOT/provenance/screen_code_sha256.json" <<'PY_SHA'
import hashlib, json, sys
from pathlib import Path
files=[
  'bdse/model/bdse_model.py',
  'bdse/model/losses.py',
  'bdse/experiments/train.py',
  'RUN_V64_3_2_APWRCCA_ACTIVATION_SCREEN_2GPU.sh',
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

python - "$OUT_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl" "$OUT_ROOT/provenance/apwcca_activation_screen.json" <<'PY'
import json, math, sys
from pathlib import Path
rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
def finite(row,k):
    try:
        v=float(row[k])
    except Exception:
        return None
    return v if math.isfinite(v) else None
def last_row():
    cand=[r for r in rows if int(r.get('epoch',-999))>=0]
    return cand[-1] if cand else {}
def anchor_row():
    cand=[r for r in rows if int(r.get('epoch',-999))==-1]
    return cand[-1] if cand else {}
def mx(k):
    vals=[]
    for r in rows:
        v=finite(r,k)
        if v is not None: vals.append(v)
    return max(vals) if vals else None
a=anchor_row(); z=last_row()
report={
  'screening_only': True,
  'epochs_seen': sorted({int(r.get('epoch',-999)) for r in rows}),
  'activation_source': 'parameter_delta_not_forward_scalar',
  'critical_adapter_parameter_delta_rms_max': mx('critical_adapter_parameter_delta_rms'),
  'critical_adapter_parameter_delta_max_abs_max': mx('critical_adapter_parameter_delta_max_abs'),
  'critical_adapter_parameter_rms_last': finite(z,'critical_adapter_parameter_rms'),
  'forward_diagnostic_abs_max': mx('critical_proposal_residual_abs_mean'),
  'forward_diagnostic_rms_max': mx('critical_proposal_residual_rms'),
  'anchor_val_critical_topm_recall': finite(a,'val_teacher_exact_winner_flip_critical_recall_topm'),
  'last_val_critical_topm_recall': finite(z,'val_teacher_exact_winner_flip_critical_recall_topm'),
  'anchor_val_critical_selected_recall': finite(a,'val_teacher_exact_winner_flip_critical_recall_selected'),
  'last_val_critical_selected_recall': finite(z,'val_teacher_exact_winner_flip_critical_recall_selected'),
  'anchor_val_proposal_decisive_recall': finite(a,'val_proposal_decisive_atom_recall'),
  'last_val_proposal_decisive_recall': finite(z,'val_proposal_decisive_atom_recall'),
  'last_val_teacher_action_match': finite(z,'val_teacher_action_match'),
}
for stem in ('critical_topm','critical_selected','proposal_decisive'):
    av=report.get('anchor_val_'+stem+'_recall')
    lv=report.get('last_val_'+stem+'_recall')
    report['delta_val_'+stem+'_recall']=(lv-av) if av is not None and lv is not None else None
required=[
  'critical_adapter_parameter_delta_rms_max',
  'anchor_val_critical_topm_recall','last_val_critical_topm_recall',
  'anchor_val_proposal_decisive_recall','last_val_proposal_decisive_recall',
]
report['screen_instrumentation_valid']=all(report.get(k) is not None for k in required)
report['apwcca_activated']=bool(report['critical_adapter_parameter_delta_rms_max'] is not None and report['critical_adapter_parameter_delta_rms_max'] > 1e-9)
# Relative-to-anchor screen avoids rejecting a healthy proposal solely because a
# 500-scene subset happens to sit below a historical 1000-scene absolute value.
report['continue_to_full_run']=bool(
    report['screen_instrumentation_valid']
    and report['apwcca_activated']
    and report['delta_val_critical_topm_recall'] is not None
    and report['delta_val_critical_topm_recall'] > 0.0
    and report['delta_val_proposal_decisive_recall'] is not None
    and report['delta_val_proposal_decisive_recall'] >= -0.02
)
Path(sys.argv[2]).write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
print(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
PY
