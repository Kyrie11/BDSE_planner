#!/usr/bin/env bash
set -euo pipefail

# V64.3.54 PDRM: exact preregistered V53 branch after both pre-execution
# trajectory-state families failed identification. Reuse the metric-safe V50.5
# paired outcomes, collect only short-horizon paired realized ego response, and
# perform mediator identification. V54 emits no t=0 runtime retention config.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BDSE_ROOT="${BDSE_ROOT:-$SCRIPT_DIR}"
cd "$BDSE_ROOT"
export PYTHONPATH="$BDSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$BDSE_ROOT/outputs}"

GPU_TREAT="${GPU_TREAT:-0}"
GPU_CONTROL="${GPU_CONTROL:-1}"
V49_ROOT="${V49_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
V50_5_ROOT="${V50_5_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_50_5_eaf_icer_pior_train_2gpu_v1}"
V51_ROOT="${V51_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_51_eaf_icer_pocr_train_v1}"
V52_ROOT="${V52_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_52_eaf_icer_hodr_train_v1}"
V53_ROOT="${V53_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_53_eaf_icer_potr_train_v1}"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_54_eaf_icer_pdrm_train_v1}"
NUPLAN_ROOT="${NUPLAN_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan}"
EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/train_log.jsonl}"
WORKERS="${WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-30}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/dynamic_response_probe"

V49_SCENE="$V49_ROOT/provenance/v64_3_49_siir_train_scene_audit.csv"
V49_CAND="$V49_ROOT/provenance/v64_3_49_siir_oof_candidate_state_audit.jsonl"
TRAIN_MANIFEST="$V50_5_ROOT/provenance/v64_3_50_pior_train_manifest.json"
TREAT_CFG="$V50_5_ROOT/provenance/v64_3_50_pior_probe_treatment.yaml"
CONTROL_CFG="$V50_5_ROOT/provenance/v64_3_50_pior_probe_control.yaml"
PAIRED="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_outcomes.jsonl"
PAIR_REPORT="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_report.json"
METRIC_AUDIT="$V50_5_ROOT/provenance/v64_3_50_5_metric_safety_full_audit.json"
V51_FIT="$V51_ROOT/provenance/v64_3_51_pocr_fit.json"
V52_FIT="$V52_ROOT/provenance/v64_3_52_hodr_fit.json"
V53_FIT="$V53_ROOT/provenance/v64_3_53_potr_fit.json"
V53_PROFILES="$V53_ROOT/provenance/v64_3_53_operator_trajectory_profiles.jsonl"

for f in "$V49_SCENE" "$V49_CAND" "$TRAIN_MANIFEST" "$TREAT_CFG" "$CONTROL_CFG" "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" "$V51_FIT" "$V52_FIT" "$V53_FIT" "$V53_PROFILES"; do
  [[ -s "$f" ]] || { echo "V54 ENGINEERING STOP missing prerequisite $f" >&2; exit 2; }
done

sha256sum -c V64_3_54_SCIENCE_MANIFEST.sha256 | tee "$OUT_ROOT/logs/v64_3_54_science_manifest.out"

python - "$V53_FIT" "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" <<'PY' | tee "$OUT_ROOT/logs/v64_3_54_parent_and_evidence_lock.out"
import hashlib,json,sys
from pathlib import Path
v53,paired,report,metric=map(Path,sys.argv[1:])
want53='9174ffeac064a85bef6c1727915d93903271f9afe1770f5e5ba3e3e51efe1b6e'
wantpair='d592baa3508ae9dc084eebcbb3505d9accadc724601ab35a1253d3536af39d43'
if hashlib.sha256(v53.read_bytes()).hexdigest()!=want53: raise SystemExit('V54 ENGINEERING STOP: V53 fit hash drift')
d=json.load(open(v53)); n=d.get('nested_crossfit',{})
if d.get('train_gate_pass') is not False or n.get('failure_diagnosis')!='preexecution_operator_trajectory_contrast_does_not_identify_effectful_outcome_order':
    raise SystemExit('V54 ENGINEERING STOP: wrong V53 preregistered branch')
if hashlib.sha256(paired.read_bytes()).hexdigest()!=wantpair: raise SystemExit('V54 ENGINEERING STOP: paired outcome hash drift')
r=json.load(open(report)); m=json.load(open(metric))
if r.get('pass') is not True or int(r.get('scenario_count',-1))!=502 or m.get('pass') is not True:
    raise SystemExit('V54 ENGINEERING STOP: V50.5 paired outcome provenance invalid')
print(json.dumps({'pass':True,'v53_fit_sha256':want53,'paired_sha256':wantpair,'branch':n['failure_diagnosis']},sort_keys=True))
PY

python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_54_eaf_icer_pdrm.py \
  bdse/tests/test_v64_3_53_eaf_icer_potr.py \
  bdse/tests/test_v64_3_52_eaf_icer_hodr.py \
  bdse/tests/test_v64_3_51_eaf_icer_pocr.py \
  bdse/tests/test_v64_3_50_6_pior_fit_repair.py \
  bdse/tests/test_v64_3_50_eaf_icer_pior.py \
  bdse/tests/test_v64_3_50_5_pior_metric_safety.py \
  bdse/tests/test_v64_3_48_eaf_icer_ocrr.py \
  | tee "$OUT_ROOT/logs/v64_3_54_targeted_regression.out"

if [[ -z "${EAF_CKPT:-}" ]]; then
  [[ -s "$EAF_TRAIN_LOG" ]] || { echo "V54 ENGINEERING STOP missing EAF_TRAIN_LOG=$EAF_TRAIN_LOG; set EAF_CKPT explicitly" >&2; exit 2; }
  python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
    --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_54_PDRM \
    --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_54.json" \
    > "$OUT_ROOT/logs/v64_3_13_reaudit_v64_3_54.out"
  SELECTED_EPOCH=$(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_54.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))['selected_epoch']))
PY
)
  EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"
fi
export EAF_CKPT
[[ -s "$EAF_CKPT" ]] || { echo "V54 ENGINEERING STOP missing checkpoint $EAF_CKPT" >&2; exit 2; }

DYN_JSONL="$OUT_ROOT/provenance/v64_3_54_paired_dynamic_response_profiles.jsonl"
DYN_REPORT="$OUT_ROOT/provenance/v64_3_54_dynamic_response_probe_report.json"
if [[ -s "$DYN_JSONL" && -s "$DYN_REPORT" && "${V54_FORCE_DYNAMIC_REPLAY:-0}" != "1" ]]; then
  python - "$DYN_JSONL" "$DYN_REPORT" <<'PY' | tee "$OUT_ROOT/logs/v64_3_54_dynamic_response_reuse.out"
import json,sys
p,r=sys.argv[1:]; rows=[json.loads(x) for x in open(p) if x.strip()]; d=json.load(open(r))
if len(rows)!=502 or len({x['scenario_token'] for x in rows})!=502 or d.get('pass') is not True or d.get('paired_outcome_labels_recollected') is not False:
    raise SystemExit('V54 ENGINEERING STOP: existing paired dynamic response replay invalid')
print('PASS V54: reusing completed short-horizon paired dynamic response replay')
PY
else
  rm -rf "$OUT_ROOT/dynamic_response_probe"; mkdir -p "$OUT_ROOT/dynamic_response_probe"
  python -m bdse.tools.run_v64_3_54_dynamic_response_probe \
    --manifest "$TRAIN_MANIFEST" --treatment-config "$TREAT_CFG" --control-config "$CONTROL_CFG" \
    --checkpoint "$EAF_CKPT" --nuplan-root "$NUPLAN_ROOT" --v53-operator-profiles "$V53_PROFILES" \
    --output-root "$OUT_ROOT/dynamic_response_probe" --output-profiles "$DYN_JSONL" --output-report "$DYN_REPORT" \
    --gpu-treatment "$GPU_TREAT" --gpu-control "$GPU_CONTROL" --workers "$WORKERS" --batch-size "$BATCH_SIZE" \
    --heartbeat-seconds "$HEARTBEAT_SECONDS" --resume \
    | tee "$OUT_ROOT/logs/v64_3_54_dynamic_response_probe.out"
fi

FIT_REPORT="$OUT_ROOT/provenance/v64_3_54_pdrm_fit.json"
ANALYSIS_TABLE="$OUT_ROOT/provenance/v64_3_54_analysis_population_snapshot.jsonl"
set +e
python -m bdse.tools.fit_v64_3_54_eaf_icer_pdrm \
  --v49-candidate-audit "$V49_CAND" --v49-scene-audit "$V49_SCENE" --paired-outcomes "$PAIRED" --v50-5-root "$V50_5_ROOT" \
  --v51-fit-report "$V51_FIT" --v52-fit-report "$V52_FIT" --v53-fit-report "$V53_FIT" --dynamic-profiles "$DYN_JSONL" \
  --output-report "$FIT_REPORT" --output-analysis-table "$ANALYSIS_TABLE" 2>&1 | tee "$OUT_ROOT/logs/v64_3_54_pdrm_fit.out"
FIT_STATUS=${PIPESTATUS[0]}
set -e

cp "$V53_FIT" "$OUT_ROOT/provenance/v64_3_53_potr_fit_reused.json"
cp "$DYN_REPORT" "$OUT_ROOT/provenance/v64_3_54_dynamic_response_probe_report_reused.json"
sha256sum "$PAIRED" "$V51_FIT" "$V52_FIT" "$V53_FIT" "$V53_PROFILES" "$DYN_JSONL" > "$OUT_ROOT/provenance/V64_3_54_REUSED_EVIDENCE_AND_STATE_SHA256SUMS.txt"

if [[ $FIT_STATUS -ne 0 ]]; then
  echo "V64.3.54 PDRM mediator scientific STOP. Do not consume untouched validation; follow v64_3_54_pdrm_fit.json." >&2
  exit "$FIT_STATUS"
fi

echo "V64.3.54 PDRM mediator identification PASS. This is not yet a t=0 deployable policy: do not run untouched validation. Follow the preregistered next branch in v64_3_54_pdrm_fit.json."
