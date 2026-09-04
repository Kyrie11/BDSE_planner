#!/usr/bin/env bash
set -euo pipefail

# V64.3.53 POTR: preregistered V52 support-identified / conditional-order-failed branch.
# One treatment-only state replay acquires pre-execution proposal-vs-incumbent
# trajectory contrast. The V50.5 502/502 paired outcome labels are reused exactly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BDSE_ROOT="${BDSE_ROOT:-$SCRIPT_DIR}"
cd "$BDSE_ROOT"
export PYTHONPATH="$BDSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$BDSE_ROOT/outputs}"

GPU0="${GPU0:-0}"
V49_ROOT="${V49_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
V50_5_ROOT="${V50_5_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_50_5_eaf_icer_pior_train_2gpu_v1}"
V51_ROOT="${V51_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_51_eaf_icer_pocr_train_v1}"
V52_ROOT="${V52_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_52_eaf_icer_hodr_train_v1}"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_53_eaf_icer_potr_train_v1}"
NUPLAN_ROOT="${NUPLAN_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan}"
EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/train_log.jsonl}"
WORKERS="${WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-30}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/operator_profile_probe"

V49_FIT="$V49_ROOT/provenance/v64_3_49_siir_fit.json"
V49_SCENE="$V49_ROOT/provenance/v64_3_49_siir_train_scene_audit.csv"
V49_CAND="$V49_ROOT/provenance/v64_3_49_siir_oof_candidate_state_audit.jsonl"
V49_CONFIG="$V49_ROOT/provenance/v64_3_49_siir.yaml"
TRAIN_MANIFEST="$V50_5_ROOT/provenance/v64_3_50_pior_train_manifest.json"
TREAT_CFG="$V50_5_ROOT/provenance/v64_3_50_pior_probe_treatment.yaml"
PAIRED="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_outcomes.jsonl"
PAIR_REPORT="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_report.json"
METRIC_AUDIT="$V50_5_ROOT/provenance/v64_3_50_5_metric_safety_full_audit.json"
V51_FIT="$V51_ROOT/provenance/v64_3_51_pocr_fit.json"
V52_FIT="$V52_ROOT/provenance/v64_3_52_hodr_fit.json"

for f in "$V49_FIT" "$V49_SCENE" "$V49_CAND" "$V49_CONFIG" "$TRAIN_MANIFEST" "$TREAT_CFG" "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" "$V51_FIT" "$V52_FIT"; do
  [[ -s "$f" ]] || { echo "V53 ENGINEERING STOP missing prerequisite $f" >&2; exit 2; }
done

sha256sum -c V64_3_53_SCIENCE_MANIFEST.sha256 | tee "$OUT_ROOT/logs/v64_3_53_science_manifest.out"

python - "$V52_FIT" "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" <<'PY' | tee "$OUT_ROOT/logs/v64_3_53_parent_and_evidence_lock.out"
import hashlib,json,sys
from pathlib import Path
v52,paired,report,metric=map(Path,sys.argv[1:])
want_v52='7a21fead5383ebd6aafaeb5da586346a77e5707050cb359511a06971b742a16b'
want_pair='d592baa3508ae9dc084eebcbb3505d9accadc724601ab35a1253d3536af39d43'
if hashlib.sha256(v52.read_bytes()).hexdigest()!=want_v52: raise SystemExit('V53 ENGINEERING STOP: V52 fit hash drift')
d=json.load(open(v52)); n=d.get('nested_crossfit',{})
if d.get('train_gate_pass') is not False or n.get('failure_diagnosis')!='effect_support_identified_but_operator_state_does_not_identify_conditional_outcome_order':
    raise SystemExit('V53 ENGINEERING STOP: wrong V52 preregistered branch')
for a in ('hurdle_sign','hurdle_pareto'):
    ri=n.get('arms',{}).get(a,{}).get('identification',{})
    if ri.get('support_identified') is not True or ri.get('functional_identified') is not False:
        raise SystemExit(f'V53 ENGINEERING STOP: V52 {a} signature drift')
if hashlib.sha256(paired.read_bytes()).hexdigest()!=want_pair: raise SystemExit('V53 ENGINEERING STOP: paired outcome hash drift')
r=json.load(open(report)); m=json.load(open(metric))
if r.get('pass') is not True or int(r.get('scenario_count',-1))!=502 or m.get('pass') is not True:
    raise SystemExit('V53 ENGINEERING STOP: paired evidence/metric-safe audit invalid')
print(json.dumps({'pass':True,'v52_fit_sha256':want_v52,'paired_sha256':want_pair,'branch':n['failure_diagnosis']},sort_keys=True))
PY

python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_53_eaf_icer_potr.py \
  bdse/tests/test_v64_3_52_eaf_icer_hodr.py \
  bdse/tests/test_v64_3_51_eaf_icer_pocr.py \
  bdse/tests/test_v64_3_50_6_pior_fit_repair.py \
  bdse/tests/test_v64_3_50_eaf_icer_pior.py \
  bdse/tests/test_v64_3_50_5_pior_metric_safety.py \
  bdse/tests/test_v64_3_48_eaf_icer_ocrr.py \
  | tee "$OUT_ROOT/logs/v64_3_53_targeted_regression.out"

if [[ -z "${EAF_CKPT:-}" ]]; then
  [[ -s "$EAF_TRAIN_LOG" ]] || { echo "V53 ENGINEERING STOP missing EAF_TRAIN_LOG=$EAF_TRAIN_LOG; set EAF_CKPT explicitly" >&2; exit 2; }
  python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
    --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_53_POTR \
    --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_53.json" \
    > "$OUT_ROOT/logs/v64_3_13_reaudit_v64_3_53.out"
  SELECTED_EPOCH=$(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_53.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print(int(r['selected_epoch']))
PY
)
  EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"
fi
export EAF_CKPT
[[ -s "$EAF_CKPT" ]] || { echo "V53 ENGINEERING STOP missing checkpoint $EAF_CKPT" >&2; exit 2; }

PROFILE_JSONL="$OUT_ROOT/provenance/v64_3_53_operator_trajectory_profiles.jsonl"
PROFILE_REPORT="$OUT_ROOT/provenance/v64_3_53_operator_profile_probe_report.json"
if [[ -s "$PROFILE_JSONL" && -s "$PROFILE_REPORT" && "${V53_FORCE_PROFILE_REPLAY:-0}" != "1" ]]; then
  python - "$PROFILE_JSONL" "$PROFILE_REPORT" <<'PY' | tee "$OUT_ROOT/logs/v64_3_53_operator_profile_reuse.out"
import json,sys
p,r=sys.argv[1:]; rows=[json.loads(x) for x in open(p) if x.strip()]; d=json.load(open(r))
if len(rows)!=502 or len({x['scenario_token'] for x in rows})!=502 or d.get('pass') is not True or d.get('scalar_D_exact_replay_all') is not True:
    raise SystemExit('V53 ENGINEERING STOP: existing state-only profile is not a valid 502/502 replay')
print('PASS V53: reusing completed state-only profile replay')
PY
else
  rm -rf "$OUT_ROOT/operator_profile_probe"
  mkdir -p "$OUT_ROOT/operator_profile_probe"
  python -m bdse.tools.run_v64_3_53_operator_profile_probe \
    --manifest "$TRAIN_MANIFEST" --treatment-config "$TREAT_CFG" --checkpoint "$EAF_CKPT" \
    --nuplan-root "$NUPLAN_ROOT" --challenge closed_loop_nonreactive_agents \
    --output-root "$OUT_ROOT/operator_profile_probe" --output-profiles "$PROFILE_JSONL" --output-report "$PROFILE_REPORT" \
    --gpu "$GPU0" --workers "$WORKERS" --batch-size "$BATCH_SIZE" --heartbeat-seconds "$HEARTBEAT_SECONDS" \
    | tee "$OUT_ROOT/logs/v64_3_53_operator_profile_probe.out"
fi

FIT_REPORT="$OUT_ROOT/provenance/v64_3_53_potr_fit.json"
POTR_CONFIG="$OUT_ROOT/provenance/v64_3_53_potr.yaml"
set +e
python -m bdse.tools.fit_v64_3_53_eaf_icer_potr \
  --v49-fit-report "$V49_FIT" --v49-candidate-audit "$V49_CAND" --v49-scene-audit "$V49_SCENE" --v49-siir-config "$V49_CONFIG" \
  --paired-outcomes "$PAIRED" --v50-5-root "$V50_5_ROOT" --v51-fit-report "$V51_FIT" --v52-fit-report "$V52_FIT" \
  --operator-profiles "$PROFILE_JSONL" --output-config "$POTR_CONFIG" --output-report "$FIT_REPORT" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_53_potr_fit.out"
FIT_STATUS=${PIPESTATUS[0]}
set -e

cp "$V52_FIT" "$OUT_ROOT/provenance/v64_3_52_hodr_fit_reused.json"
cp "$METRIC_AUDIT" "$OUT_ROOT/provenance/v64_3_50_5_metric_safety_full_audit_reused.json"
sha256sum "$PAIRED" "$V51_FIT" "$V52_FIT" "$PROFILE_JSONL" "$PROFILE_REPORT" > "$OUT_ROOT/provenance/V64_3_53_REUSED_EVIDENCE_AND_STATE_SHA256SUMS.txt"

if [[ $FIT_STATUS -ne 0 ]]; then
  echo "V64.3.53 POTR scientific STOP. Do not consume untouched validation; follow v64_3_53_potr_fit.json." >&2
  exit "$FIT_STATUS"
fi

echo "V64.3.53 POTR TRAIN PASS. Freeze the preferred state immediately; do not tune with TRAIN or consume validation except under the frozen next-stage protocol."
