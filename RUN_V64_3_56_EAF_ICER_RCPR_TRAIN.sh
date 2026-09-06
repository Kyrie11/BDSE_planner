#!/usr/bin/env bash
set -euo pipefail

# V64.3.56 RCPR — final preregistered internal state-family test.
# Branch A collects/uses the realized treatment-control constraint process over
# the exact V54 first-replan window while freezing V55's Pareto functional.
# Branch B is fit only if Branch A fully closes identification + deployment and
# distills the same mediator from t0-available frozen V53 planned geometry.
# No final closed-loop outcome label or nuPlan metric is recollected.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BDSE_ROOT="${BDSE_ROOT:-$SCRIPT_DIR}"
cd "$BDSE_ROOT"
export PYTHONPATH="$BDSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$BDSE_ROOT/outputs}"

GPU_TREAT="${GPU_TREAT:-0}"
GPU_CONTROL="${GPU_CONTROL:-1}"
V49_ROOT="${V49_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
V50_5_ROOT="${V50_5_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_50_5_eaf_icer_pior_train_2gpu_v1}"
V52_ROOT="${V52_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_52_eaf_icer_hodr_train_v1}"
V53_ROOT="${V53_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_53_eaf_icer_potr_train_v1}"
V54_ROOT="${V54_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_54_eaf_icer_pdrm_train_v1}"
V55_ROOT="${V55_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_55_eaf_icer_dmor_train_v1}"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_56_eaf_icer_rcpr_train_v1}"
NUPLAN_ROOT="${NUPLAN_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan}"
EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/train_log.jsonl}"
WORKERS="${WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-30}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/constraint_process_probe"

V49_SCENE="$V49_ROOT/provenance/v64_3_49_siir_train_scene_audit.csv"
V49_CAND="$V49_ROOT/provenance/v64_3_49_siir_oof_candidate_state_audit.jsonl"
TRAIN_MANIFEST="$V50_5_ROOT/provenance/v64_3_50_pior_train_manifest.json"
TREAT_CFG="$V50_5_ROOT/provenance/v64_3_50_pior_probe_treatment.yaml"
CONTROL_CFG="$V50_5_ROOT/provenance/v64_3_50_pior_probe_control.yaml"
PAIRED="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_outcomes.jsonl"
PAIR_REPORT="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_report.json"
METRIC_AUDIT="$V50_5_ROOT/provenance/v64_3_50_5_metric_safety_full_audit.json"
V52_FIT="$V52_ROOT/provenance/v64_3_52_hodr_fit.json"
V53_PROFILES="$V53_ROOT/provenance/v64_3_53_operator_trajectory_profiles.jsonl"
V54_FIT="$V54_ROOT/provenance/v64_3_54_pdrm_fit.json"
V54_DYNAMIC="$V54_ROOT/provenance/v64_3_54_paired_dynamic_response_profiles.jsonl"
V55_FIT="$V55_ROOT/provenance/v64_3_55_dmor_fit.json"

for f in "$V49_SCENE" "$V49_CAND" "$TRAIN_MANIFEST" "$TREAT_CFG" "$CONTROL_CFG" "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" "$V52_FIT" "$V53_PROFILES" "$V54_FIT" "$V54_DYNAMIC" "$V55_FIT"; do
  [[ -s "$f" ]] || { echo "V56 ENGINEERING STOP missing prerequisite $f" >&2; exit 2; }
done

sha256sum -c V64_3_56_SCIENCE_MANIFEST.sha256 | tee "$OUT_ROOT/logs/v64_3_56_science_manifest.out"

python - "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" "$V52_FIT" "$V53_PROFILES" "$V54_FIT" "$V54_DYNAMIC" "$V55_FIT" <<'PY' | tee "$OUT_ROOT/logs/v64_3_56_parent_and_evidence_lock.out"
import hashlib,json,sys
from pathlib import Path
paired,report,metric,v52,v53p,v54,dyn,v55=map(Path,sys.argv[1:])
want={
 'paired':'d592baa3508ae9dc084eebcbb3505d9accadc724601ab35a1253d3536af39d43',
 'v52':'7a21fead5383ebd6aafaeb5da586346a77e5707050cb359511a06971b742a16b',
 'v53p':'9a69c196a1d76e9c5d424068df223ec26f0e481252f25f67d5bb17fd355aaef6',
 'v54':'10f3e60c82bb8b82f1f688e866a27008e1498b67d2e194b0c7aadec5368536d8',
 'dyn':'dd2bdd809a757ce74973d7ce2c3189fad60dc0d3e0125d7fcec9ca7ad1bda373',
 'v55':'cf7d91b9cf20d62978e766e6b8c739eee75e00011f5a208c2892af419e56dc88',
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for key,p in [('paired',paired),('v52',v52),('v53p',v53p),('v54',v54),('dyn',dyn),('v55',v55)]:
    got=sha(p)
    if got!=want[key]: raise SystemExit(f'V56 ENGINEERING STOP: {key} hash drift {got}')
r=json.load(open(report)); m=json.load(open(metric))
if r.get('pass') is not True or int(r.get('scenario_count',-1))!=502 or m.get('pass') is not True:
    raise SystemExit('V56 ENGINEERING STOP: V50.5 paired evidence provenance invalid')
d=json.load(open(v55)); n=d.get('nested_crossfit',{}); a=n.get('arms',{}).get('realized_dominance',{}); b=n.get('arms',{}).get('predicted_dominance',{})
if d.get('train_gate_pass') is not False or n.get('failure_diagnosis')!='realized_mediator_plus_static_pareto_functional_still_deployment_insufficient':
    raise SystemExit('V56 ENGINEERING STOP: wrong V55 preregistered branch')
if a.get('identification',{}).get('functional_identified') is not True or a.get('deployment_gate',{}).get('pass') is not False:
    raise SystemExit('V56 ENGINEERING STOP: V55 realized-dominance signature drift')
if b.get('status')!='NOT_EVALUATED_BY_PREREGISTERED_BRANCH_ORDER':
    raise SystemExit('V56 ENGINEERING STOP: V55 predicted branch was unexpectedly consumed')
print(json.dumps({'pass':True,**want,'v55_branch':n['failure_diagnosis']},sort_keys=True))
PY

python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_56_eaf_icer_rcpr.py \
  bdse/tests/test_v64_3_55_eaf_icer_dmor.py \
  bdse/tests/test_v64_3_54_eaf_icer_pdrm.py \
  bdse/tests/test_v64_3_53_eaf_icer_potr.py \
  bdse/tests/test_v64_3_52_eaf_icer_hodr.py \
  bdse/tests/test_v64_3_51_eaf_icer_pocr.py \
  bdse/tests/test_v64_3_50_eaf_icer_pior.py \
  bdse/tests/test_v64_3_50_5_pior_metric_safety.py \
  bdse/tests/test_v64_3_48_eaf_icer_ocrr.py \
  | tee "$OUT_ROOT/logs/v64_3_56_targeted_regression.out"

if [[ -z "${EAF_CKPT:-}" ]]; then
  [[ -s "$EAF_TRAIN_LOG" ]] || { echo "V56 ENGINEERING STOP missing EAF_TRAIN_LOG=$EAF_TRAIN_LOG; set EAF_CKPT explicitly" >&2; exit 2; }
  python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
    --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_56_RCPR \
    --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_56.json" \
    > "$OUT_ROOT/logs/v64_3_13_reaudit_v64_3_56.out"
  SELECTED_EPOCH=$(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_56.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))['selected_epoch']))
PY
)
  EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"
fi
export EAF_CKPT
[[ -s "$EAF_CKPT" ]] || { echo "V56 ENGINEERING STOP missing checkpoint $EAF_CKPT" >&2; exit 2; }

PROC_JSONL="$OUT_ROOT/provenance/v64_3_56_paired_constraint_process_profiles.jsonl"
PROC_REPORT="$OUT_ROOT/provenance/v64_3_56_constraint_process_probe_report.json"
if [[ -s "$PROC_JSONL" && -s "$PROC_REPORT" && "${V56_FORCE_PROCESS_REPLAY:-0}" != "1" ]]; then
  python - "$PROC_JSONL" "$PROC_REPORT" "$V53_PROFILES" <<'PY' | tee "$OUT_ROOT/logs/v64_3_56_constraint_process_reuse.out"
import hashlib,json,sys
p,r,v53=sys.argv[1:]
rows=[json.loads(x) for x in open(p) if x.strip()]; d=json.load(open(r)); planned={}
for x in open(v53):
    if x.strip():
        q=json.loads(x); planned[str(q['scenario_token'])]=float(q['execution_contrast_linf'])
if len(rows)!=502 or len({x['scenario_token'] for x in rows})!=502 or d.get('pass') is not True or d.get('paired_outcome_labels_recollected') is not False:
    raise SystemExit('V56 ENGINEERING STOP: existing constraint-process replay invalid')
if hashlib.sha256(open(p,'rb').read()).hexdigest()!=d.get('profile_sha256') or int(d.get('planned_physical_equal_count',-1))!=38:
    raise SystemExit('V56 ENGINEERING STOP: existing constraint-process provenance/signature invalid')
for x in rows:
    tok=str(x['scenario_token']); vals=[abs(float(v)) for v in x['constraint_support_delta_process']]
    if planned[tok] <= 1e-10 and max(vals,default=0.0)>1e-6:
        raise SystemExit(f'V56 ENGINEERING STOP: planned-equal process diverged {tok}')
print('PASS V56: reusing completed short-horizon paired constraint-process replay')
PY
else
  rm -rf "$OUT_ROOT/constraint_process_probe"; mkdir -p "$OUT_ROOT/constraint_process_probe"
  python -m bdse.tools.run_v64_3_56_constraint_process_probe \
    --manifest "$TRAIN_MANIFEST" --treatment-config "$TREAT_CFG" --control-config "$CONTROL_CFG" \
    --checkpoint "$EAF_CKPT" --nuplan-root "$NUPLAN_ROOT" --v53-operator-profiles "$V53_PROFILES" \
    --output-root "$OUT_ROOT/constraint_process_probe" --output-profiles "$PROC_JSONL" --output-report "$PROC_REPORT" \
    --gpu-treatment "$GPU_TREAT" --gpu-control "$GPU_CONTROL" --workers "$WORKERS" --batch-size "$BATCH_SIZE" \
    --heartbeat-seconds "$HEARTBEAT_SECONDS" --resume \
    | tee "$OUT_ROOT/logs/v64_3_56_constraint_process_probe.out"
fi

FIT_REPORT="$OUT_ROOT/provenance/v64_3_56_rcpr_fit.json"
RUNTIME_ARTIFACT="$OUT_ROOT/provenance/v64_3_56_rcpr_runtime_artifact.json"
set +e
python -m bdse.tools.fit_v64_3_56_eaf_icer_rcpr \
  --v49-candidate-audit "$V49_CAND" --v49-scene-audit "$V49_SCENE" \
  --paired-outcomes "$PAIRED" --v50-5-root "$V50_5_ROOT" --v52-fit-report "$V52_FIT" \
  --v53-operator-profiles "$V53_PROFILES" --v54-fit-report "$V54_FIT" --v54-dynamic-profiles "$V54_DYNAMIC" \
  --v55-fit-report "$V55_FIT" --constraint-profiles "$PROC_JSONL" \
  --output-report "$FIT_REPORT" --output-runtime-artifact "$RUNTIME_ARTIFACT" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_56_rcpr_fit.out"
FIT_STATUS=${PIPESTATUS[0]}
set -e

sha256sum "$PAIRED" "$V52_FIT" "$V53_PROFILES" "$V54_FIT" "$V54_DYNAMIC" "$V55_FIT" "$PROC_JSONL" > "$OUT_ROOT/provenance/V64_3_56_REUSED_EVIDENCE_AND_STATE_SHA256SUMS.txt"

if [[ $FIT_STATUS -eq 3 ]]; then
  echo "V64.3.56 RCPR preregistered scientific STOP. Internal algorithm search has converged by the V56 stopping protocol; do not tune another state/feature family." >&2
  exit 3
elif [[ $FIT_STATUS -ne 0 ]]; then
  echo "V64.3.56 RCPR ENGINEERING STOP during fit (status=$FIT_STATUS). Do not perform algorithm attribution; repair engineering only." >&2
  exit "$FIT_STATUS"
fi

echo "V64.3.56 RCPR deployable TRAIN gate PASS. FREEZE immediately: no more TRAIN algorithm tuning. Next: engineering-only runtime integration, untouched paired validation, then external baselines/official benchmarking."
