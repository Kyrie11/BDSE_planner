#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export V47_ROOT="${V47_ROOT:-outputs_v64_3_47_eaf_icer_fsfr_screen_2gpu_v1}"
export V49_ROOT="${V49_ROOT:-outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_50_eaf_icer_sior_screen_2gpu_v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"
export NUPLAN_ROOT="${NUPLAN_ROOT:-/data0/senzeyu2/dataset/nuplan}"
export NUPLAN_MAP_ROOT="${NUPLAN_MAP_ROOT:-$/data0/senzeyu2/dataset/CapPlan/data/nuplan/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-$NUPLAN_ROOT/exp}"
export NUPLAN_DB_ROOT="${NUPLAN_DB_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan/nuplan-v1.1/splits/}"
export CL_CHALLENGE="${CL_CHALLENGE:-closed_loop_reactive_agents}"
export PYTHONUNBUFFERED=1

V47_RSMR="$V47_ROOT/provenance/v64_3_47_rsmr.yaml"
V49_FIT="$V49_ROOT/provenance/v64_3_49_siir_fit.json"
V49_SCENE="$V49_ROOT/provenance/v64_3_49_siir_train_scene_audit.csv"
V49_CAND="$V49_ROOT/provenance/v64_3_49_siir_oof_candidate_state_audit.jsonl"
V49_CFG="$V49_ROOT/provenance/v64_3_49_siir.yaml"
V49_SERVER_TEST="$V49_ROOT/logs/targeted_regression.out"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/paired_train"
TIMING="$OUT_ROOT/provenance/v64_3_50_stage_timing.tsv"; :>"$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >>"$TIMING"; }

for f in V64_3_50_SOURCE_MANIFEST.sha256 "$EAF_TRAIN_LOG" "$V47_RSMR" "$V49_FIT" "$V49_SCENE" "$V49_CAND" "$V49_CFG" "$V49_SERVER_TEST"; do
  [[ -s "$f" ]] || { echo "STOP V50 missing $f" >&2; exit 2; }
done
[[ -d "$NUPLAN_ROOT" && -d "$NUPLAN_MAP_ROOT" && -d "$NUPLAN_DB_ROOT" ]] || { echo 'STOP V50 missing nuPlan root/maps/TRAIN DB root; override NUPLAN_ROOT/NUPLAN_MAP_ROOT/NUPLAN_DB_ROOT' >&2; exit 2; }

stage_start source_and_v49_science_lock
sha256sum -c V64_3_50_SOURCE_MANIFEST.sha256 | tee "$OUT_ROOT/logs/v64_3_50_source_manifest.out"
python - "$V49_FIT" "$V49_SCENE" "$V49_CAND" "$V49_CFG" "$V49_SERVER_TEST" <<'PY'
import hashlib,json,sys
from pathlib import Path
paths=list(map(Path,sys.argv[1:]))
expected=[
'805c4f8088051413edeb568623bc6d225d1b3c301c52612f89109216b38be296',
'7243d1b4a07094ca691b4e0ba13ebd0a8ec1ad08d4da0b512a3714e1a9b5cd80',
'e51b8d38ec262a15a0b969e40e130a900984e9435d1cd2ae9c3d23bfd23f8f6c',
'bde69241cb44da2e605a0e9e3897d024df7a1724868ba15a2e25cdc238dee405',
'403dd6d1fff0ad36eefc65a2afc4120c3f7dd72c30f0803baa663b068819cab0',
]
for p,h in zip(paths,expected):
    g=hashlib.sha256(p.read_bytes()).hexdigest()
    if g!=h: raise SystemExit(f'STOP V50: V49 result science lock changed: {p} {g}')
r=json.loads(paths[0].read_text()); n=r.get('nested_crossfit',{})
if n.get('train_gate_pass') is not False or n.get('failure_diagnosis')!='selection_interventional_risk_does_not_outperform_observational_selected_risk_close_current_offline_selected_risk_family':
    raise SystemExit('STOP V50: V49 preregistered TRAIN failure signature changed')
ident=n.get('risk_identification',{})
if abs(float(ident.get('siir',{}).get('aggregate_nonpositive_risk_auc',-1))-0.6081222524597028)>1e-12:
    raise SystemExit('STOP V50: V49 SIIR AUC signature changed')
scene=paths[1].read_text().splitlines(); cand=paths[2].read_text().splitlines(); tests=paths[4].read_text(errors='replace')
if len(scene)!=783 or len(cand)!=782: raise SystemExit(f'STOP V50: V49 audit cardinality changed scene={len(scene)-1} candidate={len(cand)}')
if '242 passed' not in tests: raise SystemExit('STOP V50: V49 server targeted regression signature changed')
print('PASS V50 prerequisite: exact uploaded V49 scientific failure + 782-scene audits + 242-pass server regression locked')
PY
stage_end

stage_start prerequisites_and_regression
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_50_SIOR --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_50.json" > "$OUT_ROOT/logs/v64_3_13_reaudit_v64_3_50.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_50.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False): raise SystemExit('STOP V50 EAF prerequisite')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"; fi
export EAF_CKPT; [[ -s "$EAF_CKPT" ]] || { echo "STOP V50 missing EAF checkpoint: $EAF_CKPT" >&2; exit 2; }
python -m compileall -q bdse
pytest -q bdse/tests/test_v64_3_{13_eaf_dmvr,14_eaf_ocfi,15_eaf_eair,16_eaf_raer,17_eaf_daler,18_eaf_dacer,19_eaf_icer,20_eaf_icer_dc,21_eaf_icer_mcr,22_eaf_icer_tcr,23_eaf_icer_rcr,24_eaf_icer_arc,25_eaf_icer_drc,26_eaf_icer_sarc,27_eaf_icer_trcc,28_eaf_icer_ptmc,29_eaf_icer_fcr,30_2_eaf_icer_fbic_pure,30_3_eaf_icer_fbic_pure_auditfix,30_eaf_icer_fbic,31_eaf_icer_scir,32_1_eaf_icer_ssir_weightfix,32_eaf_icer_ssir,33_eaf_icer_spcr,34_eaf_icer_rsmr,35_eaf_icer_fbcsr,36_eaf_icer_sgrr,37_eaf_icer_pvr,38_eaf_icer_davr,39_eaf_icer_cfsr,40_eaf_icer_sdfr,41_eaf_icer_epvr,42_eaf_icer_ovdr,43_eaf_icer_cfrv,44_eaf_icer_pcor,45_eaf_icer_pirf,46_eaf_icer_dirp,47_eaf_icer_fsfr,48_eaf_icer_ocrr,49_eaf_icer_siir,50_eaf_icer_sior}.py bdse/tests/test_v64_3_48_2_eaf_icer_ocrr_provenance_repair.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

stage_start freeze_full_set_rsmr_train_population
TRAIN_TOKENS="$OUT_ROOT/provenance/v64_3_50_frozen_rsmr_selected_train502_tokens.txt"
CONTROL_CFG="$OUT_ROOT/provenance/v64_3_50_sior_probe_control.yaml"
TREATMENT_CFG="$OUT_ROOT/provenance/v64_3_50_sior_probe_treatment.yaml"
python -m bdse.tools.select_v64_3_50_sior_train_tokens --v49-scene-audit "$V49_SCENE" --output "$TRAIN_TOKENS" | tee "$OUT_ROOT/logs/v64_3_50_freeze_train_population.out"
python -m bdse.tools.prepare_v64_3_50_eaf_icer_sior_probe_configs --rsmr-config "$V47_RSMR" --control-output "$CONTROL_CFG" --treatment-output "$TREATMENT_CFG" | tee "$OUT_ROOT/logs/v64_3_50_prepare_probe_configs.out"
stage_end

stage_start paired_closed_loop_selected_outcome_collection
python -m bdse.tools.run_v64_3_50_paired_selected_outcome_collection \
  --control-config "$CONTROL_CFG" --treatment-config "$TREATMENT_CFG" --checkpoint "$EAF_CKPT" \
  --scenario-token-file "$TRAIN_TOKENS" --output-root "$OUT_ROOT/paired_train" --gpus "$GPU0,$GPU1" \
  --challenge "$CL_CHALLENGE" --nuplan-data-root "$NUPLAN_ROOT" --nuplan-map-root "$NUPLAN_MAP_ROOT" \
  --nuplan-exp-root "$NUPLAN_EXP_ROOT" --nuplan-db-root "$NUPLAN_DB_ROOT" --resume \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_50_paired_closed_loop_collection.out"
stage_end

stage_start nested_train_selected_outcome_gate
SIOR_CFG="$OUT_ROOT/provenance/v64_3_50_sior.yaml"
FIT_REPORT="$OUT_ROOT/provenance/v64_3_50_sior_fit.json"
SCENE_AUDIT="$OUT_ROOT/provenance/v64_3_50_sior_train_scene_audit.csv"
set +e
python -m bdse.tools.fit_v64_3_50_eaf_icer_sior \
  --v49-fit-report "$V49_FIT" --v49-scene-audit "$V49_SCENE" --v49-candidate-audit "$V49_CAND" \
  --paired-outcomes "$OUT_ROOT/paired_train/paired_selected_outcomes.csv" --v49-siir-config "$V49_CFG" \
  --output-config "$SIOR_CFG" --output-report "$FIT_REPORT" --output-scene-audit "$SCENE_AUDIT" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_50_sior_fit.out"
FIT_STATUS=${PIPESTATUS[0]}; set -e; stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >>"$TIMING"
[[ $FIT_STATUS -eq 0 ]] || exit "$FIT_STATUS"

echo 'PASS V64.3.50 SIOR paired closed-loop TRAIN gate.'
echo 'SCIENTIFIC STOP HERE: V50 launcher deliberately does not select or consume fresh scenes.'
echo 'Freeze this fit first; only then run the separately preregistered untouched paired closed-loop A500 and B500 protocol.'
