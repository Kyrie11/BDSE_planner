#!/usr/bin/env bash
set -euo pipefail

# V64.3.50 EAF-ICER-PIOR
# Paired Interventional Outcome Retention.
# Scientific rule inherited from V49 preregistration:
#   V49 nested TRAIN failure closes the offline selected-risk family.
#   V50 may change the supervision/evidence source, but must not tune the
#   offline feature/loss/threshold family.  We therefore collect paired one-shot
#   closed-loop outcomes for the *actual frozen full-set RSMR proposal* versus
#   the same incumbent, then fit the unchanged low-capacity Q/P/E risk law.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
OUT_ROOT="${OUT_ROOT:-../outputs_v64_3_50_1_eaf_icer_pior_train_2gpu_v1}"
V49_ROOT="${V49_ROOT:-../outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
NUPLAN_ROOT="${NUPLAN_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan}"
NUPLAN_DB_SPLIT_ROOT="${NUPLAN_DB_SPLIT_ROOT:-$NUPLAN_ROOT/nuplan-v1.1/splits}"
EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-../outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/train_log.jsonl}"
WORKERS_PER_ARM="${WORKERS_PER_ARM:-4}"
# Engineering-only speed/resume knobs. They do not change the 502-scene
# population, full-set RSMR proposal, one-shot treatment/control action, metric
# definition, or PIOR fitting gate.
PIOR_BATCH_SIZE="${PIOR_BATCH_SIZE:-64}"
PIOR_FIRST_BATCH_SIZE="${PIOR_FIRST_BATCH_SIZE:-4}"
PIOR_HEARTBEAT_SECONDS="${PIOR_HEARTBEAT_SECONDS:-30}"
PIOR_RESUME="${PIOR_RESUME:-1}"
PIOR_SERIALIZE_GPU_INFERENCE="${PIOR_SERIALIZE_GPU_INFERENCE:-0}"
PIOR_PROFILE_CLOSED_LOOP="${PIOR_PROFILE_CLOSED_LOOP:-1}"

V49_FIT="$V49_ROOT/provenance/v64_3_49_siir_fit.json"
V49_SCENE="$V49_ROOT/provenance/v64_3_49_siir_train_scene_audit.csv"
V49_CAND="$V49_ROOT/provenance/v64_3_49_siir_oof_candidate_state_audit.jsonl"
V49_CONFIG="$V49_ROOT/provenance/v64_3_49_siir.yaml"
V49_MANIFEST="$V49_ROOT/provenance/V64_3_49_SOURCE_MANIFEST.sha256"

mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/closed_loop_train"
TIMING="$OUT_ROOT/provenance/v64_3_50_stage_timing.tsv"; : > "$TIMING"
ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >> "$TIMING"; }

for f in "$V49_FIT" "$V49_SCENE" "$V49_CAND" "$V49_CONFIG" "$V49_MANIFEST" V64_3_50_SOURCE_MANIFEST.sha256; do
  [[ -s "$f" ]] || { echo "STOP V50 missing prerequisite $f" >&2; exit 2; }
done
for d in \
  "$BDSE_TRAIN_CACHE/train_boston" \
  "$BDSE_TRAIN_CACHE/train_pittsburgh" \
  "$BDSE_TRAIN_CACHE/train_singapore" \
  "$BDSE_TRAIN_CACHE/train_vegas_2" \
  "$NUPLAN_DB_SPLIT_ROOT/train_boston" \
  "$NUPLAN_DB_SPLIT_ROOT/train_pittsburgh" \
  "$NUPLAN_DB_SPLIT_ROOT/train_singapore" \
  "$NUPLAN_DB_SPLIT_ROOT/train_vegas"; do
  [[ -d "$d" ]] || { echo "STOP V50 missing dataset directory $d" >&2; exit 2; }
done
for d in train_boston train_pittsburgh train_singapore train_vegas; do
  compgen -G "$NUPLAN_DB_SPLIT_ROOT/$d/*.db" >/dev/null || { echo "STOP V50 no direct .db files under $NUPLAN_DB_SPLIT_ROOT/$d" >&2; exit 2; }
done

stage_start source_and_v49_failure_lock
sha256sum -c V64_3_50_SOURCE_MANIFEST.sha256 | tee "$OUT_ROOT/logs/v64_3_50_source_manifest.out"
python - "$V49_FIT" "$V49_SCENE" "$V49_CAND" "$V49_CONFIG" "$V49_MANIFEST" <<'PY'
import hashlib,json,sys
from pathlib import Path
fit,scene,cand,cfg,manifest=map(Path,sys.argv[1:])
expected={
 fit:'805c4f8088051413edeb568623bc6d225d1b3c301c52612f89109216b38be296',
 scene:'7243d1b4a07094ca691b4e0ba13ebd0a8ec1ad08d4da0b512a3714e1a9b5cd80',
 cand:'e51b8d38ec262a15a0b969e40e130a900984e9435d1cd2ae9c3d23bfd23f8f6c',
 cfg:'bde69241cb44da2e605a0e9e3897d024df7a1724868ba15a2e25cdc238dee405',
 manifest:'0baf1f02cdd774ae2754d8bf74f4a64583376c1f04e29dada9e84503ff01f48c',
}
for p,h in expected.items():
 g=hashlib.sha256(p.read_bytes()).hexdigest()
 if g!=h: raise SystemExit(f'STOP V50: V49 prerequisite identity changed {p}: {g}')
r=json.loads(fit.read_text()); n=r.get('nested_crossfit',{}); ri=n.get('risk_identification',{})
if r.get('train_gate_pass') is not False or n.get('train_gate_pass') is not False:
 raise SystemExit('STOP V50: V49 must remain preregistered nested-TRAIN failure')
if n.get('failure_diagnosis')!='selection_interventional_risk_does_not_outperform_observational_selected_risk_close_current_offline_selected_risk_family':
 raise SystemExit('STOP V50: V49 failure branch changed')
if abs(float(ri.get('aggregate_obs_sign_auc'))-0.6139192605594113)>1e-12 or abs(float(ri.get('aggregate_siir_auc'))-0.6081222524597028)>1e-12:
 raise SystemExit('STOP V50: V49 risk-identification signature changed')
print('PASS V50 prerequisite: exact V49 nested-TRAIN scientific STOP; offline selected-risk family is closed')
PY
cp V64_3_50_SOURCE_MANIFEST.sha256 "$OUT_ROOT/provenance/"
cp "$V49_MANIFEST" "$OUT_ROOT/provenance/V64_3_49_SOURCE_MANIFEST.sha256"
stage_end

stage_start regression_and_checkpoint
python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_{13_eaf_dmvr,14_eaf_ocfi,15_eaf_eair,16_eaf_raer,17_eaf_daler,18_eaf_dacer,19_eaf_icer,20_eaf_icer_dc,21_eaf_icer_mcr,22_eaf_icer_tcr,23_eaf_icer_rcr,24_eaf_icer_arc,25_eaf_icer_drc,26_eaf_icer_sarc,27_eaf_icer_trcc,28_eaf_icer_ptmc,29_eaf_icer_fcr,30_2_eaf_icer_fbic_pure,30_3_eaf_icer_fbic_pure_auditfix,30_eaf_icer_fbic,31_eaf_icer_scir,32_1_eaf_icer_ssir_weightfix,32_eaf_icer_ssir,33_eaf_icer_spcr,34_eaf_icer_rsmr,35_eaf_icer_fbcsr,36_eaf_icer_sgrr,37_eaf_icer_pvr,38_eaf_icer_davr,39_eaf_icer_cfsr,40_eaf_icer_sdfr,41_eaf_icer_epvr,42_eaf_icer_ovdr,43_eaf_icer_cfrv,44_eaf_icer_pcor,45_eaf_icer_pirf,46_eaf_icer_dirp,47_eaf_icer_fsfr,48_eaf_icer_ocrr,49_eaf_icer_siir,50_eaf_icer_pior}.py \
  bdse/tests/test_v64_3_48_2_eaf_icer_ocrr_provenance_repair.py | tee "$OUT_ROOT/logs/targeted_regression.out"
if [[ -z "${EAF_CKPT:-}" ]]; then
  [[ -s "$EAF_TRAIN_LOG" ]] || { echo "STOP V50 missing $EAF_TRAIN_LOG; set EAF_CKPT explicitly" >&2; exit 2; }
  python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
    --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_50_PIOR \
    --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_50.json" \
    > "$OUT_ROOT/logs/v64_3_13_reaudit_v64_3_50.out"
  SELECTED_EPOCH=$(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_50.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False): raise SystemExit('STOP V50 V13 prerequisites')
print(int(r['selected_epoch']))
PY
)
  EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"
fi
export EAF_CKPT
[[ -s "$EAF_CKPT" ]] || { echo "STOP V50 missing EAF checkpoint $EAF_CKPT" >&2; exit 2; }
stage_end

stage_start exact_train_population_and_probe_configs
TRAIN_MANIFEST="$OUT_ROOT/provenance/v64_3_50_pior_train_manifest.json"
TRAIN_TOKENS="$OUT_ROOT/provenance/v64_3_50_pior_train_tokens.txt"
TREAT_CFG="$OUT_ROOT/provenance/v64_3_50_pior_probe_treatment.yaml"
CTRL_CFG="$OUT_ROOT/provenance/v64_3_50_pior_probe_control.yaml"
python -m bdse.tools.build_v64_3_50_pior_train_manifest \
  --v49-candidate-audit "$V49_CAND" --train-cache "$BDSE_TRAIN_CACHE" \
  --raw-split-root "$NUPLAN_DB_SPLIT_ROOT" --output-manifest "$TRAIN_MANIFEST" --output-token-file "$TRAIN_TOKENS" \
  | tee "$OUT_ROOT/logs/v64_3_50_train_manifest.out"
python -m bdse.tools.make_v64_3_50_pior_probe_configs \
  --v49-siir-config "$V49_CONFIG" --output-treatment "$TREAT_CFG" --output-control "$CTRL_CFG" \
  | tee "$OUT_ROOT/logs/v64_3_50_probe_configs.out"
stage_end

stage_start paired_one_shot_closed_loop_outcome_collection
PAIRED="$OUT_ROOT/provenance/v64_3_50_pior_paired_closed_loop_outcomes.jsonl"
PAIR_REPORT="$OUT_ROOT/provenance/v64_3_50_pior_paired_closed_loop_report.json"
PIOR_RESUME_ARGS=()
if [[ "$PIOR_RESUME" == "1" ]]; then
  PIOR_RESUME_ARGS+=(--resume --allow-legacy-full-arm-resume)
fi
python -m bdse.tools.run_v64_3_50_pior_paired_closed_loop \
  --manifest "$TRAIN_MANIFEST" --treatment-config "$TREAT_CFG" --control-config "$CTRL_CFG" \
  --checkpoint "$EAF_CKPT" --nuplan-root "$NUPLAN_ROOT" --challenge closed_loop_nonreactive_agents \
  --output-root "$OUT_ROOT/closed_loop_train" --gpus "$GPU0,$GPU1" --workers-per-arm "$WORKERS_PER_ARM" \
  --batch-size "$PIOR_BATCH_SIZE" --first-batch-size "$PIOR_FIRST_BATCH_SIZE" --heartbeat-seconds "$PIOR_HEARTBEAT_SECONDS" \
  --serialize-gpu-inference "$PIOR_SERIALIZE_GPU_INFERENCE" --profile-closed-loop "$PIOR_PROFILE_CLOSED_LOOP" \
  "${PIOR_RESUME_ARGS[@]}" \
  --output-paired-outcomes "$PAIRED" --output-report "$PAIR_REPORT" \
  | tee "$OUT_ROOT/logs/v64_3_50_paired_closed_loop.out"
stage_end

stage_start nested_train_pior_gate
PIOR_CONFIG="$OUT_ROOT/provenance/v64_3_50_pior.yaml"
FIT_REPORT="$OUT_ROOT/provenance/v64_3_50_pior_fit.json"
set +e
python -m bdse.tools.fit_v64_3_50_eaf_icer_pior \
  --v49-fit-report "$V49_FIT" --v49-candidate-audit "$V49_CAND" --v49-scene-audit "$V49_SCENE" \
  --v49-siir-config "$V49_CONFIG" --paired-outcomes "$PAIRED" \
  --output-config "$PIOR_CONFIG" --output-report "$FIT_REPORT" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_50_pior_fit.out"
FIT_STATUS=${PIPESTATUS[0]}
set -e
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"

if [[ $FIT_STATUS -ne 0 ]]; then
  echo "V64.3.50 PIOR scientific STOP: paired closed-loop TRAIN gate failed. Do not tune offline features/loss/threshold and do not consume untouched closed-loop validation." >&2
  exit "$FIT_STATUS"
fi

echo "V64.3.50 PIOR TRAIN PASS. Freeze the emitted PIOR artifact. Next step is untouched paired closed-loop validation; do not use these TRAIN outcomes to tune features/loss/threshold."
