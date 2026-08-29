#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export V47_ROOT="${V47_ROOT:-outputs_v64_3_47_eaf_icer_fsfr_screen_2gpu_v1}"
export V49_ROOT="${V49_ROOT:-outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_50_eaf_icer_sior_screen_2gpu_v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"

# ---------------------------------------------------------------------------
# Dataset layout contract (engineering-only; V50 science is unchanged)
# ---------------------------------------------------------------------------
# BDSE NPZ caches and native nuPlan DB/maps are intentionally separate.
#
# User/preprocessed NPZ layout:
#   /data0/senzeyu2/dataset/nuplan/data/cache/
#       bdse_train_v2/{train_boston,train_pittsburgh,train_singapore,train_vegas_2}/<log>/*.npz
#       bdse_val_v2/val/<log>/*.npz
#       bdse_test_2/public_set_test/<log>/*.npz
#
# Native nuPlan layout used ONLY by closed-loop ScenarioBuilder:
#   /data0/senzeyu2/dataset/CapPlan/data/nuplan/
#       maps/...
#       nuplan-v1.1/splits/{train_boston,train_pittsburgh,train_singapore,train_vegas,val,test}/*.db
#
# Do not point NUPLAN_DB_* at the NPZ cache. V50 paired closed-loop must run on
# the original nuPlan SQLite log DBs.
export BDSE_CACHE_ROOT="${BDSE_CACHE_ROOT:-/data0/senzeyu2/dataset/nuplan/data/cache}"
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-$BDSE_CACHE_ROOT/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-$BDSE_CACHE_ROOT/bdse_val_v2}"
export BDSE_TEST_CACHE="${BDSE_TEST_CACHE:-$BDSE_CACHE_ROOT/bdse_test_2}"

export NUPLAN_ROOT="${NUPLAN_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan}"
export NUPLAN_MAP_ROOT="${NUPLAN_MAP_ROOT:-$NUPLAN_ROOT/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-$SCRIPT_DIR/$OUT_ROOT/nuplan_exp}"
export NUPLAN_SPLITS_ROOT="${NUPLAN_SPLITS_ROOT:-$NUPLAN_ROOT/nuplan-v1.1/splits}"
export CL_CHALLENGE="${CL_CHALLENGE:-closed_loop_reactive_agents}"
export PYTHONUNBUFFERED=1

# Native TRAIN DB selection.  The user's official DBs are flat inside each
# city split directory, which is directly consumable by nuPlan.  We therefore
# pass the four TRAIN directories as scenario_builder.db_files; val/test DBs
# are never mixed into V50 TRAIN paired supervision.
#
# Backward-compatible overrides:
#   NUPLAN_DB_ROOT=/some/single/root
# or
#   NUPLAN_TRAIN_DB_ROOTS=/dir/a:/dir/b:/dir/c
# The latter takes precedence over the automatic four-city defaults.
NUPLAN_DB_ARGS=()
NUPLAN_TRAIN_DB_DIRS=()
if [[ -n "${NUPLAN_TRAIN_DB_ROOTS:-}" ]]; then
  IFS=':' read -r -a NUPLAN_TRAIN_DB_DIRS <<< "$NUPLAN_TRAIN_DB_ROOTS"
  NUPLAN_DB_ARGS=(--nuplan-db-files "${NUPLAN_TRAIN_DB_DIRS[@]}")
elif [[ -n "${NUPLAN_DB_ROOT:-}" ]]; then
  NUPLAN_DB_ARGS=(--nuplan-db-root "$NUPLAN_DB_ROOT")
else
  NUPLAN_TRAIN_DB_DIRS=(
    "$NUPLAN_SPLITS_ROOT/train_boston"
    "$NUPLAN_SPLITS_ROOT/train_pittsburgh"
    "$NUPLAN_SPLITS_ROOT/train_singapore"
    "$NUPLAN_SPLITS_ROOT/train_vegas"
  )
  NUPLAN_DB_ARGS=(--nuplan-db-files "${NUPLAN_TRAIN_DB_DIRS[@]}")
fi

V47_RSMR="$V47_ROOT/provenance/v64_3_47_rsmr.yaml"
V49_FIT="$V49_ROOT/provenance/v64_3_49_siir_fit.json"
V49_SCENE="$V49_ROOT/provenance/v64_3_49_siir_train_scene_audit.csv"
V49_CAND="$V49_ROOT/provenance/v64_3_49_siir_oof_candidate_state_audit.jsonl"
V49_CFG="$V49_ROOT/provenance/v64_3_49_siir.yaml"
V49_SERVER_TEST="$V49_ROOT/logs/targeted_regression.out"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/paired_train" "$NUPLAN_EXP_ROOT"
TIMING="$OUT_ROOT/provenance/v64_3_50_stage_timing.tsv"; :>"$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >>"$TIMING"; }

for f in V64_3_50_SOURCE_MANIFEST.sha256 "$EAF_TRAIN_LOG" "$V47_RSMR" "$V49_FIT" "$V49_SCENE" "$V49_CAND" "$V49_CFG" "$V49_SERVER_TEST"; do
  [[ -s "$f" ]] || { echo "STOP V50 missing $f" >&2; exit 2; }
done
# NPZ caches are NOT consumed by the V50 paired ScenarioBuilder path.  Keep
# their canonical locations visible for provenance, but do not fail V50 merely
# because an unused val/test NPZ cache is absent on a closed-loop worker.
[[ -d "$NUPLAN_ROOT" ]] || { echo "STOP V50 missing native nuPlan root: $NUPLAN_ROOT" >&2; exit 2; }
[[ -d "$NUPLAN_MAP_ROOT" ]] || { echo "STOP V50 missing nuPlan maps root: $NUPLAN_MAP_ROOT" >&2; exit 2; }
[[ -s "$NUPLAN_MAP_ROOT/nuplan-maps-v1.0.json" ]] || { echo "STOP V50 missing nuPlan map metadata: $NUPLAN_MAP_ROOT/nuplan-maps-v1.0.json" >&2; exit 2; }

if [[ ${#NUPLAN_TRAIN_DB_DIRS[@]} -gt 0 ]]; then
  for d in "${NUPLAN_TRAIN_DB_DIRS[@]}"; do
    [[ -d "$d" ]] || { echo "STOP V50 missing native TRAIN DB directory: $d" >&2; exit 2; }
    [[ -n "$(find "$d" -maxdepth 1 -type f -name '*.db' -print -quit 2>/dev/null)" ]] || {
      echo "STOP V50 native TRAIN DB directory contains no direct .db files: $d" >&2
      echo "V50 closed-loop needs original nuPlan SQLite DBs, not BDSE NPZ cache files." >&2
      exit 2
    }
  done
elif [[ -n "${NUPLAN_DB_ROOT:-}" ]]; then
  [[ -e "$NUPLAN_DB_ROOT" ]] || { echo "STOP V50 missing NUPLAN_DB_ROOT: $NUPLAN_DB_ROOT" >&2; exit 2; }
fi

echo '[V50 dataset layout] BDSE NPZ caches (offline only):'
echo "  train=$BDSE_TRAIN_CACHE (present=$([[ -d "$BDSE_TRAIN_CACHE" ]] && echo yes || echo no); not consumed by V50 closed-loop)"
echo "  val=$BDSE_VAL_CACHE (present=$([[ -d "$BDSE_VAL_CACHE" ]] && echo yes || echo no); not consumed by V50 closed-loop)"
echo "  test=$BDSE_TEST_CACHE (present=$([[ -d "$BDSE_TEST_CACHE" ]] && echo yes || echo no); not consumed by V50 closed-loop)"
echo '[V50 dataset layout] native nuPlan closed-loop inputs:'
echo "  data_root=$NUPLAN_ROOT"
echo "  map_root=$NUPLAN_MAP_ROOT"
echo "  exp_root=$NUPLAN_EXP_ROOT"
if [[ ${#NUPLAN_TRAIN_DB_DIRS[@]} -gt 0 ]]; then
  printf '  train_db_dir=%s\n' "${NUPLAN_TRAIN_DB_DIRS[@]}"
else
  echo "  train_db_root=$NUPLAN_DB_ROOT"
fi

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
  --nuplan-exp-root "$NUPLAN_EXP_ROOT" "${NUPLAN_DB_ARGS[@]}" --resume \
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
