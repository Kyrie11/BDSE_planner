#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export V44_ROOT="${V44_ROOT:-outputs_v64_3_44_eaf_icer_pcor_screen_2gpu_v1}"
export V47_ROOT="${V47_ROOT:-outputs_v64_3_47_eaf_icer_fsfr_screen_2gpu_v1}"
export V48_ROOT="${V48_ROOT:-outputs_v64_3_48_2_eaf_icer_ocrr_screen_2gpu_v1}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_32_design_exclude_v64_3_30_3_screen_tokens.txt}"
export FROZEN_TRAIN_TOKENS="${FROZEN_TRAIN_TOKENS:-bdse/configs/v64_3_29_frozen_v28_train_3000_tokens.txt}"
export V48_CONSUMED_FRESH_TOKENS="${V48_CONSUMED_FRESH_TOKENS:-bdse/configs/v64_3_48_consumed_fresh1000_tokens.txt}"
export V48_2_CONSUMED_FRESH_TOKENS="${V48_2_CONSUMED_FRESH_TOKENS:-bdse/configs/v64_3_48_2_consumed_fresh1000_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.49-eaf-icer-siir-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"

RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
V44_PRESERVE="$V44_ROOT/provenance/v64_3_44_preserve.yaml"
V44_EDGES="$V44_ROOT/provenance/train_v20_plan_conditioned_response_edges.jsonl"
V47_SIDE="$V47_ROOT/provenance/v64_3_47_crossfit_future_state_observables.jsonl"
V47_AUDIT="$V47_ROOT/provenance/v64_3_47_fsfr_train_scene_audit.csv"
V47_PLAN="$V47_ROOT/provenance/v64_3_47_plan_control.yaml"
V47_EGO="$V47_ROOT/provenance/v64_3_47_ego_ref.yaml"
RSMR_CONFIG="$V47_ROOT/provenance/v64_3_47_rsmr.yaml"
QUALITY_CONFIG="$V47_ROOT/provenance/v64_3_47_quality.yaml"
V48_FIT="$V48_ROOT/provenance/v64_3_48_2_ocrr_fit.json"
V48_SCREEN="$V48_ROOT/provenance/v64_3_48_2_eaf_icer_ocrr_double_fresh_screen.json"
V48_OBS_CONFIG="$V48_ROOT/provenance/v64_3_48_2_sign_nomult.yaml"
V48_FRESH_LEDGER="$V48_ROOT/provenance/val_screen_fresh1000_tokens.txt"

mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"
TIMING="$OUT_ROOT/provenance/v64_3_49_stage_timing.tsv"; :>"$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >>"$TIMING"; }

for f in "$RAW_CONFIG" "$V20_CONFIG" "$V44_PRESERVE" "$V44_EDGES" "$V47_SIDE" "$V47_AUDIT" "$V47_PLAN" "$V47_EGO" "$RSMR_CONFIG" "$QUALITY_CONFIG" "$V48_FIT" "$V48_SCREEN" "$V48_OBS_CONFIG" "$V48_FRESH_LEDGER" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$V48_CONSUMED_FRESH_TOKENS" "$V48_2_CONSUMED_FRESH_TOKENS" V64_3_49_SOURCE_MANIFEST.sha256 V64_3_48_OCRR_SCIENCE_LOCK.sha256; do
  [[ -s "$f" ]] || { echo "STOP V49 missing $f" >&2; exit 2; }
done
[[ -d "$BDSE_VAL_CACHE" ]] || { echo 'STOP V49 missing val cache' >&2; exit 2; }

stage_start source_and_v48_failure_gate
sha256sum -c V64_3_49_SOURCE_MANIFEST.sha256 | tee "$OUT_ROOT/logs/v64_3_49_source_manifest.out"
sha256sum -c V64_3_48_OCRR_SCIENCE_LOCK.sha256 | tee "$OUT_ROOT/logs/v64_3_49_v48_ocrr_science_lock.out"
python - "$V48_FIT" "$V48_SCREEN" "$V48_FRESH_LEDGER" "$V48_2_CONSUMED_FRESH_TOKENS" "$FROZEN_TRAIN_TOKENS" <<'PY'
import hashlib,json,sys
from pathlib import Path
fit,screen,fresh,ledger,train=map(Path,sys.argv[1:])
expected={
 fit:'000ddef7523f6e79004c49f7d21c9f8e83b243162e72d4fd7fec4db5e8330317',
 screen:'f22d6f75f67df35188e370c2a8073c922857dc37b86817669a480a1e078b3df4',
 fresh:'d9822a47a2442c3c0591834b7e00cea70a8f30f393379487bc1761728f6fb9dc',
 ledger:'d9822a47a2442c3c0591834b7e00cea70a8f30f393379487bc1761728f6fb9dc',
 train:'b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4',
}
for p,h in expected.items():
 g=hashlib.sha256(p.read_bytes()).hexdigest()
 if g!=h: raise SystemExit(f'STOP V49 prerequisite identity changed: {p} {g}')
if fresh.read_bytes()!=ledger.read_bytes(): raise SystemExit('STOP V49: packaged V48.2 consumed-fresh ledger differs from scientific output')
xs=[x.strip() for x in ledger.read_text().splitlines() if x.strip()]
if len(xs)!=1000 or len(set(xs))!=1000: raise SystemExit('STOP V49: V48.2 consumed ledger must be 1000 unique')
r=json.loads(fit.read_text());n=r.get('nested_crossfit',{})
if n.get('preferred_promotion_arm')!='sign_mult' or n.get('train_gate_pass') is not True: raise SystemExit('STOP V49: V48 TRAIN signature changed')
s=json.loads(screen.read_text())
if s.get('pass') is not False or s.get('split_A_pass') is not False or s.get('split_B_pass') is not False or s.get('next_action')!='STOP_no_promotion_do_not_pool_A_B_or_tune': raise SystemExit('STOP V49: V48.2 fresh STOP signature changed')
print('PASS V49 prerequisite: exact V48.2 fresh failure + consumed 1000 + frozen TRAIN identities')
PY
cp V64_3_49_SOURCE_MANIFEST.sha256 "$OUT_ROOT/provenance/"
cp V64_3_48_OCRR_SCIENCE_LOCK.sha256 "$OUT_ROOT/provenance/"
cp "$V48_2_CONSUMED_FRESH_TOKENS" "$OUT_ROOT/provenance/v64_3_48_2_consumed_fresh1000_tokens.txt"
stage_end

stage_start prerequisites_and_regression
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_49_SIIR --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_49.json" > "$OUT_ROOT/logs/v64_3_13_reaudit_v64_3_49.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_49.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False): raise SystemExit('STOP V49 prerequisites')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"; fi
export EAF_CKPT; [[ -s "$EAF_CKPT" ]] || { echo 'STOP V49 missing EAF checkpoint' >&2; exit 2; }
python -m compileall -q bdse
pytest -q bdse/tests/test_v64_3_{13_eaf_dmvr,14_eaf_ocfi,15_eaf_eair,16_eaf_raer,17_eaf_daler,18_eaf_dacer,19_eaf_icer,20_eaf_icer_dc,21_eaf_icer_mcr,22_eaf_icer_tcr,23_eaf_icer_rcr,24_eaf_icer_arc,25_eaf_icer_drc,26_eaf_icer_sarc,27_eaf_icer_trcc,28_eaf_icer_ptmc,29_eaf_icer_fcr,30_2_eaf_icer_fbic_pure,30_3_eaf_icer_fbic_pure_auditfix,30_eaf_icer_fbic,31_eaf_icer_scir,32_1_eaf_icer_ssir_weightfix,32_eaf_icer_ssir,33_eaf_icer_spcr,34_eaf_icer_rsmr,35_eaf_icer_fbcsr,36_eaf_icer_sgrr,37_eaf_icer_pvr,38_eaf_icer_davr,39_eaf_icer_cfsr,40_eaf_icer_sdfr,41_eaf_icer_epvr,42_eaf_icer_ovdr,43_eaf_icer_cfrv,44_eaf_icer_pcor,45_eaf_icer_pirf,46_eaf_icer_dirp,47_eaf_icer_fsfr,48_eaf_icer_ocrr,49_eaf_icer_siir}.py bdse/tests/test_v64_3_48_2_eaf_icer_ocrr_provenance_repair.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

stage_start nested_train_selection_intervention_gate
SIIR_CONFIG="$OUT_ROOT/provenance/v64_3_49_siir.yaml"
FIT_REPORT="$OUT_ROOT/provenance/v64_3_49_siir_fit.json"
SCENE_AUDIT="$OUT_ROOT/provenance/v64_3_49_siir_train_scene_audit.csv"
CAND_AUDIT="$OUT_ROOT/provenance/v64_3_49_siir_oof_candidate_state_audit.jsonl"
set +e
python -m bdse.tools.fit_v64_3_49_eaf_icer_siir \
  --v44-train-frontier-edges "$V44_EDGES" \
  --v47-fsfr-sidecar "$V47_SIDE" --v47-scene-audit "$V47_AUDIT" \
  --v47-plan-config "$V47_PLAN" --v47-ego-ref-config "$V47_EGO" \
  --v48-fit-report "$V48_FIT" --v48-screen-report "$V48_SCREEN" \
  --output-siir-config "$SIIR_CONFIG" --output-report "$FIT_REPORT" \
  --output-scene-audit "$SCENE_AUDIT" --output-candidate-audit "$CAND_AUDIT" 2>&1 | tee "$OUT_ROOT/logs/v64_3_49_siir_fit.out"
FIT_STATUS=${PIPESTATUS[0]}; set -e; stage_end; [[ $FIT_STATUS -eq 0 ]] || exit "$FIT_STATUS"

stage_start double_fresh_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh1000_tokens.txt"; TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"; TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"; EX="$OUT_ROOT/provenance/v64_3_49_selection_exclude.txt"
cat "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$V48_CONSUMED_FRESH_TOKENS" "$V48_2_CONSUMED_FRESH_TOKENS" | awk 'NF&&!seen[$0]++' > "$EX"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$EX" --count 1000 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" --audit-output "$OUT_ROOT/provenance/v64_3_49_fresh1000_selection_audit.json" > "$OUT_ROOT/logs/fresh_selection.out"
head -n500 "$TOK1000">"$TOKA"; tail -n500 "$TOK1000">"$TOKB"
python - "$TOKA" "$TOKB" "$EX" <<'PY'
import sys
A=[x.strip() for x in open(sys.argv[1]) if x.strip()];B=[x.strip() for x in open(sys.argv[2]) if x.strip()];E={x.strip() for x in open(sys.argv[3]) if x.strip()}
if len(A)!=500 or len(B)!=500 or len(set(A))!=500 or len(set(B))!=500 or set(A)&set(B) or (set(A)|set(B))&E: raise SystemExit('STOP V49 fresh independence')
print('PASS V49 independent untouched A500+B500 selection')
PY
stage_end

run_eval(){ local gpu="$1" cfg="$2" tok="$3" tag="$4"; CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios 500 --scenario-token-file "$tok" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/${tag}_rows.jsonl" --frontier-edge-output "$OUT_ROOT/provenance/${tag}_edges.jsonl" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${tag}.out" 2>&1; }
for sp in A B; do
 tok="$TOKA"; [[ "$sp" == B ]] && tok="$TOKB"
 tags=(raw v20 preserve rsmr quality plan_control ego_ref obs_sign siir)
 cfgs=("$RAW_CONFIG" "$V20_CONFIG" "$V44_PRESERVE" "$RSMR_CONFIG" "$QUALITY_CONFIG" "$V47_PLAN" "$V47_EGO" "$V48_OBS_CONFIG" "$SIIR_CONFIG")
 for ((i=0;i<9;i+=2)); do
  stage_start "fresh_${sp}_wave_$i"; s=0
  run_eval "$GPU0" "${cfgs[$i]}" "$tok" "${sp}_${tags[$i]}" & p0=$!
  if ((i+1<9)); then run_eval "$GPU1" "${cfgs[$((i+1))]}" "$tok" "${sp}_${tags[$((i+1))]}" & p1=$!; else p1=''; fi
  wait "$p0" || s=1; if [[ -n "$p1" ]]; then wait "$p1" || s=1; fi
  [[ $s -eq 0 ]] || exit 2; stage_end
 done
done

stage_start double_fresh_screen
for sp in A B; do
 args=()
 for tag in raw v20 preserve rsmr quality plan_control ego_ref obs_sign siir; do
  x=${tag//_/-}; args+=("--${x}-metrics" "$OUT_ROOT/provenance/${sp}_${tag}_metrics.json" "--${x}-rows" "$OUT_ROOT/provenance/${sp}_${tag}_rows.jsonl")
  [[ "$tag" == raw ]] || args+=("--${x}-edges" "$OUT_ROOT/provenance/${sp}_${tag}_edges.jsonl")
 done
 python -m bdse.tools.check_v64_3_49_eaf_icer_siir_split --split-name "$sp" "${args[@]}" --output "$OUT_ROOT/provenance/v64_3_49_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_49_split_${sp}.out"
done
set +e
python -m bdse.tools.check_v64_3_49_eaf_icer_siir_screen --split-a "$OUT_ROOT/provenance/v64_3_49_split_A_screen.json" --split-b "$OUT_ROOT/provenance/v64_3_49_split_B_screen.json" --fit-report "$FIT_REPORT" --output "$OUT_ROOT/provenance/v64_3_49_eaf_icer_siir_double_fresh_screen.json" | tee "$OUT_ROOT/logs/v64_3_49_screen.out"
SCREEN_STATUS=${PIPESTATUS[0]}; set -e
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >>"$TIMING"
exit "$SCREEN_STATUS"
