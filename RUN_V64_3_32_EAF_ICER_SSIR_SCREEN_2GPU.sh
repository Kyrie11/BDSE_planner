#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export V30_2_ROOT="${V30_2_ROOT:-outputs_v64_3_30_2_eaf_icer_fbic_pure_screen_2gpu_v1}"
export V30_3_ROOT="${V30_3_ROOT:-outputs_v64_3_30_3_eaf_icer_fbic_pure_auditfix_screen_2gpu_v1}"
export V31_ROOT="${V31_ROOT:-outputs_v64_3_31_eaf_icer_scir_screen_2gpu_v1}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_32_eaf_icer_ssir_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_32_design_exclude_v64_3_30_3_screen_tokens.txt}"
export FROZEN_TRAIN_TOKENS="${FROZEN_TRAIN_TOKENS:-bdse/configs/v64_3_29_frozen_v28_train_3000_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.32-eaf-icer-ssir-cal500-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"

RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
TRAIN_EDGES="$V30_2_ROOT/provenance/train_b16_v20_edges.jsonl"
V30_3_SCREEN="$V30_3_ROOT/provenance/v64_3_30_3_eaf_icer_fbic_pure_auditfix_double_fresh_screen.json"
V31_FIT="$V31_ROOT/provenance/v64_3_31_scir_fit.json"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"
TIMING="$OUT_ROOT/provenance/v64_3_32_stage_timing.tsv"; : > "$TIMING"
ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >> "$TIMING"; }

[[ -d "$BDSE_VAL_CACHE" ]] || { echo 'STOP: missing val cache' >&2; exit 2; }
for f in "$RAW_CONFIG" "$V20_CONFIG" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$TRAIN_EDGES" "$V30_3_SCREEN" "$V31_FIT"; do
  [[ -s "$f" ]] || { echo "STOP: missing $f" >&2; exit 2; }
done

stage_start prior_causal_closure_and_v31_failure_reproduction
python - "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$V30_3_SCREEN" "$V31_FIT" <<'PY'
import hashlib,json,math,sys
ex=[x.strip() for x in open(sys.argv[1]) if x.strip()]
tr=[x.strip() for x in open(sys.argv[2]) if x.strip()]
v30=json.load(open(sys.argv[3])); v31=json.load(open(sys.argv[4]))
if len(ex)!=10700 or len(set(ex))!=10700: raise SystemExit(f'STOP DATA: V32 design exclusion !=10700 unique rows={len(ex)} unique={len(set(ex))}')
sha=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()
if sha!='041ec824756777576391756ef3721617459bb0c0a45f7f43226b52254d951473': raise SystemExit('STOP DATA: V32 design exclusion SHA changed: '+sha)
if len(tr)!=3000 or len(set(tr))!=3000: raise SystemExit('STOP DATA: frozen TRAIN !=3000 unique')
trsha=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
if trsha!='b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4': raise SystemExit('STOP DATA: frozen TRAIN SHA changed: '+trsha)
want='retained_capacity_is_not_the_reproducible_first_order_missing_mediator_under_current_frozen_consumer'
if not v30.get('engineering_valid',False) or v30.get('scientific_conclusion')!=want: raise SystemExit('STOP V32: V30.3 capacity closure prerequisite changed')
cf=v31.get('crossfit',{})
checks=[
    v31.get('frozen_train_scenes')==3000,
    v31.get('direct_support_positive_training_edges')==9394,
    v31.get('direct_support_positive_training_scenes')==782,
    v31.get('train_gate_pass') is False,
    cf.get('fold_pass_count')==0,
    cf.get('selected_count')==612,
    cf.get('selected_positive_count')==342,
    abs(float(cf.get('teacher_improvement_sum',999))-(-22.073251969552103))<1e-9,
    abs(float(cf.get('teacher_improvement_worst',999))-(-4.041263536178889))<1e-9,
    abs(float(cf.get('positive_capture_rate',999))-0.5958188153310104)<1e-12,
]
if not all(checks): raise SystemExit('STOP V32: V31 uploaded TRAIN failure does not reproduce exact audited result')
print('PASS V32 prerequisite: V30.3 capacity branch closed; exact V31 0/5 mean-ranking failure reproduced; no CAL/fresh was spent by V31')
PY
stage_end

stage_start prerequisites_and_regression
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
  --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_32_SSIR \
  --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" \
  > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False): raise SystemExit('STOP prerequisites')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"; fi
export EAF_CKPT
[[ -s "$EAF_CKPT" ]] || { echo "STOP: missing EAF checkpoint $EAF_CKPT" >&2; exit 2; }
python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_32_eaf_icer_ssir.py \
  bdse/tests/test_v64_3_31_eaf_icer_scir.py \
  bdse/tests/test_v64_3_30_3_eaf_icer_fbic_pure_auditfix.py \
  bdse/tests/test_v64_3_30_2_eaf_icer_fbic_pure.py \
  bdse/tests/test_v64_3_30_eaf_icer_fbic.py \
  bdse/tests/test_v64_3_29_eaf_icer_fcr.py \
  bdse/tests/test_v64_3_28_eaf_icer_ptmc.py \
  bdse/tests/test_v64_3_27_eaf_icer_trcc.py \
  bdse/tests/test_v64_3_26_eaf_icer_sarc.py \
  bdse/tests/test_v64_3_25_eaf_icer_drc.py \
  bdse/tests/test_v64_3_24_eaf_icer_arc.py \
  bdse/tests/test_v64_3_23_eaf_icer_rcr.py \
  bdse/tests/test_v64_3_22_eaf_icer_tcr.py \
  bdse/tests/test_v64_3_21_eaf_icer_mcr.py \
  bdse/tests/test_v64_3_20_eaf_icer_dc.py \
  bdse/tests/test_v64_3_19_eaf_icer.py \
  bdse/tests/test_v64_3_18_eaf_dacer.py \
  bdse/tests/test_v64_3_17_eaf_daler.py \
  bdse/tests/test_v64_3_16_eaf_raer.py \
  bdse/tests/test_v64_3_15_eaf_eair.py \
  bdse/tests/test_v64_3_14_eaf_ocfi.py \
  bdse/tests/test_v64_3_13_eaf_dmvr.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

stage_start train_nested_full_ssir_gate
PRESERVE_CONFIG="$OUT_ROOT/provenance/v64_3_32_preserve_control.yaml"
MEAN_CONFIG="$OUT_ROOT/provenance/v64_3_32_ssir_mean.yaml"
FIT_REPORT="$OUT_ROOT/provenance/v64_3_32_ssir_fit.json"
TRAIN_SCENE_AUDIT="$OUT_ROOT/provenance/v64_3_32_ssir_train_scene_audit.csv"
python -m bdse.tools.fit_v64_3_32_eaf_icer_ssir \
  --train-frontier-edges "$TRAIN_EDGES" --base-config "$V20_CONFIG" \
  --output-preserve-config "$PRESERVE_CONFIG" --output-mean-config "$MEAN_CONFIG" \
  --output-report "$FIT_REPORT" --output-scene-audit "$TRAIN_SCENE_AUDIT" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_32_ssir_fit.out"
stage_end

stage_start calibration_and_fresh_selection
TOK1500="$OUT_ROOT/provenance/val_screen_cal500_fresh1500_tokens.txt"
TOKCAL="$OUT_ROOT/provenance/val_screen_calibration_500_tokens.txt"
TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"
TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"
SELECT_EXCLUDE="$OUT_ROOT/provenance/v64_3_32_selection_exclude_design_plus_train.txt"
cat "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" | awk 'NF && !seen[$0]++' > "$SELECT_EXCLUDE"
python -m bdse.tools.select_fresh_preprocessed_tokens \
  --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$SELECT_EXCLUDE" --count 1500 \
  --hash-seed "$FRESH_HASH_SEED" --output "$TOK1500" \
  --audit-output "$OUT_ROOT/provenance/v64_3_32_cal500_fresh1500_selection_audit.json" \
  > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1500" > "$TOKCAL"
sed -n '501,1000p' "$TOK1500" > "$TOKA"
tail -n 500 "$TOK1500" > "$TOKB"
python - "$TOK1500" "$TOKCAL" "$TOKA" "$TOKB" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" <<'PY'
import sys
allv=[x.strip() for x in open(sys.argv[1]) if x.strip()]; cal=[x.strip() for x in open(sys.argv[2]) if x.strip()]; A=[x.strip() for x in open(sys.argv[3]) if x.strip()]; B=[x.strip() for x in open(sys.argv[4]) if x.strip()]
ex={x.strip() for x in open(sys.argv[5]) if x.strip()}; tr={x.strip() for x in open(sys.argv[6]) if x.strip()}
if len(allv)!=1500 or len(set(allv))!=1500 or any(len(x)!=500 or len(set(x))!=500 for x in [cal,A,B]): raise SystemExit('STOP DATA: V32 CAL/A/B cardinality or uniqueness failure')
if set(cal)&set(A) or set(cal)&set(B) or set(A)&set(B): raise SystemExit('STOP DATA: V32 CAL/A/B overlap')
if set(allv)!=(set(cal)|set(A)|set(B)): raise SystemExit('STOP DATA: V32 partition mismatch')
if set(allv)&ex or set(allv)&tr: raise SystemExit('STOP DATA: V32 selected data overlaps spent design/TRAIN')
print('PASS V32 label-free 1500 selection: CAL500 + untouched A500 + untouched B500, all disjoint from 10700 design + TRAIN')
PY
stage_end

run_eval_tok(){
  local gpu="$1" cfg="$2" tok="$3" maxn="$4" tag="$5"
  CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop \
    --config "$cfg" --checkpoint "$EAF_CKPT" --split val \
    --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios "$maxn" \
    --scenario-token-file "$tok" --require-all-scenario-tokens \
    --output "$OUT_ROOT/provenance/${tag}_metrics.json" \
    --per-sample-output "$OUT_ROOT/provenance/${tag}_rows.jsonl" \
    --frontier-edge-output "$OUT_ROOT/provenance/${tag}_edges.jsonl" \
    --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${tag}.out" 2>&1
}

stage_start independent_calibration_mean_replay
run_eval_tok "$GPU0" "$MEAN_CONFIG" "$TOKCAL" 500 cal_mean
stage_end

stage_start independent_direct_domain_simultaneous_calibration
MAIN_CONFIG="$OUT_ROOT/provenance/v64_3_32_ssir_main.yaml"
CAL_REPORT="$OUT_ROOT/provenance/v64_3_32_ssir_calibration.json"
python -m bdse.tools.calibrate_v64_3_32_eaf_icer_ssir \
  --calibration-rows "$OUT_ROOT/provenance/cal_mean_rows.jsonl" \
  --calibration-edges "$OUT_ROOT/provenance/cal_mean_edges.jsonl" --mean-config "$MEAN_CONFIG" \
  --output-main-config "$MAIN_CONFIG" --output-report "$CAL_REPORT" --alpha 0.05 \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_32_ssir_calibration.out"
python -m bdse.tools.check_v64_3_32_eaf_icer_ssir_contract \
  --v20-config "$V20_CONFIG" --preserve-config "$PRESERVE_CONFIG" --mean-config "$MEAN_CONFIG" --main-config "$MAIN_CONFIG" \
  --calibration-report "$CAL_REPORT" --output "$OUT_ROOT/provenance/v64_3_32_ssir_contract.json" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_32_ssir_contract.out"
stage_end

for sp in A B; do
  tok="$TOKA"; [[ "$sp" == B ]] && tok="$TOKB"
  stage_start "fresh_${sp}_wave1"
  run_eval_tok "$GPU0" "$RAW_CONFIG" "$tok" 500 "${sp}_raw" & p0=$!
  run_eval_tok "$GPU1" "$V20_CONFIG" "$tok" 500 "${sp}_v20" & p1=$!
  s=0; wait "$p0" || s=1; wait "$p1" || s=1; [[ $s -eq 0 ]] || exit 2
  stage_end
  stage_start "fresh_${sp}_wave2"
  run_eval_tok "$GPU0" "$PRESERVE_CONFIG" "$tok" 500 "${sp}_preserve" & p0=$!
  run_eval_tok "$GPU1" "$MEAN_CONFIG" "$tok" 500 "${sp}_mean" & p1=$!
  s=0; wait "$p0" || s=1; wait "$p1" || s=1; [[ $s -eq 0 ]] || exit 2
  stage_end
  stage_start "fresh_${sp}_wave3"
  run_eval_tok "$GPU0" "$MAIN_CONFIG" "$tok" 500 "${sp}_main"
  stage_end
done

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); tags=['raw','v20','preserve','mean','main']
for sp,tf in [('A',sys.argv[2]),('B',sys.argv[3])]:
    want=[x.strip() for x in open(tf) if x.strip()]; orders=[]
    if len(want)!=500 or len(set(want))!=500: raise SystemExit(f'STOP DATA: {sp} manifest !=500 unique')
    for tag in tags:
        rs=[json.loads(x) for x in open(root/f'{sp}_{tag}_rows.jsonl') if x.strip()]; got=[str(r['scenario_token']) for r in rs]
        if len(got)!=500 or len(set(got))!=500 or set(got)!=set(want): raise SystemExit(f'STOP DATA: {sp}/{tag} identity mismatch')
        orders.append(got); m=json.load(open(root/f'{sp}_{tag}_metrics.json'))
        if not m.get('scenario_token_prefilter_active',False): raise SystemExit(f'STOP DATA/SPEED: {sp}/{tag} token prefilter inactive')
    if any(x!=orders[0] for x in orders[1:]): raise SystemExit(f'STOP DATA: {sp} row order differs across five arms')
print('PASS V32 paired identity across all 10 untouched A/B arms')
PY
stage_end

stage_start screen
for sp in A B; do
  python -m bdse.tools.check_v64_3_32_eaf_icer_ssir_split --split-name "$sp" \
    --raw-metrics "$OUT_ROOT/provenance/${sp}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${sp}_raw_rows.jsonl" \
    --v20-metrics "$OUT_ROOT/provenance/${sp}_v20_metrics.json" --v20-rows "$OUT_ROOT/provenance/${sp}_v20_rows.jsonl" --v20-edges "$OUT_ROOT/provenance/${sp}_v20_edges.jsonl" \
    --preserve-metrics "$OUT_ROOT/provenance/${sp}_preserve_metrics.json" --preserve-rows "$OUT_ROOT/provenance/${sp}_preserve_rows.jsonl" --preserve-edges "$OUT_ROOT/provenance/${sp}_preserve_edges.jsonl" \
    --mean-metrics "$OUT_ROOT/provenance/${sp}_mean_metrics.json" --mean-rows "$OUT_ROOT/provenance/${sp}_mean_rows.jsonl" --mean-edges "$OUT_ROOT/provenance/${sp}_mean_edges.jsonl" \
    --main-metrics "$OUT_ROOT/provenance/${sp}_main_metrics.json" --main-rows "$OUT_ROOT/provenance/${sp}_main_rows.jsonl" --main-edges "$OUT_ROOT/provenance/${sp}_main_edges.jsonl" \
    --output "$OUT_ROOT/provenance/v64_3_32_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_32_split_${sp}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_32_eaf_icer_ssir_double_fresh_screen.json"
python -m bdse.tools.check_v64_3_32_eaf_icer_ssir_screen \
  --split-a "$OUT_ROOT/provenance/v64_3_32_split_A_screen.json" --split-b "$OUT_ROOT/provenance/v64_3_32_split_B_screen.json" \
  --calibration-report "$CAL_REPORT" --fit-report "$FIT_REPORT" --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_32_double_fresh_screen.out"
stage_end

# Finalize timing before hashing so provenance is self-consistent.
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"
(cd "$OUT_ROOT/provenance" && find . -maxdepth 1 -type f ! -name 'v64_3_32_provenance_sha256.txt' -print0 | sort -z | xargs -0 sha256sum > v64_3_32_provenance_sha256.txt)
python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for k in ['engineering_valid_both','train_nested_full_mechanism_gate_pass','independent_calibration_valid','double_fresh_promotion_pass','scientific_conclusion','next_action']:
    print(k,'=',r.get(k))
print('V32 protocol: exact V31 failure prerequisite -> nested TRAIN full SSIR gate -> independent CAL500 direct-domain scene-simultaneous calibration -> untouched A500/B500 judged separately. No pooled rescue and no alpha/scale/ridge/threshold sweep.')
PY
