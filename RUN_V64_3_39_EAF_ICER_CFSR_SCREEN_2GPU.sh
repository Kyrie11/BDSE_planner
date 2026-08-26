#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export V30_2_ROOT="${V30_2_ROOT:-outputs_v64_3_30_2_eaf_icer_fbic_pure_screen_2gpu_v1}"
export V38_ROOT="${V38_ROOT:-outputs_v64_3_38_eaf_icer_davr_screen_2gpu_v1}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_39_eaf_icer_cfsr_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_32_design_exclude_v64_3_30_3_screen_tokens.txt}"
export FROZEN_TRAIN_TOKENS="${FROZEN_TRAIN_TOKENS:-bdse/configs/v64_3_29_frozen_v28_train_3000_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.39-eaf-icer-cfsr-cal500-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"

RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
TRAIN_EDGES="$V30_2_ROOT/provenance/train_b16_v20_edges.jsonl"
V38_FIT="$V38_ROOT/provenance/v64_3_38_davr_fit.json"
V38_AUDIT="$V38_ROOT/provenance/v64_3_38_davr_train_scene_audit.csv"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"
TIMING="$OUT_ROOT/provenance/v64_3_39_stage_timing.tsv"; : > "$TIMING"
ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >> "$TIMING"; }

[[ -d "$BDSE_VAL_CACHE" ]] || { echo 'STOP: missing val cache' >&2; exit 2; }
for f in "$RAW_CONFIG" "$V20_CONFIG" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$TRAIN_EDGES" "$V38_FIT" "$V38_AUDIT"; do [[ -s "$f" ]] || { echo "STOP: missing $f" >&2; exit 2; }; done

stage_start exact_v38_failure_reproduction_and_fresh_unspent_guard
python - "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$V38_FIT" "$V38_AUDIT" "$V38_ROOT" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
ex=[x.strip() for x in open(sys.argv[1]) if x.strip()]; tr=[x.strip() for x in open(sys.argv[2]) if x.strip()]
if len(ex)!=10700 or len(set(ex))!=10700 or hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()!='041ec824756777576391756ef3721617459bb0c0a45f7f43226b52254d951473': raise SystemExit('STOP DATA: V39 design exclusion changed')
if len(tr)!=3000 or len(set(tr))!=3000 or hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()!='b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4': raise SystemExit('STOP DATA: V39 frozen TRAIN changed')
r=json.load(open(sys.argv[3])); n=r.get('nested_crossfit',{}); rr=n.get('rsmr_rank_aggregate',{}); de=n.get('dense_all_edge_value_aggregate',{}); da=n.get('davr_selected_calibrated_value_aggregate',{})
checks=[
 r.get('frozen_train_scenes')==3000, r.get('direct_support_positive_training_scenes')==782, r.get('train_gate_pass') is False,
 n.get('failure_diagnosis')=='dense_all_edge_value_does_not_improve_selected_absolute_value_identification_or_capture_tradeoff',
 rr.get('selected_count')==502, rr.get('selected_positive_count')==221, rr.get('no_positive_opportunity_false_intervention_count')==107, rr.get('catastrophic_count')==28, abs(float(rr.get('teacher_improvement_sum',999))-43.29405361274824)<1e-9,
 de.get('selected_count')==263, de.get('selected_positive_count')==138, de.get('no_positive_opportunity_false_intervention_count')==50, de.get('catastrophic_count')==23, abs(float(de.get('teacher_improvement_sum',999))-12.218356950548547)<1e-9, abs(float(de.get('positive_capture_rate',999))-0.24041811846689895)<1e-12,
 da.get('selected_count')==398, da.get('selected_positive_count')==178, da.get('no_positive_opportunity_false_intervention_count')==81, da.get('catastrophic_count')==19, abs(float(da.get('teacher_improvement_sum',999))-34.583525467380696)<1e-9, abs(float(da.get('positive_capture_rate',999))-0.31010452961672474)<1e-12,
 n.get('davr_main_gate',{}).get('pass') is False, n.get('monotone_frozen_winner_contract_valid') is True
]
if not all(checks): raise SystemExit('STOP V39: exact V38 TRAIN failure signature changed')
rows=list(csv.DictReader(open(sys.argv[4],newline='')))
if len(rows)!=782 or len({r['scenario_token'] for r in rows})!=782: raise SystemExit('STOP V39: V38 scene audit cardinality/uniqueness changed')
root=Path(sys.argv[5]); spent=[root/'provenance/val_screen_cal500_fresh1500_tokens.txt',root/'provenance/val_screen_calibration_500_tokens.txt',root/'provenance/val_screen_fresh_A_tokens.txt',root/'provenance/val_screen_fresh_B_tokens.txt']
if any(p.exists() and p.stat().st_size>0 for p in spent): raise SystemExit('STOP V39: V38 appears to have spent CAL/fresh')
print('PASS V39 prerequisite: exact V38 TRAIN failure reproduced; CAL/fresh remains unspent.')
PY
stage_end

stage_start prerequisites_and_regression
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_39_CFSR --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_39.json" > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_39.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False): raise SystemExit('STOP prerequisites')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"; fi
export EAF_CKPT; [[ -s "$EAF_CKPT" ]] || { echo "STOP: missing EAF checkpoint $EAF_CKPT" >&2; exit 2; }
python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_39_eaf_icer_cfsr.py \
  bdse/tests/test_v64_3_38_eaf_icer_davr.py \
  bdse/tests/test_v64_3_37_eaf_icer_pvr.py \
  bdse/tests/test_v64_3_36_eaf_icer_sgrr.py \
  bdse/tests/test_v64_3_35_eaf_icer_fbcsr.py \
  bdse/tests/test_v64_3_34_eaf_icer_rsmr.py \
  bdse/tests/test_v64_3_33_eaf_icer_spcr.py \
  bdse/tests/test_v64_3_32_1_eaf_icer_ssir_weightfix.py \
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

stage_start train_nested_cfsr_gate
PRESERVE_CONFIG="$OUT_ROOT/provenance/v64_3_39_preserve_control.yaml"
DENSE_CONFIG="$OUT_ROOT/provenance/v64_3_39_dense_value.yaml"
RSMR_CONFIG="$OUT_ROOT/provenance/v64_3_39_rsmr_frozen_order.yaml"
CFSR_RAW_CONFIG="$OUT_ROOT/provenance/v64_3_39_cfsr_raw.yaml"
FIT_REPORT="$OUT_ROOT/provenance/v64_3_39_cfsr_fit.json"
TRAIN_SCENE_AUDIT="$OUT_ROOT/provenance/v64_3_39_cfsr_train_scene_audit.csv"
set +e
python -m bdse.tools.fit_v64_3_39_eaf_icer_cfsr --train-frontier-edges "$TRAIN_EDGES" --base-config "$V20_CONFIG" \
  --output-preserve-config "$PRESERVE_CONFIG" --output-rsmr-config "$RSMR_CONFIG" --output-dense-config "$DENSE_CONFIG" --output-cfsr-config "$CFSR_RAW_CONFIG" \
  --output-report "$FIT_REPORT" --output-scene-audit "$TRAIN_SCENE_AUDIT" 2>&1 | tee "$OUT_ROOT/logs/v64_3_39_cfsr_fit.out"
FIT_STATUS=${PIPESTATUS[0]}; set -e; stage_end; [[ $FIT_STATUS -eq 0 ]] || exit "$FIT_STATUS"

stage_start calibration_and_fresh_selection
TOK1500="$OUT_ROOT/provenance/val_screen_cal500_fresh1500_tokens.txt"; TOKCAL="$OUT_ROOT/provenance/val_screen_calibration_500_tokens.txt"; TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"; TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"
SELECT_EXCLUDE="$OUT_ROOT/provenance/v64_3_39_selection_exclude_design_plus_train.txt"; cat "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" | awk 'NF && !seen[$0]++' > "$SELECT_EXCLUDE"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$SELECT_EXCLUDE" --count 1500 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1500" --audit-output "$OUT_ROOT/provenance/v64_3_39_cal500_fresh1500_selection_audit.json" > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1500" > "$TOKCAL"; sed -n '501,1000p' "$TOK1500" > "$TOKA"; tail -n 500 "$TOK1500" > "$TOKB"
python - "$TOK1500" "$TOKCAL" "$TOKA" "$TOKB" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" <<'PY'
import sys
allv=[x.strip() for x in open(sys.argv[1]) if x.strip()]; cal=[x.strip() for x in open(sys.argv[2]) if x.strip()]; A=[x.strip() for x in open(sys.argv[3]) if x.strip()]; B=[x.strip() for x in open(sys.argv[4]) if x.strip()]; ex={x.strip() for x in open(sys.argv[5]) if x.strip()}; tr={x.strip() for x in open(sys.argv[6]) if x.strip()}
if len(allv)!=1500 or len(set(allv))!=1500 or any(len(x)!=500 or len(set(x))!=500 for x in [cal,A,B]): raise SystemExit('STOP DATA: V39 CAL/A/B cardinality')
if set(cal)&set(A) or set(cal)&set(B) or set(A)&set(B) or set(allv)&ex or set(allv)&tr: raise SystemExit('STOP DATA: V39 independence failure')
print('PASS V39 label-free independent CAL500+A500+B500 selection')
PY
stage_end

run_eval_tok(){ local gpu="$1" cfg="$2" tok="$3" maxn="$4" tag="$5"; CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios "$maxn" --scenario-token-file "$tok" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/${tag}_rows.jsonl" --frontier-edge-output "$OUT_ROOT/provenance/${tag}_edges.jsonl" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${tag}.out" 2>&1; }

stage_start independent_CAL500_rsmr_replay_for_CFSR
run_eval_tok "$GPU0" "$RSMR_CONFIG" "$TOKCAL" 500 cal_rsmr
stage_end
stage_start independent_CAL500_translation_fit
DENSE_SHIFT_CONFIG="$OUT_ROOT/provenance/v64_3_39_dense_shift.yaml"; CFSR_MAIN_CONFIG="$OUT_ROOT/provenance/v64_3_39_cfsr.yaml"; VALUE_REPORT="$OUT_ROOT/provenance/v64_3_39_value_fit.json"
python -m bdse.tools.calibrate_v64_3_39_eaf_icer_cfsr --calibration-rows "$OUT_ROOT/provenance/cal_rsmr_rows.jsonl" --calibration-edges "$OUT_ROOT/provenance/cal_rsmr_edges.jsonl" --dense-config "$DENSE_CONFIG" --cfsr-config "$CFSR_RAW_CONFIG" --output-dense-shift-config "$DENSE_SHIFT_CONFIG" --output-cfsr-main-config "$CFSR_MAIN_CONFIG" --output-report "$VALUE_REPORT" 2>&1 | tee "$OUT_ROOT/logs/v64_3_39_value_fit.out"
stage_end

for sp in A B; do
  tok="$TOKA"; [[ "$sp" == B ]] && tok="$TOKB"
  stage_start "fresh_${sp}_wave1"; run_eval_tok "$GPU0" "$RAW_CONFIG" "$tok" 500 "${sp}_raw" & p0=$!; run_eval_tok "$GPU1" "$V20_CONFIG" "$tok" 500 "${sp}_v20" & p1=$!; s=0; wait "$p0"||s=1; wait "$p1"||s=1; [[ $s -eq 0 ]]||exit 2; stage_end
  stage_start "fresh_${sp}_wave2"; run_eval_tok "$GPU0" "$PRESERVE_CONFIG" "$tok" 500 "${sp}_preserve" & p0=$!; run_eval_tok "$GPU1" "$RSMR_CONFIG" "$tok" 500 "${sp}_rsmr" & p1=$!; s=0; wait "$p0"||s=1; wait "$p1"||s=1; [[ $s -eq 0 ]]||exit 2; stage_end
  stage_start "fresh_${sp}_wave3"; run_eval_tok "$GPU0" "$DENSE_CONFIG" "$tok" 500 "${sp}_dense" & p0=$!; run_eval_tok "$GPU1" "$DENSE_SHIFT_CONFIG" "$tok" 500 "${sp}_dense_shift" & p1=$!; s=0; wait "$p0"||s=1; wait "$p1"||s=1; [[ $s -eq 0 ]]||exit 2; stage_end
  stage_start "fresh_${sp}_wave4"; run_eval_tok "$GPU0" "$CFSR_RAW_CONFIG" "$tok" 500 "${sp}_cfsr_raw" & p0=$!; run_eval_tok "$GPU1" "$CFSR_MAIN_CONFIG" "$tok" 500 "${sp}_cfsr" & p1=$!; s=0; wait "$p0"||s=1; wait "$p1"||s=1; [[ $s -eq 0 ]]||exit 2; stage_end
done

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); tags=['raw','v20','preserve','rsmr','dense','dense_shift','cfsr_raw','cfsr']
for sp,tf in [('A',sys.argv[2]),('B',sys.argv[3])]:
 want=[x.strip() for x in open(tf) if x.strip()]; orders=[]
 for tag in tags:
  rs=[json.loads(x) for x in open(root/f'{sp}_{tag}_rows.jsonl') if x.strip()]; got=[str(r['scenario_token']) for r in rs]
  if len(got)!=500 or len(set(got))!=500 or set(got)!=set(want): raise SystemExit(f'STOP DATA: {sp}/{tag} identity mismatch')
  orders.append(got)
 if any(x!=orders[0] for x in orders[1:]): raise SystemExit(f'STOP DATA: {sp} row order differs across arms')
print('PASS V39 paired identity across all 16 untouched A/B arms')
PY
stage_end

stage_start screen
for sp in A B; do
  python -m bdse.tools.check_v64_3_39_eaf_icer_cfsr_split --split-name "$sp" \
   --raw-metrics "$OUT_ROOT/provenance/${sp}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${sp}_raw_rows.jsonl" \
   --v20-metrics "$OUT_ROOT/provenance/${sp}_v20_metrics.json" --v20-rows "$OUT_ROOT/provenance/${sp}_v20_rows.jsonl" --v20-edges "$OUT_ROOT/provenance/${sp}_v20_edges.jsonl" \
   --preserve-metrics "$OUT_ROOT/provenance/${sp}_preserve_metrics.json" --preserve-rows "$OUT_ROOT/provenance/${sp}_preserve_rows.jsonl" --preserve-edges "$OUT_ROOT/provenance/${sp}_preserve_edges.jsonl" \
   --rsmr-metrics "$OUT_ROOT/provenance/${sp}_rsmr_metrics.json" --rsmr-rows "$OUT_ROOT/provenance/${sp}_rsmr_rows.jsonl" --rsmr-edges "$OUT_ROOT/provenance/${sp}_rsmr_edges.jsonl" \
   --dense-metrics "$OUT_ROOT/provenance/${sp}_dense_metrics.json" --dense-rows "$OUT_ROOT/provenance/${sp}_dense_rows.jsonl" --dense-edges "$OUT_ROOT/provenance/${sp}_dense_edges.jsonl" \
   --dense-shift-metrics "$OUT_ROOT/provenance/${sp}_dense_shift_metrics.json" --dense-shift-rows "$OUT_ROOT/provenance/${sp}_dense_shift_rows.jsonl" --dense-shift-edges "$OUT_ROOT/provenance/${sp}_dense_shift_edges.jsonl" \
   --cfsr-raw-metrics "$OUT_ROOT/provenance/${sp}_cfsr_raw_metrics.json" --cfsr-raw-rows "$OUT_ROOT/provenance/${sp}_cfsr_raw_rows.jsonl" --cfsr-raw-edges "$OUT_ROOT/provenance/${sp}_cfsr_raw_edges.jsonl" \
   --cfsr-metrics "$OUT_ROOT/provenance/${sp}_cfsr_metrics.json" --cfsr-rows "$OUT_ROOT/provenance/${sp}_cfsr_rows.jsonl" --cfsr-edges "$OUT_ROOT/provenance/${sp}_cfsr_edges.jsonl" \
   --output "$OUT_ROOT/provenance/v64_3_39_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_39_split_${sp}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_39_eaf_icer_cfsr_double_fresh_screen.json"
python -m bdse.tools.check_v64_3_39_eaf_icer_cfsr_screen --split-a "$OUT_ROOT/provenance/v64_3_39_split_A_screen.json" --split-b "$OUT_ROOT/provenance/v64_3_39_split_B_screen.json" --value-report "$VALUE_REPORT" --fit-report "$FIT_REPORT" --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_39_double_fresh_screen.out"
stage_end

printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"
(cd "$OUT_ROOT/provenance" && find . -maxdepth 1 -type f ! -name 'v64_3_39_provenance_sha256.txt' -print0 | sort -z | xargs -0 sha256sum > v64_3_39_provenance_sha256.txt)
python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); [print(k,'=',r.get(k)) for k in ['engineering_valid_both','train_nested_rank_value_gate_pass','independent_CAL500_translation_fit_valid','double_fresh_promotion_pass','scientific_conclusion','next_action']]
PY
