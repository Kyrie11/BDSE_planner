#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export V30_2_ROOT="${V30_2_ROOT:-outputs_v64_3_30_2_eaf_icer_fbic_pure_screen_2gpu_v1}"
export V34_ROOT="${V34_ROOT:-outputs_v64_3_34_eaf_icer_rsmr_screen_2gpu_v1}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_35_eaf_icer_fbcsr_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_32_design_exclude_v64_3_30_3_screen_tokens.txt}"
export FROZEN_TRAIN_TOKENS="${FROZEN_TRAIN_TOKENS:-bdse/configs/v64_3_29_frozen_v28_train_3000_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.35-eaf-icer-fbcsr-cal500-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"

RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
TRAIN_EDGES="$V30_2_ROOT/provenance/train_b16_v20_edges.jsonl"
V34_FIT="$V34_ROOT/provenance/v64_3_34_rsmr_fit.json"
V34_AUDIT="$V34_ROOT/provenance/v64_3_34_rsmr_train_scene_audit.csv"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"
TIMING="$OUT_ROOT/provenance/v64_3_35_stage_timing.tsv"; : > "$TIMING"
ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >> "$TIMING"; }

[[ -d "$BDSE_VAL_CACHE" ]] || { echo 'STOP: missing val cache' >&2; exit 2; }
for f in "$RAW_CONFIG" "$V20_CONFIG" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$TRAIN_EDGES" "$V34_FIT" "$V34_AUDIT"; do [[ -s "$f" ]] || { echo "STOP: missing $f" >&2; exit 2; }; done

stage_start exact_v34_failure_reproduction_and_fresh_unspent_guard
python - "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$V34_FIT" "$V34_AUDIT" "$V34_ROOT" <<'PY'
import csv,hashlib,json,math,sys
from pathlib import Path
ex=[x.strip() for x in open(sys.argv[1]) if x.strip()]; tr=[x.strip() for x in open(sys.argv[2]) if x.strip()]
if len(ex)!=10700 or len(set(ex))!=10700: raise SystemExit('STOP DATA: V35 design exclusion !=10700 unique')
if hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()!='041ec824756777576391756ef3721617459bb0c0a45f7f43226b52254d951473': raise SystemExit('STOP DATA: V35 design exclusion SHA changed')
if len(tr)!=3000 or len(set(tr))!=3000 or hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()!='b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4': raise SystemExit('STOP DATA: V35 frozen TRAIN changed')
r=json.load(open(sys.argv[3])); n=r.get('nested_crossfit',{}); rr=n.get('regret_structured_margin_rank_aggregate',{}); mm=n.get('regret_structured_margin_policy_conformal_aggregate',{})
checks=[r.get('frozen_train_scenes')==3000,r.get('direct_support_positive_training_edges')==9394,r.get('direct_support_positive_training_scenes')==782,r.get('train_gate_pass') is False,
 n.get('rank_mechanism_pass') is True,n.get('fold_pass_count')==0,
 rr.get('selected_count')==502,rr.get('selected_positive_count')==221,rr.get('no_positive_opportunity_false_intervention_count')==107,rr.get('catastrophic_count')==28,
 abs(float(rr.get('teacher_improvement_sum',999))-43.29405361274824)<1e-9,abs(float(rr.get('positive_capture_rate',999))-0.38501742160278746)<1e-12,
 mm.get('selected_count')==5,mm.get('selected_positive_count')==1,mm.get('catastrophic_count')==4,abs(float(mm.get('teacher_improvement_sum',999))-(-1.746619927463273))<1e-9]
if not all(checks): raise SystemExit('STOP V35: exact V34 TRAIN failure signature changed')
rows=list(csv.DictReader(open(sys.argv[4],newline='')))
if len(rows)!=782: raise SystemExit('STOP V35: V34 scene audit cardinality changed')
root=Path(sys.argv[5]); spent=[root/'provenance/val_screen_cal500_fresh1500_tokens.txt',root/'provenance/val_screen_calibration_500_tokens.txt',root/'provenance/val_screen_fresh_A_tokens.txt',root/'provenance/val_screen_fresh_B_tokens.txt']
if any(p.exists() and p.stat().st_size>0 for p in spent): raise SystemExit('STOP V35: V34 appears to have spent CAL/fresh')
print('PASS V35 prerequisite: exact V34 RSMR failure reproduced (RANK 502/221, sum +43.2941, 107 no-op false, 28 catastrophes; MAIN 5/1, 4 catastrophes). V34 did not spend CAL/fresh.')
PY
stage_end

stage_start prerequisites_and_regression
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_35_FBCSR --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_35.json" > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_35.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False): raise SystemExit('STOP prerequisites')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"; fi
export EAF_CKPT; [[ -s "$EAF_CKPT" ]] || { echo "STOP: missing EAF checkpoint $EAF_CKPT" >&2; exit 2; }
python -m compileall -q bdse
pytest -q \
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

stage_start train_nested_fbcsr_gate
PRESERVE_CONFIG="$OUT_ROOT/provenance/v64_3_35_preserve_control.yaml"
MEAN_CONFIG="$OUT_ROOT/provenance/v64_3_35_mean_control.yaml"
RSMR_CONFIG="$OUT_ROOT/provenance/v64_3_35_rsmr_control.yaml"
FACTOR_CONFIG="$OUT_ROOT/provenance/v64_3_35_factorized_delta.yaml"
RANK_CONFIG="$OUT_ROOT/provenance/v64_3_35_fbcsr_rank.yaml"
FIT_REPORT="$OUT_ROOT/provenance/v64_3_35_fbcsr_fit.json"
TRAIN_SCENE_AUDIT="$OUT_ROOT/provenance/v64_3_35_fbcsr_train_scene_audit.csv"
set +e
python -m bdse.tools.fit_v64_3_35_eaf_icer_fbcsr --train-frontier-edges "$TRAIN_EDGES" --base-config "$V20_CONFIG" \
  --output-preserve-config "$PRESERVE_CONFIG" --output-mean-config "$MEAN_CONFIG" --output-rsmr-config "$RSMR_CONFIG" --output-factor-config "$FACTOR_CONFIG" --output-rank-config "$RANK_CONFIG" \
  --output-report "$FIT_REPORT" --output-scene-audit "$TRAIN_SCENE_AUDIT" 2>&1 | tee "$OUT_ROOT/logs/v64_3_35_fbcsr_fit.out"
FIT_STATUS=${PIPESTATUS[0]}; set -e; stage_end; [[ $FIT_STATUS -eq 0 ]] || exit "$FIT_STATUS"

stage_start calibration_and_fresh_selection
TOK1500="$OUT_ROOT/provenance/val_screen_cal500_fresh1500_tokens.txt"; TOKCAL="$OUT_ROOT/provenance/val_screen_calibration_500_tokens.txt"; TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"; TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"
SELECT_EXCLUDE="$OUT_ROOT/provenance/v64_3_35_selection_exclude_design_plus_train.txt"; cat "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" | awk 'NF && !seen[$0]++' > "$SELECT_EXCLUDE"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$SELECT_EXCLUDE" --count 1500 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1500" --audit-output "$OUT_ROOT/provenance/v64_3_35_cal500_fresh1500_selection_audit.json" > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1500" > "$TOKCAL"; sed -n '501,1000p' "$TOK1500" > "$TOKA"; tail -n 500 "$TOK1500" > "$TOKB"
python - "$TOK1500" "$TOKCAL" "$TOKA" "$TOKB" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" <<'PY'
import sys
allv=[x.strip() for x in open(sys.argv[1]) if x.strip()]; cal=[x.strip() for x in open(sys.argv[2]) if x.strip()]; A=[x.strip() for x in open(sys.argv[3]) if x.strip()]; B=[x.strip() for x in open(sys.argv[4]) if x.strip()]; ex={x.strip() for x in open(sys.argv[5]) if x.strip()}; tr={x.strip() for x in open(sys.argv[6]) if x.strip()}
if len(allv)!=1500 or len(set(allv))!=1500 or any(len(x)!=500 or len(set(x))!=500 for x in [cal,A,B]): raise SystemExit('STOP DATA: V35 CAL/A/B cardinality')
if set(cal)&set(A) or set(cal)&set(B) or set(A)&set(B) or set(allv)&ex or set(allv)&tr: raise SystemExit('STOP DATA: V35 independence failure')
print('PASS V35 label-free independent CAL500+A500+B500 selection')
PY
stage_end

run_eval_tok(){ local gpu="$1" cfg="$2" tok="$3" maxn="$4" tag="$5"; CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios "$maxn" --scenario-token-file "$tok" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/${tag}_rows.jsonl" --frontier-edge-output "$OUT_ROOT/provenance/${tag}_edges.jsonl" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${tag}.out" 2>&1; }

stage_start independent_calibration_rank_replay
run_eval_tok "$GPU0" "$RANK_CONFIG" "$TOKCAL" 500 cal_rank
stage_end
stage_start independent_selected_policy_calibration
MAIN_CONFIG="$OUT_ROOT/provenance/v64_3_35_fbcsr_main.yaml"; CAL_REPORT="$OUT_ROOT/provenance/v64_3_35_fbcsr_calibration.json"
python -m bdse.tools.calibrate_v64_3_35_eaf_icer_fbcsr --calibration-rows "$OUT_ROOT/provenance/cal_rank_rows.jsonl" --calibration-edges "$OUT_ROOT/provenance/cal_rank_edges.jsonl" --rank-config "$RANK_CONFIG" --output-main-config "$MAIN_CONFIG" --output-report "$CAL_REPORT" --alpha 0.05 2>&1 | tee "$OUT_ROOT/logs/v64_3_35_fbcsr_calibration.out"
python -m bdse.tools.check_v64_3_35_eaf_icer_fbcsr_contract --v20-config "$V20_CONFIG" --preserve-config "$PRESERVE_CONFIG" --mean-config "$MEAN_CONFIG" --rsmr-config "$RSMR_CONFIG" --factor-config "$FACTOR_CONFIG" --rank-config "$RANK_CONFIG" --main-config "$MAIN_CONFIG" --calibration-report "$CAL_REPORT" --output "$OUT_ROOT/provenance/v64_3_35_fbcsr_contract.json" | tee "$OUT_ROOT/logs/v64_3_35_fbcsr_contract.out"
stage_end

for sp in A B; do
  tok="$TOKA"; [[ "$sp" == B ]] && tok="$TOKB"
  stage_start "fresh_${sp}_wave1"; run_eval_tok "$GPU0" "$RAW_CONFIG" "$tok" 500 "${sp}_raw" & p0=$!; run_eval_tok "$GPU1" "$V20_CONFIG" "$tok" 500 "${sp}_v20" & p1=$!; s=0; wait "$p0"||s=1; wait "$p1"||s=1; [[ $s -eq 0 ]]||exit 2; stage_end
  stage_start "fresh_${sp}_wave2"; run_eval_tok "$GPU0" "$PRESERVE_CONFIG" "$tok" 500 "${sp}_preserve" & p0=$!; run_eval_tok "$GPU1" "$MEAN_CONFIG" "$tok" 500 "${sp}_mean" & p1=$!; s=0; wait "$p0"||s=1; wait "$p1"||s=1; [[ $s -eq 0 ]]||exit 2; stage_end
  stage_start "fresh_${sp}_wave3"; run_eval_tok "$GPU0" "$RSMR_CONFIG" "$tok" 500 "${sp}_rsmr" & p0=$!; run_eval_tok "$GPU1" "$FACTOR_CONFIG" "$tok" 500 "${sp}_factor" & p1=$!; s=0; wait "$p0"||s=1; wait "$p1"||s=1; [[ $s -eq 0 ]]||exit 2; stage_end
  stage_start "fresh_${sp}_wave4"; run_eval_tok "$GPU0" "$RANK_CONFIG" "$tok" 500 "${sp}_rank" & p0=$!; run_eval_tok "$GPU1" "$MAIN_CONFIG" "$tok" 500 "${sp}_main" & p1=$!; s=0; wait "$p0"||s=1; wait "$p1"||s=1; [[ $s -eq 0 ]]||exit 2; stage_end
done

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); tags=['raw','v20','preserve','mean','rsmr','factor','rank','main']
for sp,tf in [('A',sys.argv[2]),('B',sys.argv[3])]:
 want=[x.strip() for x in open(tf) if x.strip()]; orders=[]
 for tag in tags:
  rs=[json.loads(x) for x in open(root/f'{sp}_{tag}_rows.jsonl') if x.strip()]; got=[str(r['scenario_token']) for r in rs]
  if len(got)!=500 or len(set(got))!=500 or set(got)!=set(want): raise SystemExit(f'STOP DATA: {sp}/{tag} identity mismatch')
  orders.append(got)
 if any(x!=orders[0] for x in orders[1:]): raise SystemExit(f'STOP DATA: {sp} row order differs across arms')
print('PASS V35 paired identity across all 16 untouched A/B arms')
PY
stage_end

stage_start screen
for sp in A B; do
  python -m bdse.tools.check_v64_3_35_eaf_icer_fbcsr_split --split-name "$sp" \
   --raw-metrics "$OUT_ROOT/provenance/${sp}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${sp}_raw_rows.jsonl" \
   --v20-metrics "$OUT_ROOT/provenance/${sp}_v20_metrics.json" --v20-rows "$OUT_ROOT/provenance/${sp}_v20_rows.jsonl" --v20-edges "$OUT_ROOT/provenance/${sp}_v20_edges.jsonl" \
   --preserve-metrics "$OUT_ROOT/provenance/${sp}_preserve_metrics.json" --preserve-rows "$OUT_ROOT/provenance/${sp}_preserve_rows.jsonl" --preserve-edges "$OUT_ROOT/provenance/${sp}_preserve_edges.jsonl" \
   --mean-metrics "$OUT_ROOT/provenance/${sp}_mean_metrics.json" --mean-rows "$OUT_ROOT/provenance/${sp}_mean_rows.jsonl" --mean-edges "$OUT_ROOT/provenance/${sp}_mean_edges.jsonl" \
   --rsmr-metrics "$OUT_ROOT/provenance/${sp}_rsmr_metrics.json" --rsmr-rows "$OUT_ROOT/provenance/${sp}_rsmr_rows.jsonl" --rsmr-edges "$OUT_ROOT/provenance/${sp}_rsmr_edges.jsonl" \
   --factor-metrics "$OUT_ROOT/provenance/${sp}_factor_metrics.json" --factor-rows "$OUT_ROOT/provenance/${sp}_factor_rows.jsonl" --factor-edges "$OUT_ROOT/provenance/${sp}_factor_edges.jsonl" \
   --rank-metrics "$OUT_ROOT/provenance/${sp}_rank_metrics.json" --rank-rows "$OUT_ROOT/provenance/${sp}_rank_rows.jsonl" --rank-edges "$OUT_ROOT/provenance/${sp}_rank_edges.jsonl" \
   --main-metrics "$OUT_ROOT/provenance/${sp}_main_metrics.json" --main-rows "$OUT_ROOT/provenance/${sp}_main_rows.jsonl" --main-edges "$OUT_ROOT/provenance/${sp}_main_edges.jsonl" \
   --output "$OUT_ROOT/provenance/v64_3_35_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_35_split_${sp}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_35_eaf_icer_fbcsr_double_fresh_screen.json"
python -m bdse.tools.check_v64_3_35_eaf_icer_fbcsr_screen --split-a "$OUT_ROOT/provenance/v64_3_35_split_A_screen.json" --split-b "$OUT_ROOT/provenance/v64_3_35_split_B_screen.json" --calibration-report "$CAL_REPORT" --fit-report "$FIT_REPORT" --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_35_double_fresh_screen.out"
stage_end

printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"
(cd "$OUT_ROOT/provenance" && find . -maxdepth 1 -type f ! -name 'v64_3_35_provenance_sha256.txt' -print0 | sort -z | xargs -0 sha256sum > v64_3_35_provenance_sha256.txt)
python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); [print(k,'=',r.get(k)) for k in ['engineering_valid_both','train_nested_full_mechanism_gate_pass','independent_calibration_valid','double_fresh_promotion_pass','scientific_conclusion','next_action']]
PY
