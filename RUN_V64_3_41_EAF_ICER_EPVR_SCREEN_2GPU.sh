#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export V30_2_ROOT="${V30_2_ROOT:-outputs_v64_3_30_2_eaf_icer_fbic_pure_screen_2gpu_v1}"
export V40_ROOT="${V40_ROOT:-outputs_v64_3_40_eaf_icer_sdfr_screen_2gpu_v1}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_41_eaf_icer_epvr_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_32_design_exclude_v64_3_30_3_screen_tokens.txt}"
export FROZEN_TRAIN_TOKENS="${FROZEN_TRAIN_TOKENS:-bdse/configs/v64_3_29_frozen_v28_train_3000_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.41-eaf-icer-epvr-cal500-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"
RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"; V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"; TRAIN_EDGES="$V30_2_ROOT/provenance/train_b16_v20_edges.jsonl"; V40_FIT="$V40_ROOT/provenance/v64_3_40_sdfr_fit.json"; V40_AUDIT="$V40_ROOT/provenance/v64_3_40_sdfr_train_scene_audit.csv"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"; TIMING="$OUT_ROOT/provenance/v64_3_41_stage_timing.tsv"; : > "$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }; stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >> "$TIMING"; }
for f in "$RAW_CONFIG" "$V20_CONFIG" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$TRAIN_EDGES" "$V40_FIT" "$V40_AUDIT"; do [[ -s "$f" ]] || { echo "STOP: missing $f" >&2; exit 2; }; done
stage_start exact_v40_failure_reproduction_and_fresh_unspent_guard
python - "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$V40_FIT" "$V40_AUDIT" "$V40_ROOT" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
ex=[x.strip() for x in open(sys.argv[1]) if x.strip()]; tr=[x.strip() for x in open(sys.argv[2]) if x.strip()]
if len(ex)!=10700 or len(set(ex))!=10700 or hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()!='041ec824756777576391756ef3721617459bb0c0a45f7f43226b52254d951473': raise SystemExit('STOP DATA: V41 design exclusion changed')
if len(tr)!=3000 or len(set(tr))!=3000 or hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()!='b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4': raise SystemExit('STOP DATA: V41 frozen TRAIN changed')
r=json.load(open(sys.argv[3])); n=r.get('nested_crossfit',{}); A=[('rsmr_rank_aggregate',502,221,107,28,43.29405361274824,.38501742160278746),('dense_signed_mean_aggregate',263,138,50,23,12.218356950548547,.24041811846689895),('hurdle_dense_distribution_aggregate',241,125,50,20,9.725430165471074,.21777003484320556),('hurdle_sign_shift_only_aggregate',168,92,37,15,8.664215932209943,.1602787456445993),('sdfr_raw_aggregate',339,149,72,20,31.99677262243872,.259581881533101),('sdfr_main_aggregate',403,172,83,22,32.66529945558325,.29965156794425085)]
checks=[r.get('frozen_train_scenes')==3000,r.get('direct_support_positive_training_scenes')==782,r.get('train_gate_pass') is False,n.get('train_gate_pass') is False,n.get('failure_diagnosis')=='selected_distribution_adaptation_adds_tail_signal_but_zero_crossing_capture_tradeoff_remains',n.get('monotone_frozen_winner_contract_valid') is True]
for k,s,p,no,cat,sm,cap in A:
 d=n.get(k,{}); checks += [d.get('selected_count')==s,d.get('selected_positive_count')==p,d.get('no_positive_opportunity_false_intervention_count')==no,d.get('catastrophic_count')==cat,abs(float(d.get('teacher_improvement_sum',999))-sm)<1e-9,abs(float(d.get('positive_capture_rate',999))-cap)<1e-12]
if not all(checks): raise SystemExit('STOP V41: exact V40 TRAIN failure signature changed')
rows=list(csv.DictReader(open(sys.argv[4],newline='')))
if len(rows)!=782 or len({r['scenario_token'] for r in rows})!=782: raise SystemExit('STOP V41: V40 scene audit changed')
root=Path(sys.argv[5]); spent=[root/'provenance/val_screen_cal500_fresh1500_tokens.txt',root/'provenance/val_screen_calibration_500_tokens.txt',root/'provenance/val_screen_fresh_A_tokens.txt',root/'provenance/val_screen_fresh_B_tokens.txt']
if any(p.exists() and p.stat().st_size>0 for p in spent): raise SystemExit('STOP V41: V40 spent CAL/fresh')
print('PASS V41 prerequisite: exact V40 failure reproduced and fresh unspent.')
PY
stage_end
stage_start prerequisites_and_regression
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_41_EPVR --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_41.json" > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_41.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False): raise SystemExit('STOP prerequisites')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"; fi; export EAF_CKPT; [[ -s "$EAF_CKPT" ]] || { echo "STOP: missing EAF checkpoint" >&2; exit 2; }
python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_13_eaf_dmvr.py \
  bdse/tests/test_v64_3_14_eaf_ocfi.py \
  bdse/tests/test_v64_3_15_eaf_eair.py \
  bdse/tests/test_v64_3_16_eaf_raer.py \
  bdse/tests/test_v64_3_17_eaf_daler.py \
  bdse/tests/test_v64_3_18_eaf_dacer.py \
  bdse/tests/test_v64_3_19_eaf_icer.py \
  bdse/tests/test_v64_3_20_eaf_icer_dc.py \
  bdse/tests/test_v64_3_21_eaf_icer_mcr.py \
  bdse/tests/test_v64_3_22_eaf_icer_tcr.py \
  bdse/tests/test_v64_3_23_eaf_icer_rcr.py \
  bdse/tests/test_v64_3_24_eaf_icer_arc.py \
  bdse/tests/test_v64_3_25_eaf_icer_drc.py \
  bdse/tests/test_v64_3_26_eaf_icer_sarc.py \
  bdse/tests/test_v64_3_27_eaf_icer_trcc.py \
  bdse/tests/test_v64_3_28_eaf_icer_ptmc.py \
  bdse/tests/test_v64_3_29_eaf_icer_fcr.py \
  bdse/tests/test_v64_3_30_2_eaf_icer_fbic_pure.py \
  bdse/tests/test_v64_3_30_3_eaf_icer_fbic_pure_auditfix.py \
  bdse/tests/test_v64_3_30_eaf_icer_fbic.py \
  bdse/tests/test_v64_3_31_eaf_icer_scir.py \
  bdse/tests/test_v64_3_32_1_eaf_icer_ssir_weightfix.py \
  bdse/tests/test_v64_3_32_eaf_icer_ssir.py \
  bdse/tests/test_v64_3_33_eaf_icer_spcr.py \
  bdse/tests/test_v64_3_34_eaf_icer_rsmr.py \
  bdse/tests/test_v64_3_35_eaf_icer_fbcsr.py \
  bdse/tests/test_v64_3_36_eaf_icer_sgrr.py \
  bdse/tests/test_v64_3_37_eaf_icer_pvr.py \
  bdse/tests/test_v64_3_38_eaf_icer_davr.py \
  bdse/tests/test_v64_3_39_eaf_icer_cfsr.py \
  bdse/tests/test_v64_3_40_eaf_icer_sdfr.py \
  bdse/tests/test_v64_3_41_eaf_icer_epvr.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end
stage_start train_nested_epvr_gate
PRESERVE_CONFIG="$OUT_ROOT/provenance/v64_3_41_preserve.yaml"; RSMR_CONFIG="$OUT_ROOT/provenance/v64_3_41_rsmr.yaml"; DENSE_CONFIG="$OUT_ROOT/provenance/v64_3_41_dense.yaml"; ZDELTA_CONFIG="$OUT_ROOT/provenance/v64_3_41_zdelta.yaml"; DNL_CONFIG="$OUT_ROOT/provenance/v64_3_41_dnl.yaml"; EPV_RAW_CONFIG="$OUT_ROOT/provenance/v64_3_41_epv_raw.yaml"; FIT_REPORT="$OUT_ROOT/provenance/v64_3_41_epvr_fit.json"; TRAIN_SCENE_AUDIT="$OUT_ROOT/provenance/v64_3_41_epvr_train_scene_audit.csv"
set +e
python -m bdse.tools.fit_v64_3_41_eaf_icer_epvr --train-frontier-edges "$TRAIN_EDGES" --base-config "$V20_CONFIG" --output-preserve-config "$PRESERVE_CONFIG" --output-rsmr-config "$RSMR_CONFIG" --output-dense-config "$DENSE_CONFIG" --output-zdelta-config "$ZDELTA_CONFIG" --output-dnl-config "$DNL_CONFIG" --output-epv-config "$EPV_RAW_CONFIG" --output-report "$FIT_REPORT" --output-scene-audit "$TRAIN_SCENE_AUDIT" 2>&1 | tee "$OUT_ROOT/logs/v64_3_41_epvr_fit.out"
FIT_STATUS=${PIPESTATUS[0]}; set -e; stage_end; [[ $FIT_STATUS -eq 0 ]] || exit "$FIT_STATUS"
stage_start calibration_and_fresh_selection
TOK1500="$OUT_ROOT/provenance/val_screen_cal500_fresh1500_tokens.txt"; TOKCAL="$OUT_ROOT/provenance/val_screen_calibration_500_tokens.txt"; TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"; TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"; SELECT_EXCLUDE="$OUT_ROOT/provenance/v64_3_41_selection_exclude.txt"; cat "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" | awk 'NF && !seen[$0]++' > "$SELECT_EXCLUDE"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$SELECT_EXCLUDE" --count 1500 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1500" --audit-output "$OUT_ROOT/provenance/v64_3_41_cal500_fresh1500_selection_audit.json" > "$OUT_ROOT/logs/fresh_token_selection.out"; head -n500 "$TOK1500">"$TOKCAL"; sed -n '501,1000p' "$TOK1500">"$TOKA"; tail -n500 "$TOK1500">"$TOKB"
python - "$TOK1500" "$TOKCAL" "$TOKA" "$TOKB" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" <<'PY'
import sys
allv=[x.strip() for x in open(sys.argv[1]) if x.strip()]; cal=[x.strip() for x in open(sys.argv[2]) if x.strip()]; A=[x.strip() for x in open(sys.argv[3]) if x.strip()]; B=[x.strip() for x in open(sys.argv[4]) if x.strip()]; ex={x.strip() for x in open(sys.argv[5]) if x.strip()}; tr={x.strip() for x in open(sys.argv[6]) if x.strip()}
if len(allv)!=1500 or len(set(allv))!=1500 or any(len(x)!=500 or len(set(x))!=500 for x in [cal,A,B]): raise SystemExit('STOP DATA: V41 CAL/A/B cardinality')
if set(cal)&set(A) or set(cal)&set(B) or set(A)&set(B) or set(allv)&ex or set(allv)&tr: raise SystemExit('STOP DATA: V41 CAL/A/B independence')
print('PASS V41 label-free independent CAL500+A500+B500 selection')
PY
stage_end
run_eval_tok(){ local gpu="$1" cfg="$2" tok="$3" tag="$4"; CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios 500 --scenario-token-file "$tok" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/${tag}_rows.jsonl" --frontier-edge-output "$OUT_ROOT/provenance/${tag}_edges.jsonl" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${tag}.out" 2>&1; }
stage_start independent_CAL500_rsmr_replay; run_eval_tok "$GPU0" "$RSMR_CONFIG" "$TOKCAL" cal_rsmr; stage_end
stage_start independent_CAL500_translation_fit
EPVR_CONFIG="$OUT_ROOT/provenance/v64_3_41_epvr.yaml"; VALUE_REPORT="$OUT_ROOT/provenance/v64_3_41_value_fit.json"; python -m bdse.tools.calibrate_v64_3_41_eaf_icer_epvr --calibration-rows "$OUT_ROOT/provenance/cal_rsmr_rows.jsonl" --calibration-edges "$OUT_ROOT/provenance/cal_rsmr_edges.jsonl" --epv-config "$EPV_RAW_CONFIG" --output-epvr-config "$EPVR_CONFIG" --output-report "$VALUE_REPORT" | tee "$OUT_ROOT/logs/v64_3_41_value_fit.out"; stage_end
for sp in A B; do tok="$TOKA"; [[ "$sp" == B ]] && tok="$TOKB"; declare -a tags=(raw v20 preserve rsmr dense zdelta dnl epv_raw epvr); declare -a cfgs=("$RAW_CONFIG" "$V20_CONFIG" "$PRESERVE_CONFIG" "$RSMR_CONFIG" "$DENSE_CONFIG" "$ZDELTA_CONFIG" "$DNL_CONFIG" "$EPV_RAW_CONFIG" "$EPVR_CONFIG"); for i in 0 2 4 6 8; do stage_start "fresh_${sp}_wave_$i"; run_eval_tok "$GPU0" "${cfgs[$i]}" "$tok" "${sp}_${tags[$i]}" & p0=$!; if (( i+1 < 9 )); then run_eval_tok "$GPU1" "${cfgs[$((i+1))]}" "$tok" "${sp}_${tags[$((i+1))]}" & p1=$!; fi; s=0; wait "$p0"||s=1; if (( i+1 < 9 )); then wait "$p1"||s=1; fi; [[ $s -eq 0 ]]||exit 2; stage_end; done; done
stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); tags=['raw','v20','preserve','rsmr','dense','zdelta','dnl','epv_raw','epvr']
for sp,tf in [('A',sys.argv[2]),('B',sys.argv[3])]:
 want=[x.strip() for x in open(tf) if x.strip()]; orders=[]
 for tag in tags:
  rs=[json.loads(x) for x in open(root/f'{sp}_{tag}_rows.jsonl') if x.strip()]; got=[str(r['scenario_token']) for r in rs]
  if len(got)!=500 or len(set(got))!=500 or set(got)!=set(want): raise SystemExit(f'STOP DATA: {sp}/{tag} identity mismatch')
  orders.append(got)
 if any(x!=orders[0] for x in orders[1:]): raise SystemExit(f'STOP DATA: {sp} row order differs across arms')
print('PASS V41 paired identity across all A/B arms')
PY
stage_end
stage_start screen
for sp in A B; do args=(); for tag in raw v20 preserve rsmr dense zdelta dnl epv_raw epvr; do x=${tag//_/-}; args+=("--${x}-metrics" "$OUT_ROOT/provenance/${sp}_${tag}_metrics.json" "--${x}-rows" "$OUT_ROOT/provenance/${sp}_${tag}_rows.jsonl"); [[ "$tag" == raw ]] || args+=("--${x}-edges" "$OUT_ROOT/provenance/${sp}_${tag}_edges.jsonl"); done; python -m bdse.tools.check_v64_3_41_eaf_icer_epvr_split --split-name "$sp" "${args[@]}" --output "$OUT_ROOT/provenance/v64_3_41_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_41_split_${sp}.out"; done
SCREEN="$OUT_ROOT/provenance/v64_3_41_eaf_icer_epvr_double_fresh_screen.json"; python -m bdse.tools.check_v64_3_41_eaf_icer_epvr_screen --split-a "$OUT_ROOT/provenance/v64_3_41_split_A_screen.json" --split-b "$OUT_ROOT/provenance/v64_3_41_split_B_screen.json" --value-report "$VALUE_REPORT" --fit-report "$FIT_REPORT" --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_41_screen.out"; stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"
