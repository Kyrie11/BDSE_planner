#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export V41_ROOT="${V41_ROOT:-outputs_v64_3_41_eaf_icer_epvr_screen_2gpu_v1}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_42_eaf_icer_ovdr_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_32_design_exclude_v64_3_30_3_screen_tokens.txt}"
export FROZEN_TRAIN_TOKENS="${FROZEN_TRAIN_TOKENS:-bdse/configs/v64_3_29_frozen_v28_train_3000_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.42-eaf-icer-ovdr-cal500-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"
RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"; V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
V41_FIT="$V41_ROOT/provenance/v64_3_41_epvr_fit.json"; V41_AUDIT="$V41_ROOT/provenance/v64_3_41_epvr_train_scene_audit.csv"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"; TIMING="$OUT_ROOT/provenance/v64_3_42_stage_timing.tsv"; : > "$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }; stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >> "$TIMING"; }
for f in "$RAW_CONFIG" "$V20_CONFIG" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$V41_FIT" "$V41_AUDIT"; do [[ -s "$f" ]] || { echo "STOP: missing $f" >&2; exit 2; }; done
[[ -d "$BDSE_TRAIN_CACHE" && -d "$BDSE_VAL_CACHE" ]] || { echo 'STOP: missing train/val cache' >&2; exit 2; }

stage_start exact_v41_failure_reproduction_and_fresh_unspent_guard
python - "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$V41_FIT" "$V41_AUDIT" "$V41_ROOT" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
ex=[x.strip() for x in open(sys.argv[1]) if x.strip()]; tr=[x.strip() for x in open(sys.argv[2]) if x.strip()]
if len(ex)!=10700 or len(set(ex))!=10700 or hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()!='041ec824756777576391756ef3721617459bb0c0a45f7f43226b52254d951473': raise SystemExit('STOP DATA: V42 design exclusion changed')
if len(tr)!=3000 or len(set(tr))!=3000 or hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()!='b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4': raise SystemExit('STOP DATA: V42 frozen TRAIN changed')
r=json.load(open(sys.argv[3])); n=r.get('nested_crossfit',{}); A=[('rsmr_rank_aggregate',502,221,107,28,43.29405361274824,.38501742160278746),('zero_delta_aggregate',227,128,43,22,10.152048616558941,.2229965156794425),('delta_nonlinear_aggregate',317,124,60,17,30.71892468439927,.21602787456445993),('endpoint_potential_raw_aggregate',203,118,39,15,25.968878782061825,.20557491289198607),('endpoint_potential_main_aggregate',397,184,77,22,40.36294771700071,.3205574912891986)]
checks=[r.get('frozen_train_scenes')==3000,r.get('direct_support_positive_training_scenes')==782,r.get('train_gate_pass') is False,n.get('train_gate_pass') is False,n.get('failure_diagnosis')=='endpoint_potential_adds_tail_signal_but_zero_crossing_still_fails',n.get('monotone_frozen_winner_contract_valid') is True]
for k,s,p,no,cat,sm,cap in A:
 d=n.get(k,{}); checks += [d.get('selected_count')==s,d.get('selected_positive_count')==p,d.get('no_positive_opportunity_false_intervention_count')==no,d.get('catastrophic_count')==cat,abs(float(d.get('teacher_improvement_sum',999))-sm)<1e-9,abs(float(d.get('positive_capture_rate',999))-cap)<1e-12]
if not all(checks): raise SystemExit('STOP V42: exact V41 TRAIN failure signature changed')
rows=list(csv.DictReader(open(sys.argv[4],newline='')))
if len(rows)!=782 or len({r['scenario_token'] for r in rows})!=782: raise SystemExit('STOP V42: V41 scene audit changed')
root=Path(sys.argv[5]); spent=[root/'provenance/val_screen_cal500_fresh1500_tokens.txt',root/'provenance/val_screen_calibration_500_tokens.txt',root/'provenance/val_screen_fresh_A_tokens.txt',root/'provenance/val_screen_fresh_B_tokens.txt']
if any(p.exists() and p.stat().st_size>0 for p in spent): raise SystemExit('STOP V42: V41 spent CAL/fresh')
print('PASS V42 prerequisite: exact V41 failure reproduced and fresh unspent.')
PY
stage_end

stage_start prerequisites_and_regression
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_42_OVDR --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_42.json" > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_42.json" <<'PY'
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
  bdse/tests/test_v64_3_41_eaf_icer_epvr.py \
  bdse/tests/test_v64_3_42_eaf_icer_ovdr.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

stage_start train_value_observable_frontier_replay
TRAIN_INSTRUMENT_CONFIG="$OUT_ROOT/provenance/v64_3_42_train_instrument_v20.yaml"; TRAIN_EDGES="$OUT_ROOT/provenance/train_v20_value_observable_frontier_edges.jsonl"; TRAIN_COMPLETE_MARKER="$OUT_ROOT/provenance/v64_3_42_train_value_observable_frontier_complete.ok"
python - "$V20_CONFIG" "$TRAIN_INSTRUMENT_CONFIG" <<'PY'
import sys,yaml
src,out=sys.argv[1:3]; c=yaml.safe_load(open(src)); ic=c.setdefault('runtime',{}).setdefault('decisive_frontier_value',{}).setdefault('incumbent_contrastive_extremal_recovery',{}); ic['instrument_value_observables']=True; c.setdefault('metadata',{})['v64_3_42_train_instrument_only']=True; open(out,'w').write(yaml.safe_dump(c,sort_keys=False))
PY
if [[ -n "${V42_TRAIN_EDGES:-}" && -s "${V42_TRAIN_EDGES}" ]]; then TRAIN_EDGES="$V42_TRAIN_EDGES"; printf 'reuse\t%s\n' "$TRAIN_EDGES" > "$OUT_ROOT/provenance/v64_3_42_train_frontier_source.tsv";
elif [[ -s "$TRAIN_EDGES" && -s "$TRAIN_COMPLETE_MARKER" ]]; then printf 'reuse\t%s\n' "$TRAIN_EDGES" > "$OUT_ROOT/provenance/v64_3_42_train_frontier_source.tsv";
else
 printf 'replay\t%s\n' "$TRAIN_EDGES" > "$OUT_ROOT/provenance/v64_3_42_train_frontier_source.tsv"
 CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop --config "$TRAIN_INSTRUMENT_CONFIG" --checkpoint "$EAF_CKPT" --split train --preprocessed-dir "$BDSE_TRAIN_CACHE" --max-scenarios 3000 --scenario-token-file "$FROZEN_TRAIN_TOKENS" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/train_v20_value_observable_metrics.json" --frontier-edge-output "$TRAIN_EDGES" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/train_value_observable_frontier_replay.out" 2>&1
 printf 'complete\n' > "$TRAIN_COMPLETE_MARKER"
fi
python - "$TRAIN_EDGES" "$FROZEN_TRAIN_TOKENS" <<'PY'
import json,sys
from bdse.planner.value_observables import VALUE_OBSERVABLE_NAMES
want={x.strip() for x in open(sys.argv[2]) if x.strip()}; seen=set(); rows=0
for line in open(sys.argv[1]):
 if not line.strip(): continue
 r=json.loads(line); t=str(r.get('scenario_token','')); seen.add(t); rows+=1
 for n in VALUE_OBSERVABLE_NAMES:
  k='icer_value_observable_'+n
  if k not in r: raise SystemExit(f'STOP V42 TRAIN: missing observable field {k}')
if seen!=want: raise SystemExit(f'STOP V42 TRAIN: frontier scenes {len(seen)} != frozen tokens {len(want)}')
print(f'PASS V42 TRAIN observable frontier: {len(seen)} scenes / {rows} edge rows')
PY
stage_end

stage_start train_nested_ovdr_gate
PRESERVE_CONFIG="$OUT_ROOT/provenance/v64_3_42_preserve.yaml"; RSMR_CONFIG="$OUT_ROOT/provenance/v64_3_42_rsmr.yaml"; EPV_CONFIG="$OUT_ROOT/provenance/v64_3_42_epv_raw.yaml"; QUALITY_CONFIG="$OUT_ROOT/provenance/v64_3_42_quality.yaml"; RISK_CONFIG="$OUT_ROOT/provenance/v64_3_42_risk.yaml"; JOINT_CONFIG="$OUT_ROOT/provenance/v64_3_42_joint_raw.yaml"; FIT_REPORT="$OUT_ROOT/provenance/v64_3_42_ovdr_fit.json"; TRAIN_SCENE_AUDIT="$OUT_ROOT/provenance/v64_3_42_ovdr_train_scene_audit.csv"
set +e
python -m bdse.tools.fit_v64_3_42_eaf_icer_ovdr --train-frontier-edges "$TRAIN_EDGES" --base-config "$V20_CONFIG" --output-preserve-config "$PRESERVE_CONFIG" --output-rsmr-config "$RSMR_CONFIG" --output-epv-config "$EPV_CONFIG" --output-quality-config "$QUALITY_CONFIG" --output-risk-config "$RISK_CONFIG" --output-joint-config "$JOINT_CONFIG" --output-report "$FIT_REPORT" --output-scene-audit "$TRAIN_SCENE_AUDIT" 2>&1 | tee "$OUT_ROOT/logs/v64_3_42_ovdr_fit.out"
FIT_STATUS=${PIPESTATUS[0]}; set -e; stage_end; [[ $FIT_STATUS -eq 0 ]] || exit "$FIT_STATUS"

stage_start calibration_and_fresh_selection
TOK1500="$OUT_ROOT/provenance/val_screen_cal500_fresh1500_tokens.txt"; TOKCAL="$OUT_ROOT/provenance/val_screen_calibration_500_tokens.txt"; TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"; TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"; SELECT_EXCLUDE="$OUT_ROOT/provenance/v64_3_42_selection_exclude.txt"; cat "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" | awk 'NF && !seen[$0]++' > "$SELECT_EXCLUDE"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$SELECT_EXCLUDE" --count 1500 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1500" --audit-output "$OUT_ROOT/provenance/v64_3_42_cal500_fresh1500_selection_audit.json" > "$OUT_ROOT/logs/fresh_token_selection.out"; head -n500 "$TOK1500">"$TOKCAL"; sed -n '501,1000p' "$TOK1500">"$TOKA"; tail -n500 "$TOK1500">"$TOKB"
python - "$TOK1500" "$TOKCAL" "$TOKA" "$TOKB" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" <<'PY'
import sys
allv=[x.strip() for x in open(sys.argv[1]) if x.strip()]; cal=[x.strip() for x in open(sys.argv[2]) if x.strip()]; A=[x.strip() for x in open(sys.argv[3]) if x.strip()]; B=[x.strip() for x in open(sys.argv[4]) if x.strip()]; ex={x.strip() for x in open(sys.argv[5]) if x.strip()}; tr={x.strip() for x in open(sys.argv[6]) if x.strip()}
if len(allv)!=1500 or len(set(allv))!=1500 or any(len(x)!=500 or len(set(x))!=500 for x in [cal,A,B]): raise SystemExit('STOP DATA: V42 CAL/A/B cardinality')
if set(cal)&set(A) or set(cal)&set(B) or set(A)&set(B) or set(allv)&ex or set(allv)&tr: raise SystemExit('STOP DATA: V42 CAL/A/B independence')
print('PASS V42 label-free independent CAL500+A500+B500 selection')
PY
stage_end
run_eval_tok(){ local gpu="$1" cfg="$2" tok="$3" tag="$4"; CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios 500 --scenario-token-file "$tok" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/${tag}_rows.jsonl" --frontier-edge-output "$OUT_ROOT/provenance/${tag}_edges.jsonl" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${tag}.out" 2>&1; }
stage_start independent_CAL500_rsmr_replay; run_eval_tok "$GPU0" "$RSMR_CONFIG" "$TOKCAL" cal_rsmr; stage_end
stage_start independent_CAL500_translation_fit
OVDR_CONFIG="$OUT_ROOT/provenance/v64_3_42_ovdr.yaml"; VALUE_REPORT="$OUT_ROOT/provenance/v64_3_42_value_fit.json"; python -m bdse.tools.calibrate_v64_3_42_eaf_icer_ovdr --calibration-rows "$OUT_ROOT/provenance/cal_rsmr_rows.jsonl" --calibration-edges "$OUT_ROOT/provenance/cal_rsmr_edges.jsonl" --joint-config "$JOINT_CONFIG" --output-main-config "$OVDR_CONFIG" --output-report "$VALUE_REPORT" | tee "$OUT_ROOT/logs/v64_3_42_value_fit.out"; stage_end
for sp in A B; do tok="$TOKA"; [[ "$sp" == B ]] && tok="$TOKB"; declare -a tags=(raw v20 preserve rsmr epv_raw quality risk joint_raw ovdr); declare -a cfgs=("$RAW_CONFIG" "$V20_CONFIG" "$PRESERVE_CONFIG" "$RSMR_CONFIG" "$EPV_CONFIG" "$QUALITY_CONFIG" "$RISK_CONFIG" "$JOINT_CONFIG" "$OVDR_CONFIG"); for i in 0 2 4 6 8; do stage_start "fresh_${sp}_wave_$i"; run_eval_tok "$GPU0" "${cfgs[$i]}" "$tok" "${sp}_${tags[$i]}" & p0=$!; if (( i+1 < 9 )); then run_eval_tok "$GPU1" "${cfgs[$((i+1))]}" "$tok" "${sp}_${tags[$((i+1))]}" & p1=$!; fi; s=0; wait "$p0"||s=1; if (( i+1 < 9 )); then wait "$p1"||s=1; fi; [[ $s -eq 0 ]]||exit 2; stage_end; done; done
stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); tags=['raw','v20','preserve','rsmr','epv_raw','quality','risk','joint_raw','ovdr']
for sp,tf in [('A',sys.argv[2]),('B',sys.argv[3])]:
 want=[x.strip() for x in open(tf) if x.strip()]; orders=[]
 for tag in tags:
  rs=[json.loads(x) for x in open(root/f'{sp}_{tag}_rows.jsonl') if x.strip()]; got=[str(r['scenario_token']) for r in rs]
  if len(got)!=500 or len(set(got))!=500 or set(got)!=set(want): raise SystemExit(f'STOP DATA: {sp}/{tag} identity mismatch')
  orders.append(got)
 if any(x!=orders[0] for x in orders[1:]): raise SystemExit(f'STOP DATA: {sp} row order differs across arms')
print('PASS V42 paired identity across all A/B arms')
PY
stage_end
stage_start screen
for sp in A B; do args=(); for tag in raw v20 preserve rsmr epv_raw quality risk joint_raw ovdr; do x=${tag//_/-}; args+=("--${x}-metrics" "$OUT_ROOT/provenance/${sp}_${tag}_metrics.json" "--${x}-rows" "$OUT_ROOT/provenance/${sp}_${tag}_rows.jsonl"); [[ "$tag" == raw ]] || args+=("--${x}-edges" "$OUT_ROOT/provenance/${sp}_${tag}_edges.jsonl"); done; python -m bdse.tools.check_v64_3_42_eaf_icer_ovdr_split --split-name "$sp" "${args[@]}" --output "$OUT_ROOT/provenance/v64_3_42_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_42_split_${sp}.out"; done
SCREEN="$OUT_ROOT/provenance/v64_3_42_eaf_icer_ovdr_double_fresh_screen.json"; python -m bdse.tools.check_v64_3_42_eaf_icer_ovdr_screen --split-a "$OUT_ROOT/provenance/v64_3_42_split_A_screen.json" --split-b "$OUT_ROOT/provenance/v64_3_42_split_B_screen.json" --value-report "$VALUE_REPORT" --fit-report "$FIT_REPORT" --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_42_screen.out"; stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"
