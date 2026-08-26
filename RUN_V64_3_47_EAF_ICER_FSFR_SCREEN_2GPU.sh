#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export V44_ROOT="${V44_ROOT:-outputs_v64_3_44_eaf_icer_pcor_screen_2gpu_v1}"
export V45_ROOT="${V45_ROOT:-outputs_v64_3_45_eaf_icer_pirf_screen_2gpu_v1}"
export V46_ROOT="${V46_ROOT:-outputs_v64_3_46_eaf_icer_dirp_screen_2gpu_v1}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_47_eaf_icer_fsfr_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_32_design_exclude_v64_3_30_3_screen_tokens.txt}"
export FROZEN_TRAIN_TOKENS="${FROZEN_TRAIN_TOKENS:-bdse/configs/v64_3_29_frozen_v28_train_3000_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.47-eaf-icer-fsfr-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"
RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"; V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
V46_FIT="$V46_ROOT/provenance/v64_3_46_dirp_fit.json"
V45_SUP="$V45_ROOT/provenance/v64_3_45_train_response_field_supervision.jsonl"
V45_SIDECAR="$V45_ROOT/provenance/v64_3_45_crossfit_response_observables.jsonl"
V45_RF_REPORT="$V45_ROOT/provenance/v64_3_45_response_field_crossfit_report.json"
V44_EDGES="$V44_ROOT/provenance/train_v20_plan_conditioned_response_edges.jsonl"
V44_PRESERVE="$V44_ROOT/provenance/v64_3_44_preserve.yaml"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"
TIMING="$OUT_ROOT/provenance/v64_3_47_stage_timing.tsv"; :>"$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1";STAGE_START=$(date +%s);}; stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))">>"$TIMING";}
for f in "$RAW_CONFIG" "$V20_CONFIG" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$V46_FIT" "$V45_SUP" "$V45_SIDECAR" "$V45_RF_REPORT" "$V44_EDGES" "$V44_PRESERVE"; do [[ -s "$f" ]] || { echo "STOP missing $f" >&2; exit 2; }; done
[[ -d "$BDSE_TRAIN_CACHE" && -d "$BDSE_VAL_CACHE" ]] || { echo 'STOP missing train/val cache' >&2; exit 2; }

stage_start exact_v46_prerequisite_and_fresh_guard
python - "$V46_FIT" "$V46_ROOT" "$FROZEN_TRAIN_TOKENS" <<'PY'
import json,sys,hashlib
from pathlib import Path
r=json.load(open(sys.argv[1]));n=r.get('nested_crossfit',{})
e={
'rsmr_rank_aggregate':(502,221,107,28,43.29405361274824),
'quality_control_aggregate':(205,129,30,13,43.905547394411805),
'v45_plan_control_aggregate':(217,121,38,9,56.55117310290402),
'distribution_mean_aggregate':(216,118,39,9,57.52557473140379),
'temporal_profile_aggregate':(217,119,41,14,51.26324847539882),
'dirp_joint_aggregate':(207,115,39,12,55.30307441748546),
}
for k,x in e.items():
 d=n.get(k,{});g=(d.get('selected_count'),d.get('selected_positive_count'),d.get('no_positive_opportunity_false_intervention_count'),d.get('catastrophic_count'),d.get('teacher_improvement_sum'))
 if any(g[i]!=x[i] for i in range(4)) or abs(float(g[4])-x[4])>1e-9: raise SystemExit(f'STOP V47: V46 signature changed {k}: {g}')
if r.get('train_gate_pass') is not False or n.get('failure_diagnosis')!='response_second_moment_is_identifiable_but_acceleration_distribution_or_interaction_profile_still_not_absolute_value_sufficient' or not bool(n.get('distribution_identification',{}).get('identified',False)):
 raise SystemExit('STOP V47: V46 preregistered scientific-stop signature changed')
tr=[x.strip() for x in open(sys.argv[3]) if x.strip()]
if len(tr)!=3000 or len(set(tr))!=3000 or hashlib.sha256(open(sys.argv[3],'rb').read()).hexdigest()!='b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4': raise SystemExit('STOP V47: frozen TRAIN changed')
root=Path(sys.argv[2]);spent=[root/'provenance/val_screen_fresh1000_tokens.txt',root/'provenance/val_screen_fresh_A_tokens.txt',root/'provenance/val_screen_fresh_B_tokens.txt']
if any(p.exists() and p.stat().st_size>0 for p in spent): raise SystemExit('STOP V47: V46 fresh was consumed')
print('PASS V47 prerequisite: exact V46 identified-second-moment/promotion-failure signature and fresh-unspent state reproduced')
PY
stage_end

stage_start prerequisites_and_regression
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_47_FSFR --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_47.json" > "$OUT_ROOT/logs/v64_3_13_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_47.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False): raise SystemExit('STOP prerequisites')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"; fi; export EAF_CKPT; [[ -s "$EAF_CKPT" ]] || { echo 'STOP missing EAF checkpoint' >&2; exit 2; }
python -m compileall -q bdse
pytest -q bdse/tests/test_v64_3_{13_eaf_dmvr,14_eaf_ocfi,15_eaf_eair,16_eaf_raer,17_eaf_daler,18_eaf_dacer,19_eaf_icer,20_eaf_icer_dc,21_eaf_icer_mcr,22_eaf_icer_tcr,23_eaf_icer_rcr,24_eaf_icer_arc,25_eaf_icer_drc,26_eaf_icer_sarc,27_eaf_icer_trcc,28_eaf_icer_ptmc,29_eaf_icer_fcr,30_2_eaf_icer_fbic_pure,30_3_eaf_icer_fbic_pure_auditfix,30_eaf_icer_fbic,31_eaf_icer_scir,32_1_eaf_icer_ssir_weightfix,32_eaf_icer_ssir,33_eaf_icer_spcr,34_eaf_icer_rsmr,35_eaf_icer_fbcsr,36_eaf_icer_sgrr,37_eaf_icer_pvr,38_eaf_icer_davr,39_eaf_icer_cfsr,40_eaf_icer_sdfr,41_eaf_icer_epvr,42_eaf_icer_ovdr,43_eaf_icer_cfrv,44_eaf_icer_pcor,45_eaf_icer_pirf,46_eaf_icer_dirp,47_eaf_icer_fsfr}.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

stage_start future_state_supervision
AGENT_SUP="$OUT_ROOT/provenance/v64_3_47_train_agent_lateral_supervision.jsonl"; EGO_SUP="$OUT_ROOT/provenance/v64_3_47_train_ego_reference_supervision.jsonl"; SUP_AUD="$OUT_ROOT/provenance/v64_3_47_future_state_supervision_audit.json"
python -m bdse.tools.build_v64_3_47_future_state_supervision --preprocessed-dir "$BDSE_TRAIN_CACHE" --split train --scenario-token-file "$FROZEN_TRAIN_TOKENS" --base-config "$V20_CONFIG" --output-agent "$AGENT_SUP" --output-ego "$EGO_SUP" --output-audit "$SUP_AUD" | tee "$OUT_ROOT/logs/v64_3_47_future_state_supervision.out"
stage_end

stage_start crossfit_future_state_factorization
FS_SIDECAR="$OUT_ROOT/provenance/v64_3_47_crossfit_future_state_observables.jsonl"; FS_MODEL="$OUT_ROOT/provenance/v64_3_47_full_train_future_state_model.json"; FS_REPORT="$OUT_ROOT/provenance/v64_3_47_future_state_crossfit_report.json"
python -m bdse.tools.build_v64_3_47_crossfit_future_state_observables --agent-supervision "$AGENT_SUP" --ego-supervision "$EGO_SUP" --v45-supervision "$V45_SUP" --v45-sidecar "$V45_SIDECAR" --v45-response-report "$V45_RF_REPORT" --preprocessed-dir "$BDSE_TRAIN_CACHE" --split train --scenario-token-file "$FROZEN_TRAIN_TOKENS" --base-config "$V20_CONFIG" --output-sidecar "$FS_SIDECAR" --output-model "$FS_MODEL" --output-report "$FS_REPORT" | tee "$OUT_ROOT/logs/v64_3_47_future_state_crossfit.out"
stage_end

stage_start nested_train_fsfr_gate
RSMR_CONFIG="$OUT_ROOT/provenance/v64_3_47_rsmr.yaml"; QUALITY_CONFIG="$OUT_ROOT/provenance/v64_3_47_quality.yaml"; PLAN_CONFIG="$OUT_ROOT/provenance/v64_3_47_plan_control.yaml"; A2D_CONFIG="$OUT_ROOT/provenance/v64_3_47_agent_2d.yaml"; EGO_CONFIG="$OUT_ROOT/provenance/v64_3_47_ego_ref.yaml"; JOINT_CONFIG="$OUT_ROOT/provenance/v64_3_47_fsfr_joint.yaml"; FIT_REPORT="$OUT_ROOT/provenance/v64_3_47_fsfr_fit.json"; SCENE_AUDIT="$OUT_ROOT/provenance/v64_3_47_fsfr_train_scene_audit.csv"
set +e
python -m bdse.tools.fit_v64_3_47_eaf_icer_fsfr --train-frontier-edges "$V44_EDGES" --fsfr-sidecar "$FS_SIDECAR" --fsfr-model "$FS_MODEL" --fsfr-report "$FS_REPORT" --v46-fit-report "$V46_FIT" --base-config "$V20_CONFIG" --output-rsmr-config "$RSMR_CONFIG" --output-quality-config "$QUALITY_CONFIG" --output-plan-control-config "$PLAN_CONFIG" --output-agent-2d-config "$A2D_CONFIG" --output-ego-ref-config "$EGO_CONFIG" --output-fsfr-joint-config "$JOINT_CONFIG" --output-report "$FIT_REPORT" --output-scene-audit "$SCENE_AUDIT" 2>&1 | tee "$OUT_ROOT/logs/v64_3_47_fsfr_fit.out"
FIT_STATUS=${PIPESTATUS[0]};set -e;stage_end;[[ $FIT_STATUS -eq 0 ]] || exit "$FIT_STATUS"

stage_start double_fresh_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh1000_tokens.txt"; TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"; TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"; EX="$OUT_ROOT/provenance/v64_3_47_selection_exclude.txt"
cat "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" | awk 'NF&&!seen[$0]++' > "$EX"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$EX" --count 1000 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" --audit-output "$OUT_ROOT/provenance/v64_3_47_fresh1000_selection_audit.json" > "$OUT_ROOT/logs/fresh_selection.out"
head -n500 "$TOK1000">"$TOKA";tail -n500 "$TOK1000">"$TOKB"
python - "$TOKA" "$TOKB" "$EX" <<'PY'
import sys
A=[x.strip() for x in open(sys.argv[1]) if x.strip()];B=[x.strip() for x in open(sys.argv[2]) if x.strip()];E={x.strip() for x in open(sys.argv[3]) if x.strip()}
if len(A)!=500 or len(B)!=500 or len(set(A))!=500 or len(set(B))!=500 or set(A)&set(B) or (set(A)|set(B))&E: raise SystemExit('STOP V47 fresh independence')
print('PASS V47 independent A500+B500 selection')
PY
stage_end

PREFERRED=$(python - "$FIT_REPORT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['nested_crossfit']['preferred_promotion_arm'] or '')
PY
)
[[ "$PREFERRED" =~ ^(agent_2d|ego_ref|fsfr_joint)$ ]] || { echo 'STOP V47 no preregistered preferred arm' >&2; exit 2; }
run_eval(){ local gpu="$1" cfg="$2" tok="$3" tag="$4"; CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios 500 --scenario-token-file "$tok" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/${tag}_rows.jsonl" --frontier-edge-output "$OUT_ROOT/provenance/${tag}_edges.jsonl" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${tag}.out" 2>&1; }
for sp in A B; do
 tok="$TOKA";[[ "$sp" == B ]]&&tok="$TOKB";tags=(raw v20 preserve rsmr quality plan_control agent_2d ego_ref fsfr_joint);cfgs=("$RAW_CONFIG" "$V20_CONFIG" "$V44_PRESERVE" "$RSMR_CONFIG" "$QUALITY_CONFIG" "$PLAN_CONFIG" "$A2D_CONFIG" "$EGO_CONFIG" "$JOINT_CONFIG")
 for ((i=0;i<9;i+=2));do stage_start "fresh_${sp}_wave_$i";s=0;run_eval "$GPU0" "${cfgs[$i]}" "$tok" "${sp}_${tags[$i]}" & p0=$!;if ((i+1<9));then run_eval "$GPU1" "${cfgs[$((i+1))]}" "$tok" "${sp}_${tags[$((i+1))]}" & p1=$!;else p1='';fi;wait "$p0"||s=1;if [[ -n "$p1" ]];then wait "$p1"||s=1;fi;[[ $s -eq 0 ]]||exit 2;stage_end;done
done

stage_start double_fresh_screen
for sp in A B;do args=();for tag in raw v20 preserve rsmr quality plan_control agent_2d ego_ref fsfr_joint;do x=${tag//_/-};args+=("--${x}-metrics" "$OUT_ROOT/provenance/${sp}_${tag}_metrics.json" "--${x}-rows" "$OUT_ROOT/provenance/${sp}_${tag}_rows.jsonl");[[ "$tag" == raw ]]||args+=("--${x}-edges" "$OUT_ROOT/provenance/${sp}_${tag}_edges.jsonl");done;python -m bdse.tools.check_v64_3_47_eaf_icer_fsfr_split --split-name "$sp" --preferred-arm "$PREFERRED" "${args[@]}" --output "$OUT_ROOT/provenance/v64_3_47_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_47_split_${sp}.out";done
python -m bdse.tools.check_v64_3_47_eaf_icer_fsfr_screen --split-a "$OUT_ROOT/provenance/v64_3_47_split_A_screen.json" --split-b "$OUT_ROOT/provenance/v64_3_47_split_B_screen.json" --fit-report "$FIT_REPORT" --output "$OUT_ROOT/provenance/v64_3_47_eaf_icer_fsfr_double_fresh_screen.json" | tee "$OUT_ROOT/logs/v64_3_47_screen.out"
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))">>"$TIMING"
