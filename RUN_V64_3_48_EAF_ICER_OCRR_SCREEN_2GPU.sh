#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export V44_ROOT="${V44_ROOT:-outputs_v64_3_44_eaf_icer_pcor_screen_2gpu_v1}"
export V47_ROOT="${V47_ROOT:-outputs_v64_3_47_eaf_icer_fsfr_screen_2gpu_v1}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_48_eaf_icer_ocrr_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_32_design_exclude_v64_3_30_3_screen_tokens.txt}"
export FROZEN_TRAIN_TOKENS="${FROZEN_TRAIN_TOKENS:-bdse/configs/v64_3_29_frozen_v28_train_3000_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.48-eaf-icer-ocrr-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"

RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
V44_PRESERVE="$V44_ROOT/provenance/v64_3_44_preserve.yaml"
V47_FIT="$V47_ROOT/provenance/v64_3_47_fsfr_fit.json"
V47_AUDIT="$V47_ROOT/provenance/v64_3_47_fsfr_train_scene_audit.csv"
V47_PLAN="$V47_ROOT/provenance/v64_3_47_plan_control.yaml"
V47_EGO="$V47_ROOT/provenance/v64_3_47_ego_ref.yaml"

mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"
TIMING="$OUT_ROOT/provenance/v64_3_48_stage_timing.tsv"; :>"$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >>"$TIMING"; }

for f in "$RAW_CONFIG" "$V20_CONFIG" "$V44_PRESERVE" "$V47_FIT" "$V47_AUDIT" "$V47_PLAN" "$V47_EGO" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" V64_3_48_SOURCE_MANIFEST.sha256; do
  [[ -s "$f" ]] || { echo "STOP V48 missing $f" >&2; exit 2; }
done
[[ -d "$BDSE_VAL_CACHE" ]] || { echo 'STOP V48 missing val cache' >&2; exit 2; }

stage_start source_and_v47_provenance_gate
sha256sum -c V64_3_48_SOURCE_MANIFEST.sha256 | tee "$OUT_ROOT/logs/v64_3_48_source_manifest.out"
python - "$V47_FIT" "$V47_AUDIT" "$V47_PLAN" "$V47_EGO" "$V47_ROOT" "$FROZEN_TRAIN_TOKENS" <<'PY'
import hashlib,json,sys
from pathlib import Path
fit,audit,plan,ego,root,train=map(Path,sys.argv[1:])
expected={
 fit:'1bf6cee0cbfd0c1b5e9c6445a68509e6b9cad7945f81cffdd4831d9f48f64be2',
 audit:'0335316e9e8dcd1cf411f2bab172ddc29bb67c3f6483fdb0f016c1df63bb06ce',
 plan:'0587928a3baa8c0cdd6134ee54d3fb91145757adb3a49b4d425484306a5879c1',
 ego:'c4b32850637604cd8c2dafd464f44bafd8a81a19cb1173549fd673baf5c29ae5',
}
for p,h in expected.items():
 g=hashlib.sha256(p.read_bytes()).hexdigest()
 if g!=h: raise SystemExit(f'STOP V48: V47 prerequisite byte identity changed: {p} {g}')
r=json.loads(fit.read_text());n=r.get('nested_crossfit',{})
exp={
 'rsmr_rank_aggregate':(502,221,107,28,43.29405361274824),
 'quality_control_aggregate':(205,129,30,13,43.905547394411805),
 'v45_plan_control_aggregate':(217,121,38,9,56.55117310290402),
 'agent_2d_aggregate':(213,118,36,10,52.305649566059444),
 'ego_reference_aggregate':(251,136,45,9,59.53269591505746),
 'fsfr_joint_aggregate':(249,135,42,9,57.004928000622115),
}
for k,x in exp.items():
 d=n.get(k,{});g=(d.get('selected_count'),d.get('selected_positive_count'),d.get('no_positive_opportunity_false_intervention_count'),d.get('catastrophic_count'),float(d.get('teacher_improvement_sum',float('nan'))))
 if any(g[i]!=x[i] for i in range(4)) or abs(g[4]-x[4])>1e-9: raise SystemExit(f'STOP V48: V47 signature changed {k}: {g}')
if r.get('train_gate_pass') is not False or n.get('train_gate_pass') is not False:
 raise SystemExit('STOP V48: V47 unexpectedly passed TRAIN gate')
ident=n.get('future_state_identification',{})
if not bool(ident.get('agent_2d_response_identified',False)) or not bool(ident.get('ego_reference_identified',False)):
 raise SystemExit('STOP V48: V47 nuisance identification changed')
if n.get('failure_diagnosis')!='future_state_nuisances_are_identifiable_but_absolute_zero_requires_selected_deployment_decision_functional_or_more_general_future_state':
 raise SystemExit('STOP V48: V47 scientific-stop signature changed')
tr=[x.strip() for x in train.read_text().splitlines() if x.strip()]
if len(tr)!=3000 or len(set(tr))!=3000 or hashlib.sha256(train.read_bytes()).hexdigest()!='b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4':
 raise SystemExit('STOP V48: frozen TRAIN changed')
spent=[root/'provenance/val_screen_fresh1000_tokens.txt',root/'provenance/val_screen_fresh_A_tokens.txt',root/'provenance/val_screen_fresh_B_tokens.txt']
if any(p.exists() and p.stat().st_size>0 for p in spent): raise SystemExit('STOP V48: V47 fresh was consumed')
print('PASS V48 prerequisite: exact V47 bytes, preregistered representation-stop, and fresh-unspent state reproduced')
PY
cp V64_3_48_SOURCE_MANIFEST.sha256 "$OUT_ROOT/provenance/"
stage_end

stage_start prerequisites_and_regression
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_48_OCRR --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_48.json" > "$OUT_ROOT/logs/v64_3_13_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit_for_v64_3_48.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False): raise SystemExit('STOP V48 prerequisites')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"; fi
export EAF_CKPT; [[ -s "$EAF_CKPT" ]] || { echo 'STOP V48 missing EAF checkpoint' >&2; exit 2; }
python -m compileall -q bdse
pytest -q bdse/tests/test_v64_3_{13_eaf_dmvr,14_eaf_ocfi,15_eaf_eair,16_eaf_raer,17_eaf_daler,18_eaf_dacer,19_eaf_icer,20_eaf_icer_dc,21_eaf_icer_mcr,22_eaf_icer_tcr,23_eaf_icer_rcr,24_eaf_icer_arc,25_eaf_icer_drc,26_eaf_icer_sarc,27_eaf_icer_trcc,28_eaf_icer_ptmc,29_eaf_icer_fcr,30_2_eaf_icer_fbic_pure,30_3_eaf_icer_fbic_pure_auditfix,30_eaf_icer_fbic,31_eaf_icer_scir,32_1_eaf_icer_ssir_weightfix,32_eaf_icer_ssir,33_eaf_icer_spcr,34_eaf_icer_rsmr,35_eaf_icer_fbcsr,36_eaf_icer_sgrr,37_eaf_icer_pvr,38_eaf_icer_davr,39_eaf_icer_cfsr,40_eaf_icer_sdfr,41_eaf_icer_epvr,42_eaf_icer_ovdr,43_eaf_icer_cfrv,44_eaf_icer_pcor,45_eaf_icer_pirf,46_eaf_icer_dirp,47_eaf_icer_fsfr,48_eaf_icer_ocrr}.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

stage_start nested_train_operator_gate
NOMULT_CONFIG="$OUT_ROOT/provenance/v64_3_48_sign_nomult.yaml"
MULT_CONFIG="$OUT_ROOT/provenance/v64_3_48_sign_mult.yaml"
FIT_REPORT="$OUT_ROOT/provenance/v64_3_48_ocrr_fit.json"
SCENE_AUDIT="$OUT_ROOT/provenance/v64_3_48_ocrr_train_scene_audit.csv"
set +e
python -m bdse.tools.fit_v64_3_48_eaf_icer_ocrr \
  --v47-fit-report "$V47_FIT" --v47-scene-audit "$V47_AUDIT" \
  --v47-plan-config "$V47_PLAN" --v47-ego-ref-config "$V47_EGO" \
  --output-sign-nomult-config "$NOMULT_CONFIG" --output-sign-mult-config "$MULT_CONFIG" \
  --output-report "$FIT_REPORT" --output-scene-audit "$SCENE_AUDIT" 2>&1 | tee "$OUT_ROOT/logs/v64_3_48_ocrr_fit.out"
FIT_STATUS=${PIPESTATUS[0]}; set -e; stage_end; [[ $FIT_STATUS -eq 0 ]] || exit "$FIT_STATUS"

PREFERRED=$(python - "$FIT_REPORT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['nested_crossfit']['preferred_promotion_arm'] or '')
PY
)
[[ "$PREFERRED" =~ ^(sign_nomult|sign_mult)$ ]] || { echo 'STOP V48 no preregistered preferred arm' >&2; exit 2; }

stage_start double_fresh_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh1000_tokens.txt"; TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"; TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"; EX="$OUT_ROOT/provenance/v64_3_48_selection_exclude.txt"
cat "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" | awk 'NF&&!seen[$0]++' > "$EX"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$EX" --count 1000 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" --audit-output "$OUT_ROOT/provenance/v64_3_48_fresh1000_selection_audit.json" > "$OUT_ROOT/logs/fresh_selection.out"
head -n500 "$TOK1000">"$TOKA"; tail -n500 "$TOK1000">"$TOKB"
python - "$TOKA" "$TOKB" "$EX" <<'PY'
import sys
A=[x.strip() for x in open(sys.argv[1]) if x.strip()];B=[x.strip() for x in open(sys.argv[2]) if x.strip()];E={x.strip() for x in open(sys.argv[3]) if x.strip()}
if len(A)!=500 or len(B)!=500 or len(set(A))!=500 or len(set(B))!=500 or set(A)&set(B) or (set(A)|set(B))&E: raise SystemExit('STOP V48 fresh independence')
print('PASS V48 independent A500+B500 selection')
PY
stage_end

run_eval(){ local gpu="$1" cfg="$2" tok="$3" tag="$4"; CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios 500 --scenario-token-file "$tok" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/${tag}_rows.jsonl" --frontier-edge-output "$OUT_ROOT/provenance/${tag}_edges.jsonl" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${tag}.out" 2>&1; }
RSMR_CONFIG="$V47_ROOT/provenance/v64_3_47_rsmr.yaml"; QUALITY_CONFIG="$V47_ROOT/provenance/v64_3_47_quality.yaml"
for sp in A B; do
 tok="$TOKA"; [[ "$sp" == B ]] && tok="$TOKB"
 tags=(raw v20 preserve rsmr quality plan_control ego_ref sign_nomult sign_mult)
 cfgs=("$RAW_CONFIG" "$V20_CONFIG" "$V44_PRESERVE" "$RSMR_CONFIG" "$QUALITY_CONFIG" "$V47_PLAN" "$V47_EGO" "$NOMULT_CONFIG" "$MULT_CONFIG")
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
 for tag in raw v20 preserve rsmr quality plan_control ego_ref sign_nomult sign_mult; do
  x=${tag//_/-}; args+=("--${x}-metrics" "$OUT_ROOT/provenance/${sp}_${tag}_metrics.json" "--${x}-rows" "$OUT_ROOT/provenance/${sp}_${tag}_rows.jsonl")
  [[ "$tag" == raw ]] || args+=("--${x}-edges" "$OUT_ROOT/provenance/${sp}_${tag}_edges.jsonl")
 done
 python -m bdse.tools.check_v64_3_48_eaf_icer_ocrr_split --split-name "$sp" --preferred-arm "$PREFERRED" "${args[@]}" --output "$OUT_ROOT/provenance/v64_3_48_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_48_split_${sp}.out"
done
python -m bdse.tools.check_v64_3_48_eaf_icer_ocrr_screen --split-a "$OUT_ROOT/provenance/v64_3_48_split_A_screen.json" --split-b "$OUT_ROOT/provenance/v64_3_48_split_B_screen.json" --fit-report "$FIT_REPORT" --output "$OUT_ROOT/provenance/v64_3_48_eaf_icer_ocrr_double_fresh_screen.json" | tee "$OUT_ROOT/logs/v64_3_48_screen.out"
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >>"$TIMING"
