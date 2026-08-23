#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_30_eaf_icer_fbic_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_30_design_exclude_v64_3_29_screen_tokens.txt}"
export FROZEN_TRAIN_TOKENS="${FROZEN_TRAIN_TOKENS:-bdse/configs/v64_3_29_frozen_v28_train_3000_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.30-eaf-icer-fbic-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"

RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
B16_V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
B24_V20_CONFIG="bdse/configs/v64_3_30_eaf_icer_fbic_v20.yaml"
BASE_FIT_DIR="$OUT_ROOT/configs/baseline_v25_b16"
FBIC_FIT_DIR="$OUT_ROOT/configs/fbic_b24"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$BASE_FIT_DIR" "$FBIC_FIT_DIR"
TIMING="$OUT_ROOT/provenance/v64_3_30_stage_timing.tsv"; : > "$TIMING"
ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >> "$TIMING"; }

[[ -d "$BDSE_TRAIN_CACHE" && -d "$BDSE_VAL_CACHE" ]] || { echo 'STOP: missing train/val cache' >&2; exit 2; }
for f in "$RAW_CONFIG" "$B16_V20_CONFIG" "$B24_V20_CONFIG" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS"; do
  [[ -s "$f" ]] || { echo "STOP: missing $f" >&2; exit 2; }
done
python - "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" <<'PY'
import hashlib,sys
ex=[x.strip() for x in open(sys.argv[1]) if x.strip()]
tr=[x.strip() for x in open(sys.argv[2]) if x.strip()]
if len(ex)!=8700 or len(set(ex))!=8700:
    raise SystemExit(f'STOP DATA: V30 design exclusion must be exactly 8700 unique already-inspected validation tokens, got rows={len(ex)} unique={len(set(ex))}')
if len(tr)!=3000 or len(set(tr))!=3000:
    raise SystemExit(f'STOP DATA: frozen TRAIN manifest must be exactly 3000 unique tokens, got rows={len(tr)} unique={len(set(tr))}')
sha=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
if sha!='b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4':
    raise SystemExit('STOP DATA: frozen 3000 TRAIN token SHA changed: '+sha)
print('PASS V30 identity contracts: 8700 inspected-val exclusion + frozen 3000 TRAIN SHA')
PY

stage_start prerequisites
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
  --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_30 \
  --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" \
  > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False):
    raise SystemExit('STOP prerequisites')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then
  EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"
fi
export EAF_CKPT
[[ -s "$EAF_CKPT" ]] || { echo "STOP: missing EAF checkpoint $EAF_CKPT" >&2; exit 2; }
python -m compileall -q bdse
pytest -q \
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
python -m bdse.tools.check_v64_3_30_eaf_icer_fbic_contract \
  --config "$B24_V20_CONFIG" --frozen-v20-config "$B16_V20_CONFIG" \
  --output "$OUT_ROOT/provenance/v64_3_30_fbic_contract.json" \
  > "$OUT_ROOT/logs/v64_3_30_fbic_contract.out"
stage_end

train_eval(){
  local gpu="$1" cfg="$2" tag="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop \
    --config "$cfg" --checkpoint "$EAF_CKPT" --split train \
    --preprocessed-dir "$BDSE_TRAIN_CACHE" --max-scenarios 3000 \
    --scenario-token-file "$FROZEN_TRAIN_TOKENS" --require-all-scenario-tokens \
    --output "$OUT_ROOT/provenance/train_${tag}_metrics.json" \
    --per-sample-output "$OUT_ROOT/provenance/train_${tag}_rows.jsonl" \
    --frontier-edge-output "$OUT_ROOT/provenance/train_${tag}_edges.jsonl" \
    --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/train_${tag}.out" 2>&1
}

# Paired TRAIN replay. The B24 arm changes only retained-interface capacity;
# its underlying AOCC baseline still uses B16 and M remains 24.
stage_start paired_train_replay
train_eval "$GPU0" "$B16_V20_CONFIG" b16_v20 & p0=$!
train_eval "$GPU1" "$B24_V20_CONFIG" b24_v20 & p1=$!
s=0; wait "$p0" || s=1; wait "$p1" || s=1; [[ $s -eq 0 ]] || { echo 'STOP TRAIN replay failure' >&2; exit 2; }
python - "$OUT_ROOT/provenance/train_b16_v20_rows.jsonl" "$OUT_ROOT/provenance/train_b24_v20_rows.jsonl" "$FROZEN_TRAIN_TOKENS" <<'PY'
import json,sys
want={x.strip() for x in open(sys.argv[3]) if x.strip()}; orders=[]
for p in sys.argv[1:3]:
    rows=[json.loads(x) for x in open(p) if x.strip()]
    got=[str(r['scenario_token']) for r in rows]
    if len(got)!=3000 or len(set(got))!=3000 or set(got)!=want:
        raise SystemExit(f'STOP DATA: paired TRAIN identity mismatch {p}: rows={len(got)} unique={len(set(got))} overlap={len(set(got)&want)}')
    orders.append(got)
if orders[0]!=orders[1]: raise SystemExit('STOP DATA: paired TRAIN emitted row order differs')
# Capacity must be real on safe-domain scenes and must not issue more queries.
b16={json.loads(x)['scenario_token']:json.loads(x) for x in open(sys.argv[1]) if x.strip()}
b24={json.loads(x)['scenario_token']:json.loads(x) for x in open(sys.argv[2]) if x.strip()}
safe=[t for t,r in b16.items() if float(r.get('all_actions_safety_flagged_rate',0))<0.5]
apply=sum(float(b24[t].get('selector_full_bank_capacity_probe_applied',0))>=0.5 for t in safe)/max(len(safe),1)
inc=sum(float(b24[t].get('decision_budget_atom_count',0))-float(b16[t].get('decision_budget_atom_count',0)) for t in safe)/max(len(safe),1)
for t in b24:
    if abs(float(b24[t].get('upstream_configured_decision_budget_atom_count',float('nan')))-16.0)>1e-9:
        raise SystemExit(f'STOP CAPACITY CONTRACT: B24 upstream selector budget accounting is not 16 at {t}')
    if abs(float(b24[t].get('configured_decision_budget_atom_count',float('nan')))-24.0)>1e-9:
        raise SystemExit(f'STOP CAPACITY CONTRACT: B24 retained interface budget accounting is not 24 at {t}')
    if float(b24[t].get('retained_interface_atom_budget_pass',0.0))<0.5:
        raise SystemExit(f'STOP CAPACITY CONTRACT: B24 retained interface falsely exceeds configured ceiling at {t}')
for k in ['action_atom_query_count','proposal_candidate_atom_count','effective_query_action_count']:
    if any(abs(float(b16[t].get(k,0))-float(b24[t].get(k,0)))>1e-9 for t in b16):
        raise SystemExit(f'STOP CAPACITY CONTRACT: query count changed for {k}')
if apply<0.90 or inc<4.0:
    raise SystemExit(f'STOP CAPACITY CONTRACT: FBIC not materially active on TRAIN safe domain applied={apply:.4f} mean_atom_increase={inc:.4f}')
print(f'PASS V30 paired TRAIN capacity isolation: safe_applied={apply:.4f}, mean_retained_atom_increase={inc:.4f}, exact query parity')
PY
stage_end

stage_start baseline_v25_refit
python -m bdse.tools.fit_v64_3_25_eaf_icer_drc \
  --train-frontier-edges "$OUT_ROOT/provenance/train_b16_v20_edges.jsonl" \
  --base-v20-dual-config "$B16_V20_CONFIG" \
  --output-dir "$BASE_FIT_DIR" \
  --output-train-token-file "$OUT_ROOT/provenance/v64_3_30_baseline_train_tokens.txt" \
  --output-report "$OUT_ROOT/provenance/v64_3_30_baseline_v25_train_fit.json" \
  > "$OUT_ROOT/logs/v64_3_30_baseline_v25_fit.out" 2>&1
python - "$OUT_ROOT/provenance/v64_3_30_baseline_v25_train_fit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); exp={'train_scene_count':3000,'frontier_row_count':75133,'replacement_edges':1455,'replacement_scenes':310}
for k,v in exp.items():
    if int(r.get(k,-1))!=v: raise SystemExit(f'STOP BASELINE PROVENANCE: {k}={r.get(k)} expected {v}')
cf=r['crossfit']['aggregate_downside']
if not r.get('train_gate_pass',False) or int(cf.get('fold_pass_count',0))!=5 or int(cf.get('selected_count',0))!=71:
    raise SystemExit('STOP BASELINE REPRODUCTION: historical V25 TRAIN gate/selection changed')
if abs(float(cf.get('teacher_improvement_sum',float('nan'))) - 5.527642) > 1e-5:
    raise SystemExit('STOP BASELINE REPRODUCTION: historical V25 teacher-improvement sum drifted')
print('PASS historical B16 V25 TRAIN provenance reproduction')
PY
stage_end

stage_start fbic_b24_fit
python -m bdse.tools.fit_v64_3_30_eaf_icer_fbic \
  --train-frontier-edges "$OUT_ROOT/provenance/train_b24_v20_edges.jsonl" \
  --base-fbic-v20-config "$B24_V20_CONFIG" \
  --output-dir "$FBIC_FIT_DIR" \
  --output-train-token-file "$OUT_ROOT/provenance/v64_3_30_fbic_train_tokens.txt" \
  --output-report "$OUT_ROOT/provenance/v64_3_30_fbic_train_fit.json" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_30_fbic_fit.out"
stage_end
B16_DRC="$BASE_FIT_DIR/v64_3_25_aggregate_downside.yaml"
B24_DRC="$FBIC_FIT_DIR/v64_3_30_fbic_aggregate_downside.yaml"
[[ -s "$B16_DRC" && -s "$B24_DRC" ]] || { echo 'STOP: missing fitted DRC configs' >&2; exit 2; }
python - "$B24_DRC" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1])); p=c['selector']['full_bank_capacity_probe']
assert c['evidence']['budget']==16 and c['fallback']['budget_stages']==[16] and c['selector']['proposal_top_m']==24 and p['enabled'] and p['baseline_selector_budget']==16 and p['interface_budget']==24
assert c['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']['replacement_regret_risk_enabled'] is True
assert c['metadata']['algorithm_version']=='V64.3.30-EAF-ICER-FBIC-DRC'
print('PASS generated FBIC DRC config keeps upstream B16 and post-selector B24 retained-interface contract')
PY
python - "$FROZEN_TRAIN_TOKENS" "$OUT_ROOT/provenance/v64_3_30_baseline_train_tokens.txt" "$OUT_ROOT/provenance/v64_3_30_fbic_train_tokens.txt" <<'PY'
import hashlib,sys
want=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()
for p in sys.argv[2:]:
    got=hashlib.sha256(open(p,'rb').read()).hexdigest()
    if got!=want: raise SystemExit(f'STOP TRAIN TOKEN SHA: {p} {got} != {want}')
print('PASS B16/B24 fit token manifests exactly match frozen TRAIN')
PY

stage_start fresh_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh_1000_tokens.txt"
TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"
TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"
python -m bdse.tools.select_fresh_preprocessed_tokens \
  --preprocessed-dir "$BDSE_VAL_CACHE" --split val \
  --exclude-tokens "$DESIGN_EXCLUDE_TOKENS" --count 1000 \
  --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" \
  --audit-output "$OUT_ROOT/provenance/v64_3_30_fresh_1000_audit.json" \
  > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1000" > "$TOKA"; tail -n 500 "$TOK1000" > "$TOKB"
python - "$TOK1000" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" <<'PY'
import sys
fr={x.strip() for x in open(sys.argv[1]) if x.strip()}; ex={x.strip() for x in open(sys.argv[2]) if x.strip()}; tr={x.strip() for x in open(sys.argv[3]) if x.strip()}
if len(fr)!=1000 or len(ex)!=8700 or len(tr)!=3000 or fr&ex or fr&tr:
    raise SystemExit(f'STOP DATA: V30 fresh isolation failure fresh={len(fr)} exclude={len(ex)} train={len(tr)} fresh_ex={len(fr&ex)} fresh_train={len(fr&tr)}')
print('PASS V30 untouched fresh isolation')
PY
stage_end

run_eval(){
  local gpu="$1" cfg="$2" sp="$3" tag="$4"; local tok="$TOKA"; [[ "$sp" == B ]] && tok="$TOKB"
  CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop \
    --config "$cfg" --checkpoint "$EAF_CKPT" --split val \
    --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios 500 \
    --scenario-token-file "$tok" --require-all-scenario-tokens \
    --output "$OUT_ROOT/provenance/${sp}_${tag}_metrics.json" \
    --per-sample-output "$OUT_ROOT/provenance/${sp}_${tag}_rows.jsonl" \
    --frontier-edge-output "$OUT_ROOT/provenance/${sp}_${tag}_edges.jsonl" \
    --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${sp}_${tag}.out" 2>&1
}

# Five paired arms. B24 is a *capacity ceiling*, not a candidate paper method.
for sp in A B; do
  stage_start "fresh_${sp}_wave1"
  run_eval "$GPU0" "$RAW_CONFIG" "$sp" raw & p0=$!
  run_eval "$GPU1" "$B16_V20_CONFIG" "$sp" b16_v20 & p1=$!
  s=0; wait "$p0" || s=1; wait "$p1" || s=1; [[ $s -eq 0 ]] || exit 2
  stage_end

  stage_start "fresh_${sp}_wave2"
  run_eval "$GPU0" "$B16_DRC" "$sp" b16_drc & p0=$!
  run_eval "$GPU1" "$B24_V20_CONFIG" "$sp" b24_v20 & p1=$!
  s=0; wait "$p0" || s=1; wait "$p1" || s=1; [[ $s -eq 0 ]] || exit 2
  stage_end

  stage_start "fresh_${sp}_wave3"
  run_eval "$GPU0" "$B24_DRC" "$sp" b24_drc
  stage_end
done

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); tags=['raw','b16_v20','b16_drc','b24_v20','b24_drc']
for sp,tf in [('A',sys.argv[2]),('B',sys.argv[3])]:
    want={x.strip() for x in open(tf) if x.strip()}; orders=[]
    if len(want)!=500: raise SystemExit(f'STOP DATA: {sp} manifest !=500 unique')
    for tag in tags:
        rs=[json.loads(x) for x in open(root/f'{sp}_{tag}_rows.jsonl') if x.strip()]
        got=[str(r['scenario_token']) for r in rs]
        if len(got)!=500 or len(set(got))!=500 or set(got)!=want:
            raise SystemExit(f'STOP DATA: {sp}/{tag} identity mismatch')
        orders.append(got)
        m=json.load(open(root/f'{sp}_{tag}_metrics.json'))
        if not m.get('scenario_token_prefilter_active',False): raise SystemExit(f'STOP DATA/SPEED: {sp}/{tag} token prefilter inactive')
    if any(x!=orders[0] for x in orders[1:]): raise SystemExit(f'STOP DATA: {sp} row order differs across arms')
print('PASS V30 paired identity across all 10 fresh arms')
PY
stage_end

stage_start screen
for sp in A B; do
  python -m bdse.tools.check_v64_3_30_eaf_icer_fbic_split --split-name "$sp" \
    --raw-metrics "$OUT_ROOT/provenance/${sp}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${sp}_raw_rows.jsonl" \
    --b16-v20-metrics "$OUT_ROOT/provenance/${sp}_b16_v20_metrics.json" --b16-v20-rows "$OUT_ROOT/provenance/${sp}_b16_v20_rows.jsonl" --b16-v20-edges "$OUT_ROOT/provenance/${sp}_b16_v20_edges.jsonl" \
    --b16-drc-metrics "$OUT_ROOT/provenance/${sp}_b16_drc_metrics.json" --b16-drc-rows "$OUT_ROOT/provenance/${sp}_b16_drc_rows.jsonl" --b16-drc-edges "$OUT_ROOT/provenance/${sp}_b16_drc_edges.jsonl" \
    --b24-v20-metrics "$OUT_ROOT/provenance/${sp}_b24_v20_metrics.json" --b24-v20-rows "$OUT_ROOT/provenance/${sp}_b24_v20_rows.jsonl" --b24-v20-edges "$OUT_ROOT/provenance/${sp}_b24_v20_edges.jsonl" \
    --b24-drc-metrics "$OUT_ROOT/provenance/${sp}_b24_drc_metrics.json" --b24-drc-rows "$OUT_ROOT/provenance/${sp}_b24_drc_rows.jsonl" --b24-drc-edges "$OUT_ROOT/provenance/${sp}_b24_drc_edges.jsonl" \
    --output "$OUT_ROOT/provenance/v64_3_30_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_30_split_${sp}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_30_eaf_icer_fbic_double_fresh_screen.json"
python -m bdse.tools.check_v64_3_30_eaf_icer_fbic_screen \
  --split-a-report "$OUT_ROOT/provenance/v64_3_30_split_A_screen.json" \
  --split-b-report "$OUT_ROOT/provenance/v64_3_30_split_B_screen.json" \
  --b24-train-fit-report "$OUT_ROOT/provenance/v64_3_30_fbic_train_fit.json" \
  --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_30_double_fresh_screen.out"
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"
python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for k in ['fbic_contract_both','preservation_contract_both','pure_capacity_signal_both','safe_DRC_capacity_gain_both','tail_safe_both','endpoint_noninferior_both','scientific_conclusion','next_action']:
    print(k,'=',r.get(k))
print('V30 is a causal ceiling diagnostic. Follow next_action; do not tune B=20/22/24 after seeing these blocks.')
PY
