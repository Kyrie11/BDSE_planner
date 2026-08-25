#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_29_eaf_icer_fcr_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_29_design_exclude_v64_3_28_screen_tokens.txt}"
export FROZEN_TRAIN_TOKENS="${FROZEN_TRAIN_TOKENS:-bdse/configs/v64_3_29_frozen_v28_train_3000_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.29-eaf-icer-fcr-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"

RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
FCR_V20_CONFIG="bdse/configs/v64_3_29_eaf_icer_fcr_v20.yaml"
BASE_FIT_DIR="$OUT_ROOT/configs/baseline_v25"
FCR_FIT_DIR="$OUT_ROOT/configs/fcr_v29"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$BASE_FIT_DIR" "$FCR_FIT_DIR"
TIMING="$OUT_ROOT/provenance/v64_3_29_stage_timing.tsv"; : > "$TIMING"
ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >> "$TIMING"; }

[[ -d "$BDSE_TRAIN_CACHE" && -d "$BDSE_VAL_CACHE" ]] || { echo 'STOP: missing train/val cache' >&2; exit 2; }
for f in "$RAW_CONFIG" "$V20_CONFIG" "$FCR_V20_CONFIG" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS"; do
  [[ -s "$f" ]] || { echo "STOP: missing $f" >&2; exit 2; }
done
python - "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" <<'PY'
import hashlib,sys
ex=[x.strip() for x in open(sys.argv[1]) if x.strip()]
tr=[x.strip() for x in open(sys.argv[2]) if x.strip()]
if len(ex)!=7700 or len(set(ex))!=7700:
    raise SystemExit(f'STOP DATA: V29 design exclusion must be exactly 7700 unique inspected validation tokens, got rows={len(ex)} unique={len(set(ex))}')
if len(tr)!=3000 or len(set(tr))!=3000:
    raise SystemExit(f'STOP DATA: frozen V28/V27 TRAIN manifest must be exactly 3000 unique tokens, got rows={len(tr)} unique={len(set(tr))}')
sha=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
if sha!='b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4':
    raise SystemExit('STOP DATA: frozen 3000 TRAIN token SHA changed: '+sha)
print('PASS V29 identity contracts: 7700 inspected-val exclusion + frozen 3000 TRAIN SHA')
PY

stage_start prerequisites
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
  --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_29 \
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

# TRAIN replay is deliberately paired on the *same historical 3000 tokens*.
# GPU0 reproduces the frozen V25 proposal control; GPU1 changes only the
# deterministic post-EAF evidence rebind. No validation scene is used here.
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
stage_start paired_train_replay
train_eval "$GPU0" "$V20_CONFIG" baseline_v20 & p0=$!
train_eval "$GPU1" "$FCR_V20_CONFIG" fcr_v20 & p1=$!
s=0; wait "$p0" || s=1; wait "$p1" || s=1; [[ $s -eq 0 ]] || { echo 'STOP TRAIN replay failure' >&2; exit 2; }
python - "$OUT_ROOT/provenance/train_baseline_v20_rows.jsonl" "$OUT_ROOT/provenance/train_fcr_v20_rows.jsonl" "$FROZEN_TRAIN_TOKENS" <<'PY'
import json,sys
want={x.strip() for x in open(sys.argv[3]) if x.strip()}
orders=[]
for p in sys.argv[1:3]:
    rows=[json.loads(x) for x in open(p) if x.strip()]
    got=[str(r['scenario_token']) for r in rows]
    if len(got)!=3000 or len(set(got))!=3000 or set(got)!=want:
        raise SystemExit(f'STOP DATA: paired TRAIN identity mismatch {p}: rows={len(got)} unique={len(set(got))} overlap={len(set(got)&want)}')
    orders.append(got)
if orders[0]!=orders[1]:
    raise SystemExit('STOP DATA: paired TRAIN emitted row order differs across baseline/FCR')
print('PASS V29 paired TRAIN identity: exact same frozen 3000 scenes and order')
PY
stage_end

stage_start baseline_v25_refit
python -m bdse.tools.fit_v64_3_25_eaf_icer_drc \
  --train-frontier-edges "$OUT_ROOT/provenance/train_baseline_v20_edges.jsonl" \
  --base-v20-dual-config "$V20_CONFIG" \
  --output-dir "$BASE_FIT_DIR" \
  --output-train-token-file "$OUT_ROOT/provenance/v64_3_29_baseline_train_tokens.txt" \
  --output-report "$OUT_ROOT/provenance/v64_3_29_baseline_v25_train_fit.json" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_29_baseline_v25_fit.out"
python - "$OUT_ROOT/provenance/v64_3_29_baseline_v25_train_fit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
# Historical baseline provenance is a hard causal control, not a tunable target.
exp={'train_scene_count':3000,'frontier_row_count':75133,'replacement_edges':1455,'replacement_scenes':310}
for k,v in exp.items():
    if int(r.get(k,-1))!=v: raise SystemExit(f'STOP BASELINE PROVENANCE: {k}={r.get(k)} expected {v}')
cf=r['crossfit']['aggregate_downside']
if not r.get('train_gate_pass',False) or int(cf.get('fold_pass_count',0))!=5 or int(cf.get('selected_count',0))!=71:
    raise SystemExit('STOP BASELINE REPRODUCTION: historical V25 aggregate DRC TRAIN gate/selection changed')
if abs(float(cf.get('teacher_improvement_sum',float('nan'))) - 5.527642) > 1e-5:
    raise SystemExit('STOP BASELINE REPRODUCTION: historical V25 aggregate DRC teacher-improvement sum drifted')
print('PASS historical V25 aggregate DRC TRAIN provenance reproduction')
PY
stage_end

stage_start fcr_v29_fit
set +e
python -m bdse.tools.fit_v64_3_29_eaf_icer_fcr \
  --train-frontier-edges "$OUT_ROOT/provenance/train_fcr_v20_edges.jsonl" \
  --train-rows "$OUT_ROOT/provenance/train_fcr_v20_rows.jsonl" \
  --base-fcr-v20-config "$FCR_V20_CONFIG" \
  --output-dir "$FCR_FIT_DIR" \
  --output-train-token-file "$OUT_ROOT/provenance/v64_3_29_fcr_train_tokens.txt" \
  --output-report "$OUT_ROOT/provenance/v64_3_29_fcr_train_fit.json" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_29_fcr_fit.out"
FIT_STATUS=${PIPESTATUS[0]}
set -e
stage_end
if [[ "$FIT_STATUS" -ne 0 ]]; then
  echo "STOP TRAIN FCR (status=$FIT_STATUS). Do not spend fresh GPU or tune FCR/DRC knobs." >&2
  exit "$FIT_STATUS"
fi
FCR_DRC="$FCR_FIT_DIR/v64_3_29_fcr_aggregate_downside.yaml"
BASE_DRC="$BASE_FIT_DIR/v64_3_25_aggregate_downside.yaml"
python -m bdse.tools.check_v64_3_29_eaf_icer_fcr_contract \
  --config "$FCR_DRC" --frozen-fcr-v20-config "$FCR_V20_CONFIG" \
  --output "$OUT_ROOT/provenance/v64_3_29_fcr_contract.json"
python - "$FROZEN_TRAIN_TOKENS" "$OUT_ROOT/provenance/v64_3_29_baseline_train_tokens.txt" "$OUT_ROOT/provenance/v64_3_29_fcr_train_tokens.txt" <<'PY'
import hashlib,sys
want=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()
for p in sys.argv[2:]:
    got=hashlib.sha256(open(p,'rb').read()).hexdigest()
    if got!=want: raise SystemExit(f'STOP TRAIN TOKEN SHA: {p} {got} != {want}')
print('PASS baseline/FCR fit token manifests exactly match frozen V28 TRAIN')
PY

stage_start fresh_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh_1000_tokens.txt"
TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"
TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"
python -m bdse.tools.select_fresh_preprocessed_tokens \
  --preprocessed-dir "$BDSE_VAL_CACHE" --split val \
  --exclude-tokens "$DESIGN_EXCLUDE_TOKENS" --count 1000 \
  --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" \
  --audit-output "$OUT_ROOT/provenance/v64_3_29_fresh_1000_audit.json" \
  > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1000" > "$TOKA"; tail -n 500 "$TOK1000" > "$TOKB"
python - "$TOK1000" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" <<'PY'
import sys
fr={x.strip() for x in open(sys.argv[1]) if x.strip()}; ex={x.strip() for x in open(sys.argv[2]) if x.strip()}; tr={x.strip() for x in open(sys.argv[3]) if x.strip()}
if len(fr)!=1000 or len(ex)!=7700 or len(tr)!=3000 or fr&ex or fr&tr:
    raise SystemExit(f'STOP DATA: V29 fresh isolation failure fresh={len(fr)} exclude={len(ex)} train={len(tr)} fresh_ex={len(fr&ex)} fresh_train={len(fr&tr)}')
print('PASS V29 untouched fresh isolation')
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

# Five arms. V20 is historical unsafe/high-coverage context only; V25 aggregate
# DRC is the causal control. FCR-V20 verifies the interface mechanism itself;
# FCR-DRC is the deployable main arm. PTMC is intentionally absent after V28.
for sp in A B; do
  stage_start "fresh_${sp}_wave1"
  run_eval "$GPU0" "$RAW_CONFIG" "$sp" raw & p0=$!
  run_eval "$GPU1" "$V20_CONFIG" "$sp" v20 & p1=$!
  s=0; wait "$p0" || s=1; wait "$p1" || s=1; [[ $s -eq 0 ]] || exit 2
  stage_end

  stage_start "fresh_${sp}_wave2"
  run_eval "$GPU0" "$BASE_DRC" "$sp" aggregate_downside & p0=$!
  run_eval "$GPU1" "$FCR_V20_CONFIG" "$sp" fcr_v20 & p1=$!
  s=0; wait "$p0" || s=1; wait "$p1" || s=1; [[ $s -eq 0 ]] || exit 2
  stage_end

  stage_start "fresh_${sp}_wave3"
  run_eval "$GPU0" "$FCR_DRC" "$sp" fcr_downside
  stage_end
done

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); tags=['raw','v20','aggregate_downside','fcr_v20','fcr_downside']
for sp,tf in [('A',sys.argv[2]),('B',sys.argv[3])]:
    want={x.strip() for x in open(tf) if x.strip()}; orders=[]
    if len(want)!=500: raise SystemExit(f'STOP DATA: {sp} fresh manifest !=500 unique')
    for tag in tags:
        rows=[json.loads(x) for x in open(root/f'{sp}_{tag}_rows.jsonl') if x.strip()]
        got=[str(r['scenario_token']) for r in rows]
        if len(got)!=500 or len(set(got))!=500 or set(got)!=want:
            raise SystemExit(f'STOP DATA: {sp}/{tag} identity mismatch rows={len(got)} unique={len(set(got))} overlap={len(set(got)&want)}')
        orders.append(got)
        m=json.load(open(root/f'{sp}_{tag}_metrics.json'))
        if not m.get('scenario_token_prefilter_active',False):
            raise SystemExit(f'STOP DATA/SPEED: {sp}/{tag} pre-load token filter inactive')
    if any(x!=orders[0] for x in orders[1:]):
        raise SystemExit(f'STOP DATA: {sp} emitted row order differs across arms')
print('PASS V29 paired identity: exact 500-scene set/order across all 10 fresh arms')
PY
stage_end

stage_start screen
for sp in A B; do
  python -m bdse.tools.check_v64_3_29_eaf_icer_fcr_split --split-name "$sp" \
    --raw-metrics "$OUT_ROOT/provenance/${sp}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${sp}_raw_rows.jsonl" \
    --v20-metrics "$OUT_ROOT/provenance/${sp}_v20_metrics.json" --v20-rows "$OUT_ROOT/provenance/${sp}_v20_rows.jsonl" --v20-edges "$OUT_ROOT/provenance/${sp}_v20_edges.jsonl" \
    --aggregate-downside-metrics "$OUT_ROOT/provenance/${sp}_aggregate_downside_metrics.json" --aggregate-downside-rows "$OUT_ROOT/provenance/${sp}_aggregate_downside_rows.jsonl" --aggregate-downside-edges "$OUT_ROOT/provenance/${sp}_aggregate_downside_edges.jsonl" \
    --fcr-v20-metrics "$OUT_ROOT/provenance/${sp}_fcr_v20_metrics.json" --fcr-v20-rows "$OUT_ROOT/provenance/${sp}_fcr_v20_rows.jsonl" --fcr-v20-edges "$OUT_ROOT/provenance/${sp}_fcr_v20_edges.jsonl" \
    --fcr-downside-metrics "$OUT_ROOT/provenance/${sp}_fcr_downside_metrics.json" --fcr-downside-rows "$OUT_ROOT/provenance/${sp}_fcr_downside_rows.jsonl" --fcr-downside-edges "$OUT_ROOT/provenance/${sp}_fcr_downside_edges.jsonl" \
    --output "$OUT_ROOT/provenance/v64_3_29_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_29_split_${sp}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_29_eaf_icer_fcr_double_fresh_screen.json"
python -m bdse.tools.check_v64_3_29_eaf_icer_fcr_screen \
  --split-a-report "$OUT_ROOT/provenance/v64_3_29_split_A_screen.json" \
  --split-b-report "$OUT_ROOT/provenance/v64_3_29_split_B_screen.json" \
  --train-fit-report "$OUT_ROOT/provenance/v64_3_29_fcr_train_fit.json" \
  --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_29_double_fresh_screen.out"
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"
python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for k in ['both_independent_blocks_pass','safe_coverage_gain_both','tail_noninferior_both','catastrophe_free_both','fcr_monotone_contract_both','endpoint_noninferior_both','full_promotion_to_independent_full_val_reproduction','next_action']:
    print(k,'=',r.get(k))
if not r.get('full_promotion_to_independent_full_val_reproduction',False):
    raise SystemExit('STOP SCREEN: do not tune B/FCR objective/DRC thresholds and do not run full/test/closed-loop; follow next_action')
print('PASS SCREEN ONLY: freeze V29. Next allowed stage is one independent full-val reproduction. Test/closed-loop remain forbidden.')
PY
