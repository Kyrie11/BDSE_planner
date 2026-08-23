#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_30_2_eaf_icer_fbic_pure_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_30_design_exclude_v64_3_29_screen_tokens.txt}"
export FROZEN_TRAIN_TOKENS="${FROZEN_TRAIN_TOKENS:-bdse/configs/v64_3_29_frozen_v28_train_3000_tokens.txt}"
# V30.1 stopped before fresh-token selection. Keep the exact V30 seed so this is
# the untouched pure-capacity test that the original five-arm launcher never reached.
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.30-eaf-icer-fbic-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"

RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
B16_V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
B24_V20_CONFIG="bdse/configs/v64_3_30_eaf_icer_fbic_v20.yaml"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"
TIMING="$OUT_ROOT/provenance/v64_3_30_2_stage_timing.tsv"; : > "$TIMING"
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
    raise SystemExit(f'STOP DATA: V30.2 design exclusion must remain exactly 8700 unique inspected validation tokens, got rows={len(ex)} unique={len(set(ex))}')
if len(tr)!=3000 or len(set(tr))!=3000:
    raise SystemExit(f'STOP DATA: frozen TRAIN manifest must be exactly 3000 unique tokens, got rows={len(tr)} unique={len(set(tr))}')
sha=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
if sha!='b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4':
    raise SystemExit('STOP DATA: frozen 3000 TRAIN token SHA changed: '+sha)
print('PASS V30.2 identity contracts: 8700 design exclusion + frozen 3000 TRAIN SHA')
PY

stage_start prerequisites
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
  --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_30_2 \
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
python -m bdse.tools.check_v64_3_30_eaf_icer_fbic_contract \
  --config "$B24_V20_CONFIG" --frozen-v20-config "$B16_V20_CONFIG" \
  --output "$OUT_ROOT/provenance/v64_3_30_2_fbic_contract.json" \
  > "$OUT_ROOT/logs/v64_3_30_2_fbic_contract.out"
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
        raise SystemExit(f'STOP DATA: paired TRAIN identity mismatch {p}')
    orders.append(got)
if orders[0]!=orders[1]: raise SystemExit('STOP DATA: paired TRAIN emitted row order differs')
print('PASS V30.2 paired TRAIN identity')
PY
BASE_FIT_DIR="$OUT_ROOT/configs/baseline_v25_b16"; FBIC_FIT_DIR="$OUT_ROOT/configs/fbic_b24_diagnostic_only"
mkdir -p "$BASE_FIT_DIR" "$FBIC_FIT_DIR"
python -m bdse.tools.fit_v64_3_25_eaf_icer_drc \
  --train-frontier-edges "$OUT_ROOT/provenance/train_b16_v20_edges.jsonl" \
  --base-v20-dual-config "$B16_V20_CONFIG" \
  --output-dir "$BASE_FIT_DIR" \
  --output-train-token-file "$OUT_ROOT/provenance/v64_3_30_2_baseline_train_tokens.txt" \
  --output-report "$OUT_ROOT/provenance/v64_3_30_2_baseline_v25_train_fit.json" \
  > "$OUT_ROOT/logs/v64_3_30_2_baseline_v25_fit.out" 2>&1
# Expected development failure: write the B24 report, preserve the non-zero exit,
# and explicitly continue only when the failure is the frozen fold-safety gate.
set +e
python -m bdse.tools.fit_v64_3_30_eaf_icer_fbic \
  --train-frontier-edges "$OUT_ROOT/provenance/train_b24_v20_edges.jsonl" \
  --base-fbic-v20-config "$B24_V20_CONFIG" \
  --output-dir "$FBIC_FIT_DIR" \
  --output-train-token-file "$OUT_ROOT/provenance/v64_3_30_2_fbic_train_tokens.txt" \
  --output-report "$OUT_ROOT/provenance/v64_3_30_2_fbic_train_fit.json" \
  > "$OUT_ROOT/logs/v64_3_30_2_fbic_fit.out" 2>&1
FBIC_FIT_STATUS=$?
set -e
[[ $FBIC_FIT_STATUS -ne 0 && -s "$OUT_ROOT/provenance/v64_3_30_2_fbic_train_fit.json" ]] || { echo 'STOP V30.2: expected auditable B24 DRC TRAIN gate failure did not reproduce' >&2; exit 2; }
python -m bdse.tools.audit_v64_3_30_1_fbic_train \
  --b16-metrics "$OUT_ROOT/provenance/train_b16_v20_metrics.json" \
  --b16-rows "$OUT_ROOT/provenance/train_b16_v20_rows.jsonl" \
  --b24-metrics "$OUT_ROOT/provenance/train_b24_v20_metrics.json" \
  --b24-rows "$OUT_ROOT/provenance/train_b24_v20_rows.jsonl" \
  --b16-fit-report "$OUT_ROOT/provenance/v64_3_30_2_baseline_v25_train_fit.json" \
  --b24-fit-report "$OUT_ROOT/provenance/v64_3_30_2_fbic_train_fit.json" \
  --output "$OUT_ROOT/provenance/v64_3_30_1_repaired_train_audit.json" \
  | tee "$OUT_ROOT/logs/v64_3_30_1_repaired_train_audit.out"
stage_end

# The repaired V30.1 result proved the unchanged B24 DRC is not 5/5 fold-safe.
# Fresh causal arms are exactly raw/B16-V20/B24-V20.
# V30.2 intentionally does NOT lower that gate and does NOT evaluate B24 DRC on fresh data.
# It now executes the missing independent pure capacity control only.
stage_start fresh_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh_1000_tokens.txt"
TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"
TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"
python -m bdse.tools.select_fresh_preprocessed_tokens \
  --preprocessed-dir "$BDSE_VAL_CACHE" --split val \
  --exclude-tokens "$DESIGN_EXCLUDE_TOKENS" --count 1000 \
  --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" \
  --audit-output "$OUT_ROOT/provenance/v64_3_30_2_fresh_1000_audit.json" \
  > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1000" > "$TOKA"; tail -n 500 "$TOK1000" > "$TOKB"
python - "$TOK1000" "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" <<'PY'
import sys
fr={x.strip() for x in open(sys.argv[1]) if x.strip()}; ex={x.strip() for x in open(sys.argv[2]) if x.strip()}; tr={x.strip() for x in open(sys.argv[3]) if x.strip()}
if len(fr)!=1000 or len(ex)!=8700 or len(tr)!=3000 or fr&ex or fr&tr:
    raise SystemExit(f'STOP DATA: V30.2 fresh isolation failure fresh={len(fr)} fresh_ex={len(fr&ex)} fresh_train={len(fr&tr)}')
print('PASS V30.2 untouched fresh isolation')
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

for sp in A B; do
  stage_start "fresh_${sp}_wave1"
  run_eval "$GPU0" "$RAW_CONFIG" "$sp" raw & p0=$!
  run_eval "$GPU1" "$B16_V20_CONFIG" "$sp" b16_v20 & p1=$!
  s=0; wait "$p0" || s=1; wait "$p1" || s=1; [[ $s -eq 0 ]] || exit 2
  stage_end

  stage_start "fresh_${sp}_wave2"
  run_eval "$GPU0" "$B24_V20_CONFIG" "$sp" b24_v20
  stage_end
done

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); tags=['raw','b16_v20','b24_v20']
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
print('PASS V30.2 paired identity across all 6 fresh arms')
PY
stage_end

stage_start screen
for sp in A B; do
  python -m bdse.tools.check_v64_3_30_2_eaf_icer_fbic_pure_split --split-name "$sp" \
    --raw-metrics "$OUT_ROOT/provenance/${sp}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${sp}_raw_rows.jsonl" \
    --b16-v20-metrics "$OUT_ROOT/provenance/${sp}_b16_v20_metrics.json" --b16-v20-rows "$OUT_ROOT/provenance/${sp}_b16_v20_rows.jsonl" --b16-v20-edges "$OUT_ROOT/provenance/${sp}_b16_v20_edges.jsonl" \
    --b24-v20-metrics "$OUT_ROOT/provenance/${sp}_b24_v20_metrics.json" --b24-v20-rows "$OUT_ROOT/provenance/${sp}_b24_v20_rows.jsonl" --b24-v20-edges "$OUT_ROOT/provenance/${sp}_b24_v20_edges.jsonl" \
    --output "$OUT_ROOT/provenance/v64_3_30_2_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_30_2_split_${sp}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_30_2_eaf_icer_fbic_pure_double_fresh_screen.json"
python -m bdse.tools.check_v64_3_30_2_eaf_icer_fbic_pure_screen \
  --split-a-report "$OUT_ROOT/provenance/v64_3_30_2_split_A_screen.json" \
  --split-b-report "$OUT_ROOT/provenance/v64_3_30_2_split_B_screen.json" \
  --train-audit "$OUT_ROOT/provenance/v64_3_30_1_repaired_train_audit.json" \
  --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_30_2_double_fresh_screen.out"
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"
python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for k in ['engineering_valid','pure_capacity_capture_signal_both','capacity_action_switch_nonharmful_both','capacity_common_B16_opportunity_nonharmful_both','endpoint_noninferior_both','scientific_conclusion','next_action']:
    print(k,'=',r.get(k))
print('V30.2 completes the missing pure-capacity causal test. B=16 remains an operating-point control, not the paper thesis.')
PY
