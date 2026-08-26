#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_27_eaf_icer_trcc_screen_2gpu_v1}"
# V25 fresh A/B have now been inspected and are permanently excluded.  The
# packaged manifest is the exact frozen 5700-token V23 exclusion plus the 1000
# V25 fresh tokens (A/B), with hard uniqueness/overlap checks below.
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_27_design_exclude_v64_3_26_train_stop_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.27-eaf-icer-trcc-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"

RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
TRAIN_TOKENS="$OUT_ROOT/provenance/v64_3_27_train_tokens.txt"
FIT_DIR="$OUT_ROOT/configs"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$FIT_DIR"
TIMING="$OUT_ROOT/provenance/v64_3_27_stage_timing.tsv"; : > "$TIMING"
ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >> "$TIMING"; }

[[ -d "$BDSE_TRAIN_CACHE" && -d "$BDSE_VAL_CACHE" ]] || { echo 'STOP: missing train/val cache' >&2; exit 2; }
[[ -s "$DESIGN_EXCLUDE_TOKENS" ]] || { echo 'STOP: missing V27 design exclusion' >&2; exit 2; }
python - "$DESIGN_EXCLUDE_TOKENS" <<'PY'
import sys
x=[t.strip() for t in open(sys.argv[1]) if t.strip()]
if len(x)!=6700 or len(set(x))!=6700:
    raise SystemExit(f'STOP DATA: V27 design exclusion must be exactly 6700 unique inspected validation tokens, got rows={len(x)} unique={len(set(x))}')
print('PASS V27 frozen 6700-token design exclusion (V26 TRAIN STOP consumed no fresh tokens)')
PY
for f in "$RAW_CONFIG" "$V20_CONFIG"; do [[ -s "$f" ]] || { echo "STOP: missing $f" >&2; exit 2; }; done

stage_start prerequisites
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
  --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_27 \
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

# V27 needs one new TRAIN instrumentation replay because V26 frontier provenance
# serialized only coarse family coordinates, not the 12 fixed atom-type coordinates.
# V26 TRAIN STOP consumed no fresh validation scenes.  This is instrumentation only: V20/EAF/selector/frontier are
# frozen, and the new features are diagnostics until the TRAIN fitter constructs
# the two risk-memory arms.  A previously completed V27 frontier may be reused.
stage_start train_type_frontier_reuse_or_replay
TRAIN_EDGES=""
TRAIN_COMPLETE_MARKER="$OUT_ROOT/provenance/v64_3_27_train_type_frontier_complete.ok"
if [[ -n "${V27_TRAIN_EDGES:-}" && -s "${V27_TRAIN_EDGES}" ]]; then
  TRAIN_EDGES="$V27_TRAIN_EDGES"
elif [[ -s "$OUT_ROOT/provenance/train_v20_type_resolved_frontier_edges.jsonl" && -s "$TRAIN_COMPLETE_MARKER" ]]; then
  TRAIN_EDGES="$OUT_ROOT/provenance/train_v20_type_resolved_frontier_edges.jsonl"
fi
if [[ -n "$TRAIN_EDGES" ]]; then
  printf 'reuse\t%s\n' "$TRAIN_EDGES" > "$OUT_ROOT/provenance/v64_3_27_train_frontier_source.tsv"
  echo "REUSE V27 type-resolved TRAIN frontier: $TRAIN_EDGES" | tee "$OUT_ROOT/logs/train_frontier_source.out"
else
  TRAIN_EDGES="$OUT_ROOT/provenance/train_v20_type_resolved_frontier_edges.jsonl"
  printf 'replay\t%s\n' "$TRAIN_EDGES" > "$OUT_ROOT/provenance/v64_3_27_train_frontier_source.tsv"
  echo 'Replaying frozen 3000-scene V20/EAF TRAIN frontier once to instrument fixed atom-type attribution coordinates.' | tee "$OUT_ROOT/logs/train_frontier_source.out"
  CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop \
    --config "$V20_CONFIG" --checkpoint "$EAF_CKPT" --split train \
    --preprocessed-dir "$BDSE_TRAIN_CACHE" --max-scenarios 3000 \
    --output "$OUT_ROOT/provenance/train_v20_type_resolved_metrics.json" \
    --frontier-edge-output "$TRAIN_EDGES" --disable-dense-diagnostic --device cuda \
    > "$OUT_ROOT/logs/train_type_frontier_replay.out" 2>&1
  printf 'complete\n' > "$TRAIN_COMPLETE_MARKER"
fi
[[ -s "$TRAIN_EDGES" ]] || { echo 'STOP TRAIN: missing type-resolved TRAIN frontier' >&2; exit 2; }
stage_end

stage_start train_trcc_fit
set +e
python -m bdse.tools.fit_v64_3_27_eaf_icer_trcc \
  --train-frontier-edges "$TRAIN_EDGES" \
  --base-v20-dual-config "$V20_CONFIG" \
  --output-dir "$FIT_DIR" \
  --output-train-token-file "$TRAIN_TOKENS" \
  --output-report "$OUT_ROOT/provenance/v64_3_27_train_trcc_fit.json" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_27_trcc_fit.out"
FIT_STATUS=${PIPESTATUS[0]}
set -e
stage_end
if [[ "$FIT_STATUS" -ne 0 ]]; then
  echo "STOP TRAIN TRCC (status=$FIT_STATUS). Audit report retained at $OUT_ROOT/provenance/v64_3_27_train_trcc_fit.json" >&2
  exit "$FIT_STATUS"
fi
for spec in aggregate-downside type-confirmed; do
  f=${spec//-/_}
  python -m bdse.tools.check_v64_3_27_eaf_icer_trcc_contract \
    --config "$FIT_DIR/v64_3_27_${f}.yaml" --expect "$spec" \
    --frozen-v20-dual-config "$V20_CONFIG" \
    --output "$OUT_ROOT/provenance/${f}_contract.json"
done

stage_start fresh_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh_1000_tokens.txt"
TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"
TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"
python -m bdse.tools.select_fresh_preprocessed_tokens \
  --preprocessed-dir "$BDSE_VAL_CACHE" --split val \
  --exclude-tokens "$DESIGN_EXCLUDE_TOKENS" --count 1000 \
  --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" \
  --audit-output "$OUT_ROOT/provenance/v64_3_27_fresh_1000_audit.json" \
  > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1000" > "$TOKA"; tail -n 500 "$TOK1000" > "$TOKB"
python - "$TRAIN_TOKENS" "$TOK1000" "$DESIGN_EXCLUDE_TOKENS" <<'PY'
import sys
tr={x.strip() for x in open(sys.argv[1]) if x.strip()}; fr={x.strip() for x in open(sys.argv[2]) if x.strip()}; ex={x.strip() for x in open(sys.argv[3]) if x.strip()}
if len(tr)!=3000 or len(fr)!=1000 or len(ex)!=6700 or tr & fr or ex & fr:
    raise SystemExit(f'STOP DATA: identity failure train={len(tr)} fresh={len(fr)} exclude={len(ex)} train_fresh={len(tr&fr)} design_fresh={len(ex&fr)}')
print('PASS V27 train/design/fresh identity isolation')
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
AGD="$FIT_DIR/v64_3_27_aggregate_downside.yaml"
TRCC="$FIT_DIR/v64_3_27_type_confirmed.yaml"
# Four arms per independent block. V25 aggregate DRC is the proposal control;
# V27 changes only the independent confirmation view and enforces no fallback.
for sp in A B; do
  stage_start "fresh_${sp}_wave1"
  run_eval "$GPU0" "$RAW_CONFIG" "$sp" raw & p0=$!
  run_eval "$GPU1" "$V20_CONFIG" "$sp" v20 & p1=$!
  s=0; wait "$p0" || s=1; wait "$p1" || s=1; [[ $s -eq 0 ]] || exit 2
  stage_end
  stage_start "fresh_${sp}_wave2"
  run_eval "$GPU0" "$AGD" "$sp" aggregate_downside & p0=$!
  run_eval "$GPU1" "$TRCC" "$sp" type_confirmed & p1=$!
  s=0; wait "$p0" || s=1; wait "$p1" || s=1; [[ $s -eq 0 ]] || exit 2
  stage_end
done

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); tags=['raw','v20','aggregate_downside','type_confirmed']
for sp,token_file in [('A',sys.argv[2]),('B',sys.argv[3])]:
    want=[x.strip() for x in open(token_file) if x.strip()]
    want_set=set(want)
    if len(want)!=500 or len(want_set)!=500:
        raise SystemExit(f'STOP DATA: {sp} token manifest is not exactly 500 unique scenes')
    arm_orders=[]
    for tag in tags:
        rows=[json.loads(x) for x in open(root/f'{sp}_{tag}_rows.jsonl') if x.strip()]
        got=[str(r['scenario_token']) for r in rows]
        if len(got)!=500 or len(set(got))!=500 or set(got)!=want_set:
            raise SystemExit(f'STOP DATA: {sp}/{tag} token set mismatch rows={len(got)} unique={len(set(got))} overlap={len(set(got)&want_set)}')
        arm_orders.append(got)
        m=json.load(open(root/f'{sp}_{tag}_metrics.json'))
        if not m.get('scenario_token_prefilter_active',False):
            raise SystemExit(f'STOP DATA/SPEED: {sp}/{tag} pre-load filter inactive')
    # Evaluator/cache order need not equal the hash-manifest order.  Paired
    # causal rows only require identical scene set plus identical emitted order
    # across arms; this fixes the V25 false STOP without weakening identity.
    if any(order != arm_orders[0] for order in arm_orders[1:]):
        raise SystemExit(f'STOP DATA: {sp} arm row orders differ; cannot use index-paired diagnostics safely')
print('PASS V27 paired identity: manifest sets exact and 8/8 arms share within-split row order')
PY
stage_end

stage_start screen
for sp in A B; do
  python -m bdse.tools.check_v64_3_27_eaf_icer_trcc_split --split-name "$sp" \
    --raw-metrics "$OUT_ROOT/provenance/${sp}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${sp}_raw_rows.jsonl" \
    --v20-metrics "$OUT_ROOT/provenance/${sp}_v20_metrics.json" --v20-rows "$OUT_ROOT/provenance/${sp}_v20_rows.jsonl" --v20-edges "$OUT_ROOT/provenance/${sp}_v20_edges.jsonl" \
    --aggregate-downside-metrics "$OUT_ROOT/provenance/${sp}_aggregate_downside_metrics.json" --aggregate-downside-rows "$OUT_ROOT/provenance/${sp}_aggregate_downside_rows.jsonl" --aggregate-downside-edges "$OUT_ROOT/provenance/${sp}_aggregate_downside_edges.jsonl" \
    --type-confirmed-metrics "$OUT_ROOT/provenance/${sp}_type_confirmed_metrics.json" --type-confirmed-rows "$OUT_ROOT/provenance/${sp}_type_confirmed_rows.jsonl" --type-confirmed-edges "$OUT_ROOT/provenance/${sp}_type_confirmed_edges.jsonl" \
    --output "$OUT_ROOT/provenance/v64_3_27_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_27_split_${sp}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_27_eaf_icer_trcc_double_fresh_screen.json"
python -m bdse.tools.check_v64_3_27_eaf_icer_trcc_screen \
  --split-a-report "$OUT_ROOT/provenance/v64_3_27_split_A_screen.json" \
  --split-b-report "$OUT_ROOT/provenance/v64_3_27_split_B_screen.json" \
  --train-fit-report "$OUT_ROOT/provenance/v64_3_27_train_trcc_fit.json" \
  --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_27_double_fresh_screen.out"
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"
python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print('DOUBLE_FRESH_PASS=',r.get('both_independent_blocks_pass')); print('TYPE_CONFIRMATION_TAIL_INCREMENTAL_BOTH=',r.get('type_confirmation_tail_incremental_both')); print('NEXT_ACTION=',r.get('next_action'))
if not r.get('full_promotion_to_independent_full_val_reproduction',False):
    raise SystemExit('STOP SCREEN: do not run full/test/closed-loop; follow next_action')
print('PASS SCREEN ONLY: next allowed stage is one frozen independent full-val reproduction; test/closed-loop remain forbidden.')
PY
