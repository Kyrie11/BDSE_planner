#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

# V64.3.14 is deliberately evaluation/calibration-only.  It reuses the learned
# V64.3.13 EAF-DMVR checkpoint and tests only the frontier-value -> one-sided
# intervention/preservation interface.  Acquisition, B=16, M=24, DARM and DBR
# remain frozen.
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_14_eaf_ocfi_screen_2gpu_v1}"
export VAL_SCENARIOS="${VAL_SCENARIOS:-500}"
export OCFI_ALPHA="${OCFI_ALPHA:-0.10}"
export OCFI_CALIBRATION_FRACTION="${OCFI_CALIBRATION_FRACTION:-0.40}"
export OCFI_SPLIT_SEED="${OCFI_SPLIT_SEED:-v64.3.14-eaf-ocfi-v1}"
export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export RAW_CONFIG="bdse/configs/v64_3_14_eaf_ocfi_raw_calibration.yaml"

mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/configs"
[[ -s "$EAF_TRAIN_LOG" ]] || { echo "STOP: missing V64.3.13 train log: $EAF_TRAIN_LOG" >&2; exit 2; }

# -----------------------------------------------------------------------------
# Phase 0A — repair/re-run the V64.3.13 causal audit before choosing a checkpoint.
# The fixed checker never allows an exact-scene-invalid epoch to win on a noisy
# endpoint and keeps training/runtime instrumentation separate.
# -----------------------------------------------------------------------------
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
  --train-log "$EAF_TRAIN_LOG" \
  --variant REAUDIT_FOR_V64_3_14 \
  --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" \
  > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"

read -r SELECTED_EPOCH NEXT_ACTION < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid', False):
    raise SystemExit('STOP ENGINEERING: selected V64.3.13 epoch lacks exact-scene training instrumentation.')
if not r.get('acquisition_frozen', False):
    raise SystemExit('STOP ENGINEERING: V64.3.13 causal isolation failed; do not interpret EAF.')
if not r.get('value_estimation_gain', False):
    raise SystemExit('STOP VALUE: complete-frontier pair-value estimation did not improve; OCFI is not the next causal test.')
if not r.get('preservation_interface_failure', False):
    raise SystemExit('STOP BRANCH: current evidence does not isolate a preservation-interface failure; do not force OCFI.')
print(int(r['selected_epoch']), str(r.get('next_action','')))
PY
)
echo "V64.3.13 selected_epoch=$SELECTED_EPOCH next_action=$NEXT_ACTION"

if [[ -z "${EAF_CKPT:-}" ]]; then
  EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH + 1))").pt"
fi
export EAF_CKPT
[[ -s "$EAF_CKPT" ]] || {
  echo "STOP: selected EAF checkpoint not found: $EAF_CKPT" >&2
  echo "The uploaded compact outputs archive does not contain checkpoints; point EAF_CKPT/EAF_V64_3_13_ROOT to the original server output." >&2
  exit 2
}
echo "EAF_CKPT=$EAF_CKPT"

# -----------------------------------------------------------------------------
# Phase 0B — static contracts and targeted regressions.  Do not spend GPU if any
# of these fail.
# -----------------------------------------------------------------------------
python -m compileall -q bdse
python -m bdse.tools.check_v64_3_14_eaf_ocfi_contract \
  --config "$RAW_CONFIG" --expect raw \
  --output "$OUT_ROOT/provenance/v64_3_14_eaf_ocfi_raw_contract.json"
pytest -q \
  bdse/tests/test_v64_3_14_eaf_ocfi.py \
  bdse/tests/test_v64_3_13_eaf_dmvr.py \
  bdse/tests/test_v64_3_12_cet_bdmu.py \
  bdse/tests/test_v64_3_11_btp_bdmu.py \
  bdse/tests/test_v64_3_10_hap_bdmu.py \
  bdse/tests/test_v64_3_9_af_bdmu.py \
  bdse/tests/test_v64_3_7_darm_dbr.py \
  | tee "$OUT_ROOT/logs/targeted_regression.out"

# -----------------------------------------------------------------------------
# Phase 1A — one raw runtime-instrumented replay of the frozen EAF checkpoint.
# OCFI is disabled here.  This repairs the missing V64.3.13 runtime metric
# plumbing and creates proposal-conditioned calibration rows.
# -----------------------------------------------------------------------------
RAW_METRICS="$OUT_ROOT/provenance/raw_eaf_replay_metrics.json"
RAW_ROWS="$OUT_ROOT/provenance/raw_eaf_replay_per_sample.jsonl"
CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop \
  --config "$RAW_CONFIG" \
  --checkpoint "$EAF_CKPT" \
  --split val \
  --preprocessed-dir "$BDSE_VAL_CACHE" \
  --max-scenarios "$VAL_SCENARIOS" \
  --output "$RAW_METRICS" \
  --per-sample-output "$RAW_ROWS" \
  --disable-dense-diagnostic \
  --device cuda \
  > "$OUT_ROOT/logs/raw_eaf_replay.out" 2>&1

python - "$RAW_METRICS" "$RAW_ROWS" <<'PY'
import json,math,sys
m=json.load(open(sys.argv[1]))
required={
  'decisive_frontier_value_active':0.99,
  'decisive_frontier_value_complete_star_coverage':0.99,
}
for k,lo in required.items():
    v=float(m.get(k,float('nan')))
    if not math.isfinite(v) or v < lo:
        raise SystemExit(f'STOP ENGINEERING: raw runtime EAF instrumentation {k}={v}, expected >= {lo}')
for k in ('decisive_frontier_value_residual_rms','decisive_frontier_value_attribution_scale_rms'):
    v=float(m.get(k,float('nan')))
    if not math.isfinite(v) or v <= 1e-8:
        raise SystemExit(f'STOP ENGINEERING: raw runtime EAF instrumentation {k}={v}')
rows=sum(1 for x in open(sys.argv[2]) if x.strip())
if rows < 64:
    raise SystemExit(f'STOP ENGINEERING: only {rows} raw replay rows; insufficient for OCFI screen.')
print('PASS raw EAF runtime instrumentation:', {k:m.get(k) for k in [*required,'decisive_frontier_value_residual_rms','decisive_frontier_value_attribution_scale_rms']}, 'rows=',rows)
PY

# -----------------------------------------------------------------------------
# Phase 1B — fit two group-disjoint one-sided split-conformal controls from the
# same raw rows and the same deterministic split.
#   1) attribution: q * RSS(per-selected-atom EAF contributions)  [main]
#   2) none:        q * 1                                      [control]
# The held-out evaluation token list must be byte-identical across branches.
# -----------------------------------------------------------------------------
ATTR_CAL="$OUT_ROOT/provenance/ocfi_attribution_calibration.json"
ATTR_CFG="$OUT_ROOT/configs/v64_3_14_eaf_ocfi_attribution.yaml"
ATTR_CAL_TOK="$OUT_ROOT/provenance/ocfi_attribution_calibration_tokens.txt"
ATTR_EVAL_TOK="$OUT_ROOT/provenance/ocfi_attribution_evaluation_tokens.txt"
CONST_CAL="$OUT_ROOT/provenance/ocfi_constant_calibration.json"
CONST_CFG="$OUT_ROOT/configs/v64_3_14_eaf_ocfi_constant.yaml"
CONST_CAL_TOK="$OUT_ROOT/provenance/ocfi_constant_calibration_tokens.txt"
CONST_EVAL_TOK="$OUT_ROOT/provenance/ocfi_constant_evaluation_tokens.txt"

python -m bdse.tools.calibrate_v64_3_14_eaf_ocfi \
  --per-sample "$RAW_ROWS" --base-config "$RAW_CONFIG" \
  --normalization attribution --alpha "$OCFI_ALPHA" \
  --calibration-fraction "$OCFI_CALIBRATION_FRACTION" --split-seed "$OCFI_SPLIT_SEED" \
  --output-report "$ATTR_CAL" --output-config "$ATTR_CFG" \
  --calibration-token-file "$ATTR_CAL_TOK" --evaluation-token-file "$ATTR_EVAL_TOK"

python -m bdse.tools.calibrate_v64_3_14_eaf_ocfi \
  --per-sample "$RAW_ROWS" --base-config "$RAW_CONFIG" \
  --normalization none --alpha "$OCFI_ALPHA" \
  --calibration-fraction "$OCFI_CALIBRATION_FRACTION" --split-seed "$OCFI_SPLIT_SEED" \
  --output-report "$CONST_CAL" --output-config "$CONST_CFG" \
  --calibration-token-file "$CONST_CAL_TOK" --evaluation-token-file "$CONST_EVAL_TOK"

cmp "$ATTR_CAL_TOK" "$CONST_CAL_TOK"
cmp "$ATTR_EVAL_TOK" "$CONST_EVAL_TOK"
python -m bdse.tools.check_v64_3_14_eaf_ocfi_contract \
  --config "$ATTR_CFG" --expect calibrated \
  --output "$OUT_ROOT/provenance/v64_3_14_eaf_ocfi_attribution_contract.json"
python -m bdse.tools.check_v64_3_14_eaf_ocfi_contract \
  --config "$CONST_CFG" --expect calibrated \
  --output "$OUT_ROOT/provenance/v64_3_14_eaf_ocfi_constant_contract.json"

# -----------------------------------------------------------------------------
# Phase 1C — evaluate both frozen gates on exactly the same held-out val groups.
# These are independent evaluation processes and can use the two GPUs in parallel.
# -----------------------------------------------------------------------------
ATTR_METRICS="$OUT_ROOT/provenance/ocfi_attribution_metrics.json"
CONST_METRICS="$OUT_ROOT/provenance/ocfi_constant_metrics.json"

CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop \
  --config "$ATTR_CFG" --checkpoint "$EAF_CKPT" --split val \
  --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios "$VAL_SCENARIOS" \
  --scenario-token-file "$ATTR_EVAL_TOK" \
  --output "$ATTR_METRICS" --disable-dense-diagnostic --device cuda \
  > "$OUT_ROOT/logs/ocfi_attribution_eval.out" 2>&1 &
PID_ATTR=$!
CUDA_VISIBLE_DEVICES="$GPU1" python -m bdse.experiments.evaluate_open_loop \
  --config "$CONST_CFG" --checkpoint "$EAF_CKPT" --split val \
  --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios "$VAL_SCENARIOS" \
  --scenario-token-file "$CONST_EVAL_TOK" \
  --output "$CONST_METRICS" --disable-dense-diagnostic --device cuda \
  > "$OUT_ROOT/logs/ocfi_constant_eval.out" 2>&1 &
PID_CONST=$!

STATUS=0
wait "$PID_ATTR" || STATUS=1
wait "$PID_CONST" || STATUS=1
[[ "$STATUS" -eq 0 ]] || { echo "STOP ENGINEERING: an OCFI evaluation process failed; inspect logs." >&2; exit 2; }

python -m bdse.tools.check_v64_3_14_eaf_ocfi_screen \
  --attribution-metrics "$ATTR_METRICS" \
  --attribution-calibration "$ATTR_CAL" \
  --constant-metrics "$CONST_METRICS" \
  --constant-calibration "$CONST_CAL" \
  --output "$OUT_ROOT/provenance/v64_3_14_eaf_ocfi_screen.json" \
  | tee "$OUT_ROOT/logs/v64_3_14_eaf_ocfi_screen.out"

python - "$OUT_ROOT/provenance/v64_3_14_eaf_ocfi_screen.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print('\nV64.3.14 decision summary:')
print(json.dumps({
  'selected_normalization':r.get('selected_normalization'),
  'full_promotion':r.get('full_promotion'),
  'attribution_specific_gain':r.get('attribution_specific_gain'),
  'next_action':r.get('next_action'),
  'selected_deltas':(r.get('selected') or {}).get('deltas'),
}, indent=2))
if not r.get('full_promotion', False):
    raise SystemExit('STOP SCREEN: OCFI not promoted. Do not run full/test/closed-loop; follow next_action from the report.')
if r.get('selected_normalization') == 'none':
    raise SystemExit('STOP NOVELTY CLAIM: constant calibration passed but evidence-attribution scaling was not supported. Do not claim attribution-specific OCFI; inspect the control before any paper-facing promotion.')
if not r.get('attribution_specific_gain', False):
    raise SystemExit('STOP NOVELTY CONTROL: OCFI performance may improve, but attribution scaling did not beat the constant-radius control by the predeclared effect threshold. Do not promote attribution as the mechanism.')
print('PASS SCREEN: attribution-scaled EAF-OCFI has attribution-specific support and is eligible for a separate full-val calibration/reproduction. Do not touch held-out test or closed-loop yet.')
PY
