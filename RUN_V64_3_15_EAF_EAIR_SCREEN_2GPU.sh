#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

# V64.3.15 is a selective readout-capacity experiment.  The V64.3.13 EAF
# checkpoint, acquisition, B=16, M=24, DARM and DBR are frozen.  Only a tiny
# standardized logistic reliability readout is fitted from TRAIN raw proposal
# rows.  Validation is evaluation-only; test/closed-loop are forbidden here.
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_15_eaf_eair_screen_2gpu_v1}"
export TRAIN_SCENARIOS="${TRAIN_SCENARIOS:-3000}"
export VAL_DISCOVERY_SCENARIOS="${VAL_DISCOVERY_SCENARIOS:-1200}"
export VAL_SCREEN_SCENARIOS="${VAL_SCREEN_SCENARIOS:-500}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_15_design_exclude_v64_3_14_tokens.txt}"
export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export RAW_CONFIG="bdse/configs/v64_3_15_eaf_eair_raw.yaml"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/configs"

[[ -d "$BDSE_TRAIN_CACHE" ]] || { echo "STOP: missing train cache $BDSE_TRAIN_CACHE" >&2; exit 2; }
[[ -d "$BDSE_VAL_CACHE" ]] || { echo "STOP: missing val cache $BDSE_VAL_CACHE" >&2; exit 2; }
[[ -s "$EAF_TRAIN_LOG" ]] || { echo "STOP: missing V64.3.13 train log $EAF_TRAIN_LOG" >&2; exit 2; }
[[ -s "$DESIGN_EXCLUDE_TOKENS" ]] || { echo "STOP: missing V64.3.14 design-scene exclusion list $DESIGN_EXCLUDE_TOKENS" >&2; exit 2; }

# Reuse the repaired V64.3.13 audit.  We require the same causally valid EAF
# checkpoint that exhibited pair-value signal before any reliability experiment.
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
  --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_15 \
  --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" \
  > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False):
    raise SystemExit('STOP: V64.3.13 causal prerequisites are not valid for EAIR.')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then
  EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH + 1))").pt"
fi
export EAF_CKPT
[[ -s "$EAF_CKPT" ]] || { echo "STOP: missing EAF checkpoint $EAF_CKPT" >&2; exit 2; }

echo "EAF_CKPT=$EAF_CKPT"

# Static/causal contracts before GPU work.
python -m compileall -q bdse
python -m bdse.tools.check_v64_3_15_eaf_eair_contract \
  --config "$RAW_CONFIG" --expect raw \
  --output "$OUT_ROOT/provenance/v64_3_15_eaf_eair_raw_contract.json"
pytest -q \
  bdse/tests/test_v64_3_15_eaf_eair.py \
  bdse/tests/test_v64_3_14_eaf_ocfi.py \
  bdse/tests/test_v64_3_13_eaf_dmvr.py \
  bdse/tests/test_v64_3_12_cet_bdmu.py \
  bdse/tests/test_v64_3_11_btp_bdmu.py \
  bdse/tests/test_v64_3_10_hap_bdmu.py \
  bdse/tests/test_v64_3_9_af_bdmu.py \
  bdse/tests/test_v64_3_8_bdmu.py \
  bdse/tests/test_v64_3_7_darm_dbr.py \
  bdse/tests/test_v64_3_6_bcha_lbpr.py \
  | tee "$OUT_ROOT/logs/targeted_regression.out"

# Phase 1: collect runtime-only EAF reliability features from TRAIN and build a
# validation discovery pool.  The discovery pool is used only to choose scenario
# tokens that were NOT part of the V64.3.14 design screen; it is never used to fit
# EAIR or tune the 0.5 threshold.
TRAIN_RAW_METRICS="$OUT_ROOT/provenance/train_raw_eaf_metrics.json"
TRAIN_RAW_ROWS="$OUT_ROOT/provenance/train_raw_eaf_per_sample.jsonl"
VAL_DISCOVERY_METRICS="$OUT_ROOT/provenance/val_discovery_raw_eaf_metrics.json"
VAL_DISCOVERY_ROWS="$OUT_ROOT/provenance/val_discovery_raw_eaf_per_sample.jsonl"

CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop \
  --config "$RAW_CONFIG" --checkpoint "$EAF_CKPT" --split train \
  --preprocessed-dir "$BDSE_TRAIN_CACHE" --max-scenarios "$TRAIN_SCENARIOS" \
  --output "$TRAIN_RAW_METRICS" --per-sample-output "$TRAIN_RAW_ROWS" \
  --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/train_raw_eaf.out" 2>&1 &
PID_TRAIN=$!
CUDA_VISIBLE_DEVICES="$GPU1" python -m bdse.experiments.evaluate_open_loop \
  --config "$RAW_CONFIG" --checkpoint "$EAF_CKPT" --split val \
  --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios "$VAL_DISCOVERY_SCENARIOS" \
  --output "$VAL_DISCOVERY_METRICS" --per-sample-output "$VAL_DISCOVERY_ROWS" \
  --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/val_discovery_raw_eaf.out" 2>&1 &
PID_VAL=$!
STATUS=0; wait "$PID_TRAIN" || STATUS=1; wait "$PID_VAL" || STATUS=1
[[ "$STATUS" -eq 0 ]] || { echo "STOP ENGINEERING: train/discovery raw replay failed; inspect logs." >&2; exit 2; }

python - "$TRAIN_RAW_ROWS" <<'PY'
import json,math,sys
path=sys.argv[1]
rows=[json.loads(x) for x in open(path) if x.strip()]
prop=[r for r in rows if r.get('raw_frontier_anchor_action')!=r.get('raw_frontier_proposed_action') and r.get('decisive_frontier_value_teacher_proposed_vs_anchor_margin') is not None]
if len(prop)<256: raise SystemExit(f'STOP CAPACITY: only {len(prop)} TRAIN proposal edges; need >=256')
for key in ['decisive_frontier_eair_feature_raw_margin','decisive_frontier_eair_feature_proposed_attribution_scale','decisive_frontier_eair_feature_frontier_residual_rms']:
    cov=sum(math.isfinite(float(r.get(key,float('nan')))) for r in prop)/len(prop)
    if cov<0.99: raise SystemExit(f'STOP ENGINEERING: {key} TRAIN coverage={cov:.3f}')
print('PASS TRAIN EAIR feature instrumentation')
PY

# Freeze a fresh validation screen.  Every V64.3.14 design token is excluded so
# this screen is not a re-evaluation of the data used to invent EAIR.
VAL_TOKENS="$OUT_ROOT/provenance/val_screen_fresh_tokens.txt"
python - "$VAL_DISCOVERY_ROWS" "$DESIGN_EXCLUDE_TOKENS" "$VAL_TOKENS" "$VAL_SCREEN_SCENARIOS" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
exclude={x.strip() for x in open(sys.argv[2]) if x.strip()}
need=int(sys.argv[4]); out=[]; seen=set()
for r in rows:
    t=str(r['scenario_token'])
    if t in exclude or t in seen: continue
    out.append(t); seen.add(t)
    if len(out)>=need: break
if len(out)<need:
    raise SystemExit(f'STOP DATA SPLIT: only {len(out)} fresh val scenes after excluding V64.3.14 design scenes; increase VAL_DISCOVERY_SCENARIOS')
open(sys.argv[3],'w').write('\n'.join(out)+'\n')
print(f'PASS fresh val split: {len(out)} scenes, overlap_with_v64_3_14=0')
PY

# Phase 2: fit the reliability readout on TRAIN only.  No validation threshold
# sweep is performed; deployment probability threshold is fixed at 0.5.
FIT_REPORT="$OUT_ROOT/provenance/v64_3_15_eaf_eair_fit.json"
FITTED_CFG="$OUT_ROOT/configs/v64_3_15_eaf_eair_fitted.yaml"
python -m bdse.tools.fit_v64_3_15_eaf_eair \
  --train-per-sample "$TRAIN_RAW_ROWS" --base-config "$RAW_CONFIG" \
  --output-config "$FITTED_CFG" --output-report "$FIT_REPORT" \
  > "$OUT_ROOT/logs/eair_fit.out"
python -m bdse.tools.check_v64_3_15_eaf_eair_contract \
  --config "$FITTED_CFG" --expect fitted \
  --output "$OUT_ROOT/provenance/v64_3_15_eaf_eair_fitted_contract.json"

# Phase 3: paired raw-control and EAIR replay on the exact same fresh validation
# tokens.  Run concurrently; neither job may see V64.3.14 design scenes.
VAL_RAW_METRICS="$OUT_ROOT/provenance/val_fresh_raw_eaf_metrics.json"
VAL_RAW_ROWS="$OUT_ROOT/provenance/val_fresh_raw_eaf_per_sample.jsonl"
EAIR_METRICS="$OUT_ROOT/provenance/v64_3_15_eaf_eair_metrics.json"
EAIR_ROWS="$OUT_ROOT/provenance/v64_3_15_eaf_eair_per_sample.jsonl"
CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop \
  --config "$RAW_CONFIG" --checkpoint "$EAF_CKPT" --split val \
  --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios "$VAL_SCREEN_SCENARIOS" \
  --scenario-token-file "$VAL_TOKENS" --output "$VAL_RAW_METRICS" \
  --per-sample-output "$VAL_RAW_ROWS" --disable-dense-diagnostic --device cuda \
  > "$OUT_ROOT/logs/raw_fresh_val.out" 2>&1 &
PID_RAW=$!
CUDA_VISIBLE_DEVICES="$GPU1" python -m bdse.experiments.evaluate_open_loop \
  --config "$FITTED_CFG" --checkpoint "$EAF_CKPT" --split val \
  --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios "$VAL_SCREEN_SCENARIOS" \
  --scenario-token-file "$VAL_TOKENS" --output "$EAIR_METRICS" \
  --per-sample-output "$EAIR_ROWS" --disable-dense-diagnostic --device cuda \
  > "$OUT_ROOT/logs/eair_fresh_val.out" 2>&1 &
PID_EAIR=$!
STATUS=0; wait "$PID_RAW" || STATUS=1; wait "$PID_EAIR" || STATUS=1
[[ "$STATUS" -eq 0 ]] || { echo "STOP ENGINEERING: paired fresh-val replay failed; inspect logs." >&2; exit 2; }

python -m bdse.tools.check_v64_3_15_eaf_eair_screen \
  --raw-metrics "$VAL_RAW_METRICS" --eair-metrics "$EAIR_METRICS" \
  --eair-per-sample "$EAIR_ROWS" --fit-report "$FIT_REPORT" \
  --output "$OUT_ROOT/provenance/v64_3_15_eaf_eair_screen.json" \
  | tee "$OUT_ROOT/logs/v64_3_15_eaf_eair_screen.out"

python - "$OUT_ROOT/provenance/v64_3_15_eaf_eair_screen.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print('\nV64.3.15 decision summary:')
print(json.dumps({k:r.get(k) for k in ['full_promotion','instrumentation_valid','fit_capacity_signal','preservation_gain','endpoint_gain','next_action']},indent=2))
print(json.dumps(r.get('metrics',{}),indent=2))
if not r.get('full_promotion',False):
    raise SystemExit('STOP SCREEN: EAIR not promoted. Do not run full/test/closed-loop; follow next_action from the report.')
print('PASS SCREEN: EAIR has a paired reliability/preservation/endpoint gain. Next step is a separate full-val reproduction; do not run test/closed-loop from this launcher.')
PY
