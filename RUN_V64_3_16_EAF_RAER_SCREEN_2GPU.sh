#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

# V64.3.16: acquisition/value are frozen.  Train-only readouts are fitted from
# already-computed EAF frontier diagnostics; test and closed-loop are forbidden.
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_16_eaf_raer_screen_2gpu_v1}"
export TRAIN_SCENARIOS="${TRAIN_SCENARIOS:-3000}"
export VAL_DISCOVERY_SCENARIOS="${VAL_DISCOVERY_SCENARIOS:-2500}"
export VAL_SCREEN_SCENARIOS="${VAL_SCREEN_SCENARIOS:-500}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_16_design_exclude_v64_3_15_discovery_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.16-eaf-raer-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"
export RAW_CONFIG="bdse/configs/v64_3_16_eaf_raer_raw.yaml"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/configs"

[[ -d "$BDSE_TRAIN_CACHE" ]] || { echo "STOP: missing train cache $BDSE_TRAIN_CACHE" >&2; exit 2; }
[[ -d "$BDSE_VAL_CACHE" ]] || { echo "STOP: missing val cache $BDSE_VAL_CACHE" >&2; exit 2; }
[[ -s "$EAF_TRAIN_LOG" ]] || { echo "STOP: missing V64.3.13 train log $EAF_TRAIN_LOG" >&2; exit 2; }
[[ -s "$DESIGN_EXCLUDE_TOKENS" ]] || { echo "STOP: missing V64.3.15 discovery/design exclusion list" >&2; exit 2; }
[[ "$(wc -l < "$DESIGN_EXCLUDE_TOKENS")" -ge 1200 ]] || { echo "STOP: V64.3.16 must exclude all 1200 uploaded V64.3.15 discovery scenes used in this design cycle." >&2; exit 2; }

python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
  --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_16 \
  --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" \
  > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False):
    raise SystemExit('STOP: V64.3.13 causal prerequisites invalid.')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH + 1))").pt"; fi
export EAF_CKPT; [[ -s "$EAF_CKPT" ]] || { echo "STOP: missing EAF checkpoint $EAF_CKPT" >&2; exit 2; }

echo "EAF_CKPT=$EAF_CKPT"
python -m compileall -q bdse
python -m bdse.tools.check_v64_3_16_eaf_raer_contract --config "$RAW_CONFIG" --expect raw --output "$OUT_ROOT/provenance/v64_3_16_raer_raw_contract.json"
pytest -q \
  bdse/tests/test_v64_3_16_eaf_raer.py bdse/tests/test_v64_3_15_eaf_eair.py bdse/tests/test_v64_3_14_eaf_ocfi.py \
  bdse/tests/test_v64_3_13_eaf_dmvr.py bdse/tests/test_v64_3_12_cet_bdmu.py bdse/tests/test_v64_3_11_btp_bdmu.py \
  bdse/tests/test_v64_3_10_hap_bdmu.py bdse/tests/test_v64_3_9_af_bdmu.py bdse/tests/test_v64_3_8_bdmu.py \
  bdse/tests/test_v64_3_7_darm_dbr.py bdse/tests/test_v64_3_6_bcha_lbpr.py | tee "$OUT_ROOT/logs/targeted_regression.out"

TRAIN_RAW_METRICS="$OUT_ROOT/provenance/train_raw_eaf_metrics.json"
TRAIN_RAW_ROWS="$OUT_ROOT/provenance/train_raw_eaf_per_sample.jsonl"
TRAIN_EDGES="$OUT_ROOT/provenance/train_raw_eaf_frontier_edges.jsonl"
VAL_DISCOVERY_METRICS="$OUT_ROOT/provenance/val_discovery_raw_eaf_metrics.json"
VAL_DISCOVERY_ROWS="$OUT_ROOT/provenance/val_discovery_raw_eaf_per_sample.jsonl"

CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop \
  --config "$RAW_CONFIG" --checkpoint "$EAF_CKPT" --split train --preprocessed-dir "$BDSE_TRAIN_CACHE" \
  --max-scenarios "$TRAIN_SCENARIOS" --max-scenarios-strategy uniform_blocks \
  --output "$TRAIN_RAW_METRICS" --per-sample-output "$TRAIN_RAW_ROWS" --frontier-edge-output "$TRAIN_EDGES" \
  --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/train_raw_eaf.out" 2>&1 & PID_TRAIN=$!
CUDA_VISIBLE_DEVICES="$GPU1" python -m bdse.experiments.evaluate_open_loop \
  --config "$RAW_CONFIG" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" \
  --max-scenarios "$VAL_DISCOVERY_SCENARIOS" --max-scenarios-strategy uniform_blocks \
  --output "$VAL_DISCOVERY_METRICS" --per-sample-output "$VAL_DISCOVERY_ROWS" \
  --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/val_discovery_raw_eaf.out" 2>&1 & PID_VAL=$!
STATUS=0; wait "$PID_TRAIN" || STATUS=1; wait "$PID_VAL" || STATUS=1
[[ "$STATUS" -eq 0 ]] || { echo "STOP ENGINEERING: train/discovery replay failed" >&2; exit 2; }

python - "$TRAIN_RAW_ROWS" "$TRAIN_EDGES" <<'PY'
import json,math,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
prop=[r for r in rows if r.get('raw_frontier_anchor_action')!=r.get('raw_frontier_proposed_action') and r.get('decisive_frontier_value_teacher_proposed_vs_anchor_margin') is not None]
if len(prop)<256: raise SystemExit(f'STOP: only {len(prop)} TRAIN top proposal edges')
for key in ['decisive_frontier_eair_feature_raw_margin','decisive_frontier_eair_feature_proposed_attribution_scale','decisive_frontier_eair_feature_frontier_residual_rms']:
    cov=sum(math.isfinite(float(r.get(key,float('nan')))) for r in prop)/len(prop)
    if cov<.99: raise SystemExit(f'STOP ENGINEERING: {key} coverage={cov:.3f}')
edges=sum(1 for x in open(sys.argv[2]) if x.strip())
if edges<2048: raise SystemExit(f'STOP CAPACITY: only {edges} all-frontier TRAIN edges')
print(f'PASS instrumentation: top_edges={len(prop)}, all_frontier_edges={edges}')
PY

# Fresh validation is frozen only by scenario token identity.  We do not inspect
# labels or metrics to choose scenes: eligible tokens are hash-ranked by a fixed seed.
VAL_TOKENS="$OUT_ROOT/provenance/val_screen_fresh_tokens.txt"
SPLIT_AUDIT="$OUT_ROOT/provenance/v64_3_16_fresh_split_audit.json"
python - "$VAL_DISCOVERY_ROWS" "$DESIGN_EXCLUDE_TOKENS" "$VAL_TOKENS" "$SPLIT_AUDIT" "$VAL_SCREEN_SCENARIOS" "$FRESH_HASH_SEED" <<'PY'
import hashlib,json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
exclude={x.strip() for x in open(sys.argv[2]) if x.strip()}; need=int(sys.argv[5]); seed=sys.argv[6]
disc={str(r['scenario_token']) for r in rows}; eligible=disc-exclude
rank=lambda t: hashlib.sha256((seed+'::'+t).encode()).hexdigest()
out=sorted(eligible,key=lambda t:(rank(t),t))[:need]
if len(out)<need: raise SystemExit(f'STOP DATA SPLIT: only {len(out)} clean scenes; increase VAL_DISCOVERY_SCENARIOS')
open(sys.argv[3],'w').write('\n'.join(out)+'\n')
a={'audit':'v64_3_16_fresh_split','discovery_unique':len(disc),'excluded_in_discovery':len(disc&exclude),'eligible_unique':len(eligible),'fresh_count':len(out),'fresh_overlap_design':len(set(out)&exclude),'hash_seed':seed,'selection_uses_labels':False}
open(sys.argv[4],'w').write(json.dumps(a,indent=2,sort_keys=True)); print(json.dumps(a,indent=2))
PY

EAIR_FIT="$OUT_ROOT/provenance/v64_3_16_scalar_eair_control_fit.json"; EAIR_CFG="$OUT_ROOT/configs/v64_3_16_scalar_eair_control.yaml"
python -m bdse.tools.fit_v64_3_15_eaf_eair --train-per-sample "$TRAIN_RAW_ROWS" --base-config "$RAW_CONFIG" --output-config "$EAIR_CFG" --output-report "$EAIR_FIT" > "$OUT_ROOT/logs/scalar_eair_fit.out"
RAER_FIT="$OUT_ROOT/provenance/v64_3_16_raer_fit.json"; RAER_CFG="$OUT_ROOT/configs/v64_3_16_eaf_raer_fitted.yaml"
python -m bdse.tools.fit_v64_3_16_eaf_raer --train-frontier-edges "$TRAIN_EDGES" --base-config "$RAW_CONFIG" --output-config "$RAER_CFG" --output-report "$RAER_FIT" > "$OUT_ROOT/logs/raer_fit.out"
python -m bdse.tools.check_v64_3_16_eaf_raer_contract --config "$RAER_CFG" --expect fitted --output "$OUT_ROOT/provenance/v64_3_16_raer_fitted_contract.json"

RAW_M="$OUT_ROOT/provenance/val_fresh_raw_metrics.json"; RAW_R="$OUT_ROOT/provenance/val_fresh_raw_rows.jsonl"
EAIR_M="$OUT_ROOT/provenance/val_fresh_scalar_eair_metrics.json"; EAIR_R="$OUT_ROOT/provenance/val_fresh_scalar_eair_rows.jsonl"
RAER_M="$OUT_ROOT/provenance/val_fresh_raer_metrics.json"; RAER_R="$OUT_ROOT/provenance/val_fresh_raer_rows.jsonl"; RAER_E="$OUT_ROOT/provenance/val_fresh_raer_edges.jsonl"

CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop --config "$RAW_CONFIG" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" \
  --max-scenarios "$VAL_SCREEN_SCENARIOS" --scenario-token-file "$VAL_TOKENS" --require-all-scenario-tokens \
  --output "$RAW_M" --per-sample-output "$RAW_R" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/fresh_raw.out" 2>&1 & PID0=$!
CUDA_VISIBLE_DEVICES="$GPU1" python -m bdse.experiments.evaluate_open_loop --config "$EAIR_CFG" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" \
  --max-scenarios "$VAL_SCREEN_SCENARIOS" --scenario-token-file "$VAL_TOKENS" --require-all-scenario-tokens \
  --output "$EAIR_M" --per-sample-output "$EAIR_R" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/fresh_scalar_eair.out" 2>&1 & PID1=$!
STATUS=0; wait "$PID0" || STATUS=1; wait "$PID1" || STATUS=1; [[ "$STATUS" -eq 0 ]] || { echo "STOP: raw/EAIR fresh replay failed" >&2; exit 2; }
CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop --config "$RAER_CFG" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" \
  --max-scenarios "$VAL_SCREEN_SCENARIOS" --scenario-token-file "$VAL_TOKENS" --require-all-scenario-tokens \
  --output "$RAER_M" --per-sample-output "$RAER_R" --frontier-edge-output "$RAER_E" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/fresh_raer.out" 2>&1

python -m bdse.tools.check_v64_3_16_eaf_raer_screen --raw-metrics "$RAW_M" --eair-metrics "$EAIR_M" --raer-metrics "$RAER_M" \
  --raer-edge-output "$RAER_E" --eair-fit-report "$EAIR_FIT" --raer-fit-report "$RAER_FIT" \
  --output "$OUT_ROOT/provenance/v64_3_16_eaf_raer_screen.json" | tee "$OUT_ROOT/logs/v64_3_16_eaf_raer_screen.out"

python - "$OUT_ROOT/provenance/v64_3_16_eaf_raer_screen.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); keys=['full_promotion','instrumentation_valid','capacity_signal','extremal_reranking_mechanism','preservation_gain','endpoint_gain','next_action']
print('\nV64.3.16 decision summary:'); print(json.dumps({k:r.get(k) for k in keys},indent=2)); print(json.dumps(r.get('metrics',{}),indent=2)); print(json.dumps(r.get('edge_diagnostics',{}),indent=2))
if not r.get('full_promotion',False): raise SystemExit('STOP SCREEN: do not run full/test/closed-loop; follow next_action.')
print('PASS SCREEN: run a separate independent full-val reproduction before any test/closed-loop evaluation.')
PY
