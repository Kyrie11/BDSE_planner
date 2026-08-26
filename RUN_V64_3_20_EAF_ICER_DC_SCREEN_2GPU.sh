#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

# V64.3.20 EAF-ICER-DC causal screen.
# Purpose: test ONE semantic correction only. Learned V19 TRAIN-only support and
# dominance heads are frozen bit-for-bit. In the all-actions-safety-flagged domain,
# learned ICER must no longer replace the raw proposal with the DARM anchor before
# the downstream continuous structural-risk guard. Instead it preserves the raw
# legacy proposal and delegates the scene to the unchanged structural guard.
#
# Four fresh paired arms:
#   raw EAF / frozen V19 ICER-scalar / V20 ICER-DC-scalar / V20 ICER-DC-dual.
# No fitting, threshold sweep, selector/budget change, test, or closed-loop here.
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_20_eaf_icer_dc_screen_2gpu_v1}"
export VAL_SCREEN_SCENARIOS="${VAL_SCREEN_SCENARIOS:-500}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_20_design_exclude_v64_3_19_screen_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.20-eaf-icer-dc-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"
RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V19_SCALAR_CONFIG="bdse/configs/v64_3_19_icer_scalar_frozen_uploaded.yaml"
V20_SCALAR_CONFIG="bdse/configs/v64_3_20_icer_dc_scalar.yaml"
V20_DUAL_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/configs"
TIMING="$OUT_ROOT/provenance/v64_3_20_stage_timing.tsv"; : > "$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ local now; now=$(date +%s); printf '%s\t%s\n' "$STAGE_NAME" "$((now-STAGE_START))" >> "$TIMING"; }

[[ -d "$BDSE_VAL_CACHE" ]] || { echo "STOP: missing val cache $BDSE_VAL_CACHE" >&2; exit 2; }
[[ -s "$EAF_TRAIN_LOG" ]] || { echo "STOP: missing V64.3.13 train log $EAF_TRAIN_LOG" >&2; exit 2; }
[[ -s "$DESIGN_EXCLUDE_TOKENS" ]] || { echo "STOP: missing V64.3.20 design exclusion list" >&2; exit 2; }
[[ "$(wc -l < "$DESIGN_EXCLUDE_TOKENS")" -eq 3200 ]] || { echo "STOP DATA: V64.3.20 exclusion list must contain exactly 3200 already-inspected validation tokens." >&2; exit 2; }
for f in "$RAW_CONFIG" "$V19_SCALAR_CONFIG" "$V20_SCALAR_CONFIG" "$V20_DUAL_CONFIG"; do [[ -s "$f" ]] || { echo "STOP: missing config $f" >&2; exit 2; }; done

stage_start prerequisites
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_20 --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
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
python -m compileall -q bdse
python -m bdse.tools.check_v64_3_20_eaf_icer_dc_contract --config "$RAW_CONFIG" --expect raw --output "$OUT_ROOT/provenance/v64_3_20_raw_contract.json"
python -m bdse.tools.check_v64_3_19_eaf_icer_contract --config "$V19_SCALAR_CONFIG" --expect icer-scalar --output "$OUT_ROOT/provenance/v64_3_19_scalar_frozen_control_contract.json"
python -m bdse.tools.check_v64_3_20_eaf_icer_dc_contract --config "$V20_SCALAR_CONFIG" --expect icer-dc-scalar --frozen-v19-config "$V19_SCALAR_CONFIG" --output "$OUT_ROOT/provenance/v64_3_20_scalar_contract.json"
python -m bdse.tools.check_v64_3_20_eaf_icer_dc_contract --config "$V20_DUAL_CONFIG" --expect icer-dc-dual --frozen-v19-config "bdse/configs/v64_3_19_icer_dual_frozen_uploaded.yaml" --output "$OUT_ROOT/provenance/v64_3_20_dual_contract.json"
pytest -q bdse/tests/test_v64_3_20_eaf_icer_dc.py bdse/tests/test_v64_3_19_eaf_icer.py bdse/tests/test_v64_3_18_eaf_dacer.py bdse/tests/test_v64_3_17_eaf_daler.py bdse/tests/test_v64_3_16_eaf_raer.py bdse/tests/test_v64_3_15_eaf_eair.py bdse/tests/test_v64_3_14_eaf_ocfi.py bdse/tests/test_v64_3_13_eaf_dmvr.py bdse/tests/test_v64_3_12_cet_bdmu.py bdse/tests/test_v64_3_11_btp_bdmu.py bdse/tests/test_v64_3_10_hap_bdmu.py bdse/tests/test_v64_3_9_af_bdmu.py bdse/tests/test_v64_3_8_bdmu.py bdse/tests/test_v64_3_7_darm_dbr.py bdse/tests/test_v64_3_6_bcha_lbpr.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

# Select an entirely new fresh set. The previous V19 500 scenes have already been
# added to the permanent design-exclusion set; selection uses token identity + fixed hash only.
stage_start fresh_token_selection
VAL_TOKENS="$OUT_ROOT/provenance/val_screen_fresh_tokens.txt"
SPLIT_AUDIT="$OUT_ROOT/provenance/v64_3_20_fresh_split_audit.json"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$DESIGN_EXCLUDE_TOKENS" --count "$VAL_SCREEN_SCENARIOS" --hash-seed "$FRESH_HASH_SEED" --output "$VAL_TOKENS" --audit-output "$SPLIT_AUDIT" > "$OUT_ROOT/logs/fresh_token_selection.out"
stage_end

run_eval(){
  local gpu="$1" cfg="$2" tag="$3" edge="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios "$VAL_SCREEN_SCENARIOS" --scenario-token-file "$VAL_TOKENS" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/val_fresh_${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/val_fresh_${tag}_rows.jsonl" --frontier-edge-output "$edge" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/val_fresh_${tag}.out" 2>&1
}

# Wave 1 isolates the all-flagged semantic correction with the exact same frozen
# scalar heads: V19 scalar -> V20 scalar. Wave 2 tests signed attribution increment:
# V20 scalar -> V20 dual, while raw gives the endpoint reference.
stage_start fresh_wave_raw_v19
RAW_E="$OUT_ROOT/provenance/val_fresh_raw_edges.jsonl"
V19_E="$OUT_ROOT/provenance/val_fresh_v19_scalar_edges.jsonl"
run_eval "$GPU0" "$RAW_CONFIG" raw "$RAW_E" & P0=$!
run_eval "$GPU1" "$V19_SCALAR_CONFIG" v19_scalar "$V19_E" & P1=$!
S=0; wait "$P0" || S=1; wait "$P1" || S=1; [[ "$S" -eq 0 ]] || { echo 'STOP ENGINEERING: raw/V19 fresh replay failed' >&2; exit 2; }
stage_end

stage_start fresh_wave_v20
V20_S_E="$OUT_ROOT/provenance/val_fresh_v20_scalar_edges.jsonl"
V20_D_E="$OUT_ROOT/provenance/val_fresh_v20_dual_edges.jsonl"
run_eval "$GPU0" "$V20_SCALAR_CONFIG" v20_scalar "$V20_S_E" & P0=$!
run_eval "$GPU1" "$V20_DUAL_CONFIG" v20_dual "$V20_D_E" & P1=$!
S=0; wait "$P0" || S=1; wait "$P1" || S=1; [[ "$S" -eq 0 ]] || { echo 'STOP ENGINEERING: V20 fresh replay failed' >&2; exit 2; }
stage_end

# Paired identity + the V19 speed optimization must remain active in all four arms.
python - "$VAL_TOKENS" "$OUT_ROOT/provenance" <<'PY'
import json,sys,pathlib
want=[x.strip() for x in open(sys.argv[1]) if x.strip()]; root=pathlib.Path(sys.argv[2])
for tag in ['raw','v19_scalar','v20_scalar','v20_dual']:
    rows=[json.loads(x) for x in open(root/f'val_fresh_{tag}_rows.jsonl') if x.strip()]
    got=[str(r['scenario_token']) for r in rows]
    if len(got)!=len(want) or set(got)!=set(want):
        raise SystemExit(f'STOP DATA: {tag} token mismatch {len(got)}/{len(want)}')
    m=json.load(open(root/f'val_fresh_{tag}_metrics.json'))
    if not m.get('scenario_token_prefilter_active',False):
        raise SystemExit(f'STOP SPEED/ENGINEERING: {tag} did not activate pre-load token filter')
print('PASS paired fresh token identity and pre-load cache filtering: 4/4 arms')
PY

stage_start screen_check
SCREEN="$OUT_ROOT/provenance/v64_3_20_eaf_icer_dc_screen.json"
python -m bdse.tools.check_v64_3_20_eaf_icer_dc_screen \
  --raw-metrics "$OUT_ROOT/provenance/val_fresh_raw_metrics.json" \
  --v19-scalar-metrics "$OUT_ROOT/provenance/val_fresh_v19_scalar_metrics.json" \
  --v20-scalar-metrics "$OUT_ROOT/provenance/val_fresh_v20_scalar_metrics.json" \
  --v20-dual-metrics "$OUT_ROOT/provenance/val_fresh_v20_dual_metrics.json" \
  --raw-rows "$OUT_ROOT/provenance/val_fresh_raw_rows.jsonl" \
  --v19-scalar-rows "$OUT_ROOT/provenance/val_fresh_v19_scalar_rows.jsonl" \
  --v20-scalar-rows "$OUT_ROOT/provenance/val_fresh_v20_scalar_rows.jsonl" \
  --v20-dual-rows "$OUT_ROOT/provenance/val_fresh_v20_dual_rows.jsonl" \
  --v19-scalar-edge-output "$V19_E" \
  --v20-scalar-edge-output "$V20_S_E" \
  --v20-dual-edge-output "$V20_D_E" \
  --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_20_screen.out"
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"

python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print('FULL_PROMOTION=',r.get('full_promotion'))
print('NEXT_ACTION=',r.get('next_action'))
if not r.get('full_promotion',False):
    raise SystemExit('STOP SCREEN: do not run full/test/closed-loop; follow next_action')
print('PASS SCREEN ONLY. Next allowed stage is one independent frozen full-val reproduction; test/closed-loop remain forbidden until reproduction passes.')
PY
