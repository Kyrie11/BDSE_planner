#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_24_eaf_icer_arc_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_24_design_exclude_v64_3_23_screen_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.24-eaf-icer-arc-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"
RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"; V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
TRAIN_EDGES="$OUT_ROOT/provenance/train_v20_with_full_attribution_spectrum_edges.jsonl"; TRAIN_TOKENS="$OUT_ROOT/provenance/v64_3_24_train_tokens.txt"; FIT_DIR="$OUT_ROOT/configs"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$FIT_DIR"
TIMING="$OUT_ROOT/provenance/v64_3_24_stage_timing.tsv"; : > "$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }; stage_end(){ printf '%s\t%s\n' "$STAGE_NAME" "$(( $(date +%s)-STAGE_START ))" >> "$TIMING"; }
[[ -d "$BDSE_TRAIN_CACHE" && -d "$BDSE_VAL_CACHE" ]] || { echo 'STOP: missing train/val cache' >&2; exit 2; }
[[ -s "$DESIGN_EXCLUDE_TOKENS" ]] || { echo 'STOP: missing design exclusion' >&2; exit 2; }
[[ "$(wc -l < "$DESIGN_EXCLUDE_TOKENS")" -eq 5700 ]] || { echo 'STOP DATA: V24 exclusion must be exactly 5700 inspected validation tokens' >&2; exit 2; }
for f in "$RAW_CONFIG" "$V20_CONFIG"; do [[ -s "$f" ]] || { echo "STOP: missing $f" >&2; exit 2; }; done

stage_start prerequisites
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_24 --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False): raise SystemExit('STOP prerequisites')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH+1))").pt"; fi
export EAF_CKPT; [[ -s "$EAF_CKPT" ]] || { echo "STOP: missing EAF checkpoint $EAF_CKPT" >&2; exit 2; }
python -m compileall -q bdse
pytest -q bdse/tests/test_v64_3_24_eaf_icer_arc.py bdse/tests/test_v64_3_23_eaf_icer_rcr.py bdse/tests/test_v64_3_22_eaf_icer_tcr.py bdse/tests/test_v64_3_21_eaf_icer_mcr.py bdse/tests/test_v64_3_20_eaf_icer_dc.py bdse/tests/test_v64_3_19_eaf_icer.py bdse/tests/test_v64_3_18_eaf_dacer.py bdse/tests/test_v64_3_17_eaf_daler.py bdse/tests/test_v64_3_16_eaf_raer.py bdse/tests/test_v64_3_15_eaf_eair.py bdse/tests/test_v64_3_14_eaf_ocfi.py bdse/tests/test_v64_3_13_eaf_dmvr.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

# V24 must replay TRAIN once: V23 provenance did not serialize the full B<=16 attribution spectrum.
stage_start train_full_attribution_replay
CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop --config "$V20_CONFIG" --checkpoint "$EAF_CKPT" --split train --preprocessed-dir "$BDSE_TRAIN_CACHE" --max-scenarios 3000 --output "$OUT_ROOT/provenance/train_v20_full_attr_metrics.json" --frontier-edge-output "$TRAIN_EDGES" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/train_full_attribution_frontier.out" 2>&1
[[ -s "$TRAIN_EDGES" ]] || { echo 'STOP TRAIN: missing full-attribution frontier' >&2; exit 2; }
stage_end

stage_start train_arc_fit
python -m bdse.tools.fit_v64_3_24_eaf_icer_arc --train-frontier-edges "$TRAIN_EDGES" --base-v20-dual-config "$V20_CONFIG" --output-dir "$FIT_DIR" --output-train-token-file "$TRAIN_TOKENS" --output-report "$OUT_ROOT/provenance/v64_3_24_train_arc_fit.json" 2>&1 | tee "$OUT_ROOT/logs/v64_3_24_arc_fit.out"
for spec in aggregate-meanse aggregate-downside attribution-meanse attribution-downside; do f=${spec//-/_}; python -m bdse.tools.check_v64_3_24_eaf_icer_arc_contract --config "$FIT_DIR/v64_3_24_${f}.yaml" --expect "$spec" --frozen-v20-dual-config "$V20_CONFIG" --output "$OUT_ROOT/provenance/${f}_contract.json"; done
stage_end

stage_start fresh_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh_1000_tokens.txt"; TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"; TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$DESIGN_EXCLUDE_TOKENS" --count 1000 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" --audit-output "$OUT_ROOT/provenance/v64_3_24_fresh_1000_audit.json" > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1000" > "$TOKA"; tail -n 500 "$TOK1000" > "$TOKB"
python - "$TRAIN_TOKENS" "$TOK1000" <<'PY'
import sys
tr={x.strip() for x in open(sys.argv[1]) if x.strip()}; fr={x.strip() for x in open(sys.argv[2]) if x.strip()};
if len(fr)!=1000 or tr&fr: raise SystemExit(f'STOP DATA: fresh/train identity failure fresh={len(fr)} overlap={len(tr&fr)}')
print('PASS V24 train/fresh identity isolation')
PY
stage_end

run_eval(){ local gpu="$1" cfg="$2" sp="$3" tag="$4"; local tok="$TOKA"; [[ "$sp" == B ]] && tok="$TOKB"; CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios 500 --scenario-token-file "$tok" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/${sp}_${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/${sp}_${tag}_rows.jsonl" --frontier-edge-output "$OUT_ROOT/provenance/${sp}_${tag}_edges.jsonl" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${sp}_${tag}.out" 2>&1; }
AGM="$FIT_DIR/v64_3_24_aggregate_meanse.yaml"; AGD="$FIT_DIR/v64_3_24_aggregate_downside.yaml"; ARM="$FIT_DIR/v64_3_24_attribution_meanse.yaml"; ARD="$FIT_DIR/v64_3_24_attribution_downside.yaml"
# Six arms per block, paired and independent; no pooled rescue.
for sp in A B; do
 stage_start "fresh_${sp}_wave1"; run_eval "$GPU0" "$RAW_CONFIG" "$sp" raw & p0=$!; run_eval "$GPU1" "$V20_CONFIG" "$sp" v20 & p1=$!; s=0; wait "$p0"||s=1;wait "$p1"||s=1;[[ $s -eq 0 ]]||exit 2;stage_end
 stage_start "fresh_${sp}_wave2"; run_eval "$GPU0" "$AGM" "$sp" aggregate_meanse & p0=$!; run_eval "$GPU1" "$AGD" "$sp" aggregate_downside & p1=$!; s=0; wait "$p0"||s=1;wait "$p1"||s=1;[[ $s -eq 0 ]]||exit 2;stage_end
 stage_start "fresh_${sp}_wave3"; run_eval "$GPU0" "$ARM" "$sp" attribution_meanse & p0=$!; run_eval "$GPU1" "$ARD" "$sp" attribution_downside & p1=$!; s=0; wait "$p0"||s=1;wait "$p1"||s=1;[[ $s -eq 0 ]]||exit 2;stage_end
done

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); tags=['raw','v20','aggregate_meanse','aggregate_downside','attribution_meanse','attribution_downside']
for sp,tf in [('A',sys.argv[2]),('B',sys.argv[3])]:
 want=[x.strip() for x in open(tf) if x.strip()]
 for tag in tags:
  rows=[json.loads(x) for x in open(root/f'{sp}_{tag}_rows.jsonl') if x.strip()]; got=[str(r['scenario_token']) for r in rows]
  if got!=want: raise SystemExit(f'STOP DATA: {sp}/{tag} ordered token mismatch')
  m=json.load(open(root/f'{sp}_{tag}_metrics.json'))
  if not m.get('scenario_token_prefilter_active',False): raise SystemExit(f'STOP DATA/SPEED: {sp}/{tag} pre-load filter inactive')
print('PASS V24 paired identity 12/12 arms')
PY
stage_end

stage_start screen
for sp in A B; do
 python -m bdse.tools.check_v64_3_24_eaf_icer_arc_split --split-name "$sp" \
  --raw-metrics "$OUT_ROOT/provenance/${sp}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${sp}_raw_rows.jsonl" \
  --v20-metrics "$OUT_ROOT/provenance/${sp}_v20_metrics.json" --v20-rows "$OUT_ROOT/provenance/${sp}_v20_rows.jsonl" --v20-edges "$OUT_ROOT/provenance/${sp}_v20_edges.jsonl" \
  --aggregate-meanse-metrics "$OUT_ROOT/provenance/${sp}_aggregate_meanse_metrics.json" --aggregate-meanse-rows "$OUT_ROOT/provenance/${sp}_aggregate_meanse_rows.jsonl" --aggregate-meanse-edges "$OUT_ROOT/provenance/${sp}_aggregate_meanse_edges.jsonl" \
  --aggregate-downside-metrics "$OUT_ROOT/provenance/${sp}_aggregate_downside_metrics.json" --aggregate-downside-rows "$OUT_ROOT/provenance/${sp}_aggregate_downside_rows.jsonl" --aggregate-downside-edges "$OUT_ROOT/provenance/${sp}_aggregate_downside_edges.jsonl" \
  --attribution-meanse-metrics "$OUT_ROOT/provenance/${sp}_attribution_meanse_metrics.json" --attribution-meanse-rows "$OUT_ROOT/provenance/${sp}_attribution_meanse_rows.jsonl" --attribution-meanse-edges "$OUT_ROOT/provenance/${sp}_attribution_meanse_edges.jsonl" \
  --attribution-downside-metrics "$OUT_ROOT/provenance/${sp}_attribution_downside_metrics.json" --attribution-downside-rows "$OUT_ROOT/provenance/${sp}_attribution_downside_rows.jsonl" --attribution-downside-edges "$OUT_ROOT/provenance/${sp}_attribution_downside_edges.jsonl" \
  --output "$OUT_ROOT/provenance/v64_3_24_split_${sp}_screen.json" | tee "$OUT_ROOT/logs/v64_3_24_split_${sp}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_24_eaf_icer_arc_double_fresh_screen.json"; python -m bdse.tools.check_v64_3_24_eaf_icer_arc_screen --split-a-report "$OUT_ROOT/provenance/v64_3_24_split_A_screen.json" --split-b-report "$OUT_ROOT/provenance/v64_3_24_split_B_screen.json" --train-fit-report "$OUT_ROOT/provenance/v64_3_24_train_arc_fit.json" --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_24_double_fresh_screen.out"; stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"
python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));print('DOUBLE_FRESH_PASS=',r.get('both_independent_blocks_pass'));print('DOWNSIDE_INCREMENTAL_BOTH=',r.get('downside_certificate_incremental_both'));print('ATTRIBUTION_INCREMENTAL_BOTH=',r.get('attribution_resolved_incremental_both'));print('NEXT_ACTION=',r.get('next_action'))
if not r.get('full_promotion_to_independent_full_val_reproduction',False): raise SystemExit('STOP SCREEN: do not run full/test/closed-loop; follow next_action')
print('PASS SCREEN ONLY: next allowed stage is one frozen independent full-val reproduction; test/closed-loop remain forbidden.')
PY
