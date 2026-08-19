#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

# V64.3.24 EAF-ICER-TTCR double-fresh causal screen.
# Frozen: B<=16, M=24, acquisition/selector, EAF value/frontier, exact selected-
# evidence attribution, V19/V20 support+scalar-dominance, evidence/final guards,
# structural-risk delegation.  New: type-aware contrasts from already-selected
# atoms plus material-tail coherence.  No B/M/K/SE/tau/threshold/guard sweep.
#
# Factorial causal chain on each independent fresh block:
#   V23 evidence-LCB/dominance-first
#   evidence-tail/dominance-first             (tail objective only)
#   typed-LCB/dominance-first                 (typed representation only)
#   typed-tail/dominance-first                (representation x tail)
#   typed-tail/risk-first                     (+ extremal-operator alignment)
# plus raw and frozen V20 controls.

export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_24_eaf_icer_typed_tail_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_24_design_exclude_v64_3_23_screen_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.24-eaf-icer-typed-tail-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"

RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V20_DUAL_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
TRAIN_EDGES="$OUT_ROOT/provenance/train_v20_frozen_with_typed_selected_evidence_edges.jsonl"
TRAIN_TOKENS="$OUT_ROOT/provenance/v64_3_24_train_frontier_tokens.txt"
EVIDENCE_MEMORY="$OUT_ROOT/provenance/v64_3_24_evidence_local_memory.npz"
TYPED_MEMORY="$OUT_ROOT/provenance/v64_3_24_typed_local_tail_memory.npz"
EVIDENCE_LCB_CONFIG="$OUT_ROOT/configs/v64_3_24_evidence_lcb.yaml"
EVIDENCE_TAIL_CONFIG="$OUT_ROOT/configs/v64_3_24_evidence_tail.yaml"
TYPED_LCB_CONFIG="$OUT_ROOT/configs/v64_3_24_typed_lcb.yaml"
TYPED_TAIL_DOM_CONFIG="$OUT_ROOT/configs/v64_3_24_typed_tail_dominance.yaml"
TYPED_TAIL_RISK_CONFIG="$OUT_ROOT/configs/v64_3_24_typed_tail_risk_first.yaml"

mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/configs"
TIMING="$OUT_ROOT/provenance/v64_3_24_stage_timing.tsv"; : > "$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ local now; now=$(date +%s); printf '%s\t%s\n' "$STAGE_NAME" "$((now-STAGE_START))" >> "$TIMING"; }

[[ -d "$BDSE_TRAIN_CACHE" ]] || { echo "STOP: missing train cache $BDSE_TRAIN_CACHE" >&2; exit 2; }
[[ -d "$BDSE_VAL_CACHE" ]] || { echo "STOP: missing val cache $BDSE_VAL_CACHE" >&2; exit 2; }
[[ -s "$DESIGN_EXCLUDE_TOKENS" ]] || { echo "STOP: missing V24 design exclusion" >&2; exit 2; }
[[ "$(wc -l < "$DESIGN_EXCLUDE_TOKENS")" -eq 5700 ]] || { echo "STOP DATA: V24 exclusion must be exactly 5700 previously inspected validation tokens" >&2; exit 2; }
python - "$DESIGN_EXCLUDE_TOKENS" <<'PY'
import sys
p=sys.argv[1]; toks=[x.strip() for x in open(p) if x.strip()]
if len(toks)!=5700 or len(set(toks))!=5700:
    raise SystemExit(f'STOP DATA: V24 exclusion must contain 5700 unique tokens, got lines={len(toks)} unique={len(set(toks))}')
print('PASS V24 permanent design exclusion: 5700 unique previously inspected validation tokens')
PY
for f in "$RAW_CONFIG" "$V20_DUAL_CONFIG"; do [[ -s "$f" ]] || { echo "STOP: missing config $f" >&2; exit 2; }; done

stage_start prerequisites
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_24 --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
read -r SELECTED_EPOCH < <(python - "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('training_instrumentation_valid',False) or not r.get('acquisition_frozen',False) or not r.get('value_estimation_gain',False): raise SystemExit('STOP: V64.3.13 prerequisites invalid')
print(int(r['selected_epoch']))
PY
)
if [[ -z "${EAF_CKPT:-}" ]]; then EAF_CKPT="$EAF_V64_3_13_ROOT/train/checkpoints/bdse_v64_saqa_bcc.epoch_$(printf '%04d' "$((SELECTED_EPOCH + 1))").pt"; fi
export EAF_CKPT; [[ -s "$EAF_CKPT" ]] || { echo "STOP: missing EAF checkpoint $EAF_CKPT" >&2; exit 2; }
python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_24_eaf_icer_typed_tail.py \
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
  bdse/tests/test_v64_3_13_eaf_dmvr.py \
  bdse/tests/test_v64_3_12_cet_bdmu.py \
  bdse/tests/test_v64_3_11_btp_bdmu.py \
  bdse/tests/test_v64_3_10_hap_bdmu.py \
  bdse/tests/test_v64_3_9_af_bdmu.py \
  bdse/tests/test_v64_3_8_bdmu.py \
  bdse/tests/test_v64_3_7_darm_dbr.py \
  bdse/tests/test_v64_3_6_bcha_lbpr.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

# V24 MUST replay TRAIN.  V22/V23 frontier logs did not serialize typed selected-
# evidence contrasts, so reuse would silently turn the new mechanism into missing data.
stage_start train_typed_frontier_replay
CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop \
  --config "$V20_DUAL_CONFIG" --checkpoint "$EAF_CKPT" --split train --preprocessed-dir "$BDSE_TRAIN_CACHE" \
  --max-scenarios 3000 --output "$OUT_ROOT/provenance/train_v20_typed_metrics.json" \
  --frontier-edge-output "$TRAIN_EDGES" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/train_typed_frontier.out" 2>&1
[[ -s "$TRAIN_EDGES" ]] || { echo "STOP TRAIN: missing typed TRAIN frontier edges" >&2; exit 2; }
stage_end

stage_start train_only_typed_tail_fit
python -m bdse.tools.fit_v64_3_24_eaf_icer_typed_tail \
  --train-frontier-edges "$TRAIN_EDGES" --base-v20-dual-config "$V20_DUAL_CONFIG" \
  --output-evidence-memory "$EVIDENCE_MEMORY" --output-typed-memory "$TYPED_MEMORY" \
  --output-evidence-baseline-config "$EVIDENCE_LCB_CONFIG" \
  --output-evidence-tail-config "$EVIDENCE_TAIL_CONFIG" \
  --output-typed-lcb-config "$TYPED_LCB_CONFIG" \
  --output-typed-tail-dominance-config "$TYPED_TAIL_DOM_CONFIG" \
  --output-typed-tail-config "$TYPED_TAIL_RISK_CONFIG" \
  --output-train-token-file "$TRAIN_TOKENS" --output-report "$OUT_ROOT/provenance/v64_3_24_train_only_typed_tail_fit.json" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_24_typed_tail_fit.out"

python -m bdse.tools.check_v64_3_24_eaf_icer_typed_tail_contract --config "$EVIDENCE_LCB_CONFIG" --expect evidence-lcb --frozen-v20-dual-config "$V20_DUAL_CONFIG" --output "$OUT_ROOT/provenance/evidence_lcb_contract.json"
python -m bdse.tools.check_v64_3_24_eaf_icer_typed_tail_contract --config "$EVIDENCE_TAIL_CONFIG" --expect evidence-tail --frozen-v20-dual-config "$V20_DUAL_CONFIG" --output "$OUT_ROOT/provenance/evidence_tail_contract.json"
python -m bdse.tools.check_v64_3_24_eaf_icer_typed_tail_contract --config "$TYPED_LCB_CONFIG" --expect typed-lcb --frozen-v20-dual-config "$V20_DUAL_CONFIG" --output "$OUT_ROOT/provenance/typed_lcb_contract.json"
python -m bdse.tools.check_v64_3_24_eaf_icer_typed_tail_contract --config "$TYPED_TAIL_DOM_CONFIG" --expect typed-tail-dominance --frozen-v20-dual-config "$V20_DUAL_CONFIG" --output "$OUT_ROOT/provenance/typed_tail_dominance_contract.json"
python -m bdse.tools.check_v64_3_24_eaf_icer_typed_tail_contract --config "$TYPED_TAIL_RISK_CONFIG" --expect typed-tail-risk-first --frozen-v20-dual-config "$V20_DUAL_CONFIG" --output "$OUT_ROOT/provenance/typed_tail_risk_first_contract.json"
stage_end

stage_start fresh_token_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh_1000_tokens.txt"; TOKC="$OUT_ROOT/provenance/val_screen_fresh_C_tokens.txt"; TOKD="$OUT_ROOT/provenance/val_screen_fresh_D_tokens.txt"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$DESIGN_EXCLUDE_TOKENS" --count 1000 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" --audit-output "$OUT_ROOT/provenance/v64_3_24_fresh_1000_audit.json" > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1000" > "$TOKC"; tail -n 500 "$TOK1000" > "$TOKD"
python - "$TOK1000" "$TOKC" "$TOKD" "$DESIGN_EXCLUDE_TOKENS" <<'PY'
import sys
all=[x.strip() for x in open(sys.argv[1]) if x.strip()]; C=[x.strip() for x in open(sys.argv[2]) if x.strip()]; D=[x.strip() for x in open(sys.argv[3]) if x.strip()]; old={x.strip() for x in open(sys.argv[4]) if x.strip()}
assert len(all)==len(set(all))==1000 and len(C)==len(set(C))==500 and len(D)==len(set(D))==500 and not set(C)&set(D) and set(C)|set(D)==set(all)
assert not set(all)&old
print('PASS V24 fresh identity: unseen 1000 = independent C500 + D500')
PY
python - "$TRAIN_TOKENS" "$TOK1000" <<'PY'
import sys
train={x.strip() for x in open(sys.argv[1]) if x.strip()}; fresh={x.strip() for x in open(sys.argv[2]) if x.strip()}; overlap=train&fresh
if overlap: raise SystemExit(f'STOP DATA LEAKAGE: {len(overlap)} fresh validation tokens overlap TRAIN frontier')
print(f'PASS TRAIN/fresh isolation: train={len(train)} fresh={len(fresh)} overlap=0')
PY
stage_end

run_eval(){
  local gpu="$1" cfg="$2" split="$3" tag="$4"; local tok="$TOKC"; [[ "$split" == D ]] && tok="$TOKD"
  CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop \
    --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" \
    --max-scenarios 500 --scenario-token-file "$tok" --require-all-scenario-tokens \
    --output "$OUT_ROOT/provenance/${split}_${tag}_metrics.json" \
    --per-sample-output "$OUT_ROOT/provenance/${split}_${tag}_rows.jsonl" \
    --frontier-edge-output "$OUT_ROOT/provenance/${split}_${tag}_edges.jsonl" \
    --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${split}_${tag}.out" 2>&1
}

# 7 causal arms x 2 independent blocks. No pooled rescue; no same-block tuning.
stage_start fresh_wave_1; run_eval "$GPU0" "$RAW_CONFIG" C raw & P0=$!; run_eval "$GPU1" "$V20_DUAL_CONFIG" C v20 & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_2; run_eval "$GPU0" "$EVIDENCE_LCB_CONFIG" C evidence_lcb & P0=$!; run_eval "$GPU1" "$EVIDENCE_TAIL_CONFIG" C evidence_tail & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_3; run_eval "$GPU0" "$TYPED_LCB_CONFIG" C typed_lcb & P0=$!; run_eval "$GPU1" "$TYPED_TAIL_DOM_CONFIG" C typed_tail_dominance & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_4; run_eval "$GPU0" "$TYPED_TAIL_RISK_CONFIG" C typed_tail_risk_first & P0=$!; run_eval "$GPU1" "$RAW_CONFIG" D raw & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_5; run_eval "$GPU0" "$V20_DUAL_CONFIG" D v20 & P0=$!; run_eval "$GPU1" "$EVIDENCE_LCB_CONFIG" D evidence_lcb & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_6; run_eval "$GPU0" "$EVIDENCE_TAIL_CONFIG" D evidence_tail & P0=$!; run_eval "$GPU1" "$TYPED_LCB_CONFIG" D typed_lcb & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_7; run_eval "$GPU0" "$TYPED_TAIL_DOM_CONFIG" D typed_tail_dominance & P0=$!; run_eval "$GPU1" "$TYPED_TAIL_RISK_CONFIG" D typed_tail_risk_first & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKC" "$TOKD" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); tags=['raw','v20','evidence_lcb','evidence_tail','typed_lcb','typed_tail_dominance','typed_tail_risk_first']
for split,tokfile in [('C',sys.argv[2]),('D',sys.argv[3])]:
    want={x.strip() for x in open(tokfile) if x.strip()}
    for tag in tags:
        rows=[json.loads(x) for x in open(root/f'{split}_{tag}_rows.jsonl') if x.strip()]; got={str(r['scenario_token']) for r in rows}
        if len(rows)!=500 or got!=want: raise SystemExit(f'STOP DATA: {split}/{tag} token mismatch')
        m=json.load(open(root/f'{split}_{tag}_metrics.json'))
        if not m.get('scenario_token_prefilter_active',False): raise SystemExit(f'STOP SPEED/DATA: {split}/{tag} pre-load token filter inactive')
print('PASS paired identity + pre-deserialization filtering: 14/14 arms')
PY
stage_end

stage_start double_fresh_screen
for split in C D; do
python -m bdse.tools.check_v64_3_24_eaf_icer_typed_tail_split --split-name "$split" --material-delta-threshold 0.004 \
  --raw-metrics "$OUT_ROOT/provenance/${split}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${split}_raw_rows.jsonl" \
  --v20-metrics "$OUT_ROOT/provenance/${split}_v20_metrics.json" --v20-rows "$OUT_ROOT/provenance/${split}_v20_rows.jsonl" --v20-edges "$OUT_ROOT/provenance/${split}_v20_edges.jsonl" \
  --evidence-lcb-metrics "$OUT_ROOT/provenance/${split}_evidence_lcb_metrics.json" --evidence-lcb-rows "$OUT_ROOT/provenance/${split}_evidence_lcb_rows.jsonl" --evidence-lcb-edges "$OUT_ROOT/provenance/${split}_evidence_lcb_edges.jsonl" \
  --evidence-tail-metrics "$OUT_ROOT/provenance/${split}_evidence_tail_metrics.json" --evidence-tail-rows "$OUT_ROOT/provenance/${split}_evidence_tail_rows.jsonl" --evidence-tail-edges "$OUT_ROOT/provenance/${split}_evidence_tail_edges.jsonl" \
  --typed-lcb-metrics "$OUT_ROOT/provenance/${split}_typed_lcb_metrics.json" --typed-lcb-rows "$OUT_ROOT/provenance/${split}_typed_lcb_rows.jsonl" --typed-lcb-edges "$OUT_ROOT/provenance/${split}_typed_lcb_edges.jsonl" \
  --typed-tail-dominance-metrics "$OUT_ROOT/provenance/${split}_typed_tail_dominance_metrics.json" --typed-tail-dominance-rows "$OUT_ROOT/provenance/${split}_typed_tail_dominance_rows.jsonl" --typed-tail-dominance-edges "$OUT_ROOT/provenance/${split}_typed_tail_dominance_edges.jsonl" \
  --typed-tail-risk-first-metrics "$OUT_ROOT/provenance/${split}_typed_tail_risk_first_metrics.json" --typed-tail-risk-first-rows "$OUT_ROOT/provenance/${split}_typed_tail_risk_first_rows.jsonl" --typed-tail-risk-first-edges "$OUT_ROOT/provenance/${split}_typed_tail_risk_first_edges.jsonl" \
  --output "$OUT_ROOT/provenance/v64_3_24_split_${split}_screen.json" | tee "$OUT_ROOT/logs/v64_3_24_split_${split}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_24_eaf_icer_typed_tail_double_fresh_screen.json"
python -m bdse.tools.check_v64_3_24_eaf_icer_typed_tail_screen \
  --split-c-report "$OUT_ROOT/provenance/v64_3_24_split_C_screen.json" \
  --split-d-report "$OUT_ROOT/provenance/v64_3_24_split_D_screen.json" \
  --train-fit-report "$OUT_ROOT/provenance/v64_3_24_train_only_typed_tail_fit.json" \
  --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_24_double_fresh_screen.out"
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"

python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for k in ['both_independent_500_scene_blocks_pass','selected_material_tail_safe_both','typed_representation_incremental_both','tail_objective_incremental_both','risk_first_extremal_alignment_incremental_both','next_action']:
    print(f'{k}={r.get(k)}')
if not r.get('full_promotion_to_independent_full_val_reproduction',False):
    raise SystemExit('STOP SCREEN: do not run full/test/closed-loop; follow next_action and component diagnostics')
print('PASS DOUBLE-FRESH SCREEN ONLY. Next allowed stage: one frozen independent full-val reproduction. Test/closed-loop remain forbidden.')
PY
