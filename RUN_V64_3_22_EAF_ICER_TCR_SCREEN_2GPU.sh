#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

# V64.3.22 EAF-ICER-TCR double-fresh causal screen.
# Frozen: B<=16, M=24, acquisition, EAF frontier value, exact attribution,
# V19 support/dominance heads, one-sided/evidence certificate and structural guard.
# New: TRAIN-only magnitude-weighted expected-improvement risk heads, with an
# evidence-only causal control and a planner-transition-conditioned main arm.
# No threshold, risk-weight, B/M, certificate, guard or view-weight sweep.
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_22_eaf_icer_tcr_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_22_design_exclude_v64_3_21_screen_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.22-eaf-icer-tcr-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"
RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V20_DUAL_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
V21_CONTROL="bdse/configs/v64_3_21_mcr_profile_retention_mean_frozen.yaml"
EVIDENCE_RISK_CONFIG="$OUT_ROOT/configs/v64_3_22_evidence_risk_dual.yaml"
TRANS_SCALAR_CONFIG="$OUT_ROOT/configs/v64_3_22_transition_risk_scalar.yaml"
TRANS_DUAL_CONFIG="$OUT_ROOT/configs/v64_3_22_transition_risk_dual.yaml"
TRAIN_EDGES="$OUT_ROOT/provenance/train_v20_frozen_with_transition_edges.jsonl"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/configs"
TIMING="$OUT_ROOT/provenance/v64_3_22_stage_timing.tsv"; : > "$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ local now; now=$(date +%s); printf '%s\t%s\n' "$STAGE_NAME" "$((now-STAGE_START))" >> "$TIMING"; }

[[ -d "$BDSE_TRAIN_CACHE" ]] || { echo "STOP: missing train cache $BDSE_TRAIN_CACHE" >&2; exit 2; }
[[ -d "$BDSE_VAL_CACHE" ]] || { echo "STOP: missing val cache $BDSE_VAL_CACHE" >&2; exit 2; }
[[ -s "$DESIGN_EXCLUDE_TOKENS" ]] || { echo "STOP: missing V22 design exclusion" >&2; exit 2; }
[[ "$(wc -l < "$DESIGN_EXCLUDE_TOKENS")" -eq 4700 ]] || { echo "STOP DATA: V22 exclusion must be exactly 4700 inspected validation tokens" >&2; exit 2; }
for f in "$RAW_CONFIG" "$V20_DUAL_CONFIG" "$V21_CONTROL"; do [[ -s "$f" ]] || { echo "STOP: missing config $f" >&2; exit 2; }; done

stage_start prerequisites
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_22 --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
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
pytest -q bdse/tests/test_v64_3_22_eaf_icer_tcr.py bdse/tests/test_v64_3_21_eaf_icer_mcr.py bdse/tests/test_v64_3_20_eaf_icer_dc.py bdse/tests/test_v64_3_19_eaf_icer.py bdse/tests/test_v64_3_18_eaf_dacer.py bdse/tests/test_v64_3_17_eaf_daler.py bdse/tests/test_v64_3_16_eaf_raer.py bdse/tests/test_v64_3_15_eaf_eair.py bdse/tests/test_v64_3_14_eaf_ocfi.py bdse/tests/test_v64_3_13_eaf_dmvr.py bdse/tests/test_v64_3_12_cet_bdmu.py bdse/tests/test_v64_3_11_btp_bdmu.py bdse/tests/test_v64_3_10_hap_bdmu.py bdse/tests/test_v64_3_9_af_bdmu.py bdse/tests/test_v64_3_8_bdmu.py bdse/tests/test_v64_3_7_darm_dbr.py bdse/tests/test_v64_3_6_bcha_lbpr.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

# One frozen TRAIN replay is required because V64.3.22 adds runtime-only
# candidate-trajectory transition instrumentation that does not exist in the
# historical V18 edge file.  It uses the already-fitted V20 dual policy solely
# to expose frozen support/dominance logits; teacher is still diagnostics only.
stage_start train_transition_frontier_replay
CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop \
  --config "$V20_DUAL_CONFIG" --checkpoint "$EAF_CKPT" --split train --preprocessed-dir "$BDSE_TRAIN_CACHE" \
  --max-scenarios 3000 --output "$OUT_ROOT/provenance/train_v20_transition_metrics.json" \
  --frontier-edge-output "$TRAIN_EDGES" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/train_transition_frontier.out" 2>&1
[[ -s "$TRAIN_EDGES" ]] || { echo "STOP TRAIN: missing transition frontier edges" >&2; exit 2; }
stage_end

stage_start train_only_regret_risk_fit
python -m bdse.tools.fit_v64_3_22_eaf_icer_tcr \
  --train-frontier-edges "$TRAIN_EDGES" --base-v20-dual-config "$V20_DUAL_CONFIG" \
  --output-evidence-risk-config "$EVIDENCE_RISK_CONFIG" --output-transition-scalar-config "$TRANS_SCALAR_CONFIG" --output-transition-dual-config "$TRANS_DUAL_CONFIG" \
  --output-report "$OUT_ROOT/provenance/v64_3_22_train_only_regret_risk_fit.json" | tee "$OUT_ROOT/logs/v64_3_22_regret_risk_fit.out"
python -m bdse.tools.check_v64_3_22_eaf_icer_tcr_contract --config "$EVIDENCE_RISK_CONFIG" --expect evidence-risk-dual --frozen-v20-dual-config "$V20_DUAL_CONFIG" --output "$OUT_ROOT/provenance/evidence_risk_contract.json"
python -m bdse.tools.check_v64_3_22_eaf_icer_tcr_contract --config "$TRANS_SCALAR_CONFIG" --expect transition-risk-scalar --frozen-v20-dual-config "$V20_DUAL_CONFIG" --output "$OUT_ROOT/provenance/transition_scalar_contract.json"
python -m bdse.tools.check_v64_3_22_eaf_icer_tcr_contract --config "$TRANS_DUAL_CONFIG" --expect transition-risk-dual --frozen-v20-dual-config "$V20_DUAL_CONFIG" --output "$OUT_ROOT/provenance/transition_dual_contract.json"
stage_end

stage_start fresh_token_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh_1000_tokens.txt"; TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"; TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$DESIGN_EXCLUDE_TOKENS" --count 1000 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" --audit-output "$OUT_ROOT/provenance/v64_3_22_fresh_1000_audit.json" > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1000" > "$TOKA"; tail -n 500 "$TOK1000" > "$TOKB"
python - "$TOK1000" "$TOKA" "$TOKB" <<'PY'
import sys
all=[x.strip() for x in open(sys.argv[1]) if x.strip()]; A=[x.strip() for x in open(sys.argv[2]) if x.strip()]; B=[x.strip() for x in open(sys.argv[3]) if x.strip()]
assert len(all)==len(set(all))==1000 and len(A)==len(set(A))==500 and len(B)==len(set(B))==500 and not set(A)&set(B) and set(A)|set(B)==set(all)
print('PASS V22 fresh identity: 1000 unique = independent A500 + B500')
PY
stage_end

run_eval(){
  local gpu="$1" cfg="$2" split="$3" tag="$4"; local tok="$TOKA"; [[ "$split" == B ]] && tok="$TOKB"
  CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios 500 --scenario-token-file "$tok" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/${split}_${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/${split}_${tag}_rows.jsonl" --frontier-edge-output "$OUT_ROOT/provenance/${split}_${tag}_edges.jsonl" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${split}_${tag}.out" 2>&1
}

# 5 causal arms x 2 independent blocks.  No pooled rescue.
stage_start fresh_wave_1; run_eval "$GPU0" "$RAW_CONFIG" A raw & P0=$!; run_eval "$GPU1" "$V21_CONTROL" A v21_control & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_2; run_eval "$GPU0" "$EVIDENCE_RISK_CONFIG" A evidence_risk & P0=$!; run_eval "$GPU1" "$TRANS_SCALAR_CONFIG" A transition_scalar & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_3; run_eval "$GPU0" "$TRANS_DUAL_CONFIG" A transition_dual & P0=$!; run_eval "$GPU1" "$RAW_CONFIG" B raw & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_4; run_eval "$GPU0" "$V21_CONTROL" B v21_control & P0=$!; run_eval "$GPU1" "$EVIDENCE_RISK_CONFIG" B evidence_risk & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_5; run_eval "$GPU0" "$TRANS_SCALAR_CONFIG" B transition_scalar & P0=$!; run_eval "$GPU1" "$TRANS_DUAL_CONFIG" B transition_dual & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1])
for split,tokfile in [('A',sys.argv[2]),('B',sys.argv[3])]:
    want={x.strip() for x in open(tokfile) if x.strip()}
    for tag in ['raw','v21_control','evidence_risk','transition_scalar','transition_dual']:
        rows=[json.loads(x) for x in open(root/f'{split}_{tag}_rows.jsonl') if x.strip()]; got={str(r['scenario_token']) for r in rows}
        if len(rows)!=500 or got!=want: raise SystemExit(f'STOP DATA: {split}/{tag} token mismatch')
        m=json.load(open(root/f'{split}_{tag}_metrics.json'))
        if not m.get('scenario_token_prefilter_active',False): raise SystemExit(f'STOP SPEED/DATA: {split}/{tag} pre-load token filter inactive')
print('PASS paired identity + pre-deserialization filtering: 10/10 arms')
PY
stage_end

stage_start double_fresh_screen
for split in A B; do
python -m bdse.tools.check_v64_3_22_eaf_icer_tcr_split --split-name "$split" \
  --raw-metrics "$OUT_ROOT/provenance/${split}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${split}_raw_rows.jsonl" \
  --v21-control-metrics "$OUT_ROOT/provenance/${split}_v21_control_metrics.json" --v21-control-rows "$OUT_ROOT/provenance/${split}_v21_control_rows.jsonl" --v21-control-edges "$OUT_ROOT/provenance/${split}_v21_control_edges.jsonl" \
  --evidence-risk-metrics "$OUT_ROOT/provenance/${split}_evidence_risk_metrics.json" --evidence-risk-rows "$OUT_ROOT/provenance/${split}_evidence_risk_rows.jsonl" --evidence-risk-edges "$OUT_ROOT/provenance/${split}_evidence_risk_edges.jsonl" \
  --transition-scalar-metrics "$OUT_ROOT/provenance/${split}_transition_scalar_metrics.json" --transition-scalar-rows "$OUT_ROOT/provenance/${split}_transition_scalar_rows.jsonl" --transition-scalar-edges "$OUT_ROOT/provenance/${split}_transition_scalar_edges.jsonl" \
  --transition-dual-metrics "$OUT_ROOT/provenance/${split}_transition_dual_metrics.json" --transition-dual-rows "$OUT_ROOT/provenance/${split}_transition_dual_rows.jsonl" --transition-dual-edges "$OUT_ROOT/provenance/${split}_transition_dual_edges.jsonl" \
  --output "$OUT_ROOT/provenance/v64_3_22_split_${split}_screen.json" | tee "$OUT_ROOT/logs/v64_3_22_split_${split}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_22_eaf_icer_tcr_double_fresh_screen.json"
python -m bdse.tools.check_v64_3_22_eaf_icer_tcr_screen --split-a-report "$OUT_ROOT/provenance/v64_3_22_split_A_screen.json" --split-b-report "$OUT_ROOT/provenance/v64_3_22_split_B_screen.json" --train-fit-report "$OUT_ROOT/provenance/v64_3_22_train_only_regret_risk_fit.json" --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_22_double_fresh_screen.out"
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"

python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print('DOUBLE_FRESH_PASS=',r.get('both_independent_500_scene_blocks_pass')); print('SCALAR_BOTH=',r.get('transition_scalar_both_blocks_pass')); print('NEXT_ACTION=',r.get('next_action'))
if not r.get('full_promotion_to_independent_full_val_reproduction',False): raise SystemExit('STOP SCREEN: do not run full/test/closed-loop; follow next_action')
print('PASS DOUBLE-FRESH SCREEN ONLY. Next allowed stage: one frozen independent full-val reproduction. Test/closed-loop remain forbidden.')
PY
