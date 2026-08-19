#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

# V64.3.23 EAF-ICER-RCR double-fresh causal screen.
# Frozen: B<=16, M=24, acquisition, EAF value/frontier, exact attribution,
# V19/V20 support+dominance heads, final evidence/one-sided certificate and
# structural-risk guard.  New: TRAIN-only multiscale local replacement-regret
# lower bound plus self-consistent signed-evidence extremal ranking.
# No threshold/K/SE/view-weight/B/M/certificate/guard sweep.
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_23_eaf_icer_rcr_screen_2gpu_v1}"
export V22_OUT_ROOT="${V22_OUT_ROOT:-outputs_v64_3_22_eaf_icer_tcr_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_23_design_exclude_v64_3_21_screen_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.23-eaf-icer-rcr-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"
RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V20_DUAL_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
EVIDENCE_MEMORY="$OUT_ROOT/provenance/v64_3_23_evidence_local_memory.npz"
TRANSITION_MEMORY="$OUT_ROOT/provenance/v64_3_23_transition_local_memory.npz"
EVIDENCE_SCALAR_CONFIG="$OUT_ROOT/configs/v64_3_23_evidence_local_scalar.yaml"
EVIDENCE_RCR_CONFIG="$OUT_ROOT/configs/v64_3_23_evidence_local_rcr.yaml"
TRANS_RCR_CONFIG="$OUT_ROOT/configs/v64_3_23_transition_local_rcr_ablation.yaml"
TRAIN_EDGES="$OUT_ROOT/provenance/train_v20_frozen_with_transition_edges.jsonl"
TRAIN_TOKENS="$OUT_ROOT/provenance/v64_3_23_train_frontier_tokens.txt"
V22_TRAIN_EDGES="$V22_OUT_ROOT/provenance/train_v20_frozen_with_transition_edges.jsonl"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/configs"
TIMING="$OUT_ROOT/provenance/v64_3_23_stage_timing.tsv"; : > "$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ local now; now=$(date +%s); printf '%s\t%s\n' "$STAGE_NAME" "$((now-STAGE_START))" >> "$TIMING"; }

[[ -d "$BDSE_TRAIN_CACHE" ]] || { echo "STOP: missing train cache $BDSE_TRAIN_CACHE" >&2; exit 2; }
[[ -d "$BDSE_VAL_CACHE" ]] || { echo "STOP: missing val cache $BDSE_VAL_CACHE" >&2; exit 2; }
[[ -s "$DESIGN_EXCLUDE_TOKENS" ]] || { echo "STOP: missing V23 design exclusion" >&2; exit 2; }
[[ "$(wc -l < "$DESIGN_EXCLUDE_TOKENS")" -eq 4700 ]] || { echo "STOP DATA: V23 exclusion must remain exactly 4700 inspected validation tokens because V22 never selected fresh validation" >&2; exit 2; }
for f in "$RAW_CONFIG" "$V20_DUAL_CONFIG"; do [[ -s "$f" ]] || { echo "STOP: missing config $f" >&2; exit 2; }; done

stage_start prerequisites
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_23 --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
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
pytest -q bdse/tests/test_v64_3_23_eaf_icer_rcr.py bdse/tests/test_v64_3_22_eaf_icer_tcr.py bdse/tests/test_v64_3_21_eaf_icer_mcr.py bdse/tests/test_v64_3_20_eaf_icer_dc.py bdse/tests/test_v64_3_19_eaf_icer.py bdse/tests/test_v64_3_18_eaf_dacer.py bdse/tests/test_v64_3_17_eaf_daler.py bdse/tests/test_v64_3_16_eaf_raer.py bdse/tests/test_v64_3_15_eaf_eair.py bdse/tests/test_v64_3_14_eaf_ocfi.py bdse/tests/test_v64_3_13_eaf_dmvr.py bdse/tests/test_v64_3_12_cet_bdmu.py bdse/tests/test_v64_3_11_btp_bdmu.py bdse/tests/test_v64_3_10_hap_bdmu.py bdse/tests/test_v64_3_9_af_bdmu.py bdse/tests/test_v64_3_8_bdmu.py bdse/tests/test_v64_3_7_darm_dbr.py bdse/tests/test_v64_3_6_bcha_lbpr.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

# Reuse V22's completed 3000-scene TRAIN transition frontier whenever possible.
# V22 stopped after this replay, so the artifact is TRAIN-only and safe to reuse.
stage_start train_frontier_reuse_or_replay
if [[ -s "$V22_TRAIN_EDGES" ]]; then
  ln -sf "$(python - "$V22_TRAIN_EDGES" <<'PY'
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
)" "$TRAIN_EDGES"
  echo "REUSED_V22_TRAIN_FRONTIER=$V22_TRAIN_EDGES" > "$OUT_ROOT/logs/train_frontier_reuse.out"
else
  CUDA_VISIBLE_DEVICES="$GPU0" python -m bdse.experiments.evaluate_open_loop \
    --config "$V20_DUAL_CONFIG" --checkpoint "$EAF_CKPT" --split train --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    --max-scenarios 3000 --output "$OUT_ROOT/provenance/train_v20_transition_metrics.json" \
    --frontier-edge-output "$TRAIN_EDGES" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/train_transition_frontier.out" 2>&1
fi
[[ -s "$TRAIN_EDGES" ]] || { echo "STOP TRAIN: missing transition frontier edges" >&2; exit 2; }
stage_end

stage_start train_only_rcr_fit
python -m bdse.tools.fit_v64_3_23_eaf_icer_rcr \
  --train-frontier-edges "$TRAIN_EDGES" --base-v20-dual-config "$V20_DUAL_CONFIG" \
  --output-evidence-memory "$EVIDENCE_MEMORY" --output-transition-memory "$TRANSITION_MEMORY" \
  --output-evidence-scalar-config "$EVIDENCE_SCALAR_CONFIG" --output-evidence-rcr-config "$EVIDENCE_RCR_CONFIG" --output-transition-rcr-config "$TRANS_RCR_CONFIG" \
  --output-train-token-file "$TRAIN_TOKENS" --output-report "$OUT_ROOT/provenance/v64_3_23_train_only_rcr_fit.json" 2>&1 | tee "$OUT_ROOT/logs/v64_3_23_rcr_fit.out"
python -m bdse.tools.check_v64_3_23_eaf_icer_rcr_contract --config "$EVIDENCE_SCALAR_CONFIG" --expect evidence-local-scalar --frozen-v20-dual-config "$V20_DUAL_CONFIG" --output "$OUT_ROOT/provenance/evidence_scalar_contract.json"
python -m bdse.tools.check_v64_3_23_eaf_icer_rcr_contract --config "$EVIDENCE_RCR_CONFIG" --expect evidence-local-rcr --frozen-v20-dual-config "$V20_DUAL_CONFIG" --output "$OUT_ROOT/provenance/evidence_rcr_contract.json"
python -m bdse.tools.check_v64_3_23_eaf_icer_rcr_contract --config "$TRANS_RCR_CONFIG" --expect transition-local-rcr --frozen-v20-dual-config "$V20_DUAL_CONFIG" --output "$OUT_ROOT/provenance/transition_rcr_ablation_contract.json"
stage_end

stage_start fresh_token_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh_1000_tokens.txt"; TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"; TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$DESIGN_EXCLUDE_TOKENS" --count 1000 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" --audit-output "$OUT_ROOT/provenance/v64_3_23_fresh_1000_audit.json" > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1000" > "$TOKA"; tail -n 500 "$TOK1000" > "$TOKB"
python - "$TOK1000" "$TOKA" "$TOKB" <<'PY'
import sys
all=[x.strip() for x in open(sys.argv[1]) if x.strip()]; A=[x.strip() for x in open(sys.argv[2]) if x.strip()]; B=[x.strip() for x in open(sys.argv[3]) if x.strip()]
assert len(all)==len(set(all))==1000 and len(A)==len(set(A))==500 and len(B)==len(set(B))==500 and not set(A)&set(B) and set(A)|set(B)==set(all)
print('PASS V23 fresh identity: 1000 unique = independent A500 + B500')
PY
python - "$TRAIN_TOKENS" "$TOK1000" <<'PY'
import sys
train={x.strip() for x in open(sys.argv[1]) if x.strip()}
fresh={x.strip() for x in open(sys.argv[2]) if x.strip()}
overlap=train & fresh
if overlap:
    raise SystemExit(f'STOP DATA LEAKAGE: {len(overlap)} fresh validation tokens overlap TRAIN local-memory frontier; examples={sorted(overlap)[:5]}')
print(f'PASS TRAIN/fresh identity isolation: train={len(train)} fresh={len(fresh)} overlap=0')
PY
stage_end

run_eval(){
  local gpu="$1" cfg="$2" split="$3" tag="$4"; local tok="$TOKA"; [[ "$split" == B ]] && tok="$TOKB"
  CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios 500 --scenario-token-file "$tok" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/${split}_${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/${split}_${tag}_rows.jsonl" --frontier-edge-output "$OUT_ROOT/provenance/${split}_${tag}_edges.jsonl" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${split}_${tag}.out" 2>&1
}

# 5 causal arms x 2 independent blocks. No pooled rescue.
stage_start fresh_wave_1; run_eval "$GPU0" "$RAW_CONFIG" A raw & P0=$!; run_eval "$GPU1" "$V20_DUAL_CONFIG" A v20 & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_2; run_eval "$GPU0" "$EVIDENCE_SCALAR_CONFIG" A evidence_scalar & P0=$!; run_eval "$GPU1" "$EVIDENCE_RCR_CONFIG" A evidence_rcr & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_3; run_eval "$GPU0" "$TRANS_RCR_CONFIG" A transition_rcr & P0=$!; run_eval "$GPU1" "$RAW_CONFIG" B raw & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_4; run_eval "$GPU0" "$V20_DUAL_CONFIG" B v20 & P0=$!; run_eval "$GPU1" "$EVIDENCE_SCALAR_CONFIG" B evidence_scalar & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_5; run_eval "$GPU0" "$EVIDENCE_RCR_CONFIG" B evidence_rcr & P0=$!; run_eval "$GPU1" "$TRANS_RCR_CONFIG" B transition_rcr & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1])
for split,tokfile in [('A',sys.argv[2]),('B',sys.argv[3])]:
    want={x.strip() for x in open(tokfile) if x.strip()}
    for tag in ['raw','v20','evidence_scalar','evidence_rcr','transition_rcr']:
        rows=[json.loads(x) for x in open(root/f'{split}_{tag}_rows.jsonl') if x.strip()]; got={str(r['scenario_token']) for r in rows}
        if len(rows)!=500 or got!=want: raise SystemExit(f'STOP DATA: {split}/{tag} token mismatch')
        m=json.load(open(root/f'{split}_{tag}_metrics.json'))
        if not m.get('scenario_token_prefilter_active',False): raise SystemExit(f'STOP SPEED/DATA: {split}/{tag} pre-load token filter inactive')
print('PASS paired identity + pre-deserialization filtering: 10/10 arms')
PY
stage_end

stage_start double_fresh_screen
for split in A B; do
python -m bdse.tools.check_v64_3_23_eaf_icer_rcr_split --split-name "$split" \
  --raw-metrics "$OUT_ROOT/provenance/${split}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${split}_raw_rows.jsonl" \
  --v20-metrics "$OUT_ROOT/provenance/${split}_v20_metrics.json" --v20-rows "$OUT_ROOT/provenance/${split}_v20_rows.jsonl" --v20-edges "$OUT_ROOT/provenance/${split}_v20_edges.jsonl" \
  --evidence-scalar-metrics "$OUT_ROOT/provenance/${split}_evidence_scalar_metrics.json" --evidence-scalar-rows "$OUT_ROOT/provenance/${split}_evidence_scalar_rows.jsonl" --evidence-scalar-edges "$OUT_ROOT/provenance/${split}_evidence_scalar_edges.jsonl" \
  --evidence-rcr-metrics "$OUT_ROOT/provenance/${split}_evidence_rcr_metrics.json" --evidence-rcr-rows "$OUT_ROOT/provenance/${split}_evidence_rcr_rows.jsonl" --evidence-rcr-edges "$OUT_ROOT/provenance/${split}_evidence_rcr_edges.jsonl" \
  --transition-rcr-metrics "$OUT_ROOT/provenance/${split}_transition_rcr_metrics.json" --transition-rcr-rows "$OUT_ROOT/provenance/${split}_transition_rcr_rows.jsonl" --transition-rcr-edges "$OUT_ROOT/provenance/${split}_transition_rcr_edges.jsonl" \
  --output "$OUT_ROOT/provenance/v64_3_23_split_${split}_screen.json" | tee "$OUT_ROOT/logs/v64_3_23_split_${split}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_23_eaf_icer_rcr_double_fresh_screen.json"
python -m bdse.tools.check_v64_3_23_eaf_icer_rcr_screen --split-a-report "$OUT_ROOT/provenance/v64_3_23_split_A_screen.json" --split-b-report "$OUT_ROOT/provenance/v64_3_23_split_B_screen.json" --train-fit-report "$OUT_ROOT/provenance/v64_3_23_train_only_rcr_fit.json" --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_23_double_fresh_screen.out"
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"

python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print('DOUBLE_FRESH_PASS=',r.get('both_independent_500_scene_blocks_pass')); print('TRANSITION_INCREMENTAL_BOTH=',r.get('transition_conditioning_incremental_both_diagnostic')); print('SIGNED_INCREMENTAL_BOTH=',r.get('signed_profile_ranking_incremental_both')); print('NEXT_ACTION=',r.get('next_action'))
if not r.get('full_promotion_to_independent_full_val_reproduction',False): raise SystemExit('STOP SCREEN: do not run full/test/closed-loop; follow next_action')
print('PASS DOUBLE-FRESH SCREEN ONLY. Next allowed stage: one frozen independent full-val reproduction. Test/closed-loop remain forbidden.')
PY
