#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

# V64.3.21 EAF-ICER-MCR double-fresh causal replication.
# Case-E follow-up only: candidate semantics, B/M, acquisition, EAF value,
# final certificate/guards, V19 support head and V19 scalar/profile dominance heads
# are frozen. New TRAIN-only learning is restricted to the raw selected incumbent:
# predict J_T(anchor)-J_T(incumbent) with fixed-zero MSE margin semantics.
# The main extremal operator additionally requires scalar AND signed-profile
# dominance views to be positive before equal-mean ranking. No threshold/weight sweep.
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export V18_SCREEN_ROOT="${V18_SCREEN_ROOT:-outputs_v64_3_18_eaf_dacer_screen_2gpu_v1}"
export TRAIN_EDGES="${TRAIN_EDGES:-$V18_SCREEN_ROOT/provenance/train_raw_eaf_frontier_edges.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_21_eaf_icer_mcr_screen_2gpu_v1}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_21_design_exclude_v64_3_20_screen_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.21-eaf-icer-mcr-double-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"
RAW_CONFIG="bdse/configs/v64_3_20_eaf_icer_dc_raw.yaml"
V20_CONFIG="bdse/configs/v64_3_20_icer_dc_dual.yaml"
SCALAR_RET_CONFIG="$OUT_ROOT/configs/v64_3_21_mcr_scalar_retention_mean.yaml"
PROFILE_MEAN_CONFIG="$OUT_ROOT/configs/v64_3_21_mcr_profile_retention_mean.yaml"
CONSENSUS_CONFIG="$OUT_ROOT/configs/v64_3_21_mcr_profile_retention_consensus.yaml"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/configs"
TIMING="$OUT_ROOT/provenance/v64_3_21_stage_timing.tsv"; : > "$TIMING"; ALL_START=$(date +%s)
stage_start(){ STAGE_NAME="$1"; STAGE_START=$(date +%s); }
stage_end(){ local now; now=$(date +%s); printf '%s\t%s\n' "$STAGE_NAME" "$((now-STAGE_START))" >> "$TIMING"; }

[[ -d "$BDSE_VAL_CACHE" ]] || { echo "STOP: missing val cache $BDSE_VAL_CACHE" >&2; exit 2; }
[[ -s "$TRAIN_EDGES" ]] || { echo "STOP: missing frozen V18 TRAIN frontier $TRAIN_EDGES" >&2; exit 2; }
[[ -s "$DESIGN_EXCLUDE_TOKENS" ]] || { echo "STOP: missing V21 design exclusion" >&2; exit 2; }
[[ "$(wc -l < "$DESIGN_EXCLUDE_TOKENS")" -eq 3700 ]] || { echo "STOP DATA: V21 exclusion must be exactly 3700 inspected validation tokens" >&2; exit 2; }
for f in "$RAW_CONFIG" "$V20_CONFIG"; do [[ -s "$f" ]] || { echo "STOP: missing config $f" >&2; exit 2; }; done

stage_start prerequisites
python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_21 --output "$OUT_ROOT/provenance/v64_3_13_eaf_dmvr_reaudit.json" > "$OUT_ROOT/logs/v64_3_13_eaf_dmvr_reaudit.out"
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
pytest -q bdse/tests/test_v64_3_21_eaf_icer_mcr.py bdse/tests/test_v64_3_20_eaf_icer_dc.py bdse/tests/test_v64_3_19_eaf_icer.py bdse/tests/test_v64_3_18_eaf_dacer.py bdse/tests/test_v64_3_17_eaf_daler.py bdse/tests/test_v64_3_16_eaf_raer.py bdse/tests/test_v64_3_15_eaf_eair.py bdse/tests/test_v64_3_14_eaf_ocfi.py bdse/tests/test_v64_3_13_eaf_dmvr.py bdse/tests/test_v64_3_12_cet_bdmu.py bdse/tests/test_v64_3_11_btp_bdmu.py bdse/tests/test_v64_3_10_hap_bdmu.py bdse/tests/test_v64_3_9_af_bdmu.py bdse/tests/test_v64_3_8_bdmu.py bdse/tests/test_v64_3_7_darm_dbr.py bdse/tests/test_v64_3_6_bcha_lbpr.py | tee "$OUT_ROOT/logs/targeted_regression.out"
stage_end

stage_start train_only_retention_fit
python -m bdse.tools.fit_v64_3_21_eaf_icer_mcr \
  --train-frontier-edges "$TRAIN_EDGES" --base-v20-dual-config "$V20_CONFIG" \
  --output-scalar-retention-config "$SCALAR_RET_CONFIG" --output-mean-config "$PROFILE_MEAN_CONFIG" --output-consensus-config "$CONSENSUS_CONFIG" \
  --output-report "$OUT_ROOT/provenance/v64_3_21_train_only_retention_fit.json" | tee "$OUT_ROOT/logs/v64_3_21_retention_fit.out"
python -m bdse.tools.check_v64_3_21_eaf_icer_mcr_contract --config "$SCALAR_RET_CONFIG" --expect mcr-scalar-retention --frozen-v20-dual-config "$V20_CONFIG" --output "$OUT_ROOT/provenance/v64_3_21_scalar_retention_contract.json"
python -m bdse.tools.check_v64_3_21_eaf_icer_mcr_contract --config "$PROFILE_MEAN_CONFIG" --expect mcr-mean --frozen-v20-dual-config "$V20_CONFIG" --output "$OUT_ROOT/provenance/v64_3_21_profile_mean_contract.json"
python -m bdse.tools.check_v64_3_21_eaf_icer_mcr_contract --config "$CONSENSUS_CONFIG" --expect mcr-consensus --frozen-v20-dual-config "$V20_CONFIG" --output "$OUT_ROOT/provenance/v64_3_21_consensus_contract.json"
stage_end

stage_start fresh_token_selection
TOK1000="$OUT_ROOT/provenance/val_screen_fresh_1000_tokens.txt"; TOKA="$OUT_ROOT/provenance/val_screen_fresh_A_tokens.txt"; TOKB="$OUT_ROOT/provenance/val_screen_fresh_B_tokens.txt"
python -m bdse.tools.select_fresh_preprocessed_tokens --preprocessed-dir "$BDSE_VAL_CACHE" --split val --exclude-tokens "$DESIGN_EXCLUDE_TOKENS" --count 1000 --hash-seed "$FRESH_HASH_SEED" --output "$TOK1000" --audit-output "$OUT_ROOT/provenance/v64_3_21_fresh_1000_audit.json" > "$OUT_ROOT/logs/fresh_token_selection.out"
head -n 500 "$TOK1000" > "$TOKA"; tail -n 500 "$TOK1000" > "$TOKB"
python - "$TOK1000" "$TOKA" "$TOKB" <<'PY'
import sys
all=[x.strip() for x in open(sys.argv[1]) if x.strip()]; A=[x.strip() for x in open(sys.argv[2]) if x.strip()]; B=[x.strip() for x in open(sys.argv[3]) if x.strip()]
assert len(all)==len(set(all))==1000 and len(A)==len(set(A))==500 and len(B)==len(set(B))==500 and not set(A)&set(B) and set(A)|set(B)==set(all)
print('PASS fresh double-block identity: 1000 unique = 500 A + 500 B, overlap 0')
PY
stage_end

run_eval(){
  local gpu="$1" cfg="$2" split="$3" tag="$4"; local tok="$TOKA"; [[ "$split" == B ]] && tok="$TOKB"
  CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" --max-scenarios 500 --scenario-token-file "$tok" --require-all-scenario-tokens --output "$OUT_ROOT/provenance/${split}_${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/${split}_${tag}_rows.jsonl" --frontier-edge-output "$OUT_ROOT/provenance/${split}_${tag}_edges.jsonl" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/${split}_${tag}.out" 2>&1
}
run_pair(){ local stage="$1"; shift; stage_start "$stage"; "$@"; stage_end; }

# Five two-GPU waves = 10 strict paired evaluations, no pooled rescue.
stage_start fresh_wave_1; run_eval "$GPU0" "$RAW_CONFIG" A raw & P0=$!; run_eval "$GPU1" "$V20_CONFIG" A v20 & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_2; run_eval "$GPU0" "$SCALAR_RET_CONFIG" A scalar_retention & P0=$!; run_eval "$GPU1" "$PROFILE_MEAN_CONFIG" A profile_mean & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_3; run_eval "$GPU0" "$CONSENSUS_CONFIG" A consensus & P0=$!; run_eval "$GPU1" "$RAW_CONFIG" B raw & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_4; run_eval "$GPU0" "$V20_CONFIG" B v20 & P0=$!; run_eval "$GPU1" "$SCALAR_RET_CONFIG" B scalar_retention & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end
stage_start fresh_wave_5; run_eval "$GPU0" "$PROFILE_MEAN_CONFIG" B profile_mean & P0=$!; run_eval "$GPU1" "$CONSENSUS_CONFIG" B consensus & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ $S -eq 0 ]]||exit 2; stage_end

stage_start paired_identity
python - "$OUT_ROOT/provenance" "$TOKA" "$TOKB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1])
for split,tokfile in [('A',sys.argv[2]),('B',sys.argv[3])]:
    want=[x.strip() for x in open(tokfile) if x.strip()]
    for tag in ['raw','v20','scalar_retention','profile_mean','consensus']:
        rows=[json.loads(x) for x in open(root/f'{split}_{tag}_rows.jsonl') if x.strip()]; got=[str(r['scenario_token']) for r in rows]
        if len(got)!=500 or set(got)!=set(want): raise SystemExit(f'STOP DATA: {split}/{tag} token mismatch')
        m=json.load(open(root/f'{split}_{tag}_metrics.json'))
        if not m.get('scenario_token_prefilter_active',False): raise SystemExit(f'STOP SPEED: {split}/{tag} pre-load token filter inactive')
print('PASS paired token identity + pre-deserialization filtering: 10/10 arms')
PY
stage_end

stage_start double_fresh_screen
for split in A B; do
python -m bdse.tools.check_v64_3_21_eaf_icer_mcr_split --split-name "$split" \
  --raw-metrics "$OUT_ROOT/provenance/${split}_raw_metrics.json" --raw-rows "$OUT_ROOT/provenance/${split}_raw_rows.jsonl" \
  --v20-metrics "$OUT_ROOT/provenance/${split}_v20_metrics.json" --v20-rows "$OUT_ROOT/provenance/${split}_v20_rows.jsonl" --v20-edges "$OUT_ROOT/provenance/${split}_v20_edges.jsonl" \
  --scalar-retention-metrics "$OUT_ROOT/provenance/${split}_scalar_retention_metrics.json" --scalar-retention-rows "$OUT_ROOT/provenance/${split}_scalar_retention_rows.jsonl" --scalar-retention-edges "$OUT_ROOT/provenance/${split}_scalar_retention_edges.jsonl" \
  --profile-mean-metrics "$OUT_ROOT/provenance/${split}_profile_mean_metrics.json" --profile-mean-rows "$OUT_ROOT/provenance/${split}_profile_mean_rows.jsonl" --profile-mean-edges "$OUT_ROOT/provenance/${split}_profile_mean_edges.jsonl" \
  --consensus-metrics "$OUT_ROOT/provenance/${split}_consensus_metrics.json" --consensus-rows "$OUT_ROOT/provenance/${split}_consensus_rows.jsonl" --consensus-edges "$OUT_ROOT/provenance/${split}_consensus_edges.jsonl" \
  --output "$OUT_ROOT/provenance/v64_3_21_split_${split}_screen.json" | tee "$OUT_ROOT/logs/v64_3_21_split_${split}_screen.out"
done
SCREEN="$OUT_ROOT/provenance/v64_3_21_eaf_icer_mcr_double_fresh_screen.json"
python -m bdse.tools.check_v64_3_21_eaf_icer_mcr_screen --split-a-report "$OUT_ROOT/provenance/v64_3_21_split_A_screen.json" --split-b-report "$OUT_ROOT/provenance/v64_3_21_split_B_screen.json" --output "$SCREEN" | tee "$OUT_ROOT/logs/v64_3_21_double_fresh_screen.out"
stage_end
printf 'TOTAL\t%s\n' "$(( $(date +%s)-ALL_START ))" >> "$TIMING"

python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print('DOUBLE_FRESH_PASS=',r.get('both_independent_500_scene_blocks_pass')); print('NEXT_ACTION=',r.get('next_action'))
if not r.get('full_promotion_to_independent_full_val_reproduction',False): raise SystemExit('STOP SCREEN: do not run full/test/closed-loop; follow next_action')
print('PASS DOUBLE-FRESH SCREEN ONLY. Next allowed stage: one frozen independent full-val reproduction. Test/closed-loop remain forbidden until that reproduction passes.')
PY
