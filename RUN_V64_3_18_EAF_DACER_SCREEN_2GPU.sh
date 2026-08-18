#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

# V64.3.18 EAF-DACER causal screen.
# Frozen: B<=16/M=24 planner evidence interface, acquisition, V64.3.13 EAF value,
# DARM/DBR, legacy utility refinement, final one-sided/evidence certificate and
# structural-risk guard. New learned operator only competes over the FINAL-GUARD-
# ADMISSIBLE challenger frontier; legacy utility-pool membership is diagnostic / exact-tie-break context, not a learned feature or gate.
# This launcher is screen-only: no test and no closed-loop execution.
export BDSE_TRAIN_CACHE="${BDSE_TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
export BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
export EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
export EAF_TRAIN_LOG="${EAF_TRAIN_LOG:-$EAF_V64_3_13_ROOT/train/bdse_v64_saqa_bcc.train_log.jsonl}"
export OUT_ROOT="${OUT_ROOT:-outputs_v64_3_18_eaf_dacer_screen_2gpu_v1}"
export TRAIN_SCENARIOS="${TRAIN_SCENARIOS:-3000}"
export VAL_DISCOVERY_SCENARIOS="${VAL_DISCOVERY_SCENARIOS:-4000}"
export VAL_SCREEN_SCENARIOS="${VAL_SCREEN_SCENARIOS:-500}"
export DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_18_design_exclude_v64_3_17_screen_tokens.txt}"
export FRESH_HASH_SEED="${FRESH_HASH_SEED:-v64.3.18-eaf-dacer-fresh-v1}"
export GPU0="${GPU0:-0}"; export GPU1="${GPU1:-1}"
export RAW_CONFIG="bdse/configs/v64_3_18_eaf_dacer_raw.yaml"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs" "$OUT_ROOT/configs"

[[ -d "$BDSE_TRAIN_CACHE" ]] || { echo "STOP: missing train cache $BDSE_TRAIN_CACHE" >&2; exit 2; }
[[ -d "$BDSE_VAL_CACHE" ]] || { echo "STOP: missing val cache $BDSE_VAL_CACHE" >&2; exit 2; }
[[ -s "$EAF_TRAIN_LOG" ]] || { echo "STOP: missing V64.3.13 train log $EAF_TRAIN_LOG" >&2; exit 2; }
[[ -s "$DESIGN_EXCLUDE_TOKENS" ]] || { echo "STOP: missing V64.3.18 design exclusion list" >&2; exit 2; }
[[ "$(wc -l < "$DESIGN_EXCLUDE_TOKENS")" -ge 2200 ]] || { echo "STOP DATA: V64.3.18 must exclude all 1700 earlier design scenes plus all 500 inspected V64.3.17 fresh-screen scenes." >&2; exit 2; }

python -m bdse.tools.check_v64_3_13_eaf_dmvr_screen \
  --train-log "$EAF_TRAIN_LOG" --variant REAUDIT_FOR_V64_3_18 \
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
python -m bdse.tools.check_v64_3_18_eaf_dacer_contract --config "$RAW_CONFIG" --expect raw --output "$OUT_ROOT/provenance/v64_3_18_dacer_raw_contract.json"
pytest -q \
  bdse/tests/test_v64_3_18_eaf_dacer.py bdse/tests/test_v64_3_17_eaf_daler.py bdse/tests/test_v64_3_16_eaf_raer.py \
  bdse/tests/test_v64_3_15_eaf_eair.py bdse/tests/test_v64_3_14_eaf_ocfi.py bdse/tests/test_v64_3_13_eaf_dmvr.py \
  bdse/tests/test_v64_3_12_cet_bdmu.py bdse/tests/test_v64_3_11_btp_bdmu.py bdse/tests/test_v64_3_10_hap_bdmu.py \
  bdse/tests/test_v64_3_9_af_bdmu.py bdse/tests/test_v64_3_8_bdmu.py bdse/tests/test_v64_3_7_darm_dbr.py bdse/tests/test_v64_3_6_bcha_lbpr.py \
  | tee "$OUT_ROOT/logs/targeted_regression.out"

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

python - "$TRAIN_EDGES" <<'PY'
import json,math,sys,collections
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
if len(rows)<2048: raise SystemExit(f'STOP CAPACITY: only {len(rows)} TRAIN frontier edges')
keys=sorted(k for k in rows[0] if k.startswith('dacer_feature_'))
if len(keys)!=42: raise SystemExit(f'STOP ENGINEERING: expected 42 DACER features, found {len(keys)}')
for k in keys:
    cov=sum(math.isfinite(float(r.get(k,float('nan')))) for r in rows)/len(rows)
    if cov<.99: raise SystemExit(f'STOP ENGINEERING: {k} coverage={cov:.4f}')
adm=[r for r in rows if float(r.get('dacer_admissible',0))>=.5]
g=collections.defaultdict(int)
for r in adm:g[str(r.get('scenario_token',''))]+=1
multi=sum(v>=2 for v in g.values())
if len(adm)<8192 or multi<512: raise SystemExit(f'STOP CAPACITY: guard-admissible TRAIN support edges={len(adm)}, multi_scenes={multi}')
# V64.3.17 pathology is audited, not repeated: utility prior may be tiny but must NOT define dacer_admissible.
util=sum(float(r.get('dacer_utility_prior',0))>=.5 for r in rows)
print(f'PASS DACER instrumentation: all_edges={len(rows)}, admissible_edges={len(adm)}, multi_scenes={multi}, utility_prior_edges={util}, features={len(keys)}')
PY

# Fresh validation uses identity + fixed hash only. All 2200 previously inspected
# design scenes are excluded before hash ranking. No labels/metrics participate.
VAL_TOKENS="$OUT_ROOT/provenance/val_screen_fresh_tokens.txt"
SPLIT_AUDIT="$OUT_ROOT/provenance/v64_3_18_fresh_split_audit.json"
python - "$VAL_DISCOVERY_ROWS" "$DESIGN_EXCLUDE_TOKENS" "$VAL_TOKENS" "$SPLIT_AUDIT" "$VAL_SCREEN_SCENARIOS" "$FRESH_HASH_SEED" <<'PY'
import hashlib,json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
exclude={x.strip() for x in open(sys.argv[2]) if x.strip()}; need=int(sys.argv[5]); seed=sys.argv[6]
disc={str(r['scenario_token']) for r in rows}; eligible=disc-exclude
rank=lambda t: hashlib.sha256((seed+'::'+t).encode()).hexdigest()
out=sorted(eligible,key=lambda t:(rank(t),t))[:need]
if len(out)<need: raise SystemExit(f'STOP DATA SPLIT: only {len(out)} clean scenes; increase VAL_DISCOVERY_SCENARIOS')
open(sys.argv[3],'w').write('\n'.join(out)+'\n')
a={'audit':'v64_3_18_fresh_split','discovery_unique':len(disc),'design_exclusion_count':len(exclude),'excluded_in_discovery':len(disc&exclude),'eligible_unique':len(eligible),'fresh_count':len(out),'fresh_overlap_design':len(set(out)&exclude),'hash_seed':seed,'selection_uses_labels':False}
open(sys.argv[4],'w').write(json.dumps(a,indent=2,sort_keys=True)); print(json.dumps(a,indent=2))
PY

# TRAIN-only fits. No validation threshold/objective/B/M/certificate tuning.
RAER_FIT="$OUT_ROOT/provenance/v64_3_18_raer_control_fit.json"; RAER_CFG="$OUT_ROOT/configs/v64_3_18_raer_control.yaml"
python -m bdse.tools.fit_v64_3_16_eaf_raer --train-frontier-edges "$TRAIN_EDGES" --base-config "$RAW_CONFIG" --output-config "$RAER_CFG" --output-report "$RAER_FIT" > "$OUT_ROOT/logs/raer_control_fit.out"
DALER_FIT="$OUT_ROOT/provenance/v64_3_18_v17_daler_control_fit.json"; DALER_CFG="$OUT_ROOT/configs/v64_3_18_v17_daler_control.yaml"
python -m bdse.tools.fit_v64_3_17_eaf_daler --train-frontier-edges "$TRAIN_EDGES" --base-config "$RAW_CONFIG" --output-config "$DALER_CFG" --output-report "$DALER_FIT" > "$OUT_ROOT/logs/v17_daler_control_fit.out"
GDALER_FIT="$OUT_ROOT/provenance/v64_3_18_guard_listwise_gdaler_fit.json"; GDALER_CFG="$OUT_ROOT/configs/v64_3_18_guard_listwise_gdaler.yaml"
python -m bdse.tools.fit_v64_3_18_eaf_dacer --train-frontier-edges "$TRAIN_EDGES" --base-config "$RAW_CONFIG" --feature-mode scalar --objective-mode listwise --output-config "$GDALER_CFG" --output-report "$GDALER_FIT" > "$OUT_ROOT/logs/guard_listwise_gdaler_fit.out"
DS_FIT="$OUT_ROOT/provenance/v64_3_18_dacer_scalar_fit.json"; DS_CFG="$OUT_ROOT/configs/v64_3_18_dacer_scalar.yaml"
python -m bdse.tools.fit_v64_3_18_eaf_dacer --train-frontier-edges "$TRAIN_EDGES" --base-config "$RAW_CONFIG" --feature-mode scalar --objective-mode counterfactual --output-config "$DS_CFG" --output-report "$DS_FIT" > "$OUT_ROOT/logs/dacer_scalar_fit.out"
DP_FIT="$OUT_ROOT/provenance/v64_3_18_dacer_profile_fit.json"; DP_CFG="$OUT_ROOT/configs/v64_3_18_dacer_profile.yaml"
python -m bdse.tools.fit_v64_3_18_eaf_dacer --train-frontier-edges "$TRAIN_EDGES" --base-config "$RAW_CONFIG" --feature-mode profile --objective-mode counterfactual --output-config "$DP_CFG" --output-report "$DP_FIT" > "$OUT_ROOT/logs/dacer_profile_fit.out"
python -m bdse.tools.check_v64_3_18_eaf_dacer_contract --config "$GDALER_CFG" --expect gdaler --output "$OUT_ROOT/provenance/v64_3_18_gdaler_contract.json"
python -m bdse.tools.check_v64_3_18_eaf_dacer_contract --config "$DS_CFG" --expect dacer-scalar --output "$OUT_ROOT/provenance/v64_3_18_dacer_scalar_contract.json"
python -m bdse.tools.check_v64_3_18_eaf_dacer_contract --config "$DP_CFG" --expect dacer-profile --output "$OUT_ROOT/provenance/v64_3_18_dacer_profile_contract.json"

run_eval() {
  local gpu="$1" cfg="$2" tag="$3" edge="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -m bdse.experiments.evaluate_open_loop \
    --config "$cfg" --checkpoint "$EAF_CKPT" --split val --preprocessed-dir "$BDSE_VAL_CACHE" \
    --max-scenarios "$VAL_SCREEN_SCENARIOS" --scenario-token-file "$VAL_TOKENS" --require-all-scenario-tokens \
    --output "$OUT_ROOT/provenance/val_fresh_${tag}_metrics.json" --per-sample-output "$OUT_ROOT/provenance/val_fresh_${tag}_rows.jsonl" \
    --frontier-edge-output "$edge" --disable-dense-diagnostic --device cuda > "$OUT_ROOT/logs/val_fresh_${tag}.out" 2>&1
}

# Three paired waves on the identical untouched 500-scene token set.
RAW_E="$OUT_ROOT/provenance/val_fresh_raw_edges.jsonl"; RAER_E="$OUT_ROOT/provenance/val_fresh_raer_edges.jsonl"
run_eval "$GPU0" "$RAW_CONFIG" raw "$RAW_E" & P0=$!; run_eval "$GPU1" "$RAER_CFG" raer "$RAER_E" & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ "$S" -eq 0 ]] || { echo 'STOP ENGINEERING: raw/RAER fresh replay failed' >&2; exit 2; }
DALER_E="$OUT_ROOT/provenance/val_fresh_v17_daler_edges.jsonl"; GDALER_E="$OUT_ROOT/provenance/val_fresh_gdaler_edges.jsonl"
run_eval "$GPU0" "$DALER_CFG" v17_daler "$DALER_E" & P0=$!; run_eval "$GPU1" "$GDALER_CFG" gdaler "$GDALER_E" & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ "$S" -eq 0 ]] || { echo 'STOP ENGINEERING: DALER/G-DALER fresh replay failed' >&2; exit 2; }
DS_E="$OUT_ROOT/provenance/val_fresh_dacer_scalar_edges.jsonl"; DP_E="$OUT_ROOT/provenance/val_fresh_dacer_profile_edges.jsonl"
run_eval "$GPU0" "$DS_CFG" dacer_scalar "$DS_E" & P0=$!; run_eval "$GPU1" "$DP_CFG" dacer_profile "$DP_E" & P1=$!; S=0; wait "$P0"||S=1; wait "$P1"||S=1; [[ "$S" -eq 0 ]] || { echo 'STOP ENGINEERING: DACER scalar/profile fresh replay failed' >&2; exit 2; }

# Exact same token identity in all six arms is mandatory.
python - "$VAL_TOKENS" "$OUT_ROOT/provenance" <<'PY'
import json,sys,pathlib
want=[x.strip() for x in open(sys.argv[1]) if x.strip()]; root=pathlib.Path(sys.argv[2])
for tag in ['raw','raer','v17_daler','gdaler','dacer_scalar','dacer_profile']:
    got=[str(json.loads(x)['scenario_token']) for x in open(root/f'val_fresh_{tag}_rows.jsonl') if x.strip()]
    if len(got)!=len(want) or set(got)!=set(want): raise SystemExit(f'STOP DATA: {tag} replay token mismatch {len(got)}/{len(want)}')
print('PASS paired fresh token identity: 6/6 arms')
PY

SCREEN="$OUT_ROOT/provenance/v64_3_18_eaf_dacer_screen.json"
python -m bdse.tools.check_v64_3_18_eaf_dacer_screen \
  --raw-metrics "$OUT_ROOT/provenance/val_fresh_raw_metrics.json" \
  --raer-metrics "$OUT_ROOT/provenance/val_fresh_raer_metrics.json" \
  --daler-metrics "$OUT_ROOT/provenance/val_fresh_v17_daler_metrics.json" \
  --gdaler-metrics "$OUT_ROOT/provenance/val_fresh_gdaler_metrics.json" \
  --dacer-scalar-metrics "$OUT_ROOT/provenance/val_fresh_dacer_scalar_metrics.json" \
  --dacer-profile-metrics "$OUT_ROOT/provenance/val_fresh_dacer_profile_metrics.json" \
  --raer-edge-output "$RAER_E" --daler-edge-output "$DALER_E" --gdaler-edge-output "$GDALER_E" \
  --dacer-scalar-edge-output "$DS_E" --dacer-profile-edge-output "$DP_E" \
  --raer-fit-report "$RAER_FIT" --daler-fit-report "$DALER_FIT" --gdaler-fit-report "$GDALER_FIT" \
  --dacer-scalar-fit-report "$DS_FIT" --dacer-profile-fit-report "$DP_FIT" --output "$SCREEN" \
  | tee "$OUT_ROOT/logs/v64_3_18_screen.out"

python - "$SCREEN" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print('FULL_PROMOTION=',r.get('full_promotion')); print('NEXT_ACTION=',r.get('next_action'))
if not r.get('full_promotion',False):
    raise SystemExit('STOP SCREEN: do not run full/test/closed-loop; follow next_action')
print('PASS SCREEN ONLY. Do NOT run test/closed-loop yet. Next allowed stage: independent frozen full-val reproduction.')
PY
