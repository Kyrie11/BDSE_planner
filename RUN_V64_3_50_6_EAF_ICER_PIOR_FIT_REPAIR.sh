#!/usr/bin/env bash
set -euo pipefail

# V64.3.50.6: fit-only engineering repair.
# Reuse ONLY a complete V50.5 metric-safe 502x2 paired collection. Do not rerun
# closed loop and do not consume untouched validation here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BDSE_ROOT="${BDSE_ROOT:-$SCRIPT_DIR}"
cd "$BDSE_ROOT"
export PYTHONPATH="$BDSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$BDSE_ROOT/outputs}"

V49_ROOT="${V49_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
V50_5_ROOT="${V50_5_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_50_5_eaf_icer_pior_train_2gpu_v1}"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_50_6_eaf_icer_pior_fitrepair_v1}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"

V49_FIT="$V49_ROOT/provenance/v64_3_49_siir_fit.json"
V49_SCENE="$V49_ROOT/provenance/v64_3_49_siir_train_scene_audit.csv"
V49_CAND="$V49_ROOT/provenance/v64_3_49_siir_oof_candidate_state_audit.jsonl"
V49_CONFIG="$V49_ROOT/provenance/v64_3_49_siir.yaml"
PAIRED="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_outcomes.jsonl"
PAIR_REPORT="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_report.json"
TRAIN_MANIFEST="$V50_5_ROOT/provenance/v64_3_50_pior_train_manifest.json"
METRIC_AUDIT="$V50_5_ROOT/provenance/v64_3_50_5_metric_safety_full_audit.json"

for f in "$V49_FIT" "$V49_SCENE" "$V49_CAND" "$V49_CONFIG" "$PAIRED" "$PAIR_REPORT" "$TRAIN_MANIFEST" "$METRIC_AUDIT"; do
  [[ -s "$f" ]] || { echo "STOP V50.6 missing prerequisite $f" >&2; exit 2; }
done

# Preserve the original V50.5 science/engineering locks. New V50.6 files are
# additive and the old science-critical source files must remain byte-identical.
sha256sum -c V64_3_50_SOURCE_MANIFEST.sha256 > "$OUT_ROOT/logs/v64_3_50_source_manifest_recheck.out"
sha256sum -c V64_3_50_5_ENGINEERING_MANIFEST.sha256 > "$OUT_ROOT/logs/v64_3_50_5_engineering_manifest_recheck.out"

python - "$PAIR_REPORT" "$TRAIN_MANIFEST" "$PAIRED" "$METRIC_AUDIT" <<'PY' | tee "$OUT_ROOT/logs/v64_3_50_6_paired_reuse_audit.out"
import json,sys,hashlib
from pathlib import Path
pair_p,man_p,out_p,metric_p=map(Path,sys.argv[1:])
pair=json.load(open(pair_p)); man=json.load(open(man_p)); metric=json.load(open(metric_p))
if pair.get('pass') is not True or int(pair.get('scenario_count',-1)) != 502:
    raise SystemExit('STOP V50.6: V50.5 paired report is not complete 502/502 PASS')
for arm in ('control','treatment'):
    a=pair.get(arm,{})
    if int(a.get('scenario_count',-1)) != 502:
        raise SystemExit(f'STOP V50.6: {arm} scenario_count != 502')
    if any(not bool(b.get('complete')) or int(b.get('failed',-1)) != 0 or int(b.get('successful',-1)) != int(b.get('scenario_count',-2)) or int(b.get('probe_fired_count',-1)) != int(b.get('scenario_count',-2)) for b in a.get('batches',[])):
        raise SystemExit(f'STOP V50.6: {arm} has an incomplete/failed/non-probed batch')
if metric.get('pass') is not True or int(metric.get('expected_scenarios_per_arm',-1)) != 502:
    raise SystemExit('STOP V50.6: V50.5 metric-safety full audit is not PASS')
rows=[json.loads(x) for x in open(out_p,encoding='utf-8') if x.strip()]
mtoks=[str(r['scenario_token']) for r in man.get('rows',[])]
otoks=[str(r['scenario_token']) for r in rows]
if len(rows)!=502 or len(set(otoks))!=502 or set(otoks)!=set(mtoks):
    raise SystemExit('STOP V50.6: paired outcome/token identity is not exact 502/502')
fold={str(r['scenario_token']):int(r['outer_test_fold']) for r in man['rows']}
benef=[sum(bool(r['closed_loop_beneficial']) and fold[str(r['scenario_token'])]==k for r in rows) for k in range(5)]
print(json.dumps({
    'pass':True,
    'paired_count':len(rows),
    'paired_sha256':hashlib.sha256(out_p.read_bytes()).hexdigest(),
    'beneficial_count':sum(bool(r['closed_loop_beneficial']) for r in rows),
    'beneficial_by_outer_fold':benef,
    'hard_harm_count':sum(bool(r['closed_loop_hard_harm']) for r in rows),
    'reuse_role':'same already-collected TRAIN evidence; engineering fit repair only',
},sort_keys=True))
PY

python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_48_eaf_icer_ocrr.py \
  bdse/tests/test_v64_3_50_eaf_icer_pior.py \
  bdse/tests/test_v64_3_50_5_pior_metric_safety.py \
  bdse/tests/test_v64_3_50_6_pior_fit_repair.py \
  | tee "$OUT_ROOT/logs/v64_3_50_6_targeted_regression.out"

# Re-run the V50.5 metric-safety audit against the exact reused collection.
python -m bdse.tools.audit_v64_3_50_5_pior_metric_safety \
  --closed-loop-root "$V50_5_ROOT/closed_loop_train" --expected-scenarios 502 \
  --output-report "$OUT_ROOT/provenance/v64_3_50_6_reused_metric_safety_audit.json" \
  | tee "$OUT_ROOT/logs/v64_3_50_6_reused_metric_safety_audit.out"

PIOR_CONFIG="$OUT_ROOT/provenance/v64_3_50_pior.yaml"
FIT_REPORT="$OUT_ROOT/provenance/v64_3_50_6_pior_fit.json"
set +e
python -m bdse.tools.fit_v64_3_50_6_eaf_icer_pior \
  --v49-fit-report "$V49_FIT" --v49-candidate-audit "$V49_CAND" --v49-scene-audit "$V49_SCENE" \
  --v49-siir-config "$V49_CONFIG" --paired-outcomes "$PAIRED" \
  --output-config "$PIOR_CONFIG" --output-report "$FIT_REPORT" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_50_6_pior_fit.out"
FIT_STATUS=${PIPESTATUS[0]}
set -e

cp "$PAIR_REPORT" "$OUT_ROOT/provenance/v64_3_50_5_paired_closed_loop_report_reused.json"
cp "$METRIC_AUDIT" "$OUT_ROOT/provenance/v64_3_50_5_metric_safety_full_audit_reused.json"
sha256sum "$PAIRED" "$PAIR_REPORT" "$TRAIN_MANIFEST" "$METRIC_AUDIT" > "$OUT_ROOT/provenance/V64_3_50_6_REUSED_EVIDENCE_SHA256SUMS.txt"

if [[ $FIT_STATUS -ne 0 ]]; then
  echo "V64.3.50.6 PIOR scientific STOP after repaired nested TRAIN evaluation. Do not consume untouched validation." >&2
  exit "$FIT_STATUS"
fi

echo "V64.3.50.6 PIOR TRAIN PASS after engineering repair. Freeze artifact; next step is untouched paired closed-loop validation."
