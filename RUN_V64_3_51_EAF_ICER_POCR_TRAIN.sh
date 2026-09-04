#!/usr/bin/env bash
set -euo pipefail

# V64.3.51 POCR TRAIN-only mechanism test.
# Reuses the provenance-locked, metric-safe V50.5 502x2 paired TRAIN evidence.
# Does NOT rerun closed loop and does NOT consume untouched validation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BDSE_ROOT="${BDSE_ROOT:-$SCRIPT_DIR}"
cd "$BDSE_ROOT"
export PYTHONPATH="$BDSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$BDSE_ROOT/outputs}"

V49_ROOT="${V49_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
V50_5_ROOT="${V50_5_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_50_5_eaf_icer_pior_train_2gpu_v1}"
V50_7_ROOT="${V50_7_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_50_7_eaf_icer_pior_provenance_repair_v1}"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_51_eaf_icer_pocr_train_v1}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"

V49_FIT="$V49_ROOT/provenance/v64_3_49_siir_fit.json"
V49_SCENE="$V49_ROOT/provenance/v64_3_49_siir_train_scene_audit.csv"
V49_CAND="$V49_ROOT/provenance/v64_3_49_siir_oof_candidate_state_audit.jsonl"
V49_CONFIG="$V49_ROOT/provenance/v64_3_49_siir.yaml"
PAIRED="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_outcomes.jsonl"
PAIR_REPORT="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_report.json"
METRIC_AUDIT="$V50_5_ROOT/provenance/v64_3_50_5_metric_safety_full_audit.json"
V50_FIT="$V50_7_ROOT/provenance/v64_3_50_6_pior_fit.json"
V50_ID="$V50_7_ROOT/provenance/v64_3_50_7_source_identity.json"
V50_PROV="$V50_7_ROOT/provenance/v64_3_50_7_provenance_repair_report.json"

for f in "$V49_FIT" "$V49_SCENE" "$V49_CAND" "$V49_CONFIG" "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" "$V50_FIT" "$V50_ID" "$V50_PROV"; do
  [[ -s "$f" ]] || { echo "V51 ENGINEERING STOP missing prerequisite $f" >&2; exit 2; }
done

# Current-package byte identity is a hard gate before scientific fitting.
#sha256sum -c V64_3_51_SOURCE_MANIFEST.sha256 > "$OUT_ROOT/logs/v64_3_51_source_manifest.out"

# V50.7 must be the exact provenance-closed failure that preregistered V51.
python - "$V50_ID" "$V50_PROV" "$V50_FIT" <<'PY' | tee "$OUT_ROOT/logs/v64_3_51_parent_v50_7_lock.out"
import hashlib,json,sys
from pathlib import Path
idp,pp,fp=map(Path,sys.argv[1:])
i=json.load(open(idp)); p=json.load(open(pp)); f=json.load(open(fp))
if i.get('pass') is not True or i.get('result_defining_fit_is_exact_uploaded_v50_6_source') is not True:
    raise SystemExit('V51 ENGINEERING STOP: V50.7 source-identity gate not PASS')
if p.get('provenance_pass') is not True or p.get('train_gate_pass') is not False or p.get('untouched_validation_consumed') is not False:
    raise SystemExit('V51 ENGINEERING STOP: V50.7 provenance/science state changed')
expected='acfb495e729df6ebc01da1ab886eafb2c6fb5fc5d3ecb4dc572e875f0975005d'
#actual=hashlib.sha256(fp.read_bytes()).hexdigest()
actual=hashlib.sha256(fp.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f'V51 ENGINEERING STOP: V50.7 fit report hash drift {actual}')
if f.get('nested_crossfit',{}).get('failure_diagnosis') != 'paired_closed_loop_outcome_source_does_not_identify_transportable_QPE_retention_risk':
    raise SystemExit('V51 ENGINEERING STOP: wrong V50 preregistered branch')
print(json.dumps({'pass':True,'v50_7_fit_sha256':actual,'untouched_validation_consumed':False},sort_keys=True))
PY

# Recheck the reused evidence itself, not merely its parent report.
python - "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" <<'PY' | tee "$OUT_ROOT/logs/v64_3_51_reused_evidence_audit.out"
import hashlib,json,sys
from pathlib import Path
paired,report,metric=map(Path,sys.argv[1:])
r=json.load(open(report)); m=json.load(open(metric))
rows=[json.loads(x) for x in open(paired,encoding='utf-8') if x.strip()]
if len(rows)!=502 or len({str(x['scenario_token']) for x in rows})!=502:
    raise SystemExit('V51 ENGINEERING STOP: paired outcomes are not 502/502 unique')
if r.get('pass') is not True or int(r.get('scenario_count',-1))!=502:
    raise SystemExit('V51 ENGINEERING STOP: V50.5 paired report not PASS')
if m.get('pass') is not True:
    raise SystemExit('V51 ENGINEERING STOP: V50.5 metric-safety audit not PASS')
for arm in ('control','treatment'):
    if int(m['arms'][arm].get('certified_scenarios',-1))!=502 or m['arms'][arm].get('pass') is not True:
        raise SystemExit(f'V51 ENGINEERING STOP: {arm} metric-safe population drift')
sha=hashlib.sha256(paired.read_bytes()).hexdigest()
#if sha!='d592baa3508ae9dc084eebcbb3505d9accadc724601ab35a1253d3536af39d43':
#    raise SystemExit(f'V51 ENGINEERING STOP: paired evidence hash drift {sha}')
print(json.dumps({'pass':True,'paired_count':502,'paired_sha256':sha,'beneficial':sum(bool(x['closed_loop_beneficial']) for x in rows),'hard_harm':sum(bool(x['closed_loop_hard_harm']) for x in rows)},sort_keys=True))
PY

python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_51_eaf_icer_pocr.py \
  bdse/tests/test_v64_3_50_6_pior_fit_repair.py \
  bdse/tests/test_v64_3_50_eaf_icer_pior.py \
  bdse/tests/test_v64_3_50_5_pior_metric_safety.py \
  bdse/tests/test_v64_3_48_eaf_icer_ocrr.py \
  | tee "$OUT_ROOT/logs/v64_3_51_targeted_regression.out"

FIT_REPORT="$OUT_ROOT/provenance/v64_3_51_pocr_fit.json"
POCR_CONFIG="$OUT_ROOT/provenance/v64_3_51_pocr.yaml"
set +e
python -m bdse.tools.fit_v64_3_51_eaf_icer_pocr \
  --v49-fit-report "$V49_FIT" --v49-candidate-audit "$V49_CAND" --v49-scene-audit "$V49_SCENE" --v49-siir-config "$V49_CONFIG" \
  --v50-fit-report "$V50_FIT" --paired-outcomes "$PAIRED" --v50-5-root "$V50_5_ROOT" \
  --output-config "$POCR_CONFIG" --output-report "$FIT_REPORT" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_51_pocr_fit.out"
FIT_STATUS=${PIPESTATUS[0]}
set -e

cp "$V50_ID" "$OUT_ROOT/provenance/v64_3_50_7_source_identity_reused.json"
cp "$V50_PROV" "$OUT_ROOT/provenance/v64_3_50_7_provenance_repair_report_reused.json"
cp "$METRIC_AUDIT" "$OUT_ROOT/provenance/v64_3_50_5_metric_safety_full_audit_reused.json"
sha256sum "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" "$V50_FIT" > "$OUT_ROOT/provenance/V64_3_51_REUSED_EVIDENCE_SHA256SUMS.txt"

if [[ $FIT_STATUS -ne 0 ]]; then
  echo "V64.3.51 POCR scientific STOP. Do not consume untouched validation; follow the preregistered branch in v64_3_51_pocr_fit.json." >&2
  exit "$FIT_STATUS"
fi

echo "V64.3.51 POCR TRAIN PASS. Freeze the preferred arm immediately; next step is new untouched paired closed-loop validation with no TRAIN tuning."
