#!/usr/bin/env bash
set -euo pipefail

# V64.3.52 HODR TRAIN-only structured paired-outcome functional test.
# Reuses the exact V50.5 metric-safe 502x2 paired evidence and the V51
# operator-relative QPE+D state. No closed-loop rerun and no untouched
# validation are consumed unless TRAIN passes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BDSE_ROOT="${BDSE_ROOT:-$SCRIPT_DIR}"
cd "$BDSE_ROOT"
export PYTHONPATH="$BDSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$BDSE_ROOT/outputs}"

V49_ROOT="${V49_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
V50_5_ROOT="${V50_5_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_50_5_eaf_icer_pior_train_2gpu_v1}"
V51_ROOT="${V51_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_51_eaf_icer_pocr_train_v1}"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_52_eaf_icer_hodr_train_v1}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"

V49_FIT="$V49_ROOT/provenance/v64_3_49_siir_fit.json"
V49_SCENE="$V49_ROOT/provenance/v64_3_49_siir_train_scene_audit.csv"
V49_CAND="$V49_ROOT/provenance/v64_3_49_siir_oof_candidate_state_audit.jsonl"
V49_CONFIG="$V49_ROOT/provenance/v64_3_49_siir.yaml"
PAIRED="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_outcomes.jsonl"
PAIR_REPORT="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_report.json"
METRIC_AUDIT="$V50_5_ROOT/provenance/v64_3_50_5_metric_safety_full_audit.json"
V51_FIT="$V51_ROOT/provenance/v64_3_51_pocr_fit.json"

for f in "$V49_FIT" "$V49_SCENE" "$V49_CAND" "$V49_CONFIG" "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" "$V51_FIT"; do
  [[ -s "$f" ]] || { echo "V52 ENGINEERING STOP missing prerequisite $f" >&2; exit 2; }
done

# Robust provenance policy: lock only result-defining science files. Archival
# patches, diagnostics, README and changelog are deliberately excluded so a
# stale non-executable artifact can never force users to disable science SHA.
sha256sum -c V64_3_52_SCIENCE_MANIFEST.sha256 > "$OUT_ROOT/logs/v64_3_52_science_manifest.out"

python - "$V51_FIT" <<'PY' | tee "$OUT_ROOT/logs/v64_3_52_parent_v51_lock.out"
import hashlib, json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.load(open(p))
want='54d3664e378c85ba482485c49ddcb3e10e83a59d5d04cd049ccaf05fdbb23049'
got=hashlib.sha256(p.read_bytes()).hexdigest()
n=d.get('nested_crossfit',{})
if got!=want:
    raise SystemExit(f'V52 ENGINEERING STOP: V51 fit hash drift {got}')
if d.get('train_gate_pass') is not False or n.get('failure_diagnosis')!='operator_contrast_state_identified_but_low_capacity_sign_retention_functional_insufficient':
    raise SystemExit('V52 ENGINEERING STOP: wrong V51 preregistered branch')
for a in ('qpe_dose','qpe_dose_interaction'):
    if n.get('arms',{}).get(a,{}).get('risk_identification',{}).get('identified') is not True:
        raise SystemExit(f'V52 ENGINEERING STOP: V51 {a} identification drift')
    if n.get('arms',{}).get(a,{}).get('deployment_gate',{}).get('pass') is not False:
        raise SystemExit(f'V52 ENGINEERING STOP: V51 {a} deployment signature drift')
if n.get('operator_contrast_diagnostic',{}).get('effect_support_identified') is not True:
    raise SystemExit('V52 ENGINEERING STOP: V51 effect-support signature drift')
print(json.dumps({'pass':True,'v51_fit_sha256':got,'preferred_arm':n.get('preferred_promotion_arm'),'train_gate_pass':False},sort_keys=True))
PY

python - "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" <<'PY' | tee "$OUT_ROOT/logs/v64_3_52_reused_evidence_audit.out"
import hashlib, json, sys
from pathlib import Path
paired,report,metric=map(Path,sys.argv[1:])
rows=[json.loads(x) for x in open(paired,encoding='utf-8') if x.strip()]
r=json.load(open(report)); m=json.load(open(metric))
if len(rows)!=502 or len({str(x['scenario_token']) for x in rows})!=502:
    raise SystemExit('V52 ENGINEERING STOP: paired outcomes are not 502/502 unique')
if r.get('pass') is not True or int(r.get('scenario_count',-1))!=502 or m.get('pass') is not True:
    raise SystemExit('V52 ENGINEERING STOP: paired report / metric-safety audit not PASS')
for arm in ('control','treatment'):
    if m['arms'][arm].get('pass') is not True or int(m['arms'][arm].get('certified_scenarios',-1))!=502:
        raise SystemExit(f'V52 ENGINEERING STOP: {arm} metric-safe population drift')
sha=hashlib.sha256(paired.read_bytes()).hexdigest()
want='d592baa3508ae9dc084eebcbb3505d9accadc724601ab35a1253d3536af39d43'
if sha!=want:
    raise SystemExit(f'V52 ENGINEERING STOP: paired evidence hash drift {sha}')
print(json.dumps({'pass':True,'paired_count':502,'paired_sha256':sha,'beneficial':sum(bool(x['closed_loop_beneficial']) for x in rows),'hard_harm':sum(bool(x['closed_loop_hard_harm']) for x in rows)},sort_keys=True))
PY

python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_52_eaf_icer_hodr.py \
  bdse/tests/test_v64_3_51_eaf_icer_pocr.py \
  bdse/tests/test_v64_3_50_6_pior_fit_repair.py \
  bdse/tests/test_v64_3_50_eaf_icer_pior.py \
  bdse/tests/test_v64_3_50_5_pior_metric_safety.py \
  bdse/tests/test_v64_3_48_eaf_icer_ocrr.py \
  | tee "$OUT_ROOT/logs/v64_3_52_targeted_regression.out"

FIT_REPORT="$OUT_ROOT/provenance/v64_3_52_hodr_fit.json"
HODR_CONFIG="$OUT_ROOT/provenance/v64_3_52_hodr.yaml"
set +e
python -m bdse.tools.fit_v64_3_52_eaf_icer_hodr \
  --v49-fit-report "$V49_FIT" --v49-candidate-audit "$V49_CAND" --v49-scene-audit "$V49_SCENE" --v49-siir-config "$V49_CONFIG" \
  --paired-outcomes "$PAIRED" --v50-5-root "$V50_5_ROOT" --v51-fit-report "$V51_FIT" \
  --output-config "$HODR_CONFIG" --output-report "$FIT_REPORT" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_52_hodr_fit.out"
FIT_STATUS=${PIPESTATUS[0]}
set -e

cp "$V51_FIT" "$OUT_ROOT/provenance/v64_3_51_pocr_fit_reused.json"
cp "$METRIC_AUDIT" "$OUT_ROOT/provenance/v64_3_50_5_metric_safety_full_audit_reused.json"
sha256sum "$PAIRED" "$PAIR_REPORT" "$METRIC_AUDIT" "$V51_FIT" > "$OUT_ROOT/provenance/V64_3_52_REUSED_EVIDENCE_SHA256SUMS.txt"

if [[ $FIT_STATUS -ne 0 ]]; then
  echo "V64.3.52 HODR scientific STOP. Do not consume untouched validation; follow v64_3_52_hodr_fit.json." >&2
  exit "$FIT_STATUS"
fi

echo "V64.3.52 HODR TRAIN PASS. Freeze the preferred arm immediately; next step is new untouched paired validation with no TRAIN tuning."
