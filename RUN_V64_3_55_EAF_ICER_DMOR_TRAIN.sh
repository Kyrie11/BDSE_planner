#!/usr/bin/env bash
set -euo pipefail

# V64.3.55 DMOR: fit-only follow-up to V54.  No new simulation/outcome labels.
# Branch A tests a structured Pareto functional on the identified realized
# endpoint mediator. Branch B is eligible only if A closes its gate and distills
# that mediator from the frozen V53 pre-execution planned operator profile.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BDSE_ROOT="${BDSE_ROOT:-$SCRIPT_DIR}"
cd "$BDSE_ROOT"
export PYTHONPATH="$BDSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$BDSE_ROOT/outputs}"

V49_ROOT="${V49_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1}"
V50_5_ROOT="${V50_5_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_50_5_eaf_icer_pior_train_2gpu_v1}"
V52_ROOT="${V52_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_52_eaf_icer_hodr_train_v1}"
V53_ROOT="${V53_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_53_eaf_icer_potr_train_v1}"
V54_ROOT="${V54_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_54_eaf_icer_pdrm_train_v1}"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/outputs_v64_3_55_eaf_icer_dmor_train_v1}"
mkdir -p "$OUT_ROOT/provenance" "$OUT_ROOT/logs"

V49_SCENE="$V49_ROOT/provenance/v64_3_49_siir_train_scene_audit.csv"
V49_CAND="$V49_ROOT/provenance/v64_3_49_siir_oof_candidate_state_audit.jsonl"
PAIRED="$V50_5_ROOT/provenance/v64_3_50_pior_paired_closed_loop_outcomes.jsonl"
V52_FIT="$V52_ROOT/provenance/v64_3_52_hodr_fit.json"
V53_PROFILES="$V53_ROOT/provenance/v64_3_53_operator_trajectory_profiles.jsonl"
V54_FIT="$V54_ROOT/provenance/v64_3_54_pdrm_fit.json"
V54_DYNAMIC="$V54_ROOT/provenance/v64_3_54_paired_dynamic_response_profiles.jsonl"

for f in "$V49_SCENE" "$V49_CAND" "$PAIRED" "$V52_FIT" "$V53_PROFILES" "$V54_FIT" "$V54_DYNAMIC"; do
  [[ -s "$f" ]] || { echo "V55 ENGINEERING STOP missing prerequisite $f" >&2; exit 2; }
done

sha256sum -c V64_3_55_SCIENCE_MANIFEST.sha256 | tee "$OUT_ROOT/logs/v64_3_55_science_manifest.out"

python - "$PAIRED" "$V52_FIT" "$V53_PROFILES" "$V54_FIT" "$V54_DYNAMIC" <<'PY' | tee "$OUT_ROOT/logs/v64_3_55_parent_and_evidence_lock.out"
import hashlib,json,sys
from pathlib import Path
paired,v52,v53p,v54,dyn=map(Path,sys.argv[1:])
want={
 'paired':'d592baa3508ae9dc084eebcbb3505d9accadc724601ab35a1253d3536af39d43',
 'v52':'7a21fead5383ebd6aafaeb5da586346a77e5707050cb359511a06971b742a16b',
 'v53p':'9a69c196a1d76e9c5d424068df223ec26f0e481252f25f67d5bb17fd355aaef6',
 'v54':'10f3e60c82bb8b82f1f688e866a27008e1498b67d2e194b0c7aadec5368536d8',
 'dyn':'dd2bdd809a757ce74973d7ce2c3189fad60dc0d3e0125d7fcec9ca7ad1bda373',
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for key,p in [('paired',paired),('v52',v52),('v53p',v53p),('v54',v54),('dyn',dyn)]:
    if sha(p)!=want[key]: raise SystemExit(f'V55 ENGINEERING STOP: {key} hash drift {sha(p)}')
d=json.load(open(v54)); n=d.get('nested_crossfit',{}); ep=n.get('arms',{}).get('realized_endpoint',{})
if d.get('mediator_identification_pass') is not True or n.get('preferred_mediator_arm')!='realized_endpoint':
    raise SystemExit('V55 ENGINEERING STOP: wrong V54 mediator branch')
if ep.get('identification',{}).get('identified') is not True or ep.get('retrospective_oracle_gate',{}).get('pass') is not False:
    raise SystemExit('V55 ENGINEERING STOP: V54 endpoint/sign-oracle signature drift')
print(json.dumps({'pass':True,**want},sort_keys=True))
PY

python -m compileall -q bdse
pytest -q \
  bdse/tests/test_v64_3_55_eaf_icer_dmor.py \
  bdse/tests/test_v64_3_54_eaf_icer_pdrm.py \
  bdse/tests/test_v64_3_53_eaf_icer_potr.py \
  bdse/tests/test_v64_3_52_eaf_icer_hodr.py \
  bdse/tests/test_v64_3_51_eaf_icer_pocr.py \
  bdse/tests/test_v64_3_50_eaf_icer_pior.py \
  bdse/tests/test_v64_3_50_5_pior_metric_safety.py \
  bdse/tests/test_v64_3_48_eaf_icer_ocrr.py \
  | tee "$OUT_ROOT/logs/v64_3_55_targeted_regression.out"

FIT_REPORT="$OUT_ROOT/provenance/v64_3_55_dmor_fit.json"
RUNTIME_ARTIFACT="$OUT_ROOT/provenance/v64_3_55_dmor_runtime_artifact.json"
set +e
python -m bdse.tools.fit_v64_3_55_eaf_icer_dmor \
  --v49-candidate-audit "$V49_CAND" --v49-scene-audit "$V49_SCENE" --paired-outcomes "$PAIRED" --v50-5-root "$V50_5_ROOT" \
  --v52-fit-report "$V52_FIT" --v53-operator-profiles "$V53_PROFILES" \
  --v54-fit-report "$V54_FIT" --v54-dynamic-profiles "$V54_DYNAMIC" \
  --output-report "$FIT_REPORT" --output-runtime-artifact "$RUNTIME_ARTIFACT" \
  2>&1 | tee "$OUT_ROOT/logs/v64_3_55_dmor_fit.out"
STATUS=${PIPESTATUS[0]}
set -e

cp "$V54_FIT" "$OUT_ROOT/provenance/v64_3_54_pdrm_fit_reused.json"
sha256sum "$PAIRED" "$V52_FIT" "$V53_PROFILES" "$V54_FIT" "$V54_DYNAMIC" > "$OUT_ROOT/provenance/V64_3_55_REUSED_EVIDENCE_AND_STATE_SHA256SUMS.txt"

if [[ $STATUS -ne 0 ]]; then
  echo "V64.3.55 DMOR scientific STOP. Follow v64_3_55_dmor_fit.json; do not tune TRAIN and do not run untouched validation." >&2
  exit "$STATUS"
fi

echo "V64.3.55 DMOR deployable TRAIN gate PASS. Freeze immediately. Next work is engineering-only runtime integration followed by untouched paired validation; no further TRAIN algorithm tuning."
