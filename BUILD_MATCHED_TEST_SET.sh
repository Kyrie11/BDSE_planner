#!/usr/bin/env bash
set -euo pipefail

# Fresh official public_set_test preprocessing. Build integrity is a hard gate;
# distribution shift versus val is a warning, because an honest test split is
# not expected to reproduce the validation distribution.
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/nuplan/data/cache}"
MAPS_ROOT="${MAPS_ROOT:-/data0/senzeyu2/dataset/nuplan/maps}"
MAP_VERSION="${MAP_VERSION:-nuplan-maps-v1.0}"
TEST_OUT="${TEST_OUT:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_v50_matched}"
DIAG_OUT="${DIAG_OUT:-outputs_v50_dbap_ri_dataset/diagnostics_test_complete.json}"
READINESS_OUT="${READINESS_OUT:-outputs_v50_dbap_ri_dataset/test_readiness.json}"
VAL_DIAGNOSTICS="${VAL_DIAGNOSTICS:-}"
TRAIN_CACHE="${TRAIN_CACHE:-}"
VAL_CACHE="${VAL_CACHE:-}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SCENARIO_BUILDER_WORKERS="${SCENARIO_BUILDER_WORKERS:-8}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"

mkdir -p "$TEST_OUT" "$(dirname "$DIAG_OUT")" "$(dirname "$READINESS_OUT")"

python -m bdse.experiments.preprocess \
  --config bdse/configs/full_preprocess.yaml \
  --data-root "$DATA_ROOT" \
  --maps-root "$MAPS_ROOT" \
  --map-version "$MAP_VERSION" \
  --splits public_set_test \
  --output-dir "$TEST_OUT" \
  --scenario-stride 10 \
  --scenario-iteration-policy initial \
  --num-workers "$NUM_WORKERS" \
  --max-in-flight "$NUM_WORKERS" \
  --scenario-builder-workers "$SCENARIO_BUILDER_WORKERS" \
  --teacher-cost-eval-stride 1 \
  --resume \
  --resume-validate-existing \
  --resume-require-config-match \
  --candidate-aware-agent-selection \
  --include-drivable-polygons \
  --no-include-crosswalks \
  --cache-local-scheduler \
  --cache-local-log-parallelism 1 \
  --temporal-frame-cache-max-entries 262144 \
  --temporal-frame-cache-individual-miss-threshold 32 \
  --temporal-frame-cache-coalesce-bulk \
  --skip-failed-samples \
  --profile-threshold-s 10.0

python -m bdse.experiments.diagnostics \
  --config bdse/configs/full_preprocess.yaml \
  --split public_set_test \
  --preprocessed-dir "$TEST_OUT" \
  --output "$DIAG_OUT"

readiness=(python -m bdse.tools.check_test_set_readiness
  --test-diagnostics "$DIAG_OUT"
  --test-cache "$TEST_OUT"
  --min-preliminary-samples 10000
  --max-failed-fraction 0.01
  --output "$READINESS_OUT")
[[ -n "$VAL_DIAGNOSTICS" ]] && readiness+=(--val-diagnostics "$VAL_DIAGNOSTICS")
[[ -n "$TRAIN_CACHE" ]] && readiness+=(--train-cache "$TRAIN_CACHE")
[[ -n "$VAL_CACHE" ]] && readiness+=(--val-cache "$VAL_CACHE")
[[ "$ALLOW_INCOMPLETE" == "1" ]] && readiness+=(--allow-incomplete)
"${readiness[@]}"

if [[ -n "$VAL_DIAGNOSTICS" ]]; then
  echo "[bdse] Distribution-shift report follows; failures here do not invalidate build integrity."
  set +e
  python -m bdse.tools.check_dataset_diagnostics_parity \
    "$VAL_DIAGNOSTICS" "$DIAG_OUT" \
    --min-scenarios 10000 \
    --max-route-tail-m 8.0
  parity_status=$?
  set -e
  if [[ $parity_status -ne 0 ]]; then
    echo "[bdse] WARNING: public test differs materially from val. Preserve the test split; do not tune it to pass val-parity thresholds."
  fi
fi

echo "Test diagnostics: $DIAG_OUT"
echo "Integrity/readiness: $READINESS_OUT"
