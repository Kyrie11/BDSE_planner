#!/usr/bin/env bash
set -euo pipefail

# Build a fresh public_set_test cache with the feature/label knobs visible in the
# supplied train/val diagnostics.  The new output directory is deliberate: the
# old command disabled drivable polygons and --resume did not verify config.
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/nuplan/data/cache}"
MAPS_ROOT="${MAPS_ROOT:-/data0/senzeyu2/dataset/nuplan/maps}"
MAP_VERSION="${MAP_VERSION:-nuplan-maps-v1.0}"
TEST_OUT="${TEST_OUT:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_v49_matched}"
DIAG_OUT="${DIAG_OUT:-outputs_v49_dbap_dataset/diagnostics_test_complete.json}"
VAL_DIAGNOSTICS="${VAL_DIAGNOSTICS:-}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SCENARIO_BUILDER_WORKERS="${SCENARIO_BUILDER_WORKERS:-8}"

mkdir -p "$TEST_OUT" "$(dirname "$DIAG_OUT")"

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

if [[ -n "$VAL_DIAGNOSTICS" ]]; then
  python -m bdse.tools.check_dataset_diagnostics_parity \
    "$VAL_DIAGNOSTICS" "$DIAG_OUT" \
    --min-scenarios 10000 \
    --max-route-tail-m 8.0
else
  echo "Test diagnostics written to $DIAG_OUT"
  echo "Set VAL_DIAGNOSTICS=/path/to/diagnostics_val.json to run the parity gate."
fi
