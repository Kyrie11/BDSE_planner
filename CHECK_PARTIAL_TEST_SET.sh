#!/usr/bin/env bash
set -euo pipefail
TEST_DIAGNOSTICS="${TEST_DIAGNOSTICS:?set TEST_DIAGNOSTICS}"
VAL_DIAGNOSTICS="${VAL_DIAGNOSTICS:?set VAL_DIAGNOSTICS}"
OUT="${OUT:-partial_test_readiness.json}"
args=(python -m bdse.tools.check_test_set_readiness --test-diagnostics "$TEST_DIAGNOSTICS" --val-diagnostics "$VAL_DIAGNOSTICS" --allow-incomplete --output "$OUT")
[[ -n "${TEST_CACHE:-}" ]] && args+=(--test-cache "$TEST_CACHE")
[[ -n "${TRAIN_CACHE:-}" ]] && args+=(--train-cache "$TRAIN_CACHE")
[[ -n "${VAL_CACHE:-}" ]] && args+=(--val-cache "$VAL_CACHE")
"${args[@]}"
