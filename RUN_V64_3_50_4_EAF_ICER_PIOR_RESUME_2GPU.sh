#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Engineering-only resume wrapper. It does NOT modify the V50 scientific
# population, planner, checkpoint, intervention, outcome, Q/P/E state, gates,
# batch partition, or closed-loop configuration.
export PIOR_RESUME=1
# Reduce parent-side nvidia-smi/log-tail polling only. This is not planner time.
export PIOR_HEARTBEAT_SECONDS="${PIOR_HEARTBEAT_SECONDS:-120}"
# Keep profiling ON to preserve homogeneous instrumentation with already
# certified V50.4 batches. Profiling was measured to be non-dominant.
export PIOR_PROFILE_CLOSED_LOOP="${PIOR_PROFILE_CLOSED_LOOP:-1}"

OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/outputs_v64_3_50_4_eaf_icer_pior_train_2gpu_v1}"

python -m bdse.tools.audit_v64_3_50_pior_progress \
  --output-root "$OUT_ROOT" \
  --json-out "$OUT_ROOT/provenance/v64_3_50_4_resume_progress_before.json" || true

# The original source-manifest-locked launcher remains byte-identical and is
# authoritative for all scientific work. Valid .pior_batch_complete.json files
# are reused; an incomplete batch is deliberately rerun from its beginning.
bash "$ROOT/RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh"
