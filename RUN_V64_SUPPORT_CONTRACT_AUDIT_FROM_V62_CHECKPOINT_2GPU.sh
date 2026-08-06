#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
: "${V62_CKPT:?Set V62_CKPT to the existing V62 best checkpoint}"
: "${BDSE_VAL_CACHE:?Set BDSE_VAL_CACHE to the group-disjoint validation cache root}"
GPUS="${GPUS:-0,1}"
MAX_SCENARIOS="${MAX_SCENARIOS:-1000}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
OUT_ROOT="${OUT_ROOT:-outputs_v64_support_contract_audit_from_v62_checkpoint}"
REPORT="$OUT_ROOT/v64_support_contract_audit_report.json"

rm -rf "$OUT_ROOT/parallel_suite"
mkdir -p "$OUT_ROOT"
python -m bdse.tools.run_parallel_open_loop_suite \
  --system "legacy_anchor::bdse/configs/v64_saqa_bcc_anchor_control_cl.yaml::$V62_CKPT" \
  --system "support_aware_nominal::bdse/configs/v64_saqa_bcc_local_control_cl.yaml::$V62_CKPT" \
  --system "prefix_cache::bdse/configs/v64_saqa_bcc_local_control_verified_prefix_cache_cl.yaml::$V62_CKPT" \
  --system "structural_prior::bdse/configs/v64_saqa_bcc_structural_prior_ablation_cl.yaml::$V62_CKPT" \
  --preprocessed-dir "$BDSE_VAL_CACHE" --split val_tune --max-scenarios "$MAX_SCENARIOS" \
  --output-root "$OUT_ROOT/parallel_suite" --gpus "$GPUS" \
  --workers-per-gpu "$WORKERS_PER_GPU" --device cuda
python -m bdse.tools.analyze_v64_support_contract_audit \
  --suite-root "$OUT_ROOT/parallel_suite" \
  --output "$REPORT"
