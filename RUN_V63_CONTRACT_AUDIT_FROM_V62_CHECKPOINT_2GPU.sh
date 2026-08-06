#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
: "${V62_CKPT:?Set V62_CKPT to the existing V62 best checkpoint}"
: "${BDSE_VAL_CACHE:?Set BDSE_VAL_CACHE to the group-disjoint validation cache root}"
GPUS="${GPUS:-0,1}"
MAX_SCENARIOS="${MAX_SCENARIOS:-1000}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
OUT_ROOT="${OUT_ROOT:-outputs_v63_contract_audit_from_v62_checkpoint}"
FORCE="${FORCE:-0}"
REPORT="$OUT_ROOT/v63_contract_audit_report.json"
SUITE_REPORT="$OUT_ROOT/parallel_suite/parallel_open_loop_suite_report.json"
CONFIGS=(
  bdse/configs/v63_dcqc_tfcr_local_control_cl.yaml
  bdse/configs/v63_dcqc_tfcr_local_control_no_base_prior_cl.yaml
  bdse/configs/v63_dcqc_tfcr_local_control_no_structural_prior_cl.yaml
  bdse/configs/v63_dcqc_tfcr_local_control_no_runtime_priors_cl.yaml
)

is_fresh() {
  local output="$1"; shift
  [[ "$FORCE" != "1" && -s "$output" ]] || return 1
  local input
  for input in "$@"; do
    [[ -e "$input" && "$output" -nt "$input" ]] || return 1
  done
}

if is_fresh "$REPORT" "$V62_CKPT" "${CONFIGS[@]}" \
    bdse/tools/analyze_v63_contract_audit.py "$0" && \
   is_fresh "$SUITE_REPORT" "$V62_CKPT" "${CONFIGS[@]}" "$0"; then
  echo "[v63-audit] reuse fresh report: $REPORT"
  cat "$REPORT"
  exit 0
fi

rm -rf "$OUT_ROOT/parallel_suite"
mkdir -p "$OUT_ROOT"
python -m bdse.tools.run_parallel_open_loop_suite \
  --system "nominal::${CONFIGS[0]}::$V62_CKPT" \
  --system "no_base_prior::${CONFIGS[1]}::$V62_CKPT" \
  --system "no_structural_prior::${CONFIGS[2]}::$V62_CKPT" \
  --system "no_runtime_priors::${CONFIGS[3]}::$V62_CKPT" \
  --preprocessed-dir "$BDSE_VAL_CACHE" --split val_tune --max-scenarios "$MAX_SCENARIOS" \
  --output-root "$OUT_ROOT/parallel_suite" --gpus "$GPUS" \
  --workers-per-gpu "$WORKERS_PER_GPU" --device cuda
python -m bdse.tools.analyze_v63_contract_audit \
  --suite-root "$OUT_ROOT/parallel_suite" \
  --output "$REPORT"
