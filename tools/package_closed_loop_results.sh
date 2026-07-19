#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-"$HOME/code/BDSE_planner/outputs_v32_runtime_ckpt/closed_loop"}
OUTPUT=${2:-"${ROOT%/closed_loop}_closed_loop_results.tar.gz"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/package_closed_loop_results.py" --root "$ROOT" --output "$OUTPUT" "${@:3}"
