#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_CONFIG="${TRAIN_CONFIG:-bdse/configs/training.yaml}"
TRAIN_CACHE="${TRAIN_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2}"
VAL_CACHE="${VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/outputs/train_$(date -u +%Y%m%dT%H%M%SZ)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

[[ -s "$TRAIN_CONFIG" ]] || { echo "missing training config: $TRAIN_CONFIG" >&2; exit 2; }
[[ -d "$TRAIN_CACHE" ]] || { echo "missing training cache: $TRAIN_CACHE" >&2; exit 2; }
[[ -d "$VAL_CACHE" ]] || { echo "missing validation cache: $VAL_CACHE" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT must be new: $RUN_ROOT" >&2; exit 2; }

mkdir -p "$RUN_ROOT/checkpoints"
cp "$TRAIN_CONFIG" "$RUN_ROOT/resolved_input_config.yaml"
sha256sum "$TRAIN_CONFIG" > "$RUN_ROOT/input_sha256.txt"

args=(
  -m bdse.experiments.train
  --config "$TRAIN_CONFIG"
  --preprocessed-dir "$TRAIN_CACHE"
  --split train_boston train_pittsburgh train_singapore train_vegas_2
  --val-preprocessed-dir "$VAL_CACHE"
  --val-split val
  --output "$RUN_ROOT/bdse_model.pt"
  --checkpoint-dir "$RUN_ROOT/checkpoints"
  --device cuda
  --amp
  --val-mode open_loop
  --val-strict
)

[[ -z "${WARM_START_FROM:-}" ]] || args+=(--warm-start-from "$WARM_START_FROM")
[[ -z "${RESUME_FROM:-}" ]] || args+=(--resume-from "$RESUME_FROM" --resume)

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" "${args[@]}" "$@"
