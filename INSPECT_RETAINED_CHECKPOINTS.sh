#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
SEARCH_ROOT="${FOUNDATION_SEARCH_ROOT:-.}"
REPORT="${1:-retained_checkpoint_inventory.json}"

python -m bdse.tools.resolve_foundation_checkpoint \
  --config bdse/configs/v50_bdse_dbap_ri_train_2gpu.yaml \
  --search-root "$SEARCH_ROOT" \
  --output-json "$REPORT"

echo
printf 'Checkpoint files visible under retained output directories:\n'
find "$SEARCH_ROOT" -maxdepth 5 -type f -name '*.pt' \
  \( -path '*/outputs_v4[0-9]*/*' -o -path '*/outputs_v50*/*' \) \
  -printf '%p\n' 2>/dev/null | sort || true

echo
echo "Full compatibility/provenance report: $REPORT"
