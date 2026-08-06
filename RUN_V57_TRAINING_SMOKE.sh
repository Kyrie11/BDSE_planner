#!/usr/bin/env bash
set -euo pipefail

: "${BDSE_TRAIN_CACHE:?Set BDSE_TRAIN_CACHE}"
: "${BDSE_VAL_CACHE_ORIGINAL:?Set BDSE_VAL_CACHE_ORIGINAL}"
: "${BDSE_SPLIT_CACHE:?Set BDSE_SPLIT_CACHE}"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to the frozen foundation anchor}"

GPUS="${GPUS:-0,1}"
SMOKE_OUT_ROOT="${SMOKE_OUT_ROOT:-outputs_v57_wcdcip_training_smoke}"
SMOKE_TRAIN_SCENARIOS="${SMOKE_TRAIN_SCENARIOS:-1024}"
SMOKE_VAL_SCENARIOS="${SMOKE_VAL_SCENARIOS:-256}"
BASE_CONFIG="${BASE_CONFIG:-bdse/configs/v57_wcdcip_bfar_dbap_train_2gpu.yaml}"
SMOKE_CONFIG="$SMOKE_OUT_ROOT/v57_wcdcip_bfar_dbap_smoke_2gpu.yaml"

mkdir -p "$SMOKE_OUT_ROOT"
python - "$BASE_CONFIG" "$SMOKE_CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml
src, dst = map(Path, sys.argv[1:3])
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
train = cfg.setdefault("training", {})
train["epochs"] = 1
# Force a short exact-alignment tail so this smoke run verifies the true
# deployment selector and winner-correction graph, not only the sparse path.
train["deployment_selector_full_last_n_steps"] = max(
    8, int(train.get("deployment_selector_full_last_n_steps", 0))
)
train["name"] = "v57_wcdcip_bfar_dbap_training_smoke"
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

rm -rf "$SMOKE_OUT_ROOT/train" "$SMOKE_OUT_ROOT/logs"

OUT_ROOT="$SMOKE_OUT_ROOT" \
TRAIN_CONFIG="$SMOKE_CONFIG" \
FOUNDATION_CKPT="$FOUNDATION_CKPT" \
V30_CKPT_IN="$FOUNDATION_CKPT" \
INIT_MODE=warm_start \
AUTO_RESUME=0 \
RUN_MODE=train \
MAX_TRAIN_SCENARIOS="$SMOKE_TRAIN_SCENARIOS" \
VAL_SCENARIOS="$SMOKE_VAL_SCENARIOS" \
VAL_EVERY_N_EPOCHS=1 \
SAVE_EVERY_N_EPOCHS=0 \
SAVE_EVERY_N_STEPS=0 \
SELECTOR_SCENES_PER_RANK=1 \
SELECTOR_EVERY_N_STEPS=8 \
SELECTOR_FULL_LAST_N_STEPS=8 \
EXACT_SELECTOR_CPU_BACKEND=process \
EXACT_SELECTOR_WORKERS_PER_RANK="${EXACT_SELECTOR_WORKERS_PER_RANK:-2}" \
BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-8}" \
NUM_WORKERS_PER_GPU="${TRAIN_NUM_WORKERS_PER_GPU:-4}" \
GPUS="$GPUS" \
bash run_v57_wcdcip_bfar_dbap.sh

LOG="$SMOKE_OUT_ROOT/train/bdse_v57_wcdcip_bfar_dbap.train_log.jsonl"
python - "$LOG" <<'PY'
from pathlib import Path
import json, math, sys
p = Path(sys.argv[1])
rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit("V57 smoke failed: empty training log")

def maximum(key: str) -> float:
    vals = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except Exception:
            continue
        if math.isfinite(value):
            vals.append(value)
    return max(vals) if vals else float("nan")

checks = {
    "action_family_enabled": maximum("action_family_enabled") >= 0.99,
    "selector_exact_fraction": maximum("selector_exact_fraction") > 0.0,
    "deployment_selection_loss": maximum("L_deploy_select") > 0.0,
    "pair_full_action_loss": maximum("L_pair_full_action") > 0.0,
    "winner_correction_loss": maximum("L_residual_winner_correction") > 0.0,
    "residual_uncertainty_loss": maximum("L_residual_action_uncertainty") > 0.0,
}
print(json.dumps({
    "checks": checks,
    "maxima": {
        key: maximum(key) for key in (
            "action_family_enabled",
            "selector_exact_fraction",
            "L_deploy_select",
            "L_pair_full_action",
            "L_pair_full_winner_margin",
            "L_action_potential_teacher",
            "L_residual_winner_correction",
            "L_residual_action_uncertainty",
        )
    },
}, indent=2, sort_keys=True))
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("V57 smoke failed: " + ", ".join(failed))
print("V57 training smoke: PASS")
PY
