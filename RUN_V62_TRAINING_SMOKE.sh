#!/usr/bin/env bash
set -euo pipefail

: "${BDSE_TRAIN_CACHE:?Set BDSE_TRAIN_CACHE}"
# Accept either historical variable name.  The previous wrapper required
# BDSE_VAL_CACHE_ORIGINAL but the delegated runner actually consumes
# BDSE_VAL_CACHE, which made otherwise valid launch commands environment-dependent.
BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-${BDSE_VAL_CACHE_ORIGINAL:-}}"
: "${BDSE_VAL_CACHE:?Set BDSE_VAL_CACHE (or BDSE_VAL_CACHE_ORIGINAL)}"
BDSE_VAL_CACHE_ORIGINAL="${BDSE_VAL_CACHE_ORIGINAL:-$BDSE_VAL_CACHE}"
export BDSE_VAL_CACHE BDSE_VAL_CACHE_ORIGINAL
: "${BDSE_SPLIT_CACHE:?Set BDSE_SPLIT_CACHE}"
: "${FOUNDATION_CKPT:?Set FOUNDATION_CKPT to the frozen foundation anchor}"

GPUS="${GPUS:-0,1}"
SMOKE_OUT_ROOT="${SMOKE_OUT_ROOT:-outputs_v62_dcab_ewfc_training_smoke}"
SMOKE_TRAIN_SCENARIOS="${SMOKE_TRAIN_SCENARIOS:-1024}"
SMOKE_VAL_SCENARIOS="${SMOKE_VAL_SCENARIOS:-256}"
BASE_CONFIG="${BASE_CONFIG:-bdse/configs/v62_dcab_ewfc_train_2gpu.yaml}"
SMOKE_CONFIG="$SMOKE_OUT_ROOT/v62_dcab_ewfc_smoke_2gpu.yaml"

mkdir -p "$SMOKE_OUT_ROOT"
python - "$BASE_CONFIG" "$SMOKE_CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml
src, dst = map(Path, sys.argv[1:3])
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
train = cfg.setdefault("training", {})
train["epochs"] = 1
train["deployment_selector_every_n_steps"] = min(
    4, int(train.get("deployment_selector_every_n_steps", 4))
)
train["deployment_selector_full_last_n_steps"] = max(
    12, int(train.get("deployment_selector_full_last_n_steps", 0))
)
train["name"] = "v62_dcab_ewfc_training_smoke"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

rm -rf "$SMOKE_OUT_ROOT/train" "$SMOKE_OUT_ROOT/logs" "$SMOKE_OUT_ROOT/.v62_run.lock"

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
SELECTOR_EVERY_N_STEPS=4 \
SELECTOR_FULL_LAST_N_STEPS=12 \
EXACT_SELECTOR_CPU_BACKEND=process \
EXACT_SELECTOR_WORKERS_PER_RANK="${EXACT_SELECTOR_WORKERS_PER_RANK:-2}" \
BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-8}" \
NUM_WORKERS_PER_GPU="${TRAIN_NUM_WORKERS_PER_GPU:-4}" \
GPUS="$GPUS" \
bash run_v62_dcab_ewfc.sh

LOG="$SMOKE_OUT_ROOT/train/bdse_v62_dcab_ewfc.train_log.jsonl"
TRAIN_STDOUT="$SMOKE_OUT_ROOT/logs/train_2gpu.out"
python - "$LOG" "$TRAIN_STDOUT" <<'PY'
from pathlib import Path
import json, math, re, sys
log_path, stdout_path = map(Path, sys.argv[1:3])
rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit("V62 smoke failed: empty training log")

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

stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
lr_line = next((line for line in stdout.splitlines() if "optimizer LR groups:" in line), "")
has_mean_5x = bool(re.search(r"(?:^|,\s*)x5(?:\.0+)?:", lr_line))
has_var_2x = bool(re.search(r"(?:^|,\s*)x2(?:\.0+)?:", lr_line))

checks = {
    "action_family_enabled": maximum("action_family_enabled") >= 0.99,
    "selector_exact_fraction": maximum("selector_exact_fraction") > 0.0,
    "deployment_selection_loss": maximum("L_deploy_select") > 0.0,
    "pair_full_action_loss": maximum("L_pair_full_action") > 0.0,
    "winner_correction_loss": maximum("L_residual_winner_correction") > 0.0,
    "certified_winner_loss": maximum("L_certified_residual_winner") > 0.0,
    "boundary_margin_distillation": maximum("L_residual_boundary_margin_distill") > 0.0,
    "dense_winner_proposal_loss": maximum("L_proposal_dense_winner") > 0.0,
    "exact_winner_flip_loss": maximum("L_exact_winner_flip_critical_proposal") > 0.0,
    "exact_winner_flip_metric": maximum("exact_winner_flip_critical_scene_fraction") >= 0.0,
    "dense_winner_topm_metric": maximum("proposal_dense_topm_match") >= 0.0,
    "exact_hab_sampled": maximum("proposal_exact_hab_fraction") > 0.0,
    "fast_exact_hab_diagnostic": maximum("proposal_fast_exact_mask_jaccard") >= 0.0,
    "proposal_logits_bounded": maximum("proposal_logit_rms_mean") < 20.0,
    "residual_stage_routing_logged": maximum("residual_intrinsic_correction_scene_fraction") >= 0.0,
    "residual_uncertainty_loss": maximum("L_residual_action_uncertainty") > 0.0,
    "correctable_scene_support": maximum("certified_correctable_fraction") > 0.0,
    "residual_mean_lr_5x": has_mean_5x,
    "residual_variance_lr_2x": has_var_2x,
}
keys = (
    "action_family_enabled",
    "selector_exact_fraction",
    "L_deploy_select",
    "L_pair_full_action",
    "L_pair_full_winner_margin",
    "L_action_potential_teacher",
    "L_residual_winner_correction",
    "L_certified_residual_winner",
    "L_residual_boundary_margin_distill",
    "L_proposal_dense_winner",
    "L_exact_winner_flip_critical_proposal",
    "exact_winner_flip_critical_recall_topm",
    "exact_winner_flip_critical_atom_fraction",
    "exact_winner_flip_critical_scene_fraction",
    "exact_winner_flip_teacher_aligned_scene_fraction",
    "proposal_dense_topm_match",
    "proposal_dense_correct_scene_fraction",
    "proposal_fast_hab_topm_match",
    "proposal_global_topm_match",
    "proposal_exact_hab_topm_match",
    "proposal_exact_hab_fraction",
    "proposal_fast_exact_mask_jaccard",
    "proposal_logit_rms_mean",
    "L_proposal_logit_stability",
    "residual_curriculum_scale",
    "residual_proposal_failure_scene_fraction",
    "residual_intrinsic_correction_scene_fraction",
    "L_residual_action_uncertainty",
    "certified_correctable_fraction",
    "certified_robust_margin_mean",
)
report = {
    "checks": checks,
    "maxima": {key: maximum(key) for key in keys},
    "optimizer_lr_groups": lr_line,
}
print(json.dumps(report, indent=2, sort_keys=True))
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("V62 smoke failed: " + ", ".join(failed))
print("V62 training smoke: PASS")
PY
