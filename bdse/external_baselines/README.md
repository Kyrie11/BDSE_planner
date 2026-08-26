# Matched fixed-budget external baseline adapters

This package implements controlled **budget-compatible adapters, not verbatim official reproductions**. For the full fidelity/fairness audit, protocol rationale, and run commands, see [`../../EXTERNAL_BASELINE_FAIRNESS_AUDIT.md`](../../EXTERNAL_BASELINE_FAIRNESS_AUDIT.md).

| Local key | Required paper label | Status |
|---|---|---|
| `gameformer` | GameFormer-inspired budget adapter | iterative Transformer/game-style reasoning; no official joint multimodal agent prediction stack |
| `dtpp` | DTPP-inspired budget adapter | branch/tree-style candidate reasoning; no official scenario tree / joint conditional prediction-cost pipeline |
| `plantf` | PlanTF-inspired budget adapter | expert imitation + attention-token SDE; official state perturbation/object-token feature pipeline still omitted |
| `pluto` | PLUTO-inspired budget adapter | expert imitation + longitudinal/lateral decomposition; official query architecture, auxiliary loss, CIL and augmentations omitted |
| `pdm_closed_style` | PDM-Closed-style budget scorer (NOT official PDM-Closed) | deterministic static scorer on the common candidate bank; no native IDM proposal/rollout/PDM scoring stack |

## Strict comparison protocol

- Train: `/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/{train_boston,train_pittsburgh,train_singapore,train_vegas_2}`.
- Validation/model selection: `/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/val`.
- Final test token source: `/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2/public_set_test`.
- Fixed upstream proposal pool: `M=24` for all B.
- Report B in `{8,16,24}` with fallback expansion disabled.
- Train one checkpoint per B for each learnable adapter.
- Closed loop pins the same ordered `scenario_token` manifest for every system/B.
- nuPlan simulation reads raw `.db` logs from `NUPLAN_TEST_DB_ROOT`; the BDSE NPZ test cache is **not** a DB root.

## Run sequence

```bash
python -m bdse.tools.export_custom_dataset_token_manifests \
  --train-root /data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2 \
  --val-root /data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2 \
  --test-root /data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2 \
  --output-dir outputs/fairness_manifests

bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
bash RUN_FAIR_EXTERNAL_OPEN_LOOP_TEST_B8_B16_B24_2GPU.sh

export NUPLAN_ROOT=/data0/senzeyu2/dataset/nuplan
export NUPLAN_TEST_DB_ROOT=/path/to/raw/test/db/root
bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
bash RUN_V64_3_48_OWN_FIXED_BUDGET_CLOSED_LOOP_TEST_2GPU.sh
```

The learnable adapters are trained by logged-expert imitation projected onto the common candidate bank. BDSE teacher future quantities are not used as the external planner action/cost target. The common oracle evidence mask remains only as supervision for the imposed evidence-selector interface.
