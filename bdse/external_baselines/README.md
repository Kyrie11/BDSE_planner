# Budget-compatible external baselines

This package contains external baseline adapters for the BDSE paper setting.
All variants use the same runtime-only inputs, candidate bank, and evidence atoms
as BDSE.  The external model consumes at most `external_baseline.budget` evidence
atoms/tokens and outputs a candidate cost `J0`; deployment then uses
`planner.baseline_mode: external_policy` for budget/query accounting.

Implemented variants:

- `pdm_closed`: rule/PDM-Closed style centerline-progress-comfort-safety scorer;
  no training checkpoint is required.
- `gameformer`: scene Transformer plus level-k interaction refinement over the
  selected evidence tokens.
- `dtpp`: tree-policy analogue with maneuver-level branch cost plus candidate
  refinement cost.
- `plantf`: PlanTF-style token Transformer scorer with optional state dropout.
- `pluto`: PLUTO-style PlanTF extension with longitudinal/lateral decomposed
  cost heads.
- `ppad`: iterative prediction-planning update adapter; included as an optional
  indirect baseline.

Example training:

```bash
python -m bdse.external_baselines.train \
  --config bdse/configs/external_gameformer_budgeted.yaml \
  --split train_boston train_pittsburgh train_singapore train_vegas_2 \
  --preprocessed-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/ \
  --max-scenarios 50000 \
  --max-scenarios-per-split 12500 \
  --batch-size 32 \
  --num-workers 12 \
  --device cuda \
  --amp \
  --val-preprocessed-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/ \
  --val-split val \
  --val-max-scenarios 1000 \
  --epochs 30 \
  --log-file outputs/external/gameformer.train_log.jsonl \
  --output outputs/external/gameformer_budgeted.pt
```

Example open-loop evaluation:

```bash
python -m bdse.experiments.evaluate_open_loop \
  --config bdse/configs/external_gameformer_budgeted_fast_cl.yaml \
  --checkpoint outputs/external/gameformer_budgeted.best.pt \
  --split val \
  --preprocessed-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/ \
  --max-scenarios 1000 \
  --device cuda \
  --output outputs/open_loop/open_loop_external_gameformer.json \
  --per-sample-output outputs/open_loop/open_loop_external_gameformer.jsonl
```

For PDM-Closed, omit `--checkpoint` because it is deterministic and rule-based.
