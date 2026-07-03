# Recommended BDSE v7 commands

## Fast two-GPU training

```bash
torchrun --standalone --nproc_per_node=2 -m bdse.experiments.train \
  --config bdse/configs/v7_normalized_fast.yaml \
  --split train_boston train_pittsburgh train_singapore train_vegas_2 \
  --preprocessed-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/ \
  --max-scenarios 50000 \
  --max-scenarios-per-split 12500 \
  --batch-size 16 \
  --num-workers 12 \
  --prefetch-factor 1 \
  --device cuda \
  --amp \
  --val-preprocessed-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/ \
  --val-split val \
  --val-max-scenarios 512 \
  --val-mode loss \
  --val-every-n-epochs 1 \
  --best-metric val_loss \
  --best-metrics val_loss teacher_action_match full_interface_action_match teacher_regret \
  --epochs 20 \
  --save-every-n-epochs 0 \
  --log-file outputs/v7_train/bdse_v7_normalized_fast.train_log.jsonl \
  --output outputs/v7_train/bdse_v7_normalized_fast.pt
```

## Open-loop evaluation

```bash
python -m bdse.experiments.evaluate_open_loop \
  --config bdse/configs/v7_normalized_fast.yaml \
  --checkpoint outputs/v7_train/bdse_v7_normalized_fast.best.pt \
  --split val \
  --preprocessed-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/ \
  --max-scenarios 1000 \
  --device cuda \
  --output outputs/open_loop/open_loop_v7_normalized_fast_best.json \
  --per-sample-output outputs/open_loop/open_loop_v7_normalized_fast_best.jsonl
```

## Calibration

```bash
python -m bdse.experiments.calibrate \
  --config bdse/configs/v7_normalized_fast.yaml \
  --checkpoint outputs/v7_train/bdse_v7_normalized_fast.best.pt \
  --split val \
  --preprocessed-dir /data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/ \
  --max-scenarios 1000 \
  --device cuda \
  --delta 0.1 \
  --output outputs/calibration/calibration_bdse_v7_normalized_fast_best.json
```

## Closed-loop smoke test: 1 scenario, no metrics

```bash
python -m bdse.experiments.evaluate_closed_loop \
  --config bdse/configs/v7_bdse_pair_conditioned.yaml \
  --checkpoint outputs/v7_train/bdse_v7_normalized_fast.best.pt \
  --device cuda \
  --challenge closed_loop_nonreactive_agents \
  --metric-aggregator closed_loop_nonreactive_agents_weighted_average \
  --output-dir outputs/closed_loop/v7_debug_1 \
  --experiment-uid bdse_v7_debug_1 \
  --nuplan-module nuplan.planning.script.run_simulation \
  --scenario-builder nuplan \
  --worker single_machine_thread_pool \
  --hydra-full-error \
  --nuplan-data-root /data0/senzeyu2/dataset/nuplan \
  --nuplan-map-root /data0/senzeyu2/dataset/nuplan/maps \
  --nuplan-exp-root /data0/senzeyu2/dataset/nuplan/exp \
  --nuplan-db-root /data0/senzeyu2/dataset/nuplan/data/cache/val/ \
  -- \
  scenario_filter.limit_total_scenarios=1 \
  scenario_filter.shuffle=false \
  worker.max_workers=1 \
  run_metric=false
```

## Closed-loop 20-scenario BDSE test

```bash
python -m bdse.experiments.evaluate_closed_loop \
  --config bdse/configs/v7_bdse_pair_conditioned.yaml \
  --checkpoint outputs/v7_train/bdse_v7_normalized_fast.best.pt \
  --device cuda \
  --challenge closed_loop_nonreactive_agents \
  --metric-aggregator closed_loop_nonreactive_agents_weighted_average \
  --output-dir outputs/closed_loop/v7_bdse_20 \
  --experiment-uid bdse_v7_bdse_20 \
  --nuplan-module nuplan.planning.script.run_simulation \
  --scenario-builder nuplan \
  --worker single_machine_thread_pool \
  --hydra-full-error \
  --nuplan-data-root /data0/senzeyu2/dataset/nuplan \
  --nuplan-map-root /data0/senzeyu2/dataset/nuplan/maps \
  --nuplan-exp-root /data0/senzeyu2/dataset/nuplan/exp \
  --nuplan-db-root /data0/senzeyu2/dataset/nuplan/data/cache/val/ \
  -- \
  scenario_filter.limit_total_scenarios=20 \
  scenario_filter.shuffle=false \
  worker.max_workers=4 \
  run_metric=true
```

## Closed-loop baselines

Change only `--config` and output names:

- `bdse/configs/v7_baseline_base_only.yaml`
- `bdse/configs/v7_baseline_dense_full.yaml`
- `bdse/configs/v7_baseline_random_budget.yaml`
- `bdse/configs/v7_baseline_hard_safety_only.yaml`
- `bdse/configs/v7_baseline_proposal_top.yaml`
- `bdse/configs/v7_baseline_interaction_only.yaml`
- `bdse/configs/v7_baseline_rule_map_only.yaml`
- `bdse/configs/v7_baseline_risk_only.yaml`

For example:

```bash
python -m bdse.experiments.evaluate_closed_loop \
  --config bdse/configs/v7_baseline_random_budget.yaml \
  --checkpoint outputs/v7_train/bdse_v7_normalized_fast.best.pt \
  --device cuda \
  --challenge closed_loop_nonreactive_agents \
  --metric-aggregator closed_loop_nonreactive_agents_weighted_average \
  --output-dir outputs/closed_loop/v7_random_budget_20 \
  --experiment-uid bdse_v7_random_budget_20 \
  --nuplan-module nuplan.planning.script.run_simulation \
  --scenario-builder nuplan \
  --worker single_machine_thread_pool \
  --hydra-full-error \
  --nuplan-data-root /data0/senzeyu2/dataset/nuplan \
  --nuplan-map-root /data0/senzeyu2/dataset/nuplan/maps \
  --nuplan-exp-root /data0/senzeyu2/dataset/nuplan/exp \
  --nuplan-db-root /data0/senzeyu2/dataset/nuplan/data/cache/val/ \
  -- \
  scenario_filter.limit_total_scenarios=20 \
  scenario_filter.shuffle=false \
  worker.max_workers=4 \
  run_metric=true
```
