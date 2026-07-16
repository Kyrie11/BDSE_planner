# External baseline commands

Set these paths once:

```bash
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/
export BDSE_VAL_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/
export NUPLAN_ROOT=/data0/senzeyu2/dataset/nuplan
mkdir -p outputs/external outputs/open_loop outputs/closed_loop
```

## Trainable external baselines

```bash
for name in gameformer dtpp plantf pluto; do
  cfg="bdse/configs/external_${name}_budgeted.yaml"
  out="outputs/external/${name}_budgeted.pt"
  python -m bdse.external_baselines.train \
    --config "$cfg" \
    --split train_boston train_pittsburgh train_singapore train_vegas_2 \
    --preprocessed-dir "$BDSE_TRAIN_CACHE" \
    --max-scenarios 50000 \
    --max-scenarios-per-split 12500 \
    --batch-size 32 \
    --num-workers 12 \
    --device cuda \
    --amp \
    --val-preprocessed-dir "$BDSE_VAL_CACHE" \
    --val-split val \
    --val-max-scenarios 1000 \
    --val-mode loss \
    --val-every-n-epochs 1 \
    --epochs 30 \
    --log-file "outputs/external/${name}.train_log.jsonl" \
    --output "$out"
done
```

Optional indirect PPAD adapter:

```bash
python -m bdse.external_baselines.train \
  --config bdse/configs/external_ppad_budgeted.yaml \
  --split train_boston train_pittsburgh train_singapore train_vegas_2 \
  --preprocessed-dir "$BDSE_TRAIN_CACHE" \
  --max-scenarios 50000 \
  --max-scenarios-per-split 12500 \
  --batch-size 32 \
  --num-workers 12 \
  --device cuda \
  --amp \
  --val-preprocessed-dir "$BDSE_VAL_CACHE" \
  --val-split val \
  --val-max-scenarios 1000 \
  --epochs 30 \
  --log-file outputs/external/ppad.train_log.jsonl \
  --output outputs/external/ppad_budgeted.pt
```

## Open-loop evaluation

```bash
python -m bdse.experiments.evaluate_open_loop \
  --config bdse/configs/external_pdm_closed_budgeted_fast_cl.yaml \
  --split val \
  --preprocessed-dir "$BDSE_VAL_CACHE" \
  --max-scenarios 1000 \
  --device cuda \
  --output outputs/open_loop/open_loop_external_pdm_closed.json \
  --per-sample-output outputs/open_loop/open_loop_external_pdm_closed.jsonl

for name in gameformer dtpp plantf pluto; do
  python -m bdse.experiments.evaluate_open_loop \
    --config "bdse/configs/external_${name}_budgeted_fast_cl.yaml" \
    --checkpoint "outputs/external/${name}_budgeted.best.pt" \
    --split val \
    --preprocessed-dir "$BDSE_VAL_CACHE" \
    --max-scenarios 1000 \
    --device cuda \
    --output "outputs/open_loop/open_loop_external_${name}.json" \
    --per-sample-output "outputs/open_loop/open_loop_external_${name}.jsonl"
done
```

## Closed-loop 20-scenario check

```bash
python -m bdse.experiments.evaluate_closed_loop \
  --config bdse/configs/external_pdm_closed_budgeted_fast_cl.yaml \
  --device cuda \
  --challenge closed_loop_nonreactive_agents \
  --metric-aggregator closed_loop_nonreactive_agents_weighted_average \
  --output-dir outputs/closed_loop/external_pdm_closed_20 \
  --experiment-uid external_pdm_closed_20 \
  --nuplan-module nuplan.planning.script.run_simulation \
  --scenario-builder nuplan \
  --worker single_machine_thread_pool \
  --hydra-full-error \
  --nuplan-data-root "$NUPLAN_ROOT" \
  --nuplan-map-root "$NUPLAN_ROOT/maps" \
  --nuplan-exp-root "$NUPLAN_ROOT/exp" \
  --nuplan-db-root "$NUPLAN_ROOT/data/cache/val/" \
  -- \
  scenario_filter.limit_total_scenarios=20 \
  scenario_filter.shuffle=false \
  worker.max_workers=4 \
  run_metric=true

for name in gameformer dtpp plantf pluto; do
  python -m bdse.experiments.evaluate_closed_loop \
    --config "bdse/configs/external_${name}_budgeted_fast_cl.yaml" \
    --checkpoint "outputs/external/${name}_budgeted.best.pt" \
    --device cuda \
    --challenge closed_loop_nonreactive_agents \
    --metric-aggregator closed_loop_nonreactive_agents_weighted_average \
    --output-dir "outputs/closed_loop/external_${name}_20" \
    --experiment-uid "external_${name}_20" \
    --nuplan-module nuplan.planning.script.run_simulation \
    --scenario-builder nuplan \
    --worker single_machine_thread_pool \
    --hydra-full-error \
    --nuplan-data-root "$NUPLAN_ROOT" \
    --nuplan-map-root "$NUPLAN_ROOT/maps" \
    --nuplan-exp-root "$NUPLAN_ROOT/exp" \
    --nuplan-db-root "$NUPLAN_ROOT/data/cache/val/" \
    -- \
    scenario_filter.limit_total_scenarios=20 \
    scenario_filter.shuffle=false \
    worker.max_workers=4 \
    run_metric=true
done
```
