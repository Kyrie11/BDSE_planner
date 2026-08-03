# V60 external baseline runbook

The canonical commands are in `NEXT_COMMANDS_V60_EXTERNAL_BASELINES.txt` at the repository root.

## 1. Train four matched adapters

```bash
bash RUN_V60_EXTERNAL_BASELINES_MATCHED_TRAIN_2GPU.sh
```

## 2. Paired open-loop comparison

```bash
bash RUN_V60_EXTERNAL_OPEN_LOOP_2GPU.sh
```

## 3. Strict budget sweep

```bash
SWEEP_OUT=outputs/v60_external_compare/budget_sweep \
BUDGETS="8 16 24 32" \
bash RUN_V60_BUDGET_BASELINE_SWEEP.sh
```

## 4. Closed-loop

```bash
CL_PROCESSES_PER_MODEL=1 bash RUN_V60_EXTERNAL_CL20_2GPU.sh
CL_PROCESSES_PER_MODEL=1 bash RUN_V60_EXTERNAL_CL50_2GPU.sh
```

Increase `CL_PROCESSES_PER_MODEL` to 2 only after a small benchmark confirms that duplicate model copies do not cause GPU contention.
