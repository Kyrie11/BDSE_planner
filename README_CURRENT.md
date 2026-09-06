# BDSE current status — V64.3.56 converged / benchmark-ready

## Scientific state

Internal TRAIN algorithm search is **closed by preregistered falsification** at V64.3.56. The final realized constraint-process oracle passed the paired deployment gate but did not pass the required fold-stable incremental-identification gate over V55 (2/5 < 4/5). The predicted t0 branch was therefore not evaluated. Do not create V57/V58 feature/state variants or tune V56 after seeing this result.

The external benchmark uses the strongest fully t0-deployable frozen backbone:

**bounded evidence -> complete EAF -> frozen full-set RSMR extremal proposal -> same proposal/incumbent no-fallback operator.**

V54–V56 post-intervention mediators are diagnostic only and are never used by the external runtime planner.

## Primary budget protocol

- **Primary paper comparison:** B=16, M=24.
- External trainable adapters (GameFormer-/DTPP-/PlanTF-/PLUTO-inspired): independently trained and validated at B=8,16,24.
- BDSE: the learned method was developed/frozen at B=16. B=8/B=24 runs reuse the same frozen EAF/RSMR learned artifacts and are reported as **cross-budget interface robustness**, not budget-specific retraining.
- A truly matched-training BDSE B8/B24 curve would require a new preregistered EAF retrain + RSMR refit per budget before test inspection; changing only one YAML field is not equivalent.

## Closed-loop argv transport repair

The previous benchmark launcher could fail before Python startup with:

`OSError: [Errno 7] Argument list too long`

because all 66,671 scenario tokens were serialized into one Hydra argv string. The repaired runner now transports ordered scenario tokens (and a potentially long raw-DB file subset) through JSON manifests with SHA verification. The exact Hydra token override is reconstructed inside the isolated evaluator process. Manifest-backed nuPlan execution uses an in-process `runpy` transport to avoid a second `execve` with the same oversized argument.

This preserves the full ordered population and official metric aggregation; it is not scenario subsampling or shard averaging.

## Metric safety

All formal closed-loop runs use `bdse.tools.nuplan_metric_safe_run_simulation`. Planner/simulation workers remain parallel, but the stateful nuPlan metric callback is serialized within each process. Resume rejects legacy rows without the required metric-safety provenance.

## Main commands

External budget-specific training:

```bash
GPUS=0,1 BUDGETS="8 16 24" \
  bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

Primary matched B16 benchmark (BDSE + all controlled baselines):

```bash
GPUS=0,1 BUDGETS="16" CL_WORKERS_PER_JOB=4 \
  bash RUN_V64_3_56_CONVERGED_CLOSED_LOOP_B8_B16_B24_2GPU.sh
```

Full B8/B16/B24 sweep after B16 is frozen/completed:

```bash
GPUS=0,1 BUDGETS="8 16 24" CL_WORKERS_PER_JOB=4 \
  bash RUN_V64_3_56_CONVERGED_CLOSED_LOOP_B8_B16_B24_2GPU.sh
```

BDSE-only frozen cross-budget robustness:

```bash
GPUS=0,1 BUDGETS="8 16 24" CL_WORKERS_PER_JOB=4 \
  bash RUN_V64_3_56_OWN_FROZEN_B8_B16_B24_CLOSED_LOOP_2GPU.sh
```
