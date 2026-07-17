# BDSE v29 CDR-RBSR summary

## Motivation from v27 results

v27 T-RBSR did solve part of the v26 failure: fixed-budget CL20 recovered to 0.2287 with progress 0.55, collision 0.675, TTC 0.65, and drivable 0.80. This means low-delta acceptance, fixed evidence budget, and the runtime hard filter are useful.

However, v27 safety_fallback became worse than fixed_budget: CL20 score dropped to 0.1400 with collision 0.625, TTC 0.60, and drivable 0.75. Diagnostics show the selected runtime flag rate is still high: about 0.462 in fixed_budget CL20, 0.486 in safety_fallback CL20, 0.374 in fixed_budget CL50, and 0.356 in safety_fallback CL50. The hard filter availability improved versus v26, but it did not eliminate flagged selections, and route/drivable compliance remains the next bottleneck.

## v29 algorithm: CDR-RBSR

CDR-RBSR = Certificate-Drivable Risk-Bounded Safety Recovery.

v29 keeps the useful v26/v27 design choices:

1. low-delta acceptance;
2. fixed evidence budget with no expanded evidence stage;
3. hard/soft runtime risk separation;
4. critical hard evidence quota and deployment-gate-aware certificate losses.

It adds a route-aware, certificate-preserving final guard:

- `fallback.post_certificate_route_guard.enabled: true`
- The guard is applied even when fallback is disabled or not triggered.
- It does not query more evidence and does not re-run selector/tournament.
- It only reselects inside the current certificate score band / top-k actions.
- It uses runtime route distance and hard/soft risk pricing from `rule_based_runtime_scores`.
- It is designed to repair the v27 CL50 drivable bottleneck without returning to v25 conservative stop-like fallback.

## Modified files

- `bdse/planner/nuplan_planner.py`
  - Added `post_certificate_route_guard` after tournament and before trajectory output.
  - Logs `post_certificate_route_guard` diagnostics.
- `bdse/configs/v29_bdse_cdrbsr_*_fast_cl.yaml`
  - New v29 closed-loop/runtime configs.
  - Uses dual-tier hard/soft risk.
  - Enables certificate-preserving route guard.
- `bdse/configs/v29_bdse_cdrbsr_train.yaml`
  - New v29 finetune config based on v28.
- `run_v29_cdrbsr.sh`
  - Uses export-style parameters.
  - Adds `RUN_STAGE=open_loop|cl20|cl50|all` to avoid the old `RUN_CL50` ambiguity.
  - Runs 4 open-loop or CL20 evaluations concurrently by default: two jobs on GPU 0 and two jobs on GPU 1.

## Validation

Local static/unit validation:

```text
44 passed, 2 warnings
```

## Recommended execution

Runtime-only open-loop with v27 checkpoint:

```bash
export SKIP_TRAIN=1
export V29_CKPT=outputs_v27/train/bdse_v27_trbsr.best.pt
export RUN_STAGE=open_loop
export OPEN_PARALLEL4=1
bash run_v29_cdrbsr.sh
```

Runtime-only CL20 with v27 checkpoint:

```bash
export SKIP_TRAIN=1
export V29_CKPT=outputs_v27/train/bdse_v27_trbsr.best.pt
export RUN_STAGE=cl20
export CL_PARALLEL4=1
export CL_WORKERS_PER_RUN=2
bash run_v29_cdrbsr.sh
```

Finetune from v25 checkpoint:

```bash
export V25_CKPT_IN=outputs_v25/train/bdse_v25_dgcace.best.pt
export TRAIN_MAX_SCENARIOS=12000
export VAL_MAX_SCENARIOS=1000
export TRAIN_EPOCHS=5
export NPROC_PER_NODE=2
bash run_v29_cdrbsr.sh
```

CL50 with v29 checkpoint:

```bash
export SKIP_TRAIN=1
export V29_CKPT=outputs_v29/train/bdse_v29_cdrbsr.best.pt
export RUN_STAGE=cl50
export CL_WORKERS_PER_RUN=2
bash run_v29_cdrbsr.sh
```
