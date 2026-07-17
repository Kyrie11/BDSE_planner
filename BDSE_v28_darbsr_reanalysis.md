# BDSE v27 result reanalysis and v28 DA-RBSR proposal

## 1. What v27 changed relative to v26

The v27 T-RBSR direction is validated in the small CL20 setting but not in the larger CL50 setting.

### Open-loop selected diagnostics

| config | teacher_action_match | teacher_regret | budget_vs_full_match | pair_sign_acc_hard | pair_sign_acc_winner_rival | selected_hard_decisive_recall | selected_interaction_decisive_recall | fallback_would_trigger_rate | effective_query_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v26 fixed_budget | 0.193 | 12790.8 | 0.166 | 0.582 | 0.575 | 0.632 | 0.352 | 0.000 | 9155.0 |
| v26 safety_fallback | 0.193 | 12790.8 | 0.166 | 0.582 | 0.575 | 0.632 | 0.352 | 0.048 | 9155.0 |
| v26 LCB control | 0.203 | 12524.6 | 0.172 | 0.585 | 0.577 | 0.807 | 0.301 | 0.051 | 9155.0 |
| v27 fixed_budget | 0.208 | 12686.5 | 0.171 | 0.610 | 0.588 | 0.674 | 0.333 | 0.000 | 8693.3 |
| v27 safety_fallback | 0.208 | 12686.5 | 0.171 | 0.610 | 0.588 | 0.674 | 0.333 | 0.028 | 8693.3 |
| v27 LCB control | 0.209 | 12590.6 | 0.168 | 0.605 | 0.585 | 0.822 | 0.302 | 0.029 | 8693.3 |

Open-loop conclusion: v27 improves teacher_action_match, hard pair sign accuracy, winner-rival sign accuracy, hard recall, and query count. It slightly reduces interaction decisive recall and full-interface match.

### Closed-loop 20

| config | score | progress | route_progress | collision | TTC | drivable |
|---|---:|---:|---:|---:|---:|---:|
| v26 fixed_budget | 0.1394 | 0.55 | 0.459 | 0.525 | 0.50 | 0.75 |
| v26 safety_fallback | 0.1882 | 0.60 | 0.471 | 0.575 | 0.55 | 0.75 |
| v27 fixed_budget | 0.2287 | 0.55 | 0.457 | 0.675 | 0.65 | 0.80 |
| v27 safety_fallback | 0.1400 | 0.55/0.60 | 0.463 | 0.625 | 0.60 | 0.75 |
| v27 BBR/LCB | ~0.140 | 0.55 | 0.453--0.463 | 0.625 | 0.60 | 0.75 |

CL20 conclusion: v27 fixed-budget is the strongest CL20 configuration. Runtime safety tiering clearly helps fixed-budget safety and drivable compliance. Safety fallback/BBR/LCB are worse than fixed-budget on CL20, suggesting the recovery branch is now sometimes harmful.

### Closed-loop 50

| config | score | progress | route_progress | collision | TTC | comfort | drivable |
|---|---:|---:|---:|---:|---:|---:|---:|
| v26 fixed_budget | 0.2403 | 0.64 | 0.515 | 0.56 | 0.54 | 1.00 | 0.72 |
| v26 safety_fallback | 0.2601 | 0.64 | 0.519 | 0.58 | 0.56 | 1.00 | 0.72 |
| v27 fixed_budget | 0.2141 | 0.64 | 0.475 | 0.60 | 0.58 | 0.98 | 0.70 |
| v27 safety_fallback | 0.2341 | 0.64 | 0.478 | 0.62 | 0.60 | 0.98 | 0.70 |

CL50 conclusion: v27 safety fallback improves v27 fixed-budget by +0.020, mainly through +0.02 collision and +0.02 TTC. However, it is still below v26 safety fallback by -0.026. The dominant regression is drivable-area compliance and route progress.

## 2. Did the three v27 changes work?

1. **Tiered runtime safety flags:** partially worked. CL20 fixed-budget score rose from 0.1394 to 0.2287, collision from 0.525 to 0.675, TTC from 0.50 to 0.65, and drivable from 0.75 to 0.80. However, diagnostics still show a large selected_action_safety_flag rate: v27 fixed CL20 0.462, safety_fallback CL20 0.486, fixed CL50 0.374, safety_fallback CL50 0.356. So the all-candidates-unsafe failure is reduced but not eliminated.
2. **Hard/soft rule rerank:** partially worked. v27 safety_fallback CL50 improves fixed-budget by +0.020, mostly safety metrics. But it does not improve drivable compliance, and CL20 fallback branches underperform fixed-budget. Recovery/rerank is not yet consistently beneficial.
3. **More balanced training loss:** open-loop metrics improve overall: teacher_action_match +0.015, pair_sign_acc_hard +0.027, winner-rival sign +0.012, and query count lower. But CL50 score falls behind v26, meaning the finetuned model/rerank combination still over-optimizes runtime safety at the expense of route/drivable behavior.

## 3. Main remaining failure mode

v27 still uses the active runtime_safety_flags as tournament hard-filter flags. In tiered mode, if at least one soft-safe valid action exists, the soft-risk union becomes the active hard filter. That means many negotiable soft-risk actions are removed from the tournament rather than priced. This can push the planner to actions with worse route/drivable geometry, then recovery cannot fully repair the trajectory. The CL50 drivable drop to 0.70 is consistent with this failure.

The uploaded v27 closed-loop logs point to `outputs_v27/train/bdse_v27_trbsr.best.pt` for both CL20 and CL50. Therefore the uploaded CL20/CL50 logs do not preserve a clean v26-checkpoint runtime-only comparison; treat the current v27 result as the finetuned v27 result unless separate runtime-only logs are kept.

## 4. v28 DA-RBSR changes

v28 is Dual-tier Adaptive RBSR:

1. **Hard-constrained, soft-priced safety.** `runtime_safety.flag_mode=dual_tier` uses only hard violations for tournament hard masking and fallback triggering. Soft risk stays available to the tournament but is priced in rule/rerank costs and diagnostics.
2. **Drivable-aware recovery.** Route/lateral penalties are increased and progress/path reward is reduced in utility refinement and safe-progress recovery to address v27 CL50 drivable=0.70.
3. **Explicit hard/soft diagnostics.** Planner diagnostics now report hard_safe_action_available, soft_safe_action_available, active_safe_action_available, and component counts, so the next result can distinguish true hard infeasibility from soft-risk over-filtering.
4. **Execution control.** `run_v28_darbsr.sh` supports `RUN_MODE=open_loop|cl20|cl50|all`, `SKIP_OPEN_LOOP=1`, and 4-way parallel open-loop / CL20 / CL50 by default, with two jobs per GPU.

## 5. Recommended next run

Use v27 checkpoint for runtime-only v28 first:

```bash
export SKIP_TRAIN=1
export V28_CKPT=outputs_v27/train/bdse_v27_trbsr.best.pt
export RUN_MODE=open_loop
export OPEN_PARALLEL4=1
bash run_v28_darbsr.sh

export SKIP_TRAIN=1
export V28_CKPT=outputs_v27/train/bdse_v27_trbsr.best.pt
export RUN_MODE=cl20
export CL_PARALLEL4=1
export CL_WORKERS_PER_RUN=2
export RUN_CL50_ALL4=1
bash run_v28_darbsr.sh
```

If CL20 improves or at least keeps v27 fixed-budget score while reducing selected flagged/recovery rate, finetune:

```bash
export V25_CKPT_IN=outputs_v25/train/bdse_v25_dgcace.best.pt
export TRAIN_MAX_SCENARIOS=12000
export VAL_MAX_SCENARIOS=1000
export TRAIN_EPOCHS=5
export NPROC_PER_NODE=2
export RUN_MODE=all
bash run_v28_darbsr.sh
```

Then run CL50 only without rerunning open-loop. By default this runs all four CL50 variants in parallel, two jobs per GPU. Set `RUN_CL50_ALL4=0` to run only fixed_budget and safety_fallback:

```bash
export SKIP_TRAIN=1
export V28_CKPT=outputs_v28/train/bdse_v28_darbsr.best.pt
export RUN_MODE=cl50
export CL_WORKERS_PER_RUN=2
export RUN_CL50_ALL4=1
bash run_v28_darbsr.sh
```

Expected diagnostic movement: active_safe_avail_rate should approach hard_safe_avail_rate; selected_action_safety_flag_rate should fall below v27's 0.35--0.37 in CL50; drivable should recover from 0.70 toward at least 0.72; route progress should recover from 0.478 toward v26's 0.519.
