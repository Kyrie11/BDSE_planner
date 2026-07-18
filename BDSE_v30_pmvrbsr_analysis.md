# v29 AMV-RBSR result analysis and v30 PMV-RBSR update

## 1. Current v29 result summary

The uploaded v29 run contains open-loop JSON summaries, CL20 metrics, CL50 metrics, and closed-loop diagnostic JSONL logs.

### CL20

| Variant | score | progress | route progress | collision | TTC | drivable | direction |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_budget | 0.225588 | 0.55 | 0.443882 | 0.575 | 0.55 | 0.80 | 0.90 |
| safety_fallback | 0.311154 | 0.50 | 0.349826 | 0.650 | 0.65 | 0.85 | 0.95 |
| bbr_scur | 0.311154 | 0.50 | 0.349826 | 0.650 | 0.65 | 0.85 | 0.95 |
| lcb_control | **0.311161** | 0.50 | 0.342465 | 0.650 | 0.65 | **0.90** | **1.00** |

### CL50

| Variant | score | progress | route progress | collision | TTC | comfort | speed | drivable | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_budget | 0.215668 | **0.653061** | **0.491418** | 0.571429 | 0.551020 | **0.959184** | 0.995867 | 0.693878 | 1.00 |
| safety_fallback | 0.303574 | 0.620000 | 0.389429 | 0.700000 | 0.700000 | 0.920000 | 0.999118 | 0.760000 | 1.00 |
| bbr_scur | 0.303574 | 0.620000 | 0.389429 | 0.700000 | 0.700000 | 0.920000 | 0.999118 | 0.760000 | 1.00 |
| lcb_control | **0.323453** | 0.620000 | 0.382102 | **0.720000** | **0.720000** | 0.940000 | 0.999114 | **0.780000** | 1.00 |

## 2. Did v29 meet the previous acceptance criteria?

Previous criteria:

- drivable >= 0.72;
- route progress >= v28 safety fallback around 0.498;
- collision/TTC clearly higher than v28 0.56/0.54;
- final_action_safety_flag_rate clearly lower than v28;
- CL50 score higher than v28 0.2166 and v27 0.2341.

Result:

| Criterion | v29 result | Pass? |
|---|---:|---|
| drivable >= 0.72 | safety/bbr 0.76, lcb 0.78 | yes |
| route progress >= 0.498 | safety/bbr 0.389, lcb 0.382 | no |
| collision/TTC higher than v28 | safety/bbr 0.70/0.70, lcb 0.72/0.72 | yes |
| final_action_safety_flag_rate lower | safety/bbr 0.3147, lcb 0.3187 | yes |
| CL50 score > v27/v28 | best 0.323453 | yes |

Conclusion: v29 worked overall, and the CL50 score improved substantially. However, it did not solve the route-progress regression. The best v29 configuration is safer and more drivable, but too conservative in route advancement.

## 3. Did the four v29 core changes work?

### 3.1 Adaptive dual-tier runtime constraint

Worked. In CL50, safety_fallback/bbr used the soft tier for 1427/1500 replans and the hard tier for 73/1500 replans. active_safe_action_available was 0.6853, and final_action_safety_flag_rate was 0.3147. lcb_control was similar: active_safe_action_available 0.6813 and final_action_safety_flag_rate 0.3187.

Interpretation: v29 reduced the unsafe final-action rate and made hard filter useful in about 68% of replans. This is better than the v27/v28 regime where many replans were either over-flagged or hard-only priced.

Remaining limitation: about 31%--32% of replans still have no active-safe candidate. Those all-flagged replans dominate the recovery behavior.

### 3.2 Continuous min-violation risk scoring

Worked, but the current weighted-sum use is too conservative. In all-flagged cases, final hard risk remains high. For safety_fallback, final_action_hard_risk has median about 306.7 over flagged final actions. Component counts show all-flagged/final-flagged replans often have both off-route-hard and agent-hard candidates, but off-route-hard is especially common: in the final-flagged subset, off_route_hard_count has median about 20.

Interpretation: continuous risk ranking helps safety, but it is currently over-dominating progress, especially when off-route risk is used as a large absolute penalty.

### 3.3 Risk-aware rule rerank / safe-progress recovery

Worked for safety, but caused the main side effect. CL50 safety_fallback improves collision/TTC to 0.70/0.70 and drivable to 0.76, but route progress drops to 0.389. lcb_control is even safer, 0.72/0.72, with drivable 0.78, but route progress is only 0.382.

The diagnostic logs show fallback_triggered_rate around 0.325--0.336 in the safety variants. Most triggered cases go to base+safe_progress, not base+rule_rerank:

- safety_fallback: base+safe_progress 472, base+rule_rerank 10;
- lcb_control: base+safe_progress 478, base+rule_rerank 15.

Interpretation: the post-tournament recovery branch is now the main controller for difficult scenes. It needs a more progress-preserving all-flagged policy.

### 3.4 Final-action diagnostics

Worked. v29 logs now expose final_action_safety_flag, final_action_hard_flag, final_action_soft_flag, final_action_hard_risk, final_action_soft_risk, active_flag_tier, and safe-action availability. This makes it possible to distinguish pre-recovery tournament flags from the actual deployed trajectory.

## 4. Main remaining bottleneck

v29's bottleneck is no longer drivable or TTC. It is the route-progress/safety trade-off in all-flagged replans.

The current recovery cost is still close to a weighted sum:

```text
cost = lateral + final_lateral - progress + hard_risk_weight * hard_risk + soft_risk_weight * soft_risk
```

When all candidates are flagged, this can select a very low-risk but low-progress trajectory even if another candidate has almost the same risk and much better route progress. This explains why fixed_budget has route progress 0.491 but poor safety, whereas lcb_control has safety 0.72/0.72 and drivable 0.78 but route progress only 0.382.

## 5. v30 update: PMV-RBSR

The v30 update is named:

```text
PMV-RBSR: Pareto Min-Violation Risk-Bounded Safety Recovery
```

It keeps the successful v29 ideas and strengthens the part that caused route-progress regression.

### 5.1 New core idea

When a safe candidate exists:

```text
use the existing safe-first active hard/soft mask.
```

When all candidates are flagged:

```text
1. protect minimum agent-proximity violation first;
2. keep a Pareto band around the minimum continuous hard risk;
3. optionally keep a soft-risk band;
4. inside this near-min-violation set, recover progress using lateral/progress utility.
```

This is different from v29's single scalar min-violation penalty. v29 used continuous risk as a large cost; v30 uses continuous risk to form a feasible Pareto set and only then optimizes progress.

### 5.2 Code changes

Changed files:

- `bdse/planner/fallback.py`
  - added risk component outputs: `hard_agent`, `soft_agent`, `hard_off_route`, `soft_off_route`;
  - added Pareto-band all-flagged recovery in `conservative_fallback_action()`;
  - changed all-flagged risk cost from absolute risk to excess risk over the Pareto pool minimum.

- `bdse/planner/nuplan_planner.py`
  - added final-action component diagnostics: `final_action_hard_agent_risk`, `final_action_hard_offroute_risk`.

- `bdse/configs/v30_bdse_pmvrbsr_*.yaml`
  - increased progress/path reward inside recovery and utility refinement;
  - slightly relaxed hard off-route margin from 1.6m to 2.2m;
  - reduced absolute off-route risk dominance;
  - added Pareto-band recovery parameters.

- `bdse/tests/test_v30_pareto_progress_recovery.py`
  - verifies that when all candidates are flagged, v30 chooses a high-progress candidate inside a near-minimum violation band instead of blindly choosing the absolute minimum-risk low-progress one.

Validation:

```text
47 passed, 2 warnings
```

## 6. Expected v30 outcome

Compared with v29 lcb_control:

- score should remain above 0.30;
- collision/TTC should stay near 0.70 rather than falling back to v28's 0.56/0.54;
- drivable should remain >= 0.74;
- route progress should recover from 0.382 toward 0.45--0.49;
- final_action_safety_flag_rate should remain around or below 0.32.

The key regression to watch is drivable. If drivable falls below 0.72, v30 has relaxed off-route/progress too much. If route progress remains below 0.40, the Pareto band is still too narrow or progress utility is still too weak.

## 7. Recommended execution

### Runtime-only open-loop using v29 checkpoint

```bash
export SKIP_TRAIN=1
export V30_CKPT=outputs_v29/train/bdse_v29_amvrbsr.best.pt
export RUN_MODE=open_loop
export OPEN_PARALLEL4=1
bash run_v30_pmvrbsr.sh
```

### Runtime-only CL20

```bash
export SKIP_TRAIN=1
export V30_CKPT=outputs_v29/train/bdse_v29_amvrbsr.best.pt
export RUN_MODE=cl20
export CL_PARALLEL4=1
export CL_WORKERS_PER_RUN=2
bash run_v30_pmvrbsr.sh
```

### Finetune

```bash
export V25_CKPT_IN=outputs_v25/train/bdse_v25_dgcace.best.pt
export TRAIN_MAX_SCENARIOS=12000
export VAL_MAX_SCENARIOS=1000
export TRAIN_EPOCHS=5
export NPROC_PER_NODE=2
export RUN_MODE=all
bash run_v30_pmvrbsr.sh
```

### CL50 after finetune

```bash
export SKIP_TRAIN=1
export V30_CKPT=outputs_v30/train/bdse_v30_pmvrbsr.best.pt
export RUN_MODE=cl50
export RUN_CL50_ALL4=1
export CL_WORKERS_PER_RUN=2
bash run_v30_pmvrbsr.sh
```

## 8. If v30 still has a gap

If safety stays high but route progress remains low, increase:

```yaml
fallback.safe_progress_recovery.progress_quantile_floor: 0.45
fallback.safe_progress_recovery.progress_weight: 0.72
fallback.safe_progress_recovery.hard_risk_abs_margin: 36.0
```

If route progress improves but drivable drops below 0.72, revert:

```yaml
runtime_safety.hard_off_route_margin_m: 1.8
fallback.safe_progress_recovery.lateral_weight: 1.75
fallback.safe_progress_recovery.lateral_final_weight: 0.90
```

If collision/TTC drops, tighten the agent-proximity Pareto gate:

```yaml
fallback.safe_progress_recovery.agent_risk_abs_margin: 0.03
runtime_safety.risk_hard_agent_weight: 10.0
```
