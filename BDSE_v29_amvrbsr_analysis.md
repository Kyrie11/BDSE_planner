# v28 结果复盘与 v29 AMV-RBSR 修改说明

## 1. v28 是否达到上一轮预期

上一轮对 CL50 的预期是：

- selected_action_safety_flag_rate 低于 v27 的 0.35–0.37；
- drivable 从 0.70 回升到至少 0.72；
- route progress 从 0.478 向 v26 的 0.519 靠近；
- 保留 v27 的 collision/TTC 提升。

当前 v28 CL50 的实际结果：

| 配置 | score | progress | route progress | collision | TTC | drivable |
|---|---:|---:|---:|---:|---:|---:|
| v28 fixed-budget | 0.2163 | 0.64 | 0.4937 | 0.56 | 0.54 | 0.72 |
| v28 safety-fallback | 0.2166 | 0.64 | 0.4983 | 0.56 | 0.54 | 0.72 |
| v28 BBR-SCUR | 0.2166 | 0.64 | 0.4983 | 0.56 | 0.54 | 0.72 |
| v28 LCB-control | 0.2136 | 0.64 | 0.4998 | 0.60 | 0.58 | 0.72 |

结论：v28 只部分达到预期。

1. drivable 目标达到了：0.70 → 0.72。
2. route progress 明显改善：0.478 → 0.498，已经向 v26 的 0.519 靠近。
3. selected_action_safety_flag_rate 没有明确低于目标区间。CL50 中 fixed-budget 为 0.384，safety/BBR 为 0.367，LCB 为 0.369，基本仍在 0.35–0.37 附近。
4. 没有保留 v27 的 collision/TTC。v27 safety-fallback 约为 collision=0.62、TTC=0.60，而 v28 safety-fallback 降为 0.56/0.54；LCB-control 保留得最好，为 0.60/0.58，但总分仍低。

因此，v28 的 DA-RBSR 对比 v27 并没有让 CL50 综合闭环结果更好。它改善了 drivable 与 route progress，但牺牲了 collision/TTC，最终 best score 约 0.2166，低于上一轮 v27 safety-fallback 的约 0.2341。

## 2. v28 起作用的部分

### 2.1 Drivable-aware utility/recovery 起作用

v27 的主要问题之一是 drivable=0.70；v28 将 drivable 提回 0.72，同时 route progress 从 0.478 提到 0.498。说明降低过强 progress/path reward、提高 lateral/final-lateral 代价的方向是对的。

### 2.2 固定 evidence budget 仍然成立

v28 open-loop 的 effective_query_count 约 8518，低于 v27 约 8693，同时 teacher_action_match 保持在 0.206 左右，说明没有靠扩 evidence stage 换结果，fixed-budget 叙事仍可保留。

### 2.3 LCB-control 保留安全性最好

v28 LCB-control 的 CL50 collision/TTC 为 0.60/0.58，比 fixed/safety 的 0.56/0.54 更接近 v27。这说明 LCB/critical hard evidence 方向仍有价值，但当前 LCB 分支的综合 score 被 comfort 或局部 route/drivable 行为抵消。

## 3. v28 仍存在的问题与来源

### 3.1 hard-only constraint 过于宽松

v27 的 tiered 策略会在 soft-safe 候选存在时把 soft risk 纳入 hard filter；v28 改成 hard-only constraint 后，确实改善了 drivable，但也让一批 soft interaction risk 候选重新进入 tournament。结果是 collision/TTC 明显下降。

CL50 诊断显示：

| 配置 | hard-safe available | soft-safe available | selected safety flag |
|---|---:|---:|---:|
| fixed-budget | 0.616 | 0.587 | 0.384 |
| safety-fallback | 0.633 | 0.593 | 0.367 |
| LCB-control | 0.631 | 0.589 | 0.369 |

hard-safe 与 soft-safe 的差距并不大，说明完全 hard-only 没必要；可以在 soft-safe pool 充足时重新启用 soft filter，以恢复 v27 的 collision/TTC 优势。

### 3.2 所有候选都 flagged 时，v28 缺少“最小违规”排序

v28 中仍有约 36%–38% replan 没有 active-safe 候选。此时 hard filter 必然失效，safe_progress fallback 只能在 flagged candidates 里选。v28 的 fallback 对 flagged action 基本是统一 unsafe_penalty，缺少连续风险强度：

- 离 agent 0.1m 与 0.8m 都是 hard-agent flag；
- 轻微 route excess 与严重 off-route 都是 hard-off-route flag；
- all-flagged case 中 unsafe penalty 对所有候选近似相同。

这会导致 recovery 按 lateral/progress utility 选，而不是按最小 collision/TTC risk 选，解释了 v28 drivable 变好但 collision/TTC 下降。

### 3.3 诊断字段有一个重要偏差

v28 日志里的 `tournament.selected_action_safety_flag` 是 tournament 选择时的 action flag，可能发生在 rule_rerank / safe_progress 修改 action 之前。也就是说它不一定等于最终执行 action 的 safety flag。v29 已修复该诊断：新增 final-action hard/soft flag 与 continuous risk，并把 summary 中的 safety flag 统计改为 final action。

## 4. v29 AMV-RBSR 的算法修改

下一版命名为：

**AMV-RBSR: Adaptive Min-Violation Risk-Bounded Safety Recovery**

核心不是继续把 drivable 权重调大，而是在 v28 的基础上补回 v27 的 safety/TTC：

### 修改 1：Adaptive dual-tier runtime constraint

新增 `runtime_safety.flag_mode: adaptive_dual_tier`。

逻辑：

```text
if soft-safe candidate pool 足够大:
    使用 soft flags 作为 tournament constraint
else:
    使用 hard flags 作为 tournament constraint
```

这样比 v27 更稳，因为 soft filter 不会在 dense interaction 中把候选全部标死；也比 v28 更安全，因为 soft-safe 足够时不会让 soft-risk action 进入 tournament。

### 修改 2：continuous min-violation risk scoring

新增 `runtime_risk_scores()`，对每个 candidate 输出连续风险：

- hard agent proximity deficit；
- hard off-route excess；
- red-light violation；
- soft agent/off-route risk。

当所有候选都 flagged 时，不再把所有 unsafe candidate 等价处理，而是选择 lowest continuous violation risk 的候选。

### 修改 3：risk-aware rule rerank / safe-progress recovery

`rule_based_runtime_scores()` 与 `conservative_fallback_action()` 都加入 continuous risk 项。恢复策略变为：

```text
certificate / hard constraint first
then continuous hard risk
then soft risk
then drivable/lateral utility
then progress
```

这针对 v28 的核心问题：drivable 已改善，但 all-flagged recovery 没有最小风险意识。

### 修改 4：final-action diagnostics

新增字段：

- `final_action_safety_flag`
- `final_action_hard_flag`
- `final_action_soft_flag`
- `final_action_hard_risk`
- `final_action_soft_risk`
- `active_flag_tier`
- `min_hard_risk`
- `min_soft_risk`

后续 CL20/CL50 可以直接判断最终执行 action 是否真的安全，而不是只看 pre-recovery tournament action。

## 5. 下一步实验建议

### Step 1：runtime-only open-loop

使用 v28 checkpoint，不训练，验证 v29 runtime guard 不损害 open-loop：

```bash
export SKIP_TRAIN=1
export V29_CKPT=outputs_v28/train/bdse_v28_darbsr.best.pt
export RUN_MODE=open_loop
export OPEN_PARALLEL4=1
bash run_v29_amvrbsr.sh
```

### Step 2：runtime-only CL20

```bash
export SKIP_TRAIN=1
export V29_CKPT=outputs_v28/train/bdse_v28_darbsr.best.pt
export RUN_MODE=cl20
export CL_PARALLEL4=1
export CL_WORKERS_PER_RUN=2
bash run_v29_amvrbsr.sh
```

优先观察：

- final_action_safety_flag_rate 是否低于 v28；
- active_flag_tier 中 soft 占比是否明显出现；
- collision/TTC 是否向 v27 恢复；
- drivable 是否保持 0.72 附近。

### Step 3：如果 CL20 不退化，再 finetune

```bash
export V25_CKPT_IN=outputs_v25/train/bdse_v25_dgcace.best.pt
export TRAIN_MAX_SCENARIOS=12000
export VAL_MAX_SCENARIOS=1000
export TRAIN_EPOCHS=5
export NPROC_PER_NODE=2
export RUN_MODE=all
bash run_v29_amvrbsr.sh
```

### Step 4：CL50 确认

```bash
export SKIP_TRAIN=1
export V29_CKPT=outputs_v29/train/bdse_v29_amvrbsr.best.pt
export RUN_MODE=cl50
export RUN_CL50_ALL4=1
export CL_WORKERS_PER_RUN=2
bash run_v29_amvrbsr.sh
```

## 6. 预期判断标准

v29 成功的最低标准：

- drivable 不低于 0.72；
- route progress 不低于 v28 safety-fallback 的 0.498；
- collision/TTC 至少高于 v28 safety-fallback 的 0.56/0.54，理想接近 v27 的 0.62/0.60；
- final_action_safety_flag_rate 明确低于 v28；
- CL50 score 超过 v28 safety-fallback 0.2166，并尽量超过 v27 safety-fallback 0.2341。

如果 v29 CL20 显示 drivable 掉回 0.70，说明 adaptive soft filter 仍过强，需要提高 `adaptive_min_soft_safe_actions` 或降低 `adaptive_max_extra_soft_flags`；如果 collision/TTC 没改善，说明 soft constraints 触发不足，需要降低 `adaptive_min_soft_safe_ratio` 或增大 `soft_agent_radius_m`。
