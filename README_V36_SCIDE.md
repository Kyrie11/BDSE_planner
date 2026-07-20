# BDSE v35 Runtime Gate 分析与 v36 SCIDE-BDSE 方案

## 1. v35 结果结论

使用未改变的 v30 checkpoint，v35 四个配置均满足 teacher-action match、fallback 和查询预算要求，但表现出稳定的 hard/interaction 此消彼长：

| 配置 | Hard decisive recall | Interaction decisive recall | Teacher match | Effective queries | Total queries |
|---|---:|---:|---:|---:|---:|
| balanced | 0.5327 | 0.3466 | 0.228 | 5380.8 | 15879.5 |
| hard7 | 0.5761 | 0.3509 | 0.228 | 5380.8 | 15879.5 |
| soft7 | 0.4935 | 0.3632 | 0.226 | 5380.8 | 15879.5 |
| influence | 0.5287 | 0.3415 | 0.228 | 5380.8 | 15879.5 |

v34 已经能让 hard recall 超过 0.60，但 interaction recall 约 0.28；v35 将 interaction recall 提高到 0.34–0.36，却使 hard recall 降到 0.49–0.58。因为 B=16、M=64、查询图和 checkpoint 都保持不变，这种稳定的反向变化说明问题不是单个 quota 参数，而是“把结构安全约束和决策证据放入同一个稀疏证据预算”的建模冲突。

## 2. Gate 是否不合理

原 gate 在 v34/v35 单池架构下是合理的：它成功阻止了只提高 interaction recall、却丢失 hard support 的版本进入闭环。

但当算法重构为两通道架构后，继续要求 `selected_hard_decisive_recall >= 0.60` 就不再语义一致，因为 hard evidence 明确不属于 learned B=16。新的 gate 不能简单删除安全要求，而必须改为检查：

1. 所有 active hard/feasibility evidence 是否被结构安全通道覆盖；
2. 最终有效证据集合（结构安全 + B=16 decision evidence）是否覆盖 hard decisive evidence；
3. 最终动作是否仍被 runtime hard filter 判为安全；
4. learned budget 自己负责的 soft interaction/precedence evidence 是否达到门槛；
5. teacher match、pair sign、regret 和查询量不得退化。

因此 v36 gate 是架构对齐，而不是降低标准。

## 3. 是否应让所有 safety evidence 绕过 selector

结论是“结构 hard safety 应全部绕过 selector，但不是把所有 safety-related atom 都无条件加入 learned tournament”。

### 3.1 预算外的结构安全通道

以下内容是可确定计算的 viability constraints：

- red-light violation；
- drivable-area / route-corridor hard violation；
- box-aware agent overlap；
- hard TTC envelope；
- invalid candidate；
- feasibility-family hard evidence。

它们应在每次规划中全部计算，并按词典序先形成安全可行集。不能让 B=16 决定“今天是否看见红灯或碰撞约束”。

### 3.2 仍应由 B=16 选择的 decision evidence

以下 evidence 不是简单 hard constraint，而是在多个安全候选之间决定 ego action：

- soft TTC / time-gap；
- interaction priority；
- yielding / precedence；
- stop/go decision boundary；
- route/progress evidence；
- comfort / dynamic regularity。

这些 evidence 仍需在 fixed budget 下选择，这才保留论文“decision-sufficient evidence”的核心贡献。

### 3.3 不建议默认只筛选 interaction evidence

只筛 interaction 可作为 ablation，但不适合作为主方法。原因是 route/progress、decision-boundary 和 regularity 直接影响闭环 route progress、comfort、DAC 和 stop/go 稳定性。若主方法完全排除它们，容易回到 v30/v31 中“安全或进度一边倒”的问题。

v36 默认 selector 允许 family 2/3/4/5，interaction-only 仅作为单独消融配置。

## 4. v36 SCIDE-BDSE

全称：**Safety-Complete Interaction-Decisive Evidence Planning**。

核心分解为：

```
Runtime geometry / map / traffic state
               |
               +--> Structural hard-safety channel (budget-exempt)
               |       -> viable action frontier
               |
               +--> Decision-evidence proposal and selector (B=16)
                       -> pair-conditioned signed tournament inside frontier
```

### 4.1 Structural safety bypass

- `decision_budget_excludes_structural_safety: true`
- hard atom 和 feasibility-family atom 不进入 Top-M，也不消耗 B=16；
- runtime safety flags 和 continuous hard-risk 每次完整计算；
- final tournament 保留 `hard_filter_unsafe_actions: true`；
- all-flagged 时保留 minimum-risk frontier，而不是取消安全判断。

### 4.2 Viability-conditioned pair graph

selector 不再花 query 比较一个明显 unsafe action 与一个 safe action，因为该比较已由结构安全词典序决定：

- 两个以上 safe actions：只比较 safe-safe；
- 仅一个 safe action：保留小型 anchor graph 用于校准；
- 全部 flagged：比较 minimum-hard-risk frontier；
- final tournament 的安全 guard 不被移除。

### 4.3 Fixed decision evidence budget

保持：

- B=16；
- Top-M=64；
- logical pair cap=480；
- 不新增 learned parameter；
- v30 checkpoint 可直接加载。

Top-M 会删除 structural safety atoms，并在相同 M 内用 decision evidence 补足。最终 selector 的 mandatory hard quota 设为 0，因为安全已由结构通道负责。

### 4.4 Training alignment

v36 train config：

- 从 pair regression 排除 structural safety atoms；
- 从 pair regression 排除 hard-action pairs；
- 从 pair-action loss 排除 structural safety atoms；
- 保留 runtime viability filter 和 decision-only B=16 路径。

这避免 finetune 重新学习一个部署时不会使用的 hard-evidence加性分数。

## 5. 新 gate

主要门槛：

| 指标 | 门槛 | 含义 |
|---|---:|---|
| structural_hard_decisive_coverage | ≥0.98 | hard decisive evidence 被结构安全通道覆盖 |
| effective_hard_decisive_recall | ≥0.98 | 结构通道 + B=16 的有效 hard coverage |
| selected_soft_interaction_decisive_recall | ≥0.32 | B=16 自己负责的 soft interaction recall |
| effective_interaction_decisive_recall | ≥0.35 | 包含结构 interaction 后的总 interaction coverage |
| fallback_would_trigger_rate | ≤0.02 | 不增加模型失配 fallback |
| selected_action_safety_flag_rate | ≤0.005 | 最终选择动作必须几乎全部通过 hard guard |
| teacher_action_match | ≥0.215 | 决策能力不得退化 |
| effective_query_count | ≤8500 | fixed effective budget |
| total_sparse_query_count | ≤33000 | fixed compute budget |

同时相对 v35 hard7 进行非退化检查：teacher match、budget-vs-full、三组 pair sign、effective decisive recall、effective interaction recall 和 teacher regret。

旧 `selected_hard_decisive_recall` 仍报告，但只作为“learned decision subset 中碰巧选了多少 hard atoms”的诊断，不再作为主 gate。

## 6. 实验顺序

1. 继续固定 v30 checkpoint，跑 v36 runtime-only gate；
2. 同时比较 balanced、interaction10、influence、no-frontier control；
3. interaction-only 单独跑，不参与主 gate；
4. 主 gate PASS 后才跑 CL20；
5. CL20 不退化后做 v30 warm-start 的受控 finetune；
6. 最终代码和数据定义冻结后，再做 clean multi-seed training。

当前阶段不建议重新训练，因为 v36 没有新增 learned parameter，runtime-only 对照仍能最清晰地判断两通道架构是否有效。

## 7. 验证状态

已完成：

- `python -m py_compile` 全部 Python 文件；
- `bash -n run_v36_scide.sh`；
- `pytest -q`：104 passed, 5 warnings；
- 新增 structural mask、decision-only Top-M、viability pair graph、query accounting 与 gate 单元测试。

当前环境没有 nuPlan cache 和 v30 checkpoint，无法在此处声称 v36 数值一定 PASS；必须以用户环境的 1000-scenario runtime-only 结果为准。
