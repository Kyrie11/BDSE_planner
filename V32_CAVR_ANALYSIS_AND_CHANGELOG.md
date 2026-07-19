# BDSE v31 结果复盘与 v32 CAVR 优化方案

## 1. 结论摘要

v31 的五阶段流程总体执行完整：

1. 使用 v30 checkpoint 运行 v31 runtime-only open-loop；
2. 使用同一 v30 checkpoint 运行 runtime-only CL20；
3. 使用同一 v30 checkpoint 运行 runtime-only CL50；
4. 从 v30 checkpoint 进行 4 epoch v31 finetune；
5. 使用 v31 best checkpoint 运行 CL50。

没有发现 traceback、训练中断或闭环模拟未结束的问题。训练确实完成 4 个 epoch，最终 CL50 也加载了 v31 best checkpoint。不过存在两个实验记录问题：

- runtime-only 的 `v31_rhvcdsr_bbr_scur_20.zip` 只有 runner report 和 nuboard，缺少 aggregator/metric parquet；CL20 汇总 CSV 仍存在，因此更像打包竞态或漏打包，而不是仿真失败。
- 上传结果中没有闭环诊断 `*.diag.jsonl`。因此无法验证上一轮设定的 `final_action_safety_flag_rate`、all-flagged rate、frontier size 和 recovery mode 分布。最可能原因是 `OUT_ROOT`/诊断路径为相对路径，而 Hydra worker 改变了工作目录；旧代码又静默吞掉诊断写入异常。

总体判断：

> v31 runtime-only RH-VCDSR 显著恢复了闭环 route progress 和 comfort，但牺牲了一部分高速度场景的 collision/TTC；v31 finetune 没有产生有效增益，反而进一步损失 collision/TTC。v31 只部分达到最低门槛，没有达到论文理想目标，也没有形成统计稳定的综合闭环提升。

---

## 2. v30、v31 runtime-only 与 v31 finetune 的 CL50 对比

### 2.1 BBR/SCUR 配置

| 版本 | Score | Route progress | Collision | TTC | Comfort | DAC |
|---|---:|---:|---:|---:|---:|---:|
| v30 PMV-RBSR | 0.34481 | 0.39336 | 0.72 | 0.72 | 0.88 | 0.78 |
| v31 runtime-only | 0.33282 | 0.46942 | 0.70 | 0.68 | 0.96 | 0.78 |
| v31 finetune | 0.33277 | 0.46582 | 0.68 | 0.66 | 0.96 | 0.78 |

v30→v31 runtime-only 的变化：

- route progress：`+0.07606`，paired bootstrap 95% CI 为约 `[+0.038, +0.121]`，是明确、稳定的正向变化；
- comfort：`+0.08`，95% CI 约 `[+0.02, +0.16]`；
- collision：`-0.02`，置信区间跨 0；
- TTC：`-0.04`，置信区间跨 0；
- score：`-0.01199`，95% CI 约 `[-0.072, +0.046]`，不能证明综合分提高。

v31 runtime-only→v31 finetune 的变化：

- route progress：`-0.00361`；
- collision：`-0.02`；
- TTC：`-0.02`；
- score 基本不变。

因此，v31 的主要收益来自 runtime RH-VCDSR，而不是训练。

### 2.2 其他配置

v31 runtime-only 的 safety_fallback 与 BBR/SCUR 在 CL50 上完全相同；LCB 也几乎相同。这说明最终动作主要被同一套 recovery/viability 逻辑控制，不同 evidence/tournament 配置对闭环最终动作的区分度仍然不足。

LCB 的 open-loop 总 sparse query count 约 49,348，而普通 BBR/SCUR 约 32,189，但 CL50 没有相应收益。这说明当前 LCB 分支增加了计算，却没有带来足够的决策价值。

---

## 3. v31 是否达到上一轮门槛

### 3.1 最低门槛

| 最低门槛 | 结果 | 判断 |
|---|---:|---|
| Route progress 至少提高 0.015–0.02 | +0.0761 | 达到 |
| Collision/TTC 不下降超过 0.02 | Collision -0.02；TTC -0.04 | TTC 未达到 |
| Final safety flag/all-flagged 下降 3–5 个百分点 | 诊断文件缺失 | 无法判断 |
| Comfort 不下降 | +0.08 | 达到 |
| 不新增交叉口/行人/交通灯失败 | 交叉口明显改善，但新增高速度安全失败 | 部分达到 |

结论：**最低门槛只部分达到。**

### 3.2 论文理想目标

| 理想目标 | v31 runtime-only | 判断 |
|---|---:|---|
| Route progress 0.42–0.45 | 0.4694 | 达到并超过 |
| Collision/TTC ≥ 0.70 | 0.70 / 0.68 | TTC 未达到 |
| DAC ≥ 0.78 | 0.78 | 达到 |
| Final safety flag < 0.30 | 无诊断 | 无法判断 |
| 多个大规模子集上的稳定正向 paired CI | 当前仅 CL50，score CI 跨 0 | 未达到 |

v31 finetune 后 collision/TTC 进一步降到 0.68/0.66，因此训练后版本更不满足理想目标。

---

## 4. RH-VCDSR 解决了什么，又留下了什么

### 4.1 已解决或明显缓解的问题

1. **v30 的低进度保守恢复得到明显改善。**
   v30 BBR route progress 为 0.3934，v31 runtime-only 提升到 0.4694。说明 receding-horizon hard check、风险归一化和 viability/Pareto recovery 确实避免了很多“远端风险导致当前停车”的情况。

2. **舒适性明显提升。**
   Comfort 从 0.88 提升到 0.96，说明 v31 更少出现极端刹停或突兀控制。

3. **部分复杂交叉口被救回。**
   有一个 traffic-light intersection 场景 score 从 0 提高到约 0.862，collision/TTC 同时从 0 变为 1。说明“近端可行性 + 在可接受风险集合中恢复进度”的方向有效。

### 4.2 没有解决的问题

1. **固定 4 秒 hard horizon 对高速场景过短。**
   v31 的主要负面场景集中在 `high_magnitude_speed` 与 `near_high_speed_vehicle`。两个高速场景 collision/TTC 从 1 降到 0，另一个高速邻车场景 TTC 从 1 降到 0。城市低速时 4 秒能减少远端误报，但高速时制动距离对应的时间窗明显更长。

2. **Agent risk 仍使用点/圆近似和 constant velocity。**
   当前风险没有充分利用车辆长宽，也没有根据 closing speed 扩展纵向安全包络。长车、高相对速度和斜向接近会被低估。

3. **v31 的 viability guard 仍是顺序过滤。**
   Agent risk、off-route risk、certificate 按顺序过滤。候选集合会随过滤顺序变化，并不是真正的联合 viability set。

4. **Recovery 仍然压过 learned evidence。**
   Safety、BBR 和 LCB 最终结果几乎相同，说明 learned tournament 的作用在困难场景中仍然有限。

5. **训练没有对齐部署。**
   `predicted_selector_start_epoch=3`，但总共只训练 4 epoch，因此只有最后一个 epoch 使用完整 predicted-selector 路径。best checkpoint 又来自 epoch 0。训练后 teacher action match 从 0.215 降到 0.201，budget-vs-full match 从 0.173 降到 0.168，pair sign accuracy 也小幅下降。

6. **Checkpoint 指标使用了错误的“full”语义。**
   训练日志中 `val_budget_vs_full_match` 与 `val_budget_vs_sparse_full_match` 完全相同，说明当时 validation 没有运行 dense full-interface diagnostic。best checkpoint 实际根据 sparse proxy 选择，而不是论文真正关心的 dense full-interface decision preservation。

---

## 5. 当前最值得强化的部分

### 5.1 应继续强化

#### A. Pair-conditioned critical evidence learning

Proposal recall 已经接近饱和：

- proposal decisive recall ≈ 0.992；
- proposal interaction decisive recall ≈ 0.989；
- proposal hard recall = 1.0。

但 selected recall 明显较低：

- selected decisive recall ≈ 0.375；
- selected interaction decisive recall ≈ 0.343；
- selected hard decisive recall ≈ 0.724。

因此瓶颈不是“关键 atom 没进入 proposal pool”，而是：

1. pair-margin 估计不够准确；
2. selector 的 acquisition objective 没有把关键 interaction atom 留在 B=16 内；
3. 固定 quota/force-fill 把预算浪费在冗余结构证据上。

下一版应把训练重点从 proposal BCE 转向 critical pair margin、near-boundary sign、safe-vs-unsafe crossing 和 selected-budget action certificate。

#### B. Deployment-consistent selected-budget training

应从 epoch 0 就使用 predicted selector，而不是只在最后一个 epoch 才启用；并使用 dense validation 的真实 `budget_vs_full_match` 选 checkpoint。

#### C. 高速风险建模

需要把 hard horizon 与候选速度/制动时间关联，并使用车辆尺寸、closing speed 和 TTC。这个修改不增加 evidence budget，只改 runtime 风险计算，最有希望修复 v31 的主要安全退化。

#### D. Reciprocal pair cancellation

代码中 stop/go geometry pair 同时加入 `(a,b)` 和 `(b,a)`。由于 atom margin 反对称，signed coverage 同时奖励两个方向时会互相抵消。在一个现有单元测试中，明明存在能翻转 stop/go 决策的关键 atom，selector 仍返回空集合。这个问题会直接阻止模型学到的 critical atom 被预算 selector 采用。

v32 已将 reciprocal pair 折叠为一个 evidence-sensitive orientation：选择“当前可查询 atom 能产生最大正 certificate gain”的方向，不使用 teacher future 或标签。

### 5.2 应削弱或删除

1. **过大的 mandatory hard quota。**
   v31 的 mandatory hard quota 为 8，decision family quota 实际占满 16；在 B=16 下几乎不给其他关键 interaction evidence 留空间。由于 proposal hard recall 已经为 1.0，继续强制 8 个 hard atom 的边际收益很低。

2. **Proposal top-M 过大。**
   M=80 而 recall 已饱和。可缩到 64，降低计算，同时不明显损失关键 proposal recall。

3. **LCB 的高查询开销。**
   LCB 多约 17k sparse queries，但闭环结果没有明显更好。下一轮应降低 LCB seed budget，优先保留 action-rank/critical pair budget。

4. **全模型低学习率 finetune。**
   v31 训练既没有改变大多数闭环动作，又破坏了少数安全场景。下一轮应冻结 scene/action/base backbone，仅微调 evidence、pair、family/proposal 和 query heads。

5. **把 certificate 当物理安全 hard gate。**
   learned certificate 有估计误差，不应在 recovery 中先于物理风险剔除候选。它更适合作为联合 Pareto objective，而不是单独顺序 hard filter。

---

## 6. v32：CAVR-BDSE

下一版命名为：

**CAVR-BDSE: Criticality-Adaptive Viability Recovery for Budgeted Decision-Sufficient Evidence Planning**

### 6.1 速度自适应 hard horizon

对每个候选动作计算：

```text
H_hard(a) = clip(t_reaction + v_max(a) / a_decel + margin,
                 H_min, H_max)
```

默认：

- 城市低速仍保持约 4 秒 hard horizon；
- 高速候选自动扩展到最多 6.5 秒；
- soft horizon 为 hard horizon 加额外缓冲，最多 7.5 秒。

### 6.2 Box-aware + closing-speed + TTC agent risk

当 agent state 包含 length/width 时，使用 ego-agent box Minkowski sum 的椭圆近似：

- 纵向半轴由 ego/agent 长度、clearance 和 closing-speed buffer 构成；
- 横向半轴由车辆宽度和 lateral clearance 构成；
- 额外计算 gated TTC risk；
- 缺少尺寸时回退到原 circle risk。

### 6.3 联合 viability，而不是顺序过滤

将归一化后的 agent overlap、TTC 和 off-route risk 合成 Chebyshev criticality：

```text
V(a) = max(r_agent(a), r_ttc(a), r_route(a))
```

先构造 near-minimum joint viability set，再在其中计算 epsilon-Pareto frontier。Certificate、soft risk 和 progress 都作为 Pareto 目标，不再把 learned certificate 当成物理 hard gate。

### 6.4 Fixed-budget selector 重平衡

保持论文核心 evidence budget `B=16` 不变：

- proposal M：80 → 64；
- mandatory hard quota：8 → 4；
- decision-family quota：16 → 6；
- proposal fill weight：0.70 → 0.25；
- action-rank 至少保留 60% 的预算；
- candidate-aware agent selection 开启；
- reciprocal runtime pair 折叠，消除 signed certificate cancellation。

### 6.5 Critical-head finetune

冻结 scene/action/base backbone，只训练：

- evidence encoder；
- query projection；
- pair mean/variance heads；
- family/proposal heads。

训练从 epoch 0 使用 predicted selector，默认 3 epoch、`lr=2e-6`。Checkpoint 主指标改为 `fixed_budget_critical_score`，综合真实 dense budget-vs-full match、teacher match、selected interaction/hard recall、near-tie sign、sufficiency、fallback 和 regret。

### 6.6 诊断和实验完整性

- `OUT_ROOT` 强制转为绝对路径；
- 闭环诊断写入失败不再静默吞掉；
- 每个 CL20/CL50 运行后检查 diag 文件是否存在；
- runtime-only 与 finetune 输出目录严格分离。

---

## 7. v32 的实验顺序和门槛

### Stage A：严格 runtime-only，先验证安全修复

使用 v30 checkpoint，只改变 v32 runtime/selector：

1. open-loop；
2. CL20；
3. CL50 all-4。

Runtime-only gate：

- BBR route progress 不低于 v31 runtime-only 的 `0.4694 - 0.01`；
- collision ≥ 0.70；
- TTC ≥ 0.70，至少比 v31 的 0.68 提升；
- comfort ≥ 0.94；
- DAC ≥ 0.78；
- high_magnitude_speed 两个 v31 新失败场景至少救回一个；
- diag 文件完整生成；
- reciprocal-collapse 后 selected interaction recall 不低于 v31。

若 runtime-only 不满足这些条件，不应训练。

### Stage B：Critical-head finetune

训练后必须同时满足：

- dense budget-vs-full match 高于 v31 runtime 的 0.173；
- selected interaction decisive recall 高于约 0.343；
- teacher action match 不低于 0.215；
- CL50 collision/TTC 不低于 runtime-only；
- route progress 不下降超过 0.01。

### Stage C：论文级验证

CL50 只能用于快速迭代，不能支撑 SOTA 结论。最终应：

- 使用固定公开 scenario list；
- 至少 CL200/CL500，最好完整标准 split；
- 运行 3 个随机种子或多个固定子集；
- 同时报告 non-reactive 与 reactive closed-loop；
- paired bootstrap CI；
- 报告 B、M、K、effective query count、latency 和 fallback rate；
- 做 B={4,8,16,24,32} 的 frozen-support budget sweep；
- 与 random、top-magnitude、interaction-only、hard-only、oracle selector 和 full-evidence 比较；
- 将 PDM/PLUTO/DTPP 等候选规划器适配到相同 K/B/query/latency setting，或明确它们不是 budget-matched baseline。

---

## 8. “Under fixed-budget closed-loop SOTA”是否可能

有可能，但需要把 claim 定义严格：

> 在相同输入、候选动作 bank、evidence atom bank、K、B、proposal M、effective query count 和 latency cap 下，BDSE 在闭环 nuPlan 指标上优于所有 budget-matched selector/compression/planner-interface baselines。

不能用当前 CL50 直接声称 global nuPlan closed-loop SOTA。强规划器通常没有相同 evidence-interface budget，而当前 B=16 仍对应约 8k selected certificate queries 和 32k total sparse queries，因此论文必须同时报告 effective query count 和 latency，证明“固定预算”是真实的系统约束，而不是只限制最终 selected atom 数。

更适合 CCF-A 的核心贡献仍然是：

1. decision-sufficient evidence interface；
2. pair-conditioned action-margin decomposition；
3. fixed-budget critical evidence acquisition；
4. candidate-set preservation/regret 理论；
5. budget-quality-latency Pareto closed-loop benchmark。

RH-VCDSR/CAVR 应定位为 deployment policy 或 robust recovery，不宜取代 BDSE 成为论文主贡献。

---

## 9. 代码验证

v32 当前静态验证：

```text
python -m py_compile: passed
pytest: 85 passed, 5 warnings
bash -n run_v32_cavr.sh: passed
```

警告来自 PyTorch Transformer nested-tensor 设置，不是本次修改引入的运行错误。
