# BDSE v32 结果审计与 v33 CARB-BDSE 优化方案

## 0. 审计范围与限制

本轮实际读取并核查了：`bdse.zip` 全部源码、v31/v32 runtime-only 与 finetune 的 open-loop/closed-loop 产物、训练日志、场景级 parquet、v32 closed-loop diagnostics，以及上一轮“大模型建议”。

当前收到的代码压缩包和结果包中没有 `.tex`、`.pdf` 或 `.docx` 论文正文；File Library 检索也未找到与本项目匹配的 LaTeX 论文。因此，本报告能够严格验证**代码中的算法链路、数据接口、实验配置和结果**，但不能声称已经逐段核对论文文字、公式编号和实验表述。论文正文后续补入时，需要再做一次 paper-code-result 三方一致性审查。

---

## 1. 结论先行

### 1.1 v32 没有让最终闭环整体变好

以主配置 BBR/SCUR、相同 50 个场景为准：

| 版本 | Score | Route progress | Collision | TTC | Comfort | DAC |
|---|---:|---:|---:|---:|---:|---:|
| v31 runtime-only | **0.33282** | **0.46942** | **0.70** | **0.68** | 0.96 | **0.78** |
| v31 finetune | 0.33277 | 0.46582 | 0.68 | 0.66 | 0.96 | **0.78** |
| v32 runtime-only | 0.30152 | 0.46310 | 0.68 | 0.64 | **0.98** | 0.76 |
| v32 finetune | 0.32119 | 0.46242 | **0.70** | 0.66 | **0.98** | 0.76 |

v31 runtime-only → v32 runtime-only 的 paired 均值变化：

- Score：-0.03130，bootstrap 95% CI 约 `[-0.0883, +0.0084]`；
- Route progress：-0.00632；
- Collision：-0.02；
- TTC：-0.04；
- Comfort：+0.02；
- DAC：-0.02。

因此，v32 只稳定改善了舒适性，没有满足上一轮提出的 runtime-only 门槛（特别是 TTC≥0.70、DAC≥0.78），也不能得出综合闭环优于 v31 的结论。

### 1.2 v32 finetune 有窄范围安全收益，但不是一般性提升

v32 runtime-only → v32 finetune：

- Score +0.01967；
- Collision +0.02；
- TTC +0.02；
- Route progress -0.00068；
- 50 个场景中仅 2 个 score 改变，48 个完全不变。

主要收益来自 `d1f2ed3524c8591a / traversing_pickup_dropoff`：训练后 collision/TTC 从失败恢复为通过，score 约增加 0.982。说明 critical-head finetune 能改变个别决策，但尚未形成跨场景的稳定作用。

### 1.3 v32 的有效修改与失败修改可以清晰分开

**有效：**

- reciprocal pair cancellation 修复后，winner-rival / interaction / hard pair sign accuracy 分别约提升 2.3 / 1.6 / 1.6 个百分点；
- teacher action match 从 0.215 升到 0.228；
- dense validation 和 predicted-selector-from-epoch-0 已正确执行；
- diagnostics 路径已修复；
- box-aware/TTC 风险在个别高速邻车场景中救回 TTC；
- comfort 从 0.96 提升到 0.98。

**失败或负作用：**

- mandatory hard quota 8→4、decision-family quota 16→6 且去掉 feasibility family，导致 selected hard recall 从 0.724 降到 0.499；
- selected interaction recall 从 0.343 降到 0.303；
- open-loop fallback rate 从 0.5% 升到 4.6%；
- effective queries 从约 8.23k 升到 8.73k，total sparse queries 从约 32.19k 升到 33.63k，未实现“更少 proposal、计算更省”；
- speed-adaptive horizon 几乎未激活；
- relative-only joint viability 在 all-bad candidates 中缺少绝对安全下界；
- 单一 route centerline 在交叉口/分叉处制造大尺度虚假 off-route 风险。

---

## 2. v32 是否解决了上一轮总结的问题

| 上一轮问题/目标 | v32 实际结果 | 判断 |
|---|---|---|
| 修复高速场景 4s horizon 过短 | hard horizon 均值和 p90 仍约 4.0s，仅极少数扩展 | **未解决** |
| box-aware + closing speed 风险 | pair/risk 诊断改善，救回一个 near-high-speed TTC | **部分有效** |
| joint viability 避免过滤顺序依赖 | 顺序依赖已消除 | **实现** |
| joint viability 提升最终安全 | TTC 0.68→0.64，collision 0.70→0.68 | **未实现** |
| reciprocal pair cancellation | sign accuracy 明显提升 | **实现且有效** |
| fixed-budget 重分配提高 selected recall | hard/interaction recall 大幅下降 | **反向作用** |
| critical-head finetune | 个别场景救回，但 open-loop match/recall未提升 | **局部有效** |
| diagnostics 完整生成 | v32 CL diagnostics 可用 | **实现** |
| Route progress ≥0.459 | 0.4631 | 达到 |
| Collision/TTC ≥0.70 | 0.68/0.64 | **未达到** |
| Comfort ≥0.94 | 0.98 | 达到 |
| DAC ≥0.78 | 0.76 | **未达到** |

---

## 3. 根因分析

### 3.1 Proposal 不是瓶颈，selected-budget allocation 才是瓶颈

四个版本 proposal recall 基本固定：

- proposal decisive recall ≈0.992；
- proposal interaction recall ≈0.989；
- proposal hard recall =1.000。

所以 M=64/80 都能把关键 atom 放进 proposal。真正丢失发生在 B=16 的最终选择阶段。v32 把 feasibility family 从 decision-family quota 排除，同时 hard quota 减半，直接导致关键安全证据被 action-rank atoms 挤出。

### 3.2 纯相对归一化没有绝对安全语义

v32 在同一场景内把 agent/TTC/off-route 风险减去最小值再归一化，然后保留 near-minimum joint viability set。若所有候选都危险，“最不危险”仍可能是不可接受的。

CL50 diagnostics 显示：

- all-flagged recovery 约占 35.8% replans；
- recovery 中约 87.5% 改变了原动作；
- recovery selected min TTC 中位数约 0.97s；
- selected TTC risk 均值约 0.337；
- joint viability 因候选不足而放松约占 recovery 的 51%。

这说明 recovery 已经成为高频控制器，而且缺少绝对 TTC barrier。

### 3.3 单中心线 route proxy 在路口不成立

v32 recovery selected off-route risk：

- 均值约 122.45；
- 中位数约 124.98；
- p90 约 215.26。

这不是正常的 lane deviation 数量级。根因是 runtime adapter 只输出一条 stitched route centerline；在 roadblock connector、分叉和 pickup/dropoff 区域，候选可能位于合法 route interior edge 上，却离被选中的单一路径很远。该误差既提高 fallback 率，也会把 recovery 推向错误的“最小相对违反”轨迹。

### 3.4 “速度自适应”参数使其实际上退化为固定 4 秒

v32 公式为：

`H = clip(0.7 + v/5.0 + 0.35, 4.0, 6.5)`

只有候选最高速度大于约 14.75 m/s 才开始超过 4 秒。CL50 中 hard horizon 均值/p90 都约 4.0 秒，证明该机制对大多数中高速场景没有实际作用。

### 3.5 checkpoint score 允许以轻微 match 换取严重 hard-recall 退化

v32 三个 epoch 的 dense validation 和训练路径正确，但：

- selected hard recall 继续下降；
- fallback rate 固定在约 4.6%；
- fixed-budget critical score 仍略升。

原 score 对 hard recall 的权重过低、没有最低门槛，也没有对 fallback 和查询预算设置 hinge penalty。因此“best”并不代表闭环安全代理最优。

### 3.6 BDSE 主方法与 recovery 的贡献边界不够清楚

CL50 中 35% 以上 replan 进入 all-flagged recovery，且绝大多数发生动作切换。若最终得分主要由 runtime recovery 决定，审稿人会质疑：闭环增益是否来自 BDSE 的固定预算 evidence selection，还是来自一个额外的 rule-based planner。

后续必须报告：

- no-recovery / relative-recovery / CARB recovery 三组消融；
- recovery exposure rate；
- recovery changed-action rate；
- selected evidence 对最终动作的 causal flip rate；
- 相同 B、M、effective queries、latency 下的对照。

---

## 4. v33：CARB-BDSE

全称：**Criticality-Adaptive Risk-Barrier Budgeted Decision-Sufficient Evidence Planning**。

### 4.1 Criticality-adaptive fixed-budget allocation

B 保持 16，M 保持 64，不靠扩大预算：

- mandatory hard quota：4→6；
- decision family：恢复 `[feasibility, reachability_interaction, precedence, decision_boundary]`；
- decision-family quota：6→10；
- proposal fill weight：0.25→0.40；
- prioritize mandatory fill：开启；
- reciprocal pair collapse 保留。

目标不是回退到 v31 的全硬配额，而是在 hard safety 与 action-conditioned evidence 间建立中间点。

### 4.2 Absolute-Relative Risk Barrier

恢复顺序改为：

1. red-light hierarchy；
2. **绝对 TTC/agent-overlap barrier**；
3. 若无候选通过绝对 barrier，再按 best available TTC 做有限放松；
4. scene-relative joint viability；
5. conditional certificate guard；
6. epsilon-Pareto 与 progress utility。

默认绝对条件：

- predicted min TTC ≥1.5s；
- hard agent overlap deficit ≤0.02。

这样避免所有候选都危险时，纯归一化把 0.5–1.0s TTC 轨迹视为“相对可行”。

### 4.3 Route-Graph Corridor

runtime adapter 缓存 declared route 中所有 roadblock interior edge baseline，并输出 `route_corridor_centerlines`。安全检查、continuous risk 和 rule rerank 均计算到所有 route edge 的最小距离，而不只依赖单条 stitched centerline。

该修改：

- 不读取 future label；
- 不增加 evidence atom budget；
- 只使用部署时已有 route roadblock IDs 与 map API；
- 直接针对 intersection/pickup-dropoff 的虚假 off-route。

### 4.4 真正激活的速度自适应 horizon

参数调整为：

`H = clip(0.90 + v/3.0 + 0.55, 4.0, 6.5)`。

约 7.65 m/s 开始扩展，10 m/s 时约 4.78s，15 m/s 时约 6.45s。它仍保留城市低速 4 秒近端 hard check，但不再只在极高速才生效。

### 4.5 Evidence-preserving recovery

物理 barrier 之后，若至少 `min_pool` 个候选处于最优 certificate loss +0.35 normalized band 内，则优先在该集合做 Pareto。若候选不足则自动放松，certificate 永远不会排除唯一的物理更安全动作。

这使 recovery 与 BDSE evidence 真正耦合，同时避免 learned certificate 被当成物理 hard constraint。

### 4.6 Constraint-aware checkpoint score

新增 hinge penalties：

- selected hard recall <0.60；
- selected interaction recall <0.32；
- fallback rate >0.02；
- effective query >8.5k；
- total sparse query >33k。

训练仍只更新 critical heads，冻结 scene/action/base backbone。这样 checkpoint 不再允许用极小的 teacher-match 增益换取安全 evidence coverage 崩塌。

---

## 5. v33 预期验证门槛

### Open-loop 首先必须满足

- selected hard decisive recall ≥0.60；
- selected interaction decisive recall ≥0.32；
- fallback_would_trigger_rate ≤0.02（至少应显著低于 v32 的 0.046）；
- teacher action match ≥0.215；
- budget-vs-full match ≥0.170；
- effective queries ≤8.5k；
- total sparse queries ≤33k。

若 hard recall 仍低于 0.60，不进入 finetune；先做 quota ablation：`hard quota ∈ {6,7,8}`、`family quota ∈ {8,10,12}`。

### Runtime-only CL20

- 不允许相对 v31 runtime-only新增 collision/TTC failure；
- `absolute_barrier_applied` 应在 all-flagged recovery 中稳定出现；
- recovery selected min TTC 的 p50 在存在通过候选时应不低于 1.5s；
- off-route risk 中位数应从约 125 大幅下降到合理尺度。

### Runtime-only CL50

最低继续训练门槛：

- Score ≥0.333；
- Route progress ≥0.46；
- Collision ≥0.70；
- TTC ≥0.70；
- Comfort ≥0.96；
- DAC ≥0.78；
- final safety flag rate <0.30；
- 同一 50 场景 paired score CI 不应明显负向。

### Finetune 接受条件

- 相对 runtime-only 至少在 5 个以上场景产生非零决策变化，而不是仅救回一个场景；
- collision/TTC 不下降；
- selected hard/interaction recall 不下降；
- teacher action match 和 budget-vs-full 至少一项提升；
- query count 不增加。

---

## 6. 论文 novelty 与实验设计建议

当前最可信的主张不是“nuPlan global SOTA”，而是：

> 在固定 planner-interface evidence budget、固定 candidate bank、固定查询/时延上限下，CARB-BDSE 通过 criticality-adaptive evidence allocation 与 absolute-relative viability barrier，实现更高的闭环 decision sufficiency、安全性和效率。

建议形成四个清晰贡献：

1. **Fixed-budget decision sufficiency formulation**：不是压缩 token，而是在 action-pair margin 层定义 evidence sufficiency；
2. **Reciprocal-collapsed pair-conditioned selection**：消除反对称 pair cancellation；
3. **Criticality-adaptive budget allocation**：带 hard/interaction coverage constraints 的 B=16 selection；
4. **Absolute-relative evidence-conditioned viability**：物理 barrier 保底、relative Pareto 恢复效率、certificate 保持 learned decision semantics。

必须加入的消融：

- no pair collapse；
- hard quota 4/6/8；
- single centerline vs route graph corridor；
- relative-only CAVR vs absolute barrier CARB；
- no certificate guard；
- no recovery；
- B∈{4,8,16,32} budget-performance curve；
- fixed atom budget、fixed effective queries、fixed latency 三种公平预算口径；
- non-reactive + IDM reactive + stronger learned reactive agents；
- CL50 只用于开发，最终用更大固定 scenario list 和多 seed paired bootstrap。

---

## 7. 已交付代码

- `run_v33_carb.sh`；
- 5 个 v33 配置；
- route-graph runtime adapter；
- absolute TTC/overlap barrier；
- certificate-preserving guard；
- constraint-aware checkpoint score；
- closed-loop 结果打包工具；
- v33 单元测试。

本地静态/单元测试结果：

- `python -m py_compile`：通过；
- `bash -n run_v33_carb.sh`：通过；
- `pytest -q`：**90 passed, 5 warnings**。

注意：本环境没有 nuPlan 数据、GPU checkpoint 和仿真依赖，因此 v33 的性能改善是基于根因修复的理论预期，尚未替代你机器上的 runtime-only/CL20/CL50 实验验证。
