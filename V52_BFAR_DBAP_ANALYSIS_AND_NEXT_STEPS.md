# V51 结果复盘与 V52 BFAR-DBAP 设计

## 1. 结论先行

本次输出中的 **V51 gate 确实失败**，但失败发生在 residual 训练之前：

1. foundation 训练结束；
2. `check_v51_foundation_quality` 返回 FAIL；
3. V51 residual checkpoint 没有生成；
4. candidate / local-control / foundation-control 三路 replay 没有运行；
5. CL20、CL100 均没有运行。

因此不能把结果表述为“V51 FAR residual 再次被实验否定”。准确表述是：

> V51 的 foundation gate 定义把将被重置的 direct pair head 和尚未训练的预算 selector 也当成了 immutable foundation 能力，导致可用的 base+local anchor 被提前拒绝。

按 V52 的 anchor-only gate，当前 V51 checkpoint 会 PASS：

- full interface action match：0.359；
- base winner-rival sign：0.671；
- dense winner-rival sign：0.803；
- dense near-tie sign：0.706；
- dense all-pair sign：0.718；
- teacher regret：10808；
- latency p95：1214 ms。

被排除的 direct pair-head / selector 指标为：pair-full match 0.060、pair near-tie 0.342、certificate pair fraction 0.210、fallback 0.791。这些模块将在 BFAR 阶段重置或训练，不应决定 anchor 是否可用。

## 2. 训练速度诊断

### 2.1 实际耗时

| Epoch | exact selector | epoch wall time | samples/s | loss ms/step | backward ms/step |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 2796 s | 17.88 | 213.0 | 136.3 |
| 1 | 0 | 2760 s | 18.12 | 209.8 | 135.3 |
| 2 | 0 | 2509 s | 19.93 | 189.8 | 122.9 |
| 3 | 0 | 2723 s | 18.36 | 204.5 | 131.1 |
| 4 | 1 | 4725 s | 10.58 | 516.1 | 130.5 |

数据等待和 H2D 均约 1 ms/step，NPZ/DataLoader 不是主要瓶颈。非 exact 阶段的主要成本是 dense pair/loss/backward；exact epoch 又增加约 0.315 s/step 的 CPU selector 阻塞。

### 2.2 V52 速度修改

1. **直接复用 V51 immutable anchor**：当前 checkpoint 已通过修正后的 anchor-only gate，主运行无需再次花约 4.3 小时重训 foundation。
2. **factorized anchor 配置**：必须重建时，不训练 direct pair head，不运行 exact selector。
3. **每卡 batch 4 -> 8**：optimizer step 数约减半；脚本允许通过环境变量回退。
4. **Boundary Pair Sampler**：大多数 step 从完整约 118–192 个 pair 中保留 64 个；强制包含 teacher-winner、hard-crossing 和 near-tie 配额。
5. **周期性完整图**：每 4 步恢复完整 pair graph，使 exact AOCC 标签与完整图严格对齐。
6. **稀疏 exact 场景**：每 4 步每卡仅 1 个场景执行 exact CPU selector，最后 128 step 全 batch 对齐。普通 epoch exact scene fraction 为 3.125%。
7. **主预算优先**：B=16 在每个 exact 事件都监督；B=8/B=24 每 4 个 exact 事件采样一次，仅作跨预算正则。
8. **已有 process pool 保留**：exact 算法和输出 mask 不变。

预计 exact CPU 总工作量较 V51 full-exact epoch 下降约一个数量级以上；端到端提升仍需服务器热路径日志确认，不预先声称固定倍数。

## 3. 上一轮“应保留设计”的再评估

| 设计 | V51 新证据 | 判断 |
|---|---|---|
| base + local factorized decision interface | dense winner/near sign = 0.803/0.706，明显优于 direct pair 0.579/0.342 | **强力保留，升级为 immutable anchor** |
| residual selective intervention | V51 未运行；仅有 V50 beneficial 0.026 > harmful 0.008 | **保留假设，但尚未被 V51 复验** |
| AOCC exact tournament target | V51 foundation 阶段不是有效 selector 实验；certificate 0.210、fallback 0.791 | **保留算法机制，不把本次输出当有效性证据** |
| fixed budget 16/16 fill | 本次仍为 16/16 | **工程机制有效，但不等于 evidence 有效** |
| frontier/certificate representation | frontier retained 0.910，但 certified pair 仅 0.210 | **frontier 保留，certificate 学习仍需优化** |
| exact CPU process acceleration | 语义等价性已由上一轮测试验证 | **保留** |
| independent calibration + paired replay | 因前置 gate 未执行 | **实验协议保留，尚无本轮实证** |

## 4. V51 八项修改逐项判断

| # | V51 修改 | 是否起作用 | 结论与 V52 处理 |
|---:|---|---|---|
| 1 | Strong foundation gate | **机制起作用、定义有缺陷** | 成功阻止不可归因实验，但错误绑定 direct pair/selector。改为 immutable anchor-only gate。 |
| 2 | 冻结 foundation interface | **未执行** | residual stage 未启动；epoch 0 后整体退化支持“防 drift”的必要性，继续保留。 |
| 3 | semantic head reset | **未执行但证据强支持** | direct pair 显著弱于 dense local，继续重置 pair/variance head。 |
| 4 | anchor preserve/correct losses | **未执行** | 无结果可判定；V52 保留，并配合 boundary pair curriculum。 |
| 5 | full-margin certified residual gate | **未执行** | 无结果可判定；仍是防止无授权 action flip 的理论核心。 |
| 6 | three-way causal protocol | **未执行，原 health gate 还存在调度冲突** | V52 修复 sparse exact coverage 检查，继续使用三路对照。 |
| 7 | `inf-inf` 非有限修复 | **有效** | 本轮没有复现原有 pair-margin runtime warning；继续保留。零计数诊断允许为 NaN，但关键 loss 必须 finite。 |
| 8 | algorithm gate 与 latency gate 分离 | **逻辑有效但未走到该阶段** | 本次停止原因是 foundation gate，不是 latency；继续保留。 |

## 5. CCF-A / fixed-budget 闭环门槛估计

这不是官方录用线，而是按 2024–2026 nuPlan 论文竞争强度给出的工程目标：

### 开发阶段

- CL20：只用于发现 crash、token mismatch、control 不配对等工程问题，不用于论文结论。
- CL100：候选相对 matched full-budget 和 B=16 baselines 不退化，安全硬指标不能出现系统性回退。

### 投稿级主结果

- Val14 约 1.1k 场景，NR/R 两种 closed-loop；
- 官方 Test14 / Test14-hard 或等价公开协议；
- 至少 3 次独立闭环运行或 scenario bootstrap 95% CI；
- 在 **B=16 固定预算** 下，对所有同预算 baseline 达到最佳；
- 相对 full-information planner 的 R-CLS gap 最好 <= 1.0 分，最多不超过 2.0 分；
- 以当前公开强方法为参照，Val14 reactive 目标应放在约 90+；若要同时宣称通用 nuPlan SOTA，则需要逼近 93 左右的强 hybrid/rule-based 区间；
- collision、drivable、TTC 等乘法/硬安全项必须无显著下降；只靠 progress 拉高总分不足以支撑 CCF-A。

当前 V51 的闭环场景数为 0，因此与门槛之间不是“差几分”，而是缺少完整证据链：

`anchor PASS -> residual open-loop gate PASS -> paired CL20 -> paired CL100 -> Val14/Test14`

## 6. V52 BFAR-DBAP 算法主线

### 6.1 核心 idea 是否成立

主线是成立的，但必须从“重构所有 evidence”收敛为“保护可能改变 winner 的 boundary evidence”：

1. foundation 给出完整 base+local margin `m_F(a,b)`；
2. 对 teacher winner、hard-feasibility crossing、near-tie rivals 定义 flip-critical pair set；
3. 训练 proposal/selector 在成本 B=16 内保留对这些 pair 的 decisive atoms；
4. residual 只在 full foundation margin 附近、且 uncertainty-shrunk correction 足以通过认证时允许翻转；
5. 对 far-correct pair 施加 do-no-harm；
6. 最终以 action preservation、paired regret 和闭环安全验证，而不是平均 evidence reconstruction 验证。

这比“预算压缩”本身更有 novelty：它把 **pair boundary curriculum、budgeted evidence coreset 和 certified action flip** 统一到同一个决策对象上。

### 6.2 V52 新增算法点

**Boundary-Focused Anchor-Residual DBAP (BFAR-DBAP)**：

- Immutable Factorized Anchor：只要求将冻结的 base+local 接口可用；
- Quota-Constrained Boundary Pair Curriculum：winner/hard/near 三类 pair 各有保底配额，防止某一类淹没其余决策边界；
- Periodic Exact Full-Graph AOCC Distillation：稀疏 step 上以完整图和 exact B=16 selector 纠正近似训练；
- Primary-Budget Exactness：论文主预算 B=16 每次 exact 事件必训，辅助预算稀疏采样；
- Full-Margin Certified Residual：未经认证的 residual 不得改变 foundation ordering；
- Core-idea gate：新增 proposal/selected/effective decisive recall 和 interaction decisive recall 门槛。

## 7. 下一阶段最关键的优化顺序

1. 复用 V51 anchor，先确认 V52 anchor-only gate PASS；
2. 用速度版训练 residual/selector，查看 `train_pair_sample_ms_per_step`、`training_pair_fraction`、`selector_exact_fraction`；
3. 先看 residual 是否在不伤害 anchor 的前提下提高 pair-full/action match；
4. 再看 B=16 selected decisive recall、certificate fraction、fallback；
5. open-loop 三路 gate PASS 后再跑 paired CL20；
6. CL20 只要安全失败，优先优化 hard-crossing evidence 和 fallback，不再继续加开放式 reconstruction loss；
7. CL100 达标后再扩展 Val14/Test14。

## 8. 不能做出的声明

- 当前不能声称 V51 residual 无效，因为它没有训练；
- 当前不能声称任何 closed-loop 提升或 SOTA；
- V51 foundation 的 direct pair/selector 失败不能否定 base+local anchor；
- V52 的加速倍数必须由 A30 上的新训练日志确认。
