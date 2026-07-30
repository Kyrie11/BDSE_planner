# v50 结果诊断与 v51 FAR-DBAP 算法/工程优化报告

日期：2026-07-30

## 1. 结论先行

当前 `outputs_v50_dbap_ri_checkpoint_independent_2gpu_v1` **不能支持 v50 有效，更不能支持闭环 SOTA 声明**。失败并非 AOCC 预算选择器本身失效，而是以下三类问题叠加：

1. **foundation 过弱**：重建 control 的 teacher-action match 只有 `0.134`，pair-full match 只有 `0.124`；它不足以作为“稳定基座”。
2. **v50 训练破坏了已有接口**：v50 best 的 teacher-action match 降至 `0.068`，winner-rival sign accuracy 从 `0.678` 降至 `0.506`，near-tie sign accuracy 从 `0.494` 降至 `0.307`。训练日志显示 epoch 1 最好，之后整体持续恶化，符合 representation/interface drift，而非正常收敛。
3. **残差干预定义与最终决策边界不一致**：原门控只按 evidence/local margin 决定是否信任残差，没有纳入 base cost margin；near-tie 反而获得更低 trust，且符号冲突被强抑制，导致“真正需要纠错的 pair”无法得到充分干预。

因此，本次不采用继续微调 v50 超参数的方式，而是升级为：

> **v51 FAR-DBAP：Foundation-Anchored Residual Decision-Boundary Action Preservation**

核心是冻结并质量门控 foundation，重置语义不兼容的 residual heads，针对完整 foundation margin（base + local）实施“远边界不伤害、近边界可认证纠错”，并用 candidate / same-checkpoint local control / foundation control 三路配对实验完成因果归因。

---

## 2. 论文 idea 与代码实现的对齐

论文主线是：在有限 candidate bank 上，用 dense base interface 与局部 evidence atoms 表达 teacher cost，通过 pair-conditioned margins、固定预算 evidence selection、AOCC/certificate 和 tournament，在预算受限时保留足以决定动作的证据。

当前最有价值且应保留的论文结构为：

- **候选动作集合与有限预算决策问题**：问题定义清晰，适合做 decision sufficiency，而不是一般轨迹预测。
- **base + local evidence 的可解释分解**：提供了证据级诊断和预算选择的物理接口。
- **pair-conditioned rival reasoning**：比只拟合单动作 scalar score 更贴近 argmin 决策边界。
- **AOCC / one-sided certificate / rival coverage**：能形成预算压缩后的 action-preservation 理论链条。
- **exact-budget selector 与 anytime/frontier 机制**：已有实现和实验指标证明执行链路是工作的。

v51 不改变上述主线，而是在其上增加一个必要的新层：**foundation-anchored selective intervention**。这让论文从“选哪些证据”进一步回答“学习到的新交互项何时被允许改变动作”，更符合真实部署中的安全、可归因和闭环稳定需求。

---

## 3. v50 实验结果：哪些有效，哪些无效

### 3.1 核心指标

| 指标 | v50 candidate | foundation control | 差值 | 判断 |
|---|---:|---:|---:|---|
| teacher action match | 0.068 | 0.134 | -0.066 | 严重退化 |
| full-interface action match | 0.341 | 0.353 | -0.012 | 退化 |
| pair-full action match | 0.068 | 0.124 | -0.056 | 严重退化 |
| winner-rival sign accuracy | 0.506 | 0.678 | -0.172 | 决策边界退化 |
| near-tie sign accuracy | 0.307 | 0.494 | -0.187 | 最关键边界退化 |
| evidence sufficiency | 0.0770 | 0.0674 | +0.00965 | 略有提升，但不足以转化为动作收益 |
| teacher regret mean | 13113.5 | 10486.8 | +2626.7 | 退化 |
| latency p95 | 1161.4 ms | 1227.3 ms | -66.0 ms | 更快，但仍远高于 500 ms 目标 |

paired regret 也全面变差：candidate median `171.95` 对 control `30.06`，p90 `40471` 对 `36570`，CVaR90 `64854` 对 `59971`。所以不是少量异常值造成的均值偏差，而是动作质量的系统性下降。

### 3.2 已被结果支持、应保留的设计

#### A. residual intervention 作为局部纠错机制有信号

v50 best checkpoint 中：

- local pair-full match：`0.050`
- residual pair-full match：`0.068`
- beneficial residual rate：`0.026`
- harmful residual rate：`0.008`

这表明 residual branch **并非完全无效**：在当前最优 epoch，它相对 local interface 带来约 `+1.8` 个百分点，并且 beneficial > harmful。问题不是“残差思想错误”，而是基础接口、训练范围和授权规则错误。

#### B. AOCC 与固定预算执行链路有效

- AOCC certified pair fraction：`0.7520`
- calibrated bound active：`1.0`
- exact tournament target active：`1.0`
- fixed-budget fill：`16/16 = 1.0`
- frontier retained weight：`0.8038`

这些指标说明预算选择、certificate、frontier 和固定填充按预期运行。它们没有直接修复错误的 margin predictor，但也不是当前性能崩溃的首要原因。

#### C. evidence family 分配不是主要瓶颈

interaction atoms 平均占 `12.69/16 = 79.3%`，低于现有 `85%` cap；frontier retained weight 超过 `80%`。当前没有证据支持继续增加交互证据配额或进一步复杂化 selector。更高优先级是修复 margin interface。

#### D. exact CPU selector 的多进程并行应保留

上一轮实现的 persistent `spawn` workers 在代表性测试中将 exact selector 稳态耗时从 `1.679 s` 降至 `0.428 s`，mask 逐元素一致。该优化不改变算法目标，适合继续作为工程基线。

### 3.3 已被证明无效或存在严重缺陷的设计

#### A. 4-epoch rebuilt foundation 不够强

当前重建 foundation 的 teacher match 只有 `0.134`。弱 foundation 会让 residual stage 同时承担 representation learning、base/local cost reconstruction 和 boundary correction，导致实验不再能归因于 DBAP-RI。

旧 foundation 配置还存在 raw base loss 数值支配风险：未归一化的 `L_base` 量级远高于其他损失。v51 改为归一化 base loss、12 epoch、质量门控，未通过门控就停止 residual 训练。

#### B. 全模型微调导致 interface drift

训练曲线清楚显示：

- epoch 1：teacher match `0.068`，near-tie `0.307`
- epoch 3：teacher match `0.038`，near-tie `0.290`
- epoch 5：teacher match `0.055`，near-tie `0.262`
- epoch 7：teacher match `0.041`，near-tie `0.268`

同时 harmful residual 从 `0.008` 增至 `0.040`。这不是“训练不够”，而是训练越久越破坏已有决策边界。v51 主实验冻结 scene/action/base/local interface，仅训练 residual pair/variance 与 proposal-family heads。

#### C. warm-start 时错误复用了语义不同的 pair head

foundation 的 `pair_head` 是 direct pair-margin predictor，而 v50 中同形状参数被解释为 residual-over-local predictor。形状兼容不代表语义兼容；直接加载会让 residual 在 step 0 就产生大而错误的干预。

v51 在加载 foundation 后显式重置：

- `pair_head`
- `pair_var_head`

并保留 foundation 的稳定表示层。

#### D. 原 residual gate 没有针对完整决策边界

最终 pair 决策取决于：

\[
M_F(a,b)=\Delta J_0(a,b)+\sum_i \Delta g_i(a,b),
\]

但原 gate 只依据 local evidence sum。若 base margin 很大、local margin 接近零，原 gate 会错误认为该 pair 是 near-tie；反之，若 local margin 较大但与 base 抵消，原 gate 又会错过真正 near-tie。

此外原规则采用随 `|local|` 增大的 trust，使 near-tie 获得最低信任；`disagreement_penalty=1` 又把与 local 符号相反的 residual 几乎清零。这恰好阻止了 anchor-wrong pair 的必要纠错。

v51 的 gate 改为：

- anchor = normalized base margin + local evidence margin；
- trust 在完整 anchor near boundary 时最高，远离边界时只保留最小 trust；
- 非认证 flip 的 aggregate correction 被限制在不能翻转 anchor 的范围；
- 只有 residual lower confidence bound 足以跨越完整 anchor 且留出 `flip_margin` 时，才允许翻转；
- disagreement penalty 从完全禁止改为软惩罚。

#### E. v50 control 与 candidate 配置不匹配

v50 candidate 使用 v50 DBAP-RI runtime，而 control 使用 v43 control 配置。两者在 selector mode、proposal top-M、max pairs、fallback、安全项、calibration 等多处不同。即便 checkpoint 来源相同，也无法把差值严格归因于 residual algorithm。

v51 改为三路协议：

1. **candidate**：v51 checkpoint + FAR residual enabled；
2. **local control**：同一个 v51 checkpoint、同一个 selector/runtime，只关闭 residual intervention；
3. **foundation control**：foundation checkpoint，使用 matched v51 runtime/selector。

这样可分别测量 residual-only gain 和 total algorithm gain。

---

## 4. v51 FAR-DBAP 算法设计

### 4.1 Strong Foundation Quality Gate

foundation 使用新配置 `v51_strong_foundation_anchor_2gpu.yaml`：

- 12 epochs；
- normalized base loss；
- exact selector 后期开启；
- early stopping；
- 单独 open-loop replay 后执行质量门。

默认最低条件：

- teacher match ≥ `0.22`
- full-interface match ≥ `0.45`
- pair-full match ≥ `0.25`
- winner-rival sign ≥ `0.60`
- near-tie sign ≥ `0.45`
- evidence sufficiency ≥ `0.06`
- latency p95 ≤ `1500 ms`

这些阈值不是 SOTA 阈值，而是“允许研究 residual 的最低可用 anchor”。若失败，流水线停止，不允许通过调低门槛掩盖 foundation 问题。

### 4.2 Immutable Foundation Interface

主实验冻结除以下模块外的所有参数：

- `pair_head`
- `pair_var_head`
- `proposal_feature_proj`
- `family_embed`
- `family_activity_proj`
- `family_head`
- `proposal_head`

这保证 base/local action interface 不被 residual optimization 改写。论文中的 foundation 可被定义为固定 reference policy/interface，而 DBAP 只学习预算选择和有条件纠错。

### 4.3 Foundation-Anchored Do-No-Harm Objective

令 teacher pair 方向为 \(s_T\in\{-1,+1\}\)，foundation anchor margin 为 \(M_F\)，部署后 margin 为 \(M_D\)。

对于 teacher-correct 且远离边界的 pair：

\[
\mathcal L_{\text{preserve}}
=\left[\max(\rho\,s_T M_F,m_{\min})-s_T M_D\right]_+.
\]

它不要求 residual 重建所有 teacher margin，只要求不要削弱已经正确、稳定的 foundation 决策。

对于 foundation-wrong 或 near-tie pair：

\[
\mathcal L_{\text{correct}}
=\left[m_T-s_T M_D\right]_+,
\]

其中 \(m_T\) 是截断后的 teacher margin target，错误 pair 和 near-tie pair 使用更高权重。

该目标比全量 margin regression 更符合论文的 decision sufficiency：只在动作边界需要时使用额外模型容量。

### 4.4 Full-Margin Certified Residual Authorization

部署时先计算完整 anchor：

\[
M_F=\Delta J_0+\sum_i \Delta g_i^{\text{local}}.
\]

残差 aggregate correction 为 \(C_R\)，方差为 \(\sigma_R^2\)。非认证状态下约束：

\[
|C_R|\le \rho |M_F|, \quad \rho<1,
\]

因此 foundation sign 不能被翻转。

只有满足：

\[
|C_R|-\beta\sigma_R \ge |M_F|+m_{\text{flip}}
\]

并且 proposed margin 确实跨越零点时，才允许 residual flip。该规则直接作用于最终决策边界，而不是 evidence-only proxy。

### 4.5 Exact-Budget AOCC 保持不变

本轮不改变以下部分，以避免归因混乱：

- budget B=16 主路径；
- AOCC exact selector；
- certificate/frontier；
- tournament pair union；
- v50 的 selector caps 与 runtime budget 参数。

只有在 v51 通过 open-loop gate 后，才进行 selector/latency 的第二阶段优化。

---

## 5. 理论贡献路径

现有论文 theorem 主要证明：当 selected evidence 对所有关键 rivals 提供正的 one-sided lower-confidence margin，且 rival coverage 成立时，预算接口可保留 teacher action 或给出 regret bound。

v51 可增加一个 **selective intervention theorem**：

### Proposition 1: Far-anchor action preservation

对 teacher-correct、远离边界的 anchor pair，若 gate 保证未认证 correction 满足 \(|C_R|\le\rho|M_F|,\rho<1\)，则：

\[
\operatorname{sign}(M_F+C_R)=\operatorname{sign}(M_F).
\]

因此 residual 不会破坏该 pair 的 foundation ordering。

### Proposition 2: Certified boundary correction

若 residual correction 的 lower confidence bound 足以跨越完整 anchor margin并留出正 margin，则允许 flip 后的新 pair ordering 具有显式 confidence certificate。

### Theorem route: error decomposition

最终 action failure 可以分解为：

\[
P(\hat a\ne a_T)
\le P(E_F)+P(E_R\mid E_F^c)+P(E_B\mid E_F^c,E_R^c),
\]

其中：

- \(E_F\)：foundation 本身在关键 rival 上错误；
- \(E_R\)：未经认证的有害 intervention 或认证纠错失败；
- \(E_B\)：预算选择 / rival screening / AOCC coverage 失败。

这比仅证明预算证据压缩更有 novelty，因为它把学习型 interaction residual 的安全授权纳入 action-preservation 证明。

注意：以上是可写成正式 theorem/proof 的结构，不等于当前已有实验已经达到 SOTA。SOTA 必须由 fresh paired open-loop、CL20、CL100 和公开 benchmark 对比验证。

---

## 6. 工程问题与修复

### 6.1 非有限 margin

v50 日志出现 `inf - inf` warning。无效 candidate 的 base cost 为 `inf`，在 pair matrix 构造和 metrics 中直接相减会产生 NaN，污染 selector 和统计。

修复：

- tournament pair-margin matrix 先用统一 finite sentinel 替换 invalid costs；
- scale 只由 valid-valid pairs 估计；
- metrics 使用相同 finite-cost policy；
- 新增无 RuntimeWarning/NaN 单测。

### 6.2 checkpoint 语义与 provenance

- 保留 checkpoint-independent foundation resolver；
- 默认拒绝 v47/v48/v49 等 algorithm-specific checkpoint 作为论文主实验初始化；
- 新 command 使用全新 `OUT_ROOT` + `FOUNDATION_POLICY=rebuild`；
- candidate、local control、foundation control 各自独立 calibration，并记录 provenance。

### 6.3 训练/推理一致性

训练与 runtime 均调用同一 full-anchor gate 逻辑：

- training：Torch differentiable continuous path；
- runtime：NumPy exact deployment path；
- sign authorization mask 由 detached values 计算，避免伪梯度穿过离散 flip 决策。

### 6.4 三路 deterministic replay

三路评估必须使用完全相同的 scenario token/timestamp keys。gate 会拒绝：

- 缺行；
- 重复 key；
- 非有限训练指标；
- selector exact fraction < 0.99；
- calibration 未独立启用；
- candidate paired regret 相对任一 control 退化。

### 6.5 latency 归因

v50 candidate p95 为 `1161 ms`，其中 prediction stage mean `579 ms`，selector mean `78.8 ms`。多进程 exact selector 已显著改善 selector，因此下一阶段若仍未达 500 ms，应优先 profile：

- pair-conditioned prediction 的 tensor materialization；
- proposal/pair head 的重复 forward；
- CPU↔GPU synchronization；
- runtime feature/candidate preprocessing。

在动作质量通过 gate 前，不建议用近似 selector 或降低 exact coverage 换速度，否则无法区分算法与工程近似造成的变化。

---

## 7. v51 open-loop gate

默认要求：

- candidate vs foundation teacher match ≥ `+0.015`；
- candidate vs same-checkpoint local control teacher match ≥ `+0.005`；
- evidence sufficiency vs foundation ≥ `+0.010`；
- winner-rival 与 near-tie sign 不退化；
- candidate internal local interface 与 same-checkpoint local control drift ≤ `0.005`；
- residual pair-full gain ≥ `+0.005`；
- harmful residual ≤ `0.03`，且 beneficial > harmful；
- AOCC certified pair fraction ≥ `0.55`；
- frontier retained weight ≥ `0.55`；
- budget fill ≥ `0.95`；
- interaction budget fraction ≤ `0.85`；
- fallback ≤ `0.50`；
- latency p95 目标为 `500 ms`，默认作为独立工程告警；设置 `ENFORCE_LATENCY_BEFORE_CL=1` 时才成为硬门；
- paired regret median/p90 相对两个 control 均不退化。

算法质量 gate PASS 后，流水线执行 candidate、local control、foundation control 的 paired CL20。这样已知的预测延迟不会阻断算法闭环归因；但 latency 未通过时严禁声称 real-time deployment。CL100 默认关闭，先看 CL20 的 safety/progress/comfort 分解。

---

## 8. 必做消融与论文表格

主实验之后建议只做以下高信息量消融：

1. **FAR full model**：完整 v51。
2. **No anchor preservation**：去掉 `L_anchor_preserve`，验证 harmful intervention 是否上升。
3. **No certified flip**：完全禁止 flip，只允许 margin strengthening，验证纠错收益是否消失。
4. **Evidence-only gate**：复现 v50 门控，证明 full-margin authorization 的必要性。
5. **No residual / same-checkpoint local control**：测 residual-only gain。
6. **Foundation control**：测 total gain。
7. **B=8/16/24**：预算-性能-延迟曲线；训练主路径保留多预算或单独微调，评估必须配对。
8. **Exact vs learned proposal only**：仅作为速度消融，不能替代主结果。

论文结果应至少报告：teacher match、pair sign、near-tie sign、paired regret median/p90/CVaR、harmful/beneficial intervention、certificate coverage、fallback、latency、CL score、安全事件、progress、comfort，以及跨 seed 置信区间。

---

## 9. 下一步执行命令

使用新的代码目录和全新输出目录，避免自动复用弱的 4-epoch foundation：

```bash
cd /path/to/bdse_v51_far_dbap

unset V30_CKPT_IN
unset FOUNDATION_CKPT
unset CONTROL_CKPT

export NUPLAN_ROOT=/data0/senzeyu2/dataset/nuplan
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2
export BDSE_VAL_CACHE_ORIGINAL=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2
export OUT_ROOT=outputs_v51_far_dbap_2gpu_v1

PIPELINE_DETACH=1 \
FOUNDATION_POLICY=rebuild \
REBUILD_FOUNDATION_IF_MISSING=1 \
RECOVER_SAFE_FOUNDATION_COPIES=0 \
ALLOW_ALGORITHM_CHECKPOINT_INIT=0 \
RUN_CLOSED_LOOP_AFTER_GATE=1 \
RUN_CL100_AFTER_CL20=0 \
ENFORCE_LATENCY_BEFORE_CL=0 \
EXACT_SELECTOR_CPU_BACKEND=process \
EXACT_SELECTOR_WORKERS_PER_RANK=4 \
GPUS=0,1 \
bash V51_FAR_DBAP_NEXT_COMMANDS.sh
```

监控：

```bash
tail -f "$OUT_ROOT"/logs/pipeline_*.log
```

foundation gate：

```bash
cat "$OUT_ROOT"/logs/v51_foundation_quality_gate.out
```

v51 训练：

```bash
tail -f "$OUT_ROOT"/logs/train_2gpu.out
```

最终 open-loop gate：

```bash
cat "$OUT_ROOT"/logs/v51_far_dbap_gate.out
```

若 foundation gate 失败，先停止 residual 实验，不要降低论文主实验阈值。优先查看 foundation 的 base loss scale、teacher match、pair sign 曲线和 best-checkpoint selection。

---

## 10. 验证状态与声明边界

本地已完成：

- Python compile：通过；
- 5 个 v51 YAML load/schedule 校验：通过；
- Shell syntax：通过；
- 完整单元测试：`173 passed, 6 warnings`；
- full-anchor far-boundary no-flip 测试：通过；
- near-boundary certified flip 测试：通过；
- warm-start reset + freeze 测试：通过；
- invalid candidate 非有限 margin 测试：通过。

本环境没有 nuPlan 数据和两张 A30，无法实际完成 fresh training、open-loop replay 或 closed-loop simulation。因此不能预先声称 v51 一定 PASS，更不能声称已经达到 SOTA。当前交付的是一个更可归因、可证伪、理论链条更完整的算法版本和严格流水线。
