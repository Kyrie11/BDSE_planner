# BDSE v48 实验诊断与 v49 DBAP 算法升级报告

## 0. 核心结论

1. **v48 的 closed-loop 没有执行是预期 gate 行为，不是 CL20 调用丢失。** `check_v48_dbce_gate` 返回非零后，`set -euo pipefail` 终止流水线，因此没有进入 CL20。
2. **上一轮提出的五项投稿组合尚未被完整证明。** v48 只较强地验证了 tournament-active certificate frontier 与 pair-full→budget compression；没有验证因果边界 evidence、可靠的 local+residual interface，也没有真正评估固定 B=16 的 action preservation。
3. **模型仍没有学会因果边界 evidence。** Proposal head 学到了一部分 boundary-related ranking，但 pair critical loss 基本不下降，最终选择仍被 interaction family 占据，且 dense/local 的 near-tie 正确性在 sparse pair interface 中被破坏。
4. **v48 的主矛盾已经从 certificate/compression 转移到 candidate quality + dense/local→pair interface。** Pair-full→budget harmful compression 只有 0.024，而 dense→pair harmful interface rate 为 0.136。
5. v49 将核心 claim 收敛为 **Deployment-Boundary Action Preservation (DBAP)**：纯 leave-one-out 边界 criticality、受约束 sparse residual、严格 fixed-budget nested prefix、跨 family 前缀容量约束，以及 candidate/interface/compression 分层评估。

---

## 1. v48 gate 与 closed-loop

### 1.1 Gate 实际结果

| 指标 | v48 | Control/门槛 | 判断 |
|---|---:|---:|---|
| Teacher action match | 0.263 | 0.238，要求至少 +0.020 | PASS，+0.025 |
| Evidence sufficiency | 0.081607 | 0.070400，要求至少 +0.010 | PASS，+0.011207 |
| Winner–rival sign | 0.676385 | 0.638368 | PASS |
| Near-tie sign | 0.460480 | 0.583448 | FAIL，-0.122969 |
| Pair-full interface match | 0.272 | ≥0.300 | FAIL |
| Certified-pair fraction | 0.831067 | ≥0.500 | PASS |
| Frontier retained weight | 0.613374 | 诊断项 | 可接受，但需消融 |
| Interaction budget fraction | 0.948399 | ≤0.850 | FAIL |
| Fallback trigger rate | 0.162 | 相比 v47 应下降 | 明显改善 |
| Planner latency p95 | 757.20 ms | ≤500 ms | FAIL |
| Median teacher regret | 41.36 | Control 30.62 | FAIL |
| p90 teacher regret | 35,899.54 | Control 30,794.33 | FAIL |
| Exact selector fraction | 1.0 | ≥0.99 | PASS |
| Train rows / unique epochs | 16 / 16 | 必须一致 | PASS，无双写者污染 |

### 1.2 为什么 closed-loop 没执行

v48 的脚本已经包含真正的 gate 后 CL20 调用。失败原因是 open-loop gate 同时触发了 near-tie、pair-full、family collapse、latency 和 regret 五类失败，gate 命令返回非零，父流水线在 CL20 之前退出。这是正确的实验保护逻辑。

不建议绕过 gate 直接将本轮 CL20 作为论文主结果。可以单独设置 `RUN_CLOSED_LOOP_AFTER_GATE=0` 做算法调试，但主实验仍应通过 paired gate 后再进入闭环。

---

## 2. 五项投稿组合是否被完整证明

| 投稿组合 | v48 状态 | 证据与缺口 |
|---|---|---|
| 1. Leave-one-out deployment-boundary criticality | **仅实现，未学会** | 代码中已有 LOO deficit，但 `positive_support_floor=0.03`、旧 positive-support critical loss、interaction upweight 和高权重通用 proposal loss共同污染目标；`L_cf_critical_pair` 约 3.599→3.603，未下降。 |
| 2. Integrable local margin + uncertainty-shrunk sparse residual | **未验证，实际失败** | Dense/local near-tie 为约 0.683，pair interface 后仅 0.460。v48 先形成 `local+residual` 再检测冲突，平均 local 权重只有约 0.205，即 residual 仍保留约 79.5%。 |
| 3. 与最终 tournament 一致的 nested certificate frontier | **大体有效，但原 claim 不精确** | exact target rate=1.0，certified-pair=0.831，fallback=0.162，说明 frontier/certificate 有效；但平均只选 7.403 个原子，属于“认证后提前停止”，不是严格 B=16 action preservation。 |
| 4. candidate/interface/compression 三层误差分解 | **诊断框架成立** | dense full=0.336，sparse full=0.309，pair full=0.272，budget=0.263；dense→pair harmful=0.136，pair→budget harmful=0.024。该分解成功指出 interface 是主瓶颈，但还不是性能贡献本身。 |
| 5. fixed-budget action preservation | **未证明** | v48 实际平均预算支持 7.403/16；budget-vs-pair-full=0.882 只能说明提前停止前缀较少破坏自己的 pair-full 动作，不能等价于固定 B=16 曲线。 |

### 结论

组合不能按“已完整验证”写入论文。目前可以诚实表述为：

> v48 验证了 exact-tournament certificate frontier 可显著提高可认证率并降低 fallback，同时显示主要误差已集中到 candidate quality 与 local-to-pair interface；但 boundary-critical evidence learning 和严格 fixed-budget action preservation 尚未成立。

---

## 3. 模型是否学会因果边界 evidence

### 3.1 学到的部分

- Proposal decisive recall：0.8153。
- Proposal interaction-decisive recall：0.8052。
- Teacher action match 相比 control 提高 0.025。
- Winner–rival sign 相比 control 提高约 0.038。
- Certificate 与 fallback 显著改善。

这些说明模型学到了一部分“与决策相关、容易进入 proposal pool 的 evidence”。

### 3.2 没学到的关键部分

- `L_cf_critical_pair` 从约 3.599 到 3.603，几乎没有优化；而 proposal CF loss 从约 2.972 降到 2.324。
- Selected decisive recall 只有 0.2708，selected interaction-decisive recall 只有 0.3012。
- 预算 94.84% 被 interaction family 占据。
- Pair near-tie sign 0.4605，明显低于 dense/local near-tie 约 0.683。
- Pair-full teacher match 只有 0.272。
- Teacher regret 中位数与 p90 均差于 control。

因此最准确的判断是：

> Proposal head 学会了部分“边界相关候选召回”，但 pair head 没有学会稳定、可积、可部署的因果边界贡献；selector 也没有形成真实的跨 family 单位成本竞争。

这里的“因果”仅指 leave-one-out intervention over the teacher evidence decomposition，不等同于真实世界因果干预。论文当前正文明确称 pair label 不是 causal counterfactual label，因此需要同步修改术语：建议使用 **deployment-boundary intervention criticality** 或 **leave-one-out boundary criticality**。

---

## 4. v48 七项算法优化是否发挥作用

| v48 优化 | 判断 | 结果解释 |
|---|---|---|
| 1. Leave-one-out decision-boundary criticality | **部分实现，训练无效** | LOO 公式存在，但目标被 positive-support floor、旧 critical loss、interaction boost 污染；pair CF loss不降。 |
| 2. Teacher-nearest + model-confused rival union | **部分有效** | Broad winner/rival sign 提升，但 hardest near-tie 继续下降，说明 rival coverage 改善而 margin estimator 未改善。 |
| 3. Confidence-shrunk local + residual | **无效/实现错误** | local weight 均值约 0.20，且冲突检测对 `local+residual` 与 local 做比较，无法可靠识别 residual 与 local 的原始冲突。 |
| 4. Tournament-active AOCC frontier | **明显有效** | certified-pair 0.135→0.831，fallback 0.887→0.162，exact tournament target=1.0。 |
| 5. 恢复跨 family 预算竞争 | **无效** | interaction fraction 仍为 0.948；降低 quota/boost 不足以对抗候选池和训练损失的系统性偏置。 |
| 6. 降低过度保守性 | **有效** | 证书可达性、fallback 和 latency 均改善；但 latency 仍未达标。 |
| 7. 重写 best-checkpoint 标准 | **理念正确，工程实现不完整** | validation 没有计算真实 pair-full 与 family collapse，score 静默回退到 sparse-full；effective hard recall 又因 structural bypass 被报告为 1.0。Epoch 1 被选为 best，但关键接口指标未真正参与选择。 |

附加工程项：

- 单写者锁有效：16 行训练记录对应 16 个唯一 epoch。
- gate 后自动 CL20 已正确实现。
- 双 GPU open-loop 分片与合并可用。

---

## 5. 当前算法层缺陷

### 5.1 Candidate ceiling 仍低

- v48 candidate regret median 41.36，control 30.62。
- Test 的 safe-candidate-exists 仅 0.4937，val 为 0.7173。
- Candidate teacher ADE p90：test 19.32 m，val 12.92 m。

Evidence selector 不可能修复候选集中不存在的安全/合适动作。论文结果必须分开报告 candidate coverage failure 与 evidence/interface failure。

### 5.2 Pair head 的目标熵过高且互相冲突

v48 同时存在：

- LOO CF listwise target；
- 旧 positive-support critical target；
- general proposal BCE/rank；
- interaction-only/interaction boost；
- action、pair regression、cycle、certificate 等多目标。

最终 proposal loss 可以下降，但 pair head 的 boundary-critical distribution 接近均匀分布，`L_cf_critical_pair≈log(active atoms)`。

### 5.3 Residual 没有被约束为 residual

Pair residual 可以全幅覆盖 local margin，破坏可积性和 near-tie。正确结构应为：

\[
M_{ab}=M^{local}_{ab}+\alpha_{ab}R_{ab},\quad 0\le\alpha_{ab}\le\alpha_{max}<1,
\]

其中 \(\alpha\) 在高方差、near-boundary、符号冲突、幅值异常时下降。

### 5.4 “Anytime certificate”与“fixed-budget claim”混用

v48 的 selector 在认证后提前停止，平均使用 7.403 个 atom。它证明的是自适应查询效率，而论文核心说法是固定预算 B 下的 action preservation。两者应同时报告但不能互相替代：

- first-certified prefix：效率指标；
- exact-B nested prefix：核心 fixed-budget 结果。

### 5.5 Family collapse 不能只靠 soft boost 修复

Proposal pool 平均 interaction 约占绝大多数，selector 再降低 interaction boost 也无法产生不存在的 non-interaction 候选。必须在 **nested prefix 本身**施加跨 family capacity，而不是仅对 proposal logits 做软调整。

---

## 6. 当前工程层缺陷

### 6.1 Best checkpoint 使用了缺失指标的静默回退

v48 validation 没有计算：

- `val_pair_full_interface_action_match`；
- dense→pair harmful interface；
- pair→budget harmful compression；
- selector interaction fraction；
- exact-budget fill fraction。

旧 score 在缺失 pair-full 时回退到 sparse-full，因此 best checkpoint 选择并未真正对齐最终部署路径。

### 6.2 Test cache 的 `--resume` 不校验配置

旧 resume 只检查文件存在、大小或最小 schema，不比较 feature/label 参数。当前“快速生成”命令显式使用 `--no-include-drivable-polygons`，而 train/val 诊断显示 `include_drivable_polygons=true`。继续写入同一 output 目录可能形成混合 cache。

### 6.3 Test 与 train/val 存在真实分布偏移

| 数据集指标 | val | test |
|---|---:|---:|
| 样本数 | 58,418 | 18,908 |
| Quality keep | 0.9175 | 0.6395 |
| Logged ego far-from-route reject | 0.0513 | 0.3274 |
| Route distance tail p95-p90 | 3.263 m | 20.655 m |
| Safe candidate exists | 0.7173 | 0.4937 |
| Full-interface match | 0.9657 | 0.9006 |
| B16 oracle sufficiency | 0.9120 | 0.8065 |
| Candidate-to-log ADE p90 | 12.915 m | 19.316 m |

重新构建可以消除配置和缓存污染，但不能保证消除 official public test 自身的场景分布差异。后续应把“feature parity”和“distribution shift”分开报告。

### 6.4 Control freshness 未绑定 split provenance

旧 pipeline 判断 control 是否可复用时只比较 control config/checkpoint，没有比较 val_tune manifest 与 calibration split provenance。重建 split 后可能误用旧 control。v49 已修复。

### 6.5 Latency 主瓶颈是 prediction

- stage_predict：约 479.69 ms；
- stage_selector：约 1.14 ms；
- stage_tournament：约 4.98 ms；
- total mean：约 534.77 ms；
- p95：757.20 ms。

继续微调 selector 的收益有限。应先减少 pair materialization、refinement pair 数、proposal pool 中无效 interaction 候选，并在 A30 上启用稳定的 batched unique-pair scoring。

---

## 7. v49 DBAP 已实现的修改

### 7.1 纯 leave-one-out boundary target

- `positive_support_floor` 默认与配置均改为 0；
- 关闭旧 `critical_pair` / `critical_proposal`；
- 关闭旧 interaction-only critical target；
- 加入 `target_top_k_atoms=8` 与 `min_relative_gain=0.05`，避免微小正贡献把 target 重新变成全 evidence support；
- 同一稀疏 target 监督 pair utility 与 proposal ranking。

### 7.2 Bounded confidence-shrunk residual

新的部署接口直接对 residual 计算 trust：

- variance trust；
- local boundary strength；
- residual-vs-local sign conflict；
- residual/local magnitude ratio。

默认 `max_residual_weight=0.35`，即 pair head 最多提供 35% 的修正，不再能够无约束推翻 local interface。

### 7.3 Strict exact-budget nested AOCC

- 首次认证后不再结束完整 nested order；
- 继续构造 exact-B prefix；
- 同时记录 `aocc_first_certified_prefix_length/cost`，保留 anytime efficiency 结果；
- gate 新增 fixed-budget fill fraction，要求至少 0.95。

### 7.4 Cross-family prefix capacity

- 每个 nested prefix 均限制 interaction family 比例；
- v49 默认上限 0.80；
- 只有 non-interaction 已耗尽时才放松限制；
- proposal 的强制 interaction slot 从 8 降为 4，interaction boost 进一步降低。

### 7.5 真实部署路径 checkpoint 选择

Validation 现在计算：

- exact pair-full action；
- budget-vs-pair-full；
- dense→pair harmful interface；
- pair→budget harmful compression；
- selector family composition；
- planner latency；
- exact-budget fill。

缺失 pair-full/family 指标将被直接惩罚，不再静默回退。

### 7.6 Cache provenance guard

新增：

- `--resume-require-config-match`；
- 每个 split 的 `cache_provenance.json`；
- feature/label 参数不一致时拒绝 resume；
- 旧 cache 没有 provenance 时，严格 resume 会要求新目录或 overwrite。

### 7.7 双 A30 流水线

- 训练：DDP，两卡各 batch 4，global batch 8；
- open-loop：两卡场景分片并行；
- CL20/CL100：两卡 token shard 并行；
- 单写者锁与 pipeline lock；
- control freshness 绑定 split manifest/provenance。

---

## 8. Novelty 建议调整

建议把核心 novelty 从泛化的“budgeted evidence selection”收紧为以下整体：

> **Deployment-Boundary Action Preservation under a fixed planner-interface budget**：通过 teacher-defined leave-one-out boundary interventions 学习稀疏 evidence utility，以 integrable local margin 为主接口、以 uncertainty-gated pair residual 表达非可加交互，并使用与最终 tournament 相同的 exact-budget nested certificate frontier 保留动作；误差显式分解为 candidate、interface、compression 三层。

论文中应明确区分：

1. **训练 target**：teacher decomposition 上的 LOO intervention，不宣称真实世界 causal effect；
2. **部署 guarantee**：条件式 margin-preservation theorem；
3. **实验证据**：exact-B action preservation curve、first-certificate efficiency curve、三层错误分解与 paired closed-loop。

当前 theorem 只是条件保证，不证明 LOO target 最优，也不证明模型已满足 premise。v49 必须通过实验建立 premise：near-tie sign、pair-full match、fixed-B preservation、certificate calibration 和 closed-loop consistency。

---

## 9. 下一轮必须报告的结果

### 9.1 主 gate

- teacher action match gain ≥ +0.020；
- sufficiency gain ≥ +0.010；
- near-tie sign 不低于 control；
- pair-full action match ≥ 0.300；
- certified pair fraction ≥ 0.500；
- interaction budget fraction ≤ 0.850；
- fixed-budget fill ≥ 0.950；
- planner p95 ≤ 500 ms；
- median/p90 regret 均不劣于 control。

### 9.2 核心消融

1. positive-support vs pure LOO；
2. model-only rival vs teacher+model union；
3. local-only / residual-only / full residual / bounded residual；
4. early-stop certificate vs exact-B nested prefix；
5. family cap 1.0/0.9/0.8/0.7；
6. B=4/8/16/24/32；
7. certified 与 uncertified 条件性能；
8. CL20 后再 CL100，并报告 paired bootstrap confidence intervals。

### 9.3 判断算法是否真正成功

只有同时出现以下链路，才能说模型学到了 boundary evidence：

- LOO pair loss显著下降；
- proposal recall保持；
- pair-full near-tie 与 action match提高；
- exact-B budget-vs-pair-full保持；
- teacher regret改善；
- closed-loop score/安全指标在相同 token 上改善。

---

## 10. 代码验证

本次代码在当前环境完成：

- Python compileall：PASS；
- 配置加载：PASS；
- shell syntax：PASS；
- pytest：**161 passed，5 warnings**。

这只证明代码结构和单元行为通过，不等同于新算法效果已经被实验验证。v49 的实际收益仍需按下一步脚本重新训练、校准、paired replay 和 closed-loop 验证。
