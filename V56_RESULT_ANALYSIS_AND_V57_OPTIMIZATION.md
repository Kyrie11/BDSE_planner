# V56 结果诊断与 V57 WC-DCIP-BFAR 优化方案

## 0. 结论先行

本轮上传的 V56 结果不能支持“模型已经学会改善最终 action winner”。更准确的结论是：

1. **V56 的 selector/evidence 路径明显有效**：selected decisive recall 从 V55 的约 0.538 提升到 `0.6064`，effective decisive recall 为 `0.7721`，interaction decisive recall 为 `0.5751`；evidence certificate 恢复到 `0.8880`，fallback 降到 `0.111`。
2. **V56 已经在指标层面解决了 V55 的 evidence certificate 与 residual uncertainty 混合问题**，因此 minimum 指标本身没有失败项。
3. **V56 residual 没有形成可复现、净正向的 winner correction**：1000 个 paired 场景中只有 5 次 deployed candidate-local action flip，0 次把错误 action 修成 teacher winner，1 次把 teacher-correct action 改错；内部 pair-full 有 17 次 residual flip，仅 1 次有益、1 次有害，净增益为 0。
4. **Protocol FAIL 的根因是训练图被代码静默关闭，而不是 selector 结构本身失效**：V56 配置将 legacy `loss_weights.action` 设为 0，但代码把它错误地当成整个 winner/deployment loss family 的总开关，导致 exact selector、deployment selection、pair-full winner、budget winner、action-potential teacher 等关键监督全部为 0。
5. 还存在第二条隐藏断链：在多预算 B=16 的 direct-potential 训练分支中，`residual_action_potential` 没有传入最终 winner logits。即使修复总开关，selected-budget winner loss 仍可能不给 residual head 梯度。

因此，当前主要矛盾不是继续修改 selector，也不是放宽 gate，而是让 **direct evidence potential 真正接受最终 winner 级监督，并让 residual flip 严格经过独立的双证书授权**。

---

## 1. 三个 gate 为什么 FAIL

### 1.1 Protocol gate：真实的训练协议失败

失败项为：

- `last selector_exact_fraction = 0.0 < 0.015`；
- `no exact selector supervision observed`。

训练日志中 5 个 epoch 均为：

- `selector_exact_fraction = 0`；
- `L_deploy_select = 0`；
- `L_pair_full_action = 0`；
- `L_pair_full_winner_margin = 0`；
- `L_budget_preserve_pair_full = 0`；
- `L_pair_full_anchor_preserve = 0`；
- `L_action_potential_teacher = 0`。

只有 `L_residual_action_atom ≈ 0.0117` 非零，而且几乎不下降。

#### 根本原因 A：legacy action 总开关错误关闭整个 winner family

V56 配置中：

```yaml
training:
  loss_weights:
    action: 0.0
    deployment_selection: 6.0
    pair_full_action: 12.0
    pair_full_winner_margin: 10.0
    budget_preserve_pair_full: 8.0
    pair_full_anchor_preserve: 8.0
    action_potential_teacher: 8.0
    residual_action_atom: 12.0
```

但原代码使用：

```python
enable_action_loss = action_loss_started and loss_weights["action"] > 0
```

结果是：虽然所有关键子损失权重均非零，整个 action/winner/deployment 分支仍被 `action: 0.0` 静默跳过。

#### 根本原因 B：direct potential 的 B=16 winner 分支没有接入 residual

V56 的 multi-budget selected-mask 分支调用 integrable-potential logits 时，遗漏：

```python
residual_action_potential=out["residual_action_potential"]
```

因此 selected B=16 的 corrected logits 实际等于 selected-local anchor logits。这个问题会让部署级 winner correction objective 对 residual head 失去梯度。

#### 判断

Protocol FAIL 不是阈值太严，也不是日志误报，而是关键训练目标没有真实执行。当前 V56 checkpoint 不满足对算法有效性进行强结论归因的训练协议。

---

### 1.2 Minimum gate：指标已经通过，正式标签被 protocol 连带置为 FAIL

V56 gate report 中：

```text
minimum_failures = []
minimum_pass = false
```

这是因为原 gate 定义为：

```python
minimum_pass = protocol_pass and not minimum_failures
```

所以 minimum 的正式标签因 protocol FAIL 被连带置为 false；但 **minimum metrics 本身已经通过**：

| 指标 | V56 |
|---|---:|
| evidence certificate fraction | 0.8880 |
| fully-certified scene rate | 0.805 左右 |
| fallback rate | 0.111 |
| budget fill | 1.000 |
| frontier retained | 0.7814 |
| selected decisive recall | 0.6064 |
| effective decisive recall | 0.7721 |
| interaction decisive recall | 0.5751 |

#### V56 是否解决了“Evidence certificate 与 residual uncertainty certificate 混在一起”

**解决了 evidence gate 层面的核心问题。**

V55 中 residual variance 把 certificate 从 `0.888` 压到 `0.204`、fallback 从 `0.110` 推高到 `0.801`；V56 重新得到 `0.888/0.111`，说明 residual uncertainty 已不再污染 evidence certificate。

但还没有完整闭环：

1. `residual_flip_certificate_pass` 和 `dual_certificate_deployment_certified` 在聚合结果中为 NaN，说明评估指标没有完整传播；
2. 配置虽然声明 residual flip 前必须有 evidence certificate，但 tournament 没有实际消费该 certificate；
3. 一个 evidence certificate fraction 只有 `0.2` 的场景仍发生 harmful deployed residual flip。

所以结论应写成：

> V56 已解决证书定义混合，但尚未完全实现“证书定义—部署执行—指标审计”三者一致。

V57 gate 已新增 `minimum_metrics_pass`，明确区分“minimum 指标是否通过”和“包含 protocol 前置条件的正式 minimum gate 是否通过”。

---

### 1.3 Competitive gate：仍是真实算法失败

V56 关键指标：

| 指标 | Candidate | Local | Foundation |
|---|---:|---:|---:|
| teacher action match | 0.140 | 0.141 | 0.141 |
| pair-full teacher match | 0.141 | 0.141 | — |

Competitive failures：

- total teacher-match gain `-0.001 < +0.015`；
- residual gain `-0.001 < +0.005`；
- pair-full residual gain `0.000 < +0.005`；
- beneficial/harmful residual `0.000/0.001`；
- proposal decisive recall `0.79815 < 0.80`。

proposal recall 只差约 `0.00185`，不是主要问题。主要问题是 residual 没有产生净正向 winner correction。

#### 1000-scene paired action 分析

Deployed candidate vs local：

- action flip：5/1000；
- exact teacher-match beneficial：0；
- harmful：1；
- neutral：4。

内部 pair-full residual：

- action flip：17/1000；
- beneficial：1；
- harmful：1；
- neutral：15。

其中存在一个很弱的正向信号：场景 `78ad5be049265b09` 中，内部 full-support residual 将 local action 2 修正为 teacher action 1，但最终部署 action 仍为 2。它证明 residual head 不是严格恒零，但不能证明已经学会稳定改善 deployed winner。

另一个场景 `b7e29c617ae25487` 中，local action 2 本来就是 teacher winner，residual 却改为 action 9；该场景 certificate fraction 只有 0.2。这说明 residual 的方向与授权都不可靠。

#### 是否“部分学会了 winner 修正”

需要区分两种表述：

- **学会产生非零 winner perturbation：是。** 它在内部 full-support 中产生 17 次 action change。
- **学会改善最终 action winner：否。** 部署有益翻转为 0，pair-full 有益与有害相互抵消，teacher-match gain 为负。

因此不能把 V56 competitive FAIL 描述为“只差一点阈值”；它仍是算法训练失败，只是失败根因现在主要是训练图断链，而不再是 V55 的 pair/Hodge 表达本身。

---

## 2. 当前模型的主要问题

### 2.1 最终 winner objective 与 residual head 断开

这是本轮最主要的问题。V56 实际优化的是 proposal、calibration 与 atomwise residual reconstruction，而不是 deployed winner。

atomwise teacher-local correction可以让 residual 具有局部数值意义，但它不能保证：

- teacher winner 超过当前错误 anchor winner；
- teacher-correct anchor 不被 residual 破坏；
- B=16 selected evidence 下的 winner 与 full-support correction 一致；
- residual 在 gate 授权后才改变 action。

### 2.2 B=16 anchor 仍是性能瓶颈，但 selector 不应推倒重来

当前：

- dense/full-interface action match：`0.359`；
- B=16 selected-local/candidate：约 `0.141`。

说明从 dense evidence 到 B=16 后仍丢失大量决定 winner 的有效信息或局部 cost calibration。但 selected decisive recall 已达到 `0.606`、interaction recall `0.575`，比 V55 明显提高，因此 selector 已经越过“完全无效”阶段。

下一步应：

- 保留当前 selector 主结构；
- 通过 exact B=16 deployment-mask distillation 和 winner-directed loss 强化它；
- 不再把主要算力花在全 pair、全场景、每 step exact；
- 不要因 proposal recall 0.798 略低于 0.80 就大改 proposal 结构。

### 2.3 Residual uncertainty 没有独立学好

V56 训练只明确优化 residual mean 的 atomwise target，residual variance 没有与实际 correction error 建立独立、可审计的训练目标。因此 residual flip certificate 即使结构上分开，也缺少可靠 uncertainty 输入。

### 2.4 证书配置没有成为硬部署约束

V56 的 harmful low-certificate flip 表明：配置字段存在，但 runtime tournament 没有真正执行 `evidence certificate -> residual flip authorization`。

### 2.5 指标与实现缺乏可观测性

V56 gate 只检查 exact fraction，没检查：

- action-family branch 是否激活；
- winner-level losses 是否非零；
- deployment-selection loss 是否非零；
- residual certificate 指标是否真正写入 open-loop summary。

这使“代码跑完”被误认为“算法目标被训练”。

---

## 3. 哪些算法有效，哪些无效

| 模块/设计 | 判断 | 处理 |
|---|---|---|
| base + dense-local immutable anchor | 有效 | 保留 |
| boundary winner/hard/near-tie pair curriculum | 有效且显著节省训练 | 保留，不退回全 pair 每 step |
| sparse periodic exact AOCC | 有效，但 V56 被训练总开关关闭 | 修复并强化审计 |
| independent calibration | 有效 | 保留 |
| same-checkpoint local + matched foundation controls | 有效 | 保留 |
| proposal/selector decisive supervision | 部分有效 | 保留，轻量强化 interaction/winner evidence |
| dual evidence/residual certificate 定义 | 有效 | 保留并完成 runtime enforcement |
| direct per-evidence integrable action potential | 表达方向正确 | 保留并接通 winner gradient |
| arbitrary pair residual + Hodge projection | 历史无效 | 不恢复 |
| scene-level-only residual potential target | 历史无效 | 不恢复 |
| atomwise residual distillation 单独承担 winner 学习 | 无效 | 降为局部可识别性辅助目标 |
| legacy `action` 总开关 | 工程错误 | 移除其 family master-switch 语义 |
| 未训练/未审计 residual variance | 无效 | 单独监督、单独校准 |
| 只在配置中声明 evidence prerequisite | 无效 | 改成 tournament 硬约束 |
| 仅看 teacher-match 的 residual 归因 | 不充分 | 同时报告 paired regret、beneficial/harmful、flip-conditioned 指标 |

---

## 4. V57：Winner-Correction Dual-Certificate Integrable Potential

V57 保持论文主线：

```text
fixed planner-interface budget
→ winner-critical evidence
→ exact B=16 selected-local anchor
→ evidence-attributable integrable correction
→ dual-certified action intervention
→ global winner
→ paired closed loop
```

### 4.1 修复 action-family 激活逻辑

winner/deployment family 只要任一子损失权重非零就执行，不再由 legacy `loss_weights.action` 单独控制。

新增训练审计：

- `action_family_enabled`；
- max winner-level loss；
- max deployment-selection loss。

### 4.2 补齐 B=16 direct-potential 梯度路径

所有 full-support、oracle-selected 与 predicted B=16 分支都显式传入：

```python
residual_action_potential=out.get("residual_action_potential")
```

这样部署预算下的 winner loss 才真实优化 residual head。

### 4.3 Winner-directed residual correction loss

对 anchor-wrong scene：

```text
corrected teacher-winner score
>
corrected selected-local anchor-winner score + margin
```

对 anchor-correct scene：

```text
corrected teacher winner
>
strongest valid rival + preservation margin
```

这直接优化 residual 的部署职责：

- 错误时纠正 winner；
- 正确时不破坏 winner。

内部名称使用 `winner_correction`，避免把 teacher/local comparative supervision错误表述为严格因果或 counterfactual intervention。

### 4.4 独立 residual uncertainty supervision

V57 用 detached atomwise residual error 监督 residual variance head，在 log-variance 空间做稳健回归：

- uncertainty 只能学习 correction error；
- 不反向改变 residual mean target；
- 不进入 evidence certificate；
- 只服务 residual flip certificate。

### 4.5 严格 dual-certificate deployment

residual 只有同时满足以下条件才允许改变 action：

1. residual robust margin 通过；
2. score gain 通过；
3. evidence certificate fraction 达到配置阈值；
4. 结构/安全 guard 通过。

主实验默认：

```yaml
require_evidence_certificate_before_residual_flip: true
min_evidence_certificate_fraction_for_residual_flip: 1.0
```

即 residual 只在 fully evidence-certified 场景中改变 winner。低 certificate harmful flip 会被硬阻止。

同时修复 structural post-processing 可能重新引入已被 residual certificate 拒绝的 flip 的问题。

### 4.6 指标与 gate 完整对齐

open-loop 聚合新增：

- `evidence_certificate_fraction`；
- `residual_flip_proposed`；
- `residual_flip_deployed`；
- `residual_flip_certificate_pass`；
- `dual_certificate_deployment_certified`；
- evidence-certificate block diagnostics。

V57 protocol gate 会在以下任一情况直接 FAIL：

- exact selector supervision 为 0；
- action-family 从未激活；
- winner-level losses 全为 0；
- deployment-selection distillation 全为 0。

---

## 5. 与论文 novelty 的对齐建议

当前论文的核心 novelty——fixed planner-interface budget、decision evidence atoms、winner-vs-rival margin preservation、bounded regret——应保持不变。

V57 可以作为对论文方法部分的强化，而不是另起一条与 BDSE 无关的工程线：

1. 把每个 evidence 的修正明确写成 action potential `h_i(a)`；
2. pair margin 由同一个 action potential 的差得到：
   `d_i(a,b) = [g_i(b)+h_i(b)] - [g_i(a)+h_i(a)]`；
3. 强调 integrability 是结构保证，不再需要 Hodge 后处理；
4. evidence certificate 只证明 selected local evidence support；
5. residual certificate 只证明 proposed winner intervention；
6. 使用 “winner-directed/winner-correction supervision”，不要在没有真实 intervention dataset 时声称严格 causal/counterfactual label；
7. theorem 仍以 decisive margin preservation 为主，V57 的 dual certificate 可作为部署命题或 corollary，而不是替换主 theorem。

这样可以让 novelty 更集中：

> 在固定可查询 evidence budget 下，模型不仅选择 decision-sufficient evidence，还学习由 evidence 可归因、天然可积、仅在双证书通过时改变 winner 的 residual correction。

---

## 6. 下一轮实验顺序与停止条件

### Stage 0：训练链 smoke test，必须先跑

只跑 1024 train / 256 val、1 epoch。必须同时满足：

- `action_family_enabled = 1`；
- `selector_exact_fraction > 0`；
- `L_deploy_select > 0`；
- `L_pair_full_action > 0`；
- `L_residual_winner_correction > 0`；
- `L_residual_action_uncertainty > 0`。

任一为 0，立即停止，不要投入完整训练。

### Stage 1：完整 6-epoch 训练

重点监控：

- exact fraction 最后一轮 `>= 0.015`；
- winner-correction loss 与 action-potential teacher loss 有下降趋势；
- residual atom loss 不再是唯一 residual objective；
- uncertainty loss 有限且不爆炸；
- pair training fraction 保持约 0.55–0.65，不回退到全 pair。

### Stage 2：三路独立 calibration + paired 1000 open-loop

最低健康目标：

- protocol PASS；
- minimum metrics PASS；
- evidence certificate `>= 0.85`；
- fallback `<= 0.20`；
- selected decisive recall `>= 0.60`；
- interaction decisive recall `>= 0.55`；
- 不允许 evidence certificate 不足的 deployed residual flip。

Competitive 目标：

- candidate-local teacher-match gain `>= +0.005`；
- pair-full residual gain `>= +0.005`；
- beneficial residual rate `> harmful residual rate`；
- harmful residual rate尽量接近 0；
- proposal decisive recall `>= 0.80`；
- paired regret median/p90 不退化。

### Stage 3：paired NR-CL20 与 R-CL20

上传的 V56 archive 中没有任何 closed-loop 文件，因此本轮无法分析用户所述 reactive CL20；当前结论只来自训练与 open-loop。

V57 冻结后应分别运行：

- `closed_loop_nonreactive_agents` paired CL20；
- `closed_loop_reactive_agents` paired CL20；
- candidate/local/foundation 使用完全一致的 token hash。

即使 competitive 未通过，也可以在 protocol PASS 后运行 diagnostic CL20，但不能作为论文有效性结论。

### Stage 4：CL100

只有 competitive PASS、beneficial > harmful、paired CL20 无安全退化后再运行。

### Stage 5：partial test one-shot

当前 partial test 有 67,042 个唯一样本、内部重复为 0，但明显比 val 更难：

| 指标 | Val | Partial test |
|---|---:|---:|
| quality keep | 0.9175 | 0.6791 |
| safe candidate exists | 0.7173 | 0.5779 |
| teacher ADE p90 | 12.92 | 17.14 |
| full-interface sufficiency | 0.9657 | 0.9343 |
| runtime decision sufficiency | 0.7490 | 0.6407 |
| B16 oracle sufficiency | 0.9120 | 0.8399 |

只能在 checkpoint、calibration、gate threshold 全部冻结后运行一次，不得用于继续调参，也不能称为最终 held-out test。

---

## 7. 代码验证边界

已完成：

- Python compile：PASS；
- 4 个 V57 YAML：PASS；
- 5 个 shell runner：PASS；
- 全量 unit tests：`208 passed, 8 warnings`；
- legacy action master-switch regression：PASS；
- direct B=16 residual-potential gradient：PASS；
- low-evidence certificate flip blocking：PASS；
- fully certified flip allowing：PASS；
- protocol/minimum-metrics separation：PASS；
- historical V56 gate/test 文件保持不变，V57 逻辑独立。

未完成且未声称：

- 没有在当前环境运行 fresh V57 nuPlan 训练；
- 没有运行 V57 open-loop、NR-CL20、R-CL20 或 CL100；
- 不预先声称 V57 会通过 competitive gate；
- 不预先声称闭环提升、实时性、SOTA 或 CCF-A 录用级结果。
