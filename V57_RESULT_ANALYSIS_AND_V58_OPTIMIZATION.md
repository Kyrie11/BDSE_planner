# V57 结果诊断、工程审计与 V58 CSIP-BFAR-DBAP 优化

## 0. 结论先行

本轮分析对象是上传的 V57 代码和 `outputs_v57_wcdcip_bfar_dbap_fast_2gpu_v1` 结果。结论分为“原 gate 报告”和“修正后的严格工程口径”两层：

| Gate | V57 原报告 | 严格工程口径 | 解释 |
|---|---:|---:|---|
| Protocol | PASS | **INCOMPLETE / FAIL** | 训练与 paired replay 子协议通过，但 direct residual uncertainty 没有被校准，residual proposal 指标被 structural guard 污染，CL20 在第一个场景前崩溃 |
| Minimum metrics | PASS | **PASS** | evidence certificate、fallback 和 decisive recall 达标 |
| Formal minimum | PASS | **FAIL（若 protocol 是前置条件）** | minimum 数值没有失败，但严格 protocol 不完整 |
| Competitive | **FAIL** | **FAIL** | candidate、local、foundation winner 完全相同，residual 和 pair-full gain 均为 0 |

因此，V57 不能简单概括为“两个 gate 通过、一个 gate 失败”。准确说法是：

> V57 修复了 V56 的 winner/deployment 训练断链，selector 与 evidence certificate 继续有效；但 residual 没有学会可部署的 winner correction，同时三项工程问题使原 protocol PASS 偏乐观，闭环结果实际上不存在。

V58 不承诺在未运行实验前一定通过三个 gate。它做的是：消除已确认的工程误导，令训练目标、calibration、runtime certificate 和 gate 审计对齐，并提高 residual 学到可认证 winner correction 的可能性。

---

## 1. 为什么运行很慢

### 1.1 整体耗时拆解

根据 pipeline 日志和文件时间戳，从启动到 gate 完成约 **4 小时 50 分钟**：

| 阶段 | 约耗时 | 占比/判断 |
|---|---:|---|
| Anchor quality | 13 分 24 秒 | 一次性诊断，非主瓶颈 |
| 训练 6 epoch | **2 小时 9 分 18 秒** | 最大单阶段 |
| Candidate calibration | 37 分 26 秒 | 重复回放 |
| Local calibration | 37 分 50 秒 | 与 candidate 基本重复 |
| Foundation calibration | 40 分 27 秒 | 与前两路继续重复 |
| Candidate 1000-scene open-loop | 6 分 31 秒 | 两 GPU shard |
| Local open-loop | 12 分 30 秒 | 单路串行 |
| Foundation open-loop | 12 分 07 秒 | 单路串行 |
| Gate | 约 1 秒 | 可忽略 |

三次 calibration 串行总计约 **1 小时 56 分钟**，正式三路 open-loop 串行总计约 **31 分钟**。因此：

> 慢的主要原因不是“正式 open-loop 数量太多”，而是 **训练本身 + 三次重复 calibration + 三路控制评估串行执行**。

### 1.2 训练慢在哪里

6 个 epoch 墙钟分别约：

`1207 / 1134 / 1061 / 990 / 1043 / 1041 s`

其中每个 epoch：

- loss construction/相关 GPU 工作约 `711–789 s`；
- boundary pair sampling 约 `120–184 s`；
- data wait 除首 epoch 外仅约 `3.5–4.0 s`；
- H2D 约 `1.9–2.1 s`。

说明训练不是 DataLoader 或 CPU 核数不足导致。增加 CPU worker 对主耗时帮助有限。主要成本来自复杂 loss 图、pair/evidence 张量运算和模型前向/反向。

注意 PyTorch CUDA 运算是异步的，阶段计时不能被解释成完全独立的 kernel 时间；但 data wait 很低这一结论可靠。

### 1.3 Open-loop 慢在哪里

V57 candidate open-loop：

- mean latency：约 `586 ms`；
- p95：`905 ms`；
- prediction：约 `441 ms`；
- selector：约 `77 ms`；
- tournament：约 `6 ms`。

prediction 占 planner latency 约 75%，selector 约 13%。因此继续只微优化 selector 不会显著降低总时延。

### 1.4 V58 的运行速度优化

V58 已实施：

1. **Calibration 合并**  
   不再 candidate/local/foundation 各跑一遍旧 calibration。改为：
   - 两 GPU 对 `val_calib` 分片并行；
   - 一次回放同时收集共享 evidence adverse score 和 candidate residual proposal margin nonconformity；
   - controls 复用共享 evidence calibration，并显式关闭 residual calibration。

2. **所有正式 open-loop 同时进入一个有界任务池**  
   `run_parallel_open_loop_suite.py` 会：
   - candidate、local、foundation 一起排队；
   - 在所有 GPU 上按相同 modulo shard 并发运行；
   - 默认 `OPEN_LOOP_WORKERS_PER_GPU=2`；
   - 检查三路 scenario/timestamp hash 和数量完全一致。

3. **并发不是越多越好**  
   prediction 是 GPU 主瓶颈。CPU 核数多可以支持更多进程，但过多同 GPU forward 会增加 contention。提供 `BENCHMARK_V58_OPEN_LOOP_CONCURRENCY.sh`，建议在 120 个场景上比较每 GPU 1/2/3 worker，再固定最快设置。

4. **闭环保留共享模型优化**  
   不恢复“一 simulation 一 CUDA 模型”的旧问题；使用进程内共享模型与 GPU inference serialization。

当前 V58 并发实现仍会分别执行 candidate 与 local 的相同 checkpoint prediction。未来最高价值的进一步工程优化是把 candidate/local 组织成同一进程的 paired evaluator，复用一次 prediction，只分叉 tournament/certificate；但这需要较大 refactor。为了本轮先保证 attribution 正确，没有仓促引入该高风险修改。

---

## 2. 先排查工程错误

### 2.1 闭环实际上没有运行

两个 CL20 shard 都在 nuPlan callback 构建阶段崩溃：

```text
main_callback.metric_summary_callback=null
AttributeError: 'NoneType' object has no attribute '_target_'
```

因此：

- 没有任何有效 CL20 场景；
- `closed_loop/` 目录存在不代表完成；
- 不能从本轮结果推断 NR-CLS 或 R-CLS。

V58 修复：

- 不再把 callback 设为 `null`；
- 保留合法 summary callback；
- 若不需要 PDF，在成功后删除 PDF，而不是破坏 callback；
- 只有两个 shard 与 combined summary 都成功后才写 `.closed_loop_complete.json`；
- 三路 CL20 额外校验 token SHA 和场景数。

### 2.2 Residual proposal/certificate 指标被 structural guard 污染

V57 在 residual guard 之后又执行 all-flagged structural safety guard，并覆盖了 `pair_action_anchor_action`。之后 `residual_flip_proposed` 使用被覆盖后的 anchor 来比较，导致 structural guard 的 action 变化被误报为 residual proposal。

1000 场景中原报告：

- residual proposal：99；
- certificate pass：914；
- deployed flip：0。

逐行审计后：

- **86** 个是真正的 raw residual proposal，但被 margin/uncertainty certificate 拒绝；
- **13** 个是 all-flagged structural guard 造成的 action 变化，被错误记成 residual proposal；
- 总 all-flagged guard scene 为 22；
- 真正 deployed residual flip 仍为 0。

所以 `residual_flip_certificate_pass=0.914` 不能理解为“91.4% 的 residual proposal 通过”。它混入了大量 no-proposal scene 和 structural guard artifact。

V58 修复：

- tournament 在 structural post-processing 前冻结 raw anchor、raw proposed action、raw margin、sigma、residual epsilon；
- proposal-conditional certificate 与 no-proposal abstention 分开报告；
- evidence pass、residual margin pass、dual certificate pass 分开报告；
- structural guard 不再改变 residual intervention attribution。

### 2.3 V57 calibration 校准错了对象

V57 三次 calibration 使用 `calibrate_v48_adverse_bounds.py`，针对的是 legacy `pair_atom_delta/pair_atom_var` adverse bound，而 V57 部署 residual flip 使用的是新的 direct action-potential mean/variance。

三路 calibration 都处理：

- 5000 scenes；
- 12,717,544 atom pairs；
- learned variance fraction = 0；
- raw error MAE 约 `0.0209984`；
- 推荐 evidence adverse epsilon = 0。

candidate 和 local 的 MAE 几乎完全相同，进一步说明大量计算是重复的。更关键的是：

> 这三次 calibration 没有给 direct residual winner certificate 提供所需的 proposal-level conformal residual epsilon。

V58 修复：

- evidence certificate 与 residual flip certificate 各自校准；
- residual calibration 只在 candidate 上、只对实际 proposed-vs-anchor margin 误差进行；
- train/calibration/runtime 的 `beta_uncertainty` 统一为 1.0；
- evidence epsilon 只写入 AOCC/adverse certificate，不再意外改动 tournament action rule；
- control 的 residual mean、variance、certificate 与 residual epsilon 全部关闭/清零。

### 2.4 原 protocol gate 审计不完整

V57 protocol gate 成功检查了：

- paired key/hash；
- action family 已启用；
- exact-selector fraction 非零；
- winner-level loss 非零。

但没有检查：

- direct residual uncertainty 是否被独立校准；
- candidate/local/foundation calibration 是否只存在预期差异；
- train/deploy beta 是否一致；
- 每一个配置为非零的 winner loss 是否都实际执行；
- closed-loop 是否真正完成。

V58 gate 增加这些检查。因此，V57 原报告的 protocol PASS 可保留为“V57 旧定义下 PASS”，但不能作为完整双证书协议已经正确执行的证明。

### 2.5 Checkpoint selection 与研究目标错位

V57 的 best checkpoint 由 `fixed_budget_critical_score` 选择，epoch 3 被选中；epoch 5 的 proposal recall 已从 `0.7993` 上升到 `0.8013`，但 residual gain 始终为零。

旧 score 主要奖励 selector/certificate，无法拒绝 residual 完全没有 winner gain 的 checkpoint。

V58 新增 `val_competitive_score`：

- 奖励 candidate teacher match；
- 显式奖励 candidate-local gain；
- 显式奖励 pair-full-local gain；
- 奖励 beneficial-harmful intervention；
- 同时保留 selected/interaction recall 与 fallback 约束。

---

## 3. 三个 gate 的根本状态

## 3.1 Protocol gate

### 原 V57 报告

**PASS**，且训练健康指标为：

- `action_family_enabled = 1`；
- exact fraction 最大/最后为 `0.02568`；
- deployment loss 非零；
- winner-level loss 最大约 `10.0749`。

这证明 V57 修复了 V56 的 action-family 总开关断链，训练图确实执行了。

### 严格工程结论

**INCOMPLETE / 应视为 FAIL**，原因不是训练分支再次断开，而是：

1. direct residual certificate 未校准；
2. residual proposal 指标语义错误；
3. closed-loop 未运行；
4. 原 gate 未审计以上条件。

### 根本原因

> V57 的训练协议已恢复，但 deployment certificate 与评估协议没有与新 direct residual head 完整对齐。

## 3.2 Minimum gate

V57 minimum 数值全部通过：

- evidence certificate：`0.8880`；
- fallback：`0.110`；
- selected decisive recall：`0.6079`；
- effective decisive recall：`0.7735`；
- interaction decisive recall：`0.5777`；
- budget fill：`1.0`。

说明 V55/V56 延续下来的双证书拆分和 selector 主体仍有效。`minimum_failures=[]`。

因此：

- 单看 minimum metrics：**PASS**；
- 若 formal minimum 强制依赖完整 protocol：严格口径 **FAIL**；
- 这不是 selector 指标失败，而是 protocol 前置条件不完整。

## 3.3 Competitive gate

**真实算法 FAIL**。

关键结果：

- candidate/local/foundation teacher match：`0.141 / 0.141 / 0.141`；
- pair-full candidate/local：`0.141 / 0.141`；
- total action gain：`0`；
- residual gain：`0`；
- pair-full residual gain：`0`；
- beneficial/harmful deployed residual：`0 / 0`；
- deployed residual flip：`0`。

proposal decisive recall `0.799339 < 0.80` 只差 `0.000661`，不是 competitive 失败的根本原因。即使把该阈值放宽，winner gain 仍全部为零。

根本原因是：

> residual 学到了一些连续分数扰动和 proposal 尝试，但没有学到能够跨越 selected-local 决策边界、同时满足 uncertainty/conformal margin、并最终改善 teacher winner 的修正。

---

## 4. V57 的正向信号

不能因为 competitive FAIL 就判断整个设计无效。V57 有以下可靠正向信号：

1. **V56 训练断链已被修复**  
   exact selector、deployment selection、winner correction 和 uncertainty loss 都实际执行。

2. **Selector/evidence 路径继续有效**  
   - evidence certificate `0.888`；
   - fallback `0.11`；
   - selected decisive recall `0.608`；
   - interaction decisive recall `0.578`；
   - effective decisive recall `0.774`。

3. **Proposal 在训练末期跨过 0.80**  
   epoch 5 validation proposal recall 为 `0.8013`，说明边界 evidence proposal 仍在改善。

4. **Winner-level 梯度真实存在**  
   - pair-full winner margin loss：`10.07 -> 8.17`；
   - residual winner correction：`7.23 -> 5.38`；
   - uncertainty loss：`3.98 -> 2.38`。

5. **Residual head 不是严格零函数**  
   修正指标后仍有约 86 个 raw residual proposal，只是全部被 certificate 拒绝。

这些信号支持保留：

- immutable base+dense-local anchor；
- boundary pair curriculum；
- fixed B=16；
- sparse periodic exact AOCC；
- direct integrable action potential；
- evidence/residual 双证书定义；
- paired same-checkpoint local 和 matched foundation control。

---

## 5. 当前模型的主要算法问题

### 5.1 Reconstruction loss 没有学到 teacher correction

两项直接重构损失几乎不改善：

- `L_action_potential_teacher` 始终约 `0.363`；
- `L_residual_action_atom` 始终约 `0.012`。

这说明 atomwise teacher-local target 对现有 representation 仍难以拟合，或它与最终 winner correction 的监督几何不一致。

### 5.2 Winner loss 下降，但没有跨过任何 action boundary

pair-full/B16 winner 全程保持 `0.141`。因此 V57 学到的是“减少 soft margin loss”，不是“改变离散 winner”。

原因之一是 V57 winner objective 没有直接优化部署使用的：

```text
raw residual margin - beta * sigma - conformal residual epsilon
```

模型可以降低普通 margin loss，但产生的 correction 永远低于 robust flip threshold。

### 5.3 Uncertainty target 与部署证书不对齐

V57 variance head 以 atomwise residual error 为 target，但部署关心的是 proposed action 与 anchor action 的 scene-level pair margin error。两者并不等价。

V58 calibration 改为 proposal-conditional scene-level nonconformity，并在训练中用 selected-set 聚合 variance 构造同一 robust margin。

### 5.4 Dense-to-budget 信息缺口仍大

- dense full-interface action match：`0.359`；
- B16/pair-full selected-local action match：`0.141`。

即便 recall 指标较好，B=16 仍未保留足够的信息来重建 dense/teacher winner。Residual 需要在有限 evidence 上补偿这一缺口，但当前 additive per-evidence potential 可能无法表达 selected-set interaction。

V58 先采用保守方案：让目标与 selected set 和 certificate 完全对齐。如果 V58 后 pair-full residual gain 仍为 0，则下一步应增加 **set-conditioned interaction potential head**，而不是继续降低 gate 阈值或单纯放大 residual scale。

---

## 6. V58 CSIP-BFAR-DBAP 算法优化

V58 名称：

**Certified Set-Aligned Integrable-Potential Boundary-Focused Anchor-Residual Decision-Budget Action Preservation**。

论文主线不变：

```text
decision boundary
-> winner-critical evidence
-> exact B=16 selection
-> selected-local anchor
-> certified residual correction
-> final winner
-> paired closed loop
```

### 6.1 Certified residual winner loss

新增 `L_certified_residual_winner`，直接优化部署 robust margin：

```text
predicted corrected margin
- beta * predicted residual sigma
- residual conformal epsilon reserve
>= flip margin
```

仅对 teacher 有足够真实优势、且 selected-local anchor 错误的 scene 要求 flip。对 anchor 已正确的 scene，增加对 strongest rival 的 do-no-harm preservation certificate。

这样避免为了制造 flip 而破坏正确场景。

### 6.2 Calibration reserve

真实 residual epsilon 在 checkpoint 冻结后才可通过 `val_calib` 得到。训练阶段新增 `residual_epsilon_reserve=0.05`，避免模型只学到刚好超过未校准阈值的 correction，校准后又全部被拒绝。

### 6.3 Residual head 独立学习率

V57 对零初始化 residual head 使用与 selector 相同的低 LR。V58：

- residual mean head：基础 LR 的 `5x`；
- residual variance head：基础 LR 的 `2x`；
- selector/proposal/family heads：`1x`。

保留 anchor/selector 稳定性，同时让 residual 有足够更新幅度。

### 6.4 Competitive checkpoint selection

best checkpoint 不再只看 fixed-budget selector score。新 score 显式要求：

- candidate-local gain；
- pair-full-local gain；
- beneficial > harmful；
- 同时维持 recall 与 fallback。

### 6.5 更严格的 dual-certificate runtime

V58 runtime 分开计算并记录：

- raw residual margin；
- selected-set residual sigma；
- residual conformal epsilon；
- evidence certificate pass；
- residual margin certificate pass；
- proposal-conditional dual certificate pass；
- deployed flip。

无 proposal scene 记为 abstention，不再被混入 proposal pass rate。

### 6.6 Gate health checks

Protocol 必须检查：

- 所有配置为非零的 winner loss 实际非零，包括 certified winner loss；
- exact selector fraction 非零；
- direct residual uncertainty loss 非零；
- independent calibration 标记存在；
- 三路共享相同 evidence epsilon；
- candidate residual 开启，controls residual/epsilon 关闭；
- train/deploy beta 一致；
- paired scenario hash 完全一致。

---

## 7. 下一轮实验顺序与判据

### Stage A：训练 smoke test

先运行 1024 train / 256 val、1 epoch：

必须满足：

- `action_family_enabled > 0`；
- `selector_exact_fraction > 0`；
- `L_deploy_select > 0`；
- `L_pair_full_action > 0`；
- `L_residual_winner_correction > 0`；
- `L_certified_residual_winner > 0`；
- `L_residual_action_uncertainty > 0`；
- `certified_correctable_fraction > 0`；
- optimizer 中存在 `x5` 和 `x2` LR group。

任一失败都不要跑完整训练。

### Stage B：完整训练 + dual calibration + simultaneous open-loop

运行 8 epoch；calibration 两 GPU 并行；candidate/local/foundation 同时评估。

建议第一轮 `OPEN_LOOP_WORKERS_PER_GPU=2`。在正式长跑前可运行并发 benchmark，机器若出现 GPU contention 则退回 1，显存和吞吐允许则使用 3。

### Stage C：Gate 判定

目标：

- strict protocol：PASS；
- evidence certificate：`>= 0.85`；
- fallback：`<= 0.20`；
- selected decisive recall：`>= 0.60`；
- interaction decisive recall：`>= 0.55`；
- candidate-local teacher-match gain：先要求 `> 0`，再冲击 `>= +0.005`；
- pair-full residual gain：`>= +0.005`；
- beneficial residual rate：显著高于 harmful；
- 低 evidence certificate 的 deployed residual flip：0；
- proposal-conditional dual certificate 指标有限且可解释，不得为 NaN 或 no-proposal 混算。

正式 competitive gate 仍保留较高目标，不通过时不要靠改阈值制造 PASS。

### Stage D：闭环

只有 strict protocol PASS 后才解释闭环：

1. paired NR-CL20；
2. frozen paired R-CL20；
3. 检查 `.closed_loop_complete.json` 和 three-way token protocol；
4. competitive open-loop 转正且 CL20 无安全退化后才运行 CL100。

---

## 8. 若 V58 仍失败，下一步如何判断

### 情况 1：certified loss 下降，raw proposals 增加，但仍全部被拒绝

说明 sigma 或 calibrated epsilon 太大。先检查 calibration reliability 和 variance over-estimation，不要直接降低 certificate 阈值。

### 情况 2：pair-full residual gain 转正，但 B16 gain 仍为 0

说明 residual 能修 full selected set，但 selector 的 B=16 evidence 不足。强化 winner-conditioned selection/interaction evidence，而不是继续增大 residual LR。

### 情况 3：pair-full residual gain 仍为 0，atomwise/global distillation 仍平坦

说明 additive per-evidence potential 表达不足。下一版应增加轻量 set-conditioned interaction head，例如对选中 evidence embedding 做 permutation-invariant aggregation后输出 action-level interaction correction；需要保留 zero-init、control closure 和独立 certificate。

### 情况 4：open-loop gain 转正但 CL safety 下降

说明 teacher-winner correction 与 rollout safety 不一致。增加 safety-conditioned correctable mask 和闭环 failure replay，而不是扩大 residual deployment rate。

---

## 9. 验证边界

本环境完成：

- Python compile；
- 4 个 V58 YAML 解析；
- Shell syntax；
- 单元测试 `215 passed, 8 warnings`；
- calibrated winner loss、raw-proposal attribution、dual certificate、LR groups、parallel suite 和 gate 静态审计。

本环境没有运行 fresh V58 training、calibration、open-loop、NR-CL 或 R-CL，因此不提前宣称三个 gate 已通过、闭环分数提高或达到 CCF-A 竞争力。
