# BDSE v49 结果诊断与 v50 DBAP-RI 优化报告

## 1. 结论摘要

### 1.1 Test set

当前未完成的 `public_set_test` 缓存包含 12,920 个有效样本，满足以下初步完整性条件：

- 与 val 的关键预处理参数完全一致；
- `scenario_stride=10`、`scenario_iteration_policy=initial`；
- `max_samples_per_log=None`，没有再次引入每 log 512 条的截断；
- candidate-aware agent selection 已启用；
- drivable polygons 已启用，crosswalks 与 train/val 一样关闭；
- teacher cost evaluation stride 为 1；
- 缺失标签样本数为 0；
- 12,920 个 scenario identity 全部唯一，内部重复数为 0。

因此，它可以作为**冻结模型的一次性初步独立 stress test**，用于观察跨 split 泛化，不必继续把 val 冒充 test。

但它尚不能作为论文最终 test 结果，原因是：

1. 构建未完成；
2. 上传内容只有聚合 diagnostics，没有完整 cache manifest；
3. 尚不能验证 train/test、val/test identity 零重叠；
4. 尚不能验证 `failed_preprocess.jsonl` 的失败比例及其是否引入选择偏差；
5. 尚不能验证当前 12,920 条是否在城市、log 和场景类型上具有无序/均匀覆盖，而不是构建顺序的前缀。

v50 新增了 `check_test_set_readiness.py`，把“构建完整性”和“自然分布变化”分开。分布与 val 不同不再被当作构建错误。

### 1.2 Open-loop gate

v49 **没有通过 open-loop gate**。因此 closed-loop 未执行是正确的流水线行为。

失败项：

- teacher action match：0.232，control 为 0.286，下降 0.054；
- near-tie sign：0.4621，control 为 0.5933，下降 0.1312；
- pair-full action match：0.231，低于 0.300；
- planner latency p95：854.84 ms，高于 500 ms；
- teacher regret median：36.21，高于 control 的 9.91；
- teacher regret p90：36,577.44，高于 control 的 34,134.98。

通过或明显改善的部分：

- evidence sufficiency：0.078996，比 control 高 0.024737；
- winner–rival sign：0.68714，比 control 高 0.02831；
- certified pair fraction：0.88197；
- fully certified scene rate：0.793；
- fallback rate：0.115；
- fixed-budget fill：15.584/16=0.974；
- pair-full→budget harmful compression：0.009；
- budget-vs-pair-full match：0.913。

核心错误发生在 **local/dense→pair interface**，而不是最终预算压缩：

- dense full-interface action match：0.360；
- pair-full action match：0.231；
- dense→pair-full flip rate：0.692；
- harmful pair-interface rate：0.176；
- beneficial pair-interface rate：0.047；
- pair-full→budget harmful rate：仅 0.009。

## 2. Test 与 val 的差异性质

| 指标 | Val | 当前部分 Test | 判断 |
|---|---:|---:|---|
| 样本数 | 58,418 | 12,920 | test 未完成，但已足够初步诊断 |
| safe candidate exists | 0.7173 | 0.5662 | test 安全候选覆盖更弱 |
| candidate hard violation | 0.5668 | 0.6796 | test 更困难 |
| teacher ADE p50 | 5.480 | 5.154 | 中位数没有恶化 |
| teacher ADE p90 | 12.915 | 17.549 | test 长尾显著更重 |
| route-distance tail | 3.263 m | 17.585 m | test 路线长尾明显 |
| quality keep rate | 0.9175 | 0.6666 | test 更严格/更困难 |
| oracle full-interface match | 0.9657 | 0.9310 | 仍较高，但低于 val |
| oracle evidence sufficiency | 0.5922 | 0.5209 | 仍超过 0.5 |
| oracle B16 sufficiency | 0.9120 | 0.8293 | test 在 B=16 下更难 |

这些差异不能通过人为改 test 参数来“修平”。否则会造成 test-set tuning。正确做法是：

- 先用 manifest 和失败记录证明构建完整性；
- 接受 official test 的真实 distribution shift；
- 在冻结 checkpoint、冻结 calibration、冻结 gate 后只运行一次正式 test；
- 将 val 与 test 差异作为泛化结果报告，而不是继续调整算法。

## 3. v49 算法性质

### 3.1 应继续保留的有效设计

#### A. Candidate/interface/compression 三层误差分解

这是当前最有价值的诊断与论文贡献之一。它准确证明错误并不主要发生在 B=16 压缩，而发生在 pair interface：

- dense→pair harmful：0.176；
- pair→budget harmful：0.009。

v50 将它进一步细分为：

1. dense/sparse full interface；
2. local-only pair-full interface；
3. residual-combined pair-full interface；
4. fixed-budget interface。

这样可直接测量 residual 是有益修正还是有害干预。

#### B. Nested AOCC certificate frontier

该设计有效：认证比例高、fallback 低、压缩损伤小。应保留，并作为稳定的 downstream action-preservation 模块，不建议继续增加复杂规则。

可继续深化的方向仅限于：

- 报告 B=4/8/16/24/32 的 nested frontier 曲线；
- 报告 certified/uncertified 条件下的 teacher match 与 closed-loop 指标；
- 验证 frontier retained weight 和动作保持之间的单调关系。

#### C. Fixed-budget action preservation

v49 的 B=16 fill 为 0.974，pair-full→budget harmful 仅 0.009，说明 fixed-budget selector 已基本能够保护其上游动作。应保留。

v50 强制构建完整 B=16 prefix，避免“提前认证后平均只选不到 16 个”对 exact-budget 论证造成歧义。

#### D. Proposal/LOO candidate recall

proposal decisive recall 为 0.813，selected decisive recall 为 0.574，说明 LOO target 对候选召回有一定作用。该设计不应删除，但不能宣称端到端因果边界学习已经成功。

正确定位是：

- proposal 学到了一部分 boundary-relevant candidates；
- pair residual 没有把它们转化为正确动作边界。

### 3.2 当前无效或有害的设计

#### A. v49 residual interface

v49 residual trust 均值只有约 0.0386，但 dense→pair 动作仍有 69.2% 被翻转。原因不是简单的“权重还不够小”，而是两个结构问题：

1. 训练损失使用未收缩的 `local + residual`，部署才使用 confidence-shrunk residual，形成 train/deploy mismatch；
2. 收缩按 atom 执行，24 个小 residual 可以在 pair 级累积成大校正并翻转动作。

训练曲线进一步证明 pair head 越训练越差：

- epoch 1 teacher match 0.232、pair-full 0.231；
- epoch 19 teacher match 0.149、pair-full 0.151；
- dense full-interface 反而从 0.360 上升至 0.401；
- harmful pair interface 从 0.176 上升至 0.299；
- pair regression loss 从约 0.0117 上升至 0.0222；
- LOO pair loss 仅从约 3.576 降至 3.494。

这说明 residual head 不是在修正 local 的边界错误，而是在逐渐覆盖一个更稳定的 local interface。

#### B. 将 residual 当作全部 pair-margin 重建器

pair residual 的正确角色应是：

> 仅修正 local interface 在 teacher boundary 附近或符号错误的少量 pair。

而不是：

> 对所有 pair 和所有 atom 重建完整 pair margin。

v50 引入 correction-focused residual learning：

- local 符号错误的 pair：高权重；
- local 或 teacher margin 靠近边界的 pair：中高权重；
- local 已正确且远离边界的 pair：仅保留 0.1 基础权重。

#### C. Teacher+model rival union 的 near-tie 部分

winner–rival sign 提升，说明 broad rival 建模有效；near-tie sign 大幅下降，说明真正动作翻转边界没有学好。

因此 rival union 应保留，但训练和 gate 必须提高 near-tie 优先级。不能再用 broad sign improvement 掩盖 near-tie regression。

#### D. Interaction family 仍偏高

interaction evidence 占 12.76/15.584=81.9%。相比 v48 已从约 94.8% 改善，但仍接近上限，且说明跨 family 竞争尚不充分。

v50：

- interaction prefix cap 从 0.80 调到 0.75；
- soft interaction Top-M reserve 从 4 调到 2；
- decision family boost 设为 0；
- 保留真实 family competition，而非删除 interaction evidence。

## 4. v50 DBAP-RI 核心修改

新版本名称：

**DBAP-RI：Deployment-Boundary Action Preservation with Residual Intervention Control**

### 4.1 训练与部署共用 residual gate

新增 `bdse/model/residual_gate.py`，NumPy runtime 与 Torch training 使用同一个数学路径。

每个 atom 的 residual trust 同时由以下因素控制：

- residual variance；
- local boundary strength；
- local/residual 原始符号冲突；
- residual/local 幅值比例。

### 4.2 Pair-level aggregate intervention gate

对所有 atom residual 求和后，额外限制 pair 级总校正：

- 正常校正不超过 local margin 的一定比例和绝对上限；
- 无置信度的 residual 不允许翻转 local pair sign；
- 只有 residual correction 的下置信界超过 local margin 与额外 flip margin 时，才允许翻转；
- 新增 flip proposal、flip allowed、confident flip、aggregate scale 等诊断。

### 4.3 Local-only pair-full 诊断

新增：

- `local_pair_full_interface_action_match`；
- `local_pair_full_to_residual_flip_rate`；
- `harmful_residual_intervention_rate`；
- `beneficial_residual_intervention_rate`；
- `dense_to_local_pair_full_flip_rate`。

下一轮可以明确回答：

- pair graph/local tournament 是否本身错误；
- residual 是否在改善 local；
- residual 是否净有害。

### 4.4 Best-checkpoint 标准

checkpoint score 现在：

- 奖励 local pair-full 和 residual-combined pair-full；
- 直接处罚 `local_pair_full - pair_full` 的 residual interface drop；
- 处罚 harmful residual 超过 beneficial residual；
- 缺少 local/residual 诊断时视为失败，而不是使用弱代理。

### 4.5 Early stopping

v49 的最优 checkpoint 出现在 epoch 1，随后持续退化。v50 增加 validation-aware early stopping，默认连续 3 次 validation 不改善后停止，防止 residual head继续覆盖稳定 local interface。

### 4.6 Latency

v49 latency 的主要瓶颈是 prediction：

- stage predict mean 约 520 ms；
- selector mean 约 2.32 ms；
- p95 总 latency 约 855 ms。

v50 不牺牲 B=16 的情况下：

- max runtime pair queries：128→96；
- max selector pairs：64→56；
- L_infer：10→8；
- residual refine pairs：24→16；
- utility refinement top-k：6→4。

这是当前可由结果明确支持的 query/materialization 优化。若仍超过 500 ms，下一轮应基于新增 stage timing 做 CUDA profiler，而不是盲目继续修改 selector。

## 5. 工程问题及修复

| 工程问题 | v50 修复 |
|---|---|
| 训练使用 raw local+residual，部署使用 shrunk residual | NumPy/Torch 共用同一 gate |
| 多 atom residual 可累计翻转 pair | pair-level aggregate cap 与 sign-preservation gate |
| validation 不知道 local-only pair-full | 新增 local/residual interface 分解 |
| best checkpoint 无法处罚 residual 干预 | 新 score 加入 residual drop 与 harmful intervention |
| 训练后期持续退化仍跑满 20 epoch | validation-aware early stopping |
| gate 名称仍显示 v48 | 新建 v50 gate |
| test parity 将真实 split shift 当作构建错误 | 拆成 hard integrity gate 与 shift warning |
| partial test 无法验证 leakage/failure bias | 新增 manifest overlap、failed fraction、identity 检查 |
| early stop 后 pipeline 可能误判训练未完成 | final checkpoint 作为完成标记 |
| v49 interaction family 接近饱和 | 更严格 prefix cap 和较少机械 reserve |

本地验证：

- Python compileall：通过；
- YAML 加载：通过；
- shell syntax：通过；
- 单元测试：164 passed，5 warnings。

## 6. 下一步实验判断顺序

1. 使用全新 OUT_ROOT 运行 v50；
2. 检查 local-only pair-full；
3. 检查 residual harmful/beneficial intervention；
4. 检查 pair-full 与 near-tie；
5. 检查 latency；
6. 只有严格 gate PASS 后运行 CL20；
7. CL20 有正向趋势后运行 paired CL100；
8. test cache 完成并通过 manifest integrity/leakage gate 后，只运行一次最终 test。

下一轮最关键的成功条件不是 proposal recall 继续上升，而是：

- `local_pair_full_interface_action_match >= 0.30`；
- residual pair-full 不低于 local pair-full 超过 0.01；
- harmful residual intervention <=0.05，且不高于 beneficial；
- near-tie 不低于 control；
- teacher match 比 control 至少提高 0.02；
- paired regret 不退化；
- p95 latency <=500 ms。

只有满足这些条件，才能把 closed-loop 改善合理归因于 deployment-boundary action preservation，而不是偶然的预算压缩或 fallback 行为。
