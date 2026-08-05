# V61 Gate 根因复核与 V62 DCAB-EWFC 优化报告

## 1. 结论先行

本次复核后的三个 Gate 真实状态不是“全部算法失败”：

| Gate | V61 报告字段 | 复核结论 |
|---|---:|---|
| Protocol | FAIL | **工程可观测性假失败**。set-conditioned residual 在 planner/evaluator 中已经生成，但 `bdse_metrics.py` 的 qdiag 前缀过滤器没有保留 `set_conditioned_residual_*`，导致 1000 条 JSONL 的四个字段覆盖率均为 0。现有结果无法补写字段，但不能据此断言 set head 未执行。 |
| Minimum | official FAIL；metrics PASS | **Minimum 算法指标已通过**。`minimum_metrics_pass=true`，失败列表为空；`minimum_pass=false` 只是 checker 把 Protocol 作为先决条件串联后的官方状态。proposal decisive recall=0.803054，selected recall=0.611521。 |
| Competitive | FAIL | **真实算法失败**。candidate/local/foundation teacher match 都是 0.141，总 gain、residual gain、pair-full gain 均为 0；residual beneficial/harmful 均为 0。 |

因此，不能继续围绕“Minimum 没过”盲目增加 proposal 权重。V61 已经把 V60 的 atom-proposal 上游失稳修好；当前首要矛盾是 **部署 action-query interface 与训练 winner-preservation interface 不一致**，其次才是 residual 的方向性、幅度和可证翻转能力。

## 2. 三个 Gate 的根本原因

### 2.1 Protocol Gate：字段导出错误，而不是已证实的算法失效

静态调用链为：

```text
model set factors
  -> tournament.py 生成 set_conditioned_residual_* diagnostics
  -> evaluate_open_loop.py 写入 qdiag
  -> bdse_metrics.py 末端前缀过滤
  -> 字段被丢弃
```

V61 gate 唯一的 Protocol failures 正是四个 set 字段覆盖率不足；dual calibration、candidate/local/foundation 配置一致性、group-disjoint split、exact-HAB 训练健康度均通过。修复后 V62 checker 还会要求：all-valid action mode 已启用、valid-action 查询覆盖率可观测、dense→HAB→runtime 分解指标可观测、literal criticality 指标可观测，防止再次出现“代码里有路径、结果里没有证据”的情况。

### 2.2 Minimum Gate：指标通过，被 Protocol 串联遮蔽

V61 的正式报告中：

- `minimum_failures=[]`；
- `minimum_metrics_pass=true`；
- proposal decisive recall = 0.803054，高于 0.72；
- selected decisive recall = 0.611521；
- effective decisive recall = 0.777185；
- interaction recall = 0.582544。

V62 checker 将同时报告 `*_metrics_pass` 与 `*_official_pass`，不再把协议阻断误写成算法指标失败。

### 2.3 Competitive Gate：真实失败链路

V61 的层级结果为：

| 路径 | teacher/action match |
|---|---:|
| Dense full-interface | 0.359 |
| Runtime sparse-full | 0.141 |
| B=16 final | 0.141 |
| B16 vs sparse-full | 0.981 |
| B16 vs dense-full | 0.172 |

这组数值说明：

1. **B=16 selector 不是当前主要失效点**。它对实际输入的 sparse-full winner 保持率为 0.981；一旦 sparse interface 已经选错，预算内 selector 只能忠实复现错误输入。
2. **主要损失发生在 dense local evidence 到 runtime sparse interface 之间**。V61 训练时 deployment-HAB 只对 atom 维度做 hard mask，却仍对全部 action 计算 dense `g`; runtime 则仅查询 rival graph 覆盖的 action，未查询 action 的 evidence contribution 被置零。训练中的 HAB winner-preservation 指标和真实 runtime 因而不是同一个 planner interface。
3. **Residual 不是仅被 calibration 压制**。raw proposal rate=0.016，1000 场景约 16 次；逐行复核只有 1 次 proposed action 指向 teacher。即使把 residual epsilon 从 0.526467 降到 0，绝大多数 proposal 的方向也不正确。epsilon 大于训练 reserve（0.150）是次级抑制因素，不是零收益的唯一原因。
4. pair-full 与 local 完全相同，dense→pair-full flip rate=0.828。但在修复 action-query bridge 前，这个数字不能全部归因于 action potential 表示，因为 pair-full 也消费不完整的 action-query sparse tensor。

修订后的模型状态应表述为：

> V60 使用部署不一致且不稳定的 proposal surrogate，持续破坏 dense evidence 到 HAB Top-M 的 atom 桥；V61 已经修复并稳定了 atom 桥，但真实部署仍保留一个未显式建模的 action-query sparsification。已有 dense decision 信息在 action-query bridge 上被截断；residual 有梯度，但 proposal 很少、幅度弱且大多不朝向 teacher。

## 3. 排除工程错误后的判断

### 3.1 `BDSE_VAL_ORIGINAL_CACHE` / `BDSE_VAL_CACHE` 问题

上传的命令文件使用的是 `BDSE_VAL_CACHE_ORIGINAL`，而直接 runner 消费 `BDSE_VAL_CACHE`。完整 V61 pipeline 在构建并验证 group-disjoint `val_tune/val_calib` 后确实执行了：

```bash
export BDSE_VAL_CACHE="$BDSE_SPLIT_CACHE"
```

结果 provenance 也确认 tune/calibration group-disjoint、无 group overlap，所以**本次上传的 V61 结果没有加载错 validation split 的证据**。但直接调用 `run_v61_dehab_bfar_dbap.sh` 时环境变量链很脆弱。V62 runner 已加入：

```bash
BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-${BDSE_SPLIT_CACHE:-${BDSE_VAL_CACHE_ORIGINAL:-}}}"
```

主 pipeline 仍显式 export，避免依赖 fallback。

### 3.2 Set-head provenance

字段缺失来自 metrics export，而不是 evaluator 没有构造字段。修复后必须用 fresh open-loop 重新生成 JSONL；不能通过修改旧 summary 把 Protocol 追认成 PASS。

### 3.3 Signed teacher regret

partial test 的 `full_interface_teacher_regret=-1961.54` 不足以单独证明 teacher index/cost sign 错误，因为 teacher action 是 safety-first lexicographic winner，而原指标只计算 scalar `J_T(candidate)-J_T(teacher)`；一个 safety 更差的 action 可能有更低 scalar cost。V62 同时输出：

- signed `teacher_scalar_cost_delta`：用于诊断 cost 方向；
- nonnegative `teacher_regret=max(delta,0)`：用于 gate/tail 比较。

原字段保留作兼容，但不再让负 signed delta 污染 regret gate。

### 3.4 无效训练热路径

V61 顶层 `loss_weights.action=0`，但仍可能构造 CPU deployment certificate mask，随后整体乘 0。V62 在 aggregate action loss 为 0 时跳过这条 stop-gradient CPU 路径；同时当 local uncertainty 完全关闭时跳过未使用的 local variance head。两项都是不改变模型函数的热路径删除。

## 4. V61 的正向信号

### 明确值得保留

- immutable foundation anchor 与同 checkpoint local/foundation controls；
- 固定 planner-interface evidence budget `B=16`；
- HAB family hierarchy、soft-interaction reservation、structural safety bypass；
- 预算内确定性 AOCC selector 与审计证书；
- group-disjoint calibration 与 paired scenario/timestamp evaluation；
- direct integrable action potential / set-conditioned residual 的代码基础；
- proposal/residual 分阶段 routing；
- gate-feasible checkpoint selection。

### V61 已验证有效的算法信号

- proposal recall 从 foundation 0.766319 提升到 candidate 0.803054；
- selected recall 从 0.581940 提升到 0.611521；
- effective interaction recall 从 0.727941 提升到 0.748462；
- `L_prop` 从 3.3935 平稳降到 3.2839，没有 V60 的 runaway；
- proposal logit RMS 最大 1.9116；
- fast/exact HAB mask Jaccard 始终 1.0；
- 每次验证 `val_minimum_gate_feasible=1`；
- evidence certificate=0.8875，fallback=0.111。

这些信号表明，V61 的 deployment-exact HAB atom proposal、logit centering/mass normalization、checkpoint feasibility 优先级应保留，不应退回 global Top-M surrogate，也不应重新做简单 proposal 大权重搜索。

## 5. V62 已落地的算法优化

版本名：**V62 Deployment-Complete Action Bridge + Exact Winner-Flip Criticality BFAR-DBAP (DCAB-EWFC)**。

### 5.1 Deployment-complete action bridge

新增 `runtime.action_query_mode: all_valid`。对 HAB Top-M 或最终 B 个**已查询 evidence atoms**，一次性向固定 candidate bank 中全部 valid actions 展开 `g(i,a)`：

```text
query complexity <= B × K,  B=16, K<=32
```

这没有增加 evidence atom budget，也没有绕过 planner interface；它消除的是 V61 未声明的第二个 action sparsification。每个 atom 仍可审计，selector 仍只能在 B 的 evidence budget 内工作。保留 `rival_graph` 配置作为同 checkpoint ablation，验证收益确实来自 action bridge，而不是其他偶然变化。

新增 runtime/eval 指标：

- `action_query_mode_all_valid`；
- `valid_action_count`；
- `queried_valid_action_fraction`；
- `hab_topm_dense_value_action_match`；
- `hab_topm_dense_value_vs_runtime_sparse_full_match`；
- `runtime_sparse_value_bridge_flip_rate`；
- `selected_budget_dense_value_action_match`；
- `selected_budget_dense_value_vs_deployed_match`。

这样可以把失败分解为 atom proposal、action query、B16 selection、potential/residual 四层。

### 5.2 Literal exact winner-flip criticality

V62 不再用 margin deficit 代替论文定义。对每个 active atom `i`，在 dense local interface 上计算：

```text
winner_dense = argmin_a [J0(a) + sum_j g(j,a)]
winner_without_i = argmin_a [J0(a) + sum_{j!=i} g(j,a)]
critical(i) = 1[winner_without_i != winner_dense]
```

binary label 完全由 winner action 是否翻转决定；severity 只用于 critical atoms 内部排序，不会把“margin 变小但 winner 不变”的 atom 标成 critical。新增 proposal BCE/listwise loss 与 Top-M/selected recall 指标。

### 5.3 论文 novelty 的保留与表述修正

核心不变：固定 evidence budget、可审计 atoms、预算内 selector、action-flip criticality、双证书。论文修订版补充了 fixed `B×K` action expansion，并明确：这里的 “exact AOCC selector” 是**对论文定义的确定性、固定预算 AOCC operator 的精确执行与可复现审计**；当前 acquisition order 仍是 greedy/anytime，不应声称求解了全局组合最优集合。这个表述可避免 CCF-A 审稿人抓住 “exact” 与算法实现的语义漏洞。

### 5.4 Residual 的处理策略

V62 暂不通过扩大 residual scale 或降低 conformal epsilon 强行制造 flip。先完成 action bridge 后按以下顺序判断：

1. 若 sparse-full 显著提高、pair-full 仍退回 0.141 附近，再升级 action-potential projection/teacher distillation；
2. 若 raw residual teacher-directed rate 仍低，升级 residual target 为 teacher-directed certified correction，并单独优化 mean/variance；
3. 只有 raw beneficial > harmful、margin 足够且 residual epsilon 与训练 reserve 同量级后，才讨论放宽 flip guard。

这避免 residual 再次被迫补偿 upstream interface 中根本不存在的信息。

## 6. 数据集判断

Validation 共 58418 个样本：full interface=0.9657、runtime decision sufficiency=0.7490、B16 oracle=0.9120，说明 B=16 研究前提成立。但 safe-candidate-exists=0.7173<0.75，teacher candidate-log ADE p50=5.480、p90=12.915，candidate bank 覆盖仍是闭环上限之一。`decision_boundary_atom_count=0` 说明当前 diagnostics 没有显式 boundary atom 标签；V62 的 literal leave-one-out criticality不依赖该字段。

当前 test diagnostics 有 67042 个样本，但用户已说明尚未构建完成；它的 full interface=0.9343、runtime=0.6407、B16=0.8399、safe-candidate-exists=0.5779，均弱于 val，存在明显分布/覆盖差异。建议将**构建完成并通过 parity/readiness audit 后的 test**作为冻结最终 testing；当前 partial test 只用于找数据问题，禁止用于调权、选 checkpoint 或选择 V62/V63 方向。

## 7. 效率分析

V61 训练 epoch wall time（除首 epoch cache warmup 后）中位数约 1124.4s，loss hot path 是主要耗时。V62 的效率改动：

- action loss 总权重为 0 时不再构造 CPU certificate mask；
- local uncertainty 未启用时跳过 local variance head；
- exact criticality 完全张量化，不逐 atom 做 Python loop；
- exact runtime HAB 仍采用稀疏抽样，避免每 scene 每 step CPU 调用；
- 同 checkpoint `rival_graph` ablation 默认不在主 pipeline 自动执行，避免重复校准/训练。

上传 V61 中 `action_atom_query_count=690.488`、Top-M atoms 约 23.998，折算 legacy 平均查询 action 数约 28.773。固定候选上限 K=32 时，全 valid action expansion 的最坏查询数增幅约 11.2%（实际取决于 valid count），evidence atom 数仍不变。新 exact-criticality CPU 合成 microbenchmark（B=8,E=48,K=32, 1 thread）forward p50=0.333ms，forward+backward p50=0.457ms。真实 CUDA/DDP 增量必须从 fresh V62 日志确认，不能用合成数字宣称训练加速。

Open-loop p95=948.0ms，仍高于实时目标。主因是 predict（约 424.9ms）和 selector（约 74.4ms），而不是 tournament（约 6.4ms）。在 winner-level 信号转正前，不建议用模型裁剪或减少证据 budget 换速度；先用 profiler 确认 all-valid expansion 的真实增量，再做 batched query projection、cache 和 CUDA kernel 优化。

## 8. Fresh V62 的判定顺序

1. Smoke：literal critical loss 有梯度；exact HAB fraction>0；fast/exact Jaccard 可观测；proposal RMS<20。
2. Protocol：set residual 四字段覆盖率>=0.99；`action_query_mode_all_valid=1`；`queried_valid_action_fraction≈1`；bridge 与 criticality 字段覆盖完整。
3. Minimum metrics：proposal decisive recall>=0.72，selected recall 不明显退化；单独看 metric pass，不被 Protocol cascade 混淆。
4. Bridge：`hab_topm_dense_value_vs_runtime_sparse_full_match>=0.95`，`runtime_sparse_value_bridge_flip_rate<=0.05`；否则不得归因 residual/potential。
5. Criticality：Top-M literal critical recall 目标>=0.80，selected>=0.50；先看 scene support，避免无 critical scenes 时错误解读。
6. Winner：sparse-full、pair-full、candidate teacher match 必须高于 V61 的 0.141；budget-vs-sparse 继续保持高位。
7. Residual：raw teacher-directed proposal、beneficial/harmful、robust margin 和 calibrated epsilon 联合判断。
8. Protocol + Minimum + bridge 信号通过后跑 paired CL20；Competitive 通过且 CL20 无安全退化后再跑 CL100。

## 9. 验证边界

本次完成的是代码静态/单元/脚本/配置验证和 V61 结果复核，没有在当前环境执行 fresh V62 train、calibration、open-loop 或 nuPlan closed-loop。因此不能提前声称 V62 通过 gate、改善 closed-loop 或达到 SOTA。V62 的价值是把下一轮实验变成可归因检验：先证明 action bridge，再决定是否升级 potential 和 residual。
