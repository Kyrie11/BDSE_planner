# BDSE v45 Gate 失败诊断与 v46 AOCC 优化报告

## 1. 结论摘要

本次 v45 训练不再存在 v44 的 NaN 问题，checkpoint 可以视为一次真实训练结果；但它仍不能通过 strict open-loop gate，也不应进入 CL20。

核心原因不是单点故障，而是三个层级同时存在问题：

1. **训练与部署 selector 不充分对齐**：所谓 exact distillation 的实际覆盖率最低仅 6.248%，其余约 93.75% 场景仍由与 runtime selector 一致性仅约 0.23 的 GPU surrogate 提供 action-path 监督。
2. **dense pair/interface 表示仍不够强**：full-interface action match 为 0.296，处于上一轮决策树设定的 0.30 临界线附近；逐场景中有 602/1000 个场景 dense 与 budgeted 都错误。
3. **预算压缩仍破坏决策**：在 296 个 dense 正确场景中，预算化流程使 153 个场景变错；虽然预算流程又救回了 102 个 dense 错误场景，但净损失 51 个场景。
4. **最终 tournament 与 selector 内部诊断不一致**：`selector_margin_coreset_target_action_preserved=0.929` 使用的是 selector 内部 pairwise preference action，而最终 planner 使用 rival graph、soft-min、安全 guard 和 utility 的 `run_pair_conditioned_tournament`。因此 0.929 不能证明最终动作被保留。
5. **运行时 p95=901.03 ms**：超过 500 ms。双卡 open-loop 是场景吞吐并行，不会自动降低单次 replan latency。

因此，本报告建议并已实现一个 **v46 双轨版本**：

- 表示学习侧：online teacher-hard-rival mining、triangle/cycle consistency、重新启用并校准 pair uncertainty；
- selector 侧：Anytime One-Sided Adverse-Bound Certificate Coreset（代码简称 AOCC），以与最终 winner–rival certificate 一致的 nested one-sided deficit objective 替换 quota/beam/swap 搜索；
- 评估侧：新增与最终 tournament 完全相同的 pair-full-interface 指标、逐阶段 latency、严格 scenario-token paired gate；
- 训练侧：每个 local scene、每个 step 使用 exact CPU selector，强制 exact fraction ≥0.99。

AOCC 值得现在实施，但不能单独替代 dense-interface 修复。当前数据同时证明了 compression failure 和 representation failure；仅做其中一侧都不足以通过 gate。

---

## 2. 论文 Idea、算法与代码对齐审查

论文当前的核心对象是：在有限候选动作集合上，从层次化 evidence atom 集合中选择固定预算证据，保留 teacher-best 相对 decisive rivals 的正 margin，并通过 uncertainty-aware pairwise tournament 输出动作。

论文的主要组成包括：

- route-conditioned candidate bank；
- metric-aligned robust teacher；
- base cost + evidence atom local cost partition；
- pair-conditioned atom margin `d_i(a,b)`；
- hierarchical atom proposal/family allocation；
- uncertainty-aware LCB acquisition；
- risk-aware pairwise tournament；
- one-sided margin preservation theorem。

论文当前方法公式主要是 capped LCB greedy，并明确使用 uncertainty 与 calibration；但 v45 部署实际使用：

- `signed_margin_coreset / MARS`；
- backward removal + swap passes；
- `beta_uncertainty=0`、`epsilon_cal=0`；
- structural safety bypass；
- soft interaction post-fill/quota；
- 与论文算法框和 theorem 不完全相同的 tournament path。

这会在投稿时形成高风险的 theorem–algorithm–runtime mismatch。v46 AOCC 的设计目标之一，就是把论文中的 one-sided certificate、训练目标、runtime ordering 和 budget curve 统一为同一个数学对象。

另一个需要在论文中谨慎表述的问题是“query budget”。当前代码在选择前已经对 Top-M atoms 和 runtime pair union 进行神经 pair scoring，因此 B 更准确地表示 **最终保留并送入 certificate/tournament 的 interface evidence budget**，而不是完全避免所有未选 atoms 的高成本神经查询。若论文要强调 active high-cost querying，需要进一步引入 cheap proposal scorer + 按 AOCC ordering 顺序调用 high-fidelity scorer 的两阶段实现。

---

## 3. v45 Gate 失败的定量拆解

### 3.1 Gate 指标

| 指标 | Candidate | Control | 增益/门槛 | 结论 |
|---|---:|---:|---:|---|
| Teacher action match | 0.245 | 0.238 | +0.007，要求 ≥+0.020 | FAIL |
| Winner–rival sign | 0.653926 | 0.638368 | +0.015558 | 有提升，但不足以弥补 action error |
| Evidence sufficiency | 0.080182 | 0.070399 | +0.009783，要求 ≥+0.010 | 仅差 0.000217，但统计与实际意义仍不足 |
| Exact selector fraction | 0.06248 | — | 要求 ≥0.99 | FAIL |
| Planner latency p95 | 901.03 ms | — | ≤500 ms | FAIL |
| Paired median/p90 regret | 未验证 | — | candidate/control JSONL 必须可一一配对 | FAIL/不可判定 |

用户命令中 control summary 与 control JSONL 来自两个不同根目录：

- summary：`outputs_v43_sabdacc_runtime_v30ckpt/...`
- JSONL：`outputs_v43/...`

即使两个文件名称相近，也不能假定它们来自相同 checkpoint、scenario token 顺序和配置。旧 checker 又没有按 scenario token 严格 join，因此 paired regret gate 不可信。v46 checker 已强制检查 token、重复键、缺失键与 summary/JSONL 一致性。

### 3.2 逐场景误差链

1000 场景分解：

| Dense full-interface | Budgeted final | 场景数 | 含义 |
|---|---|---:|---|
| 对 | 对 | 143 | 两层均成功 |
| 错 | 对 | 102 | budget/tournament 偶然或结构性救回 |
| 对 | 错 | 153 | 预算压缩破坏正确 dense 决策 |
| 错 | 错 | 602 | 上游表示/候选/teacher-interface 失败 |

由此可见：

- dense correct = 296；
- final correct = 245；
- selector/tournament 从 dense 正确集合中损失 153；
- selector/tournament 从 dense 错误集合中救回 102；
- 净变化 = -51。

所以“full-interface 低于 0.30 就停止改 selector”在这次实验中不能机械执行。0.296 只比阈值低 0.004，而且当前 `full_interface_action_match` 还不是最终 pair-conditioned tournament 的 full-support 版本；与此同时 153 个 compression failures 已经直接证明 selector path 需要修复。正确决策是：**先补正确的 pair-full 诊断，同时并行修表示和 selector，不再做 quota/beam/swap 参数扫。**

### 3.3 Proposal 与 selector

| 指标 | 数值 |
|---|---:|
| Proposal decisive recall | 0.832009 |
| Selected decisive recall | 0.388432 |
| Proposal interaction decisive recall | 0.817510 |
| Selected interaction decisive recall | 0.334288 |
| Dense winner–rival sign | 0.801493 |
| Selected winner–rival sign | 0.653926 |
| Selector pair sign disagreement | 0.178043 |
| Tournament pair sign disagreement | 0.189179 |

Proposal pool 已覆盖多数 decisive evidence，主要损失发生在预算保留和最终 tournament。继续增加 Top-M 或做普通 proposal sweep 的优先级不高。更值得做的是：

- 让 selector objective 直接优化 one-sided winner–rival deficit；
- 使用同一个 nested ordering 服务 B=4/8/16/24/32；
- 消除 selector surrogate 与 exact runtime 的训练断层；
- 对最终 pair-conditioned tournament 做 full-support oracle replacement。

### 3.4 训练覆盖与监督质量

训练日志：

- 12 个 epoch 全部 finite；
- total loss 约 926.34–944.13；
- exact selector fraction 最低 0.0、稳定阶段约 0.06248；
- surrogate/exact agreement 最后约 0.2311；
- `L_base` 约 2412–2459，变化很小；
- pair loss 约 0.009，量级远小于 base path；
- 500 场景 validation 最后 teacher match 0.328、full-interface 0.374，但最终 1000 场景只有 0.245/0.296。

当前 fast 配置：

- local batch = 4；
- exact scenes/rank = 1；
- exact every 4 steps；

理论覆盖为 `1/4 × 1/4 = 1/16 = 6.25%`，与日志一致。其余训练使用 surrogate mask，而 surrogate/exact agreement 只有约 0.23。由此得到的 checkpoint 不能被描述为 fully deployment-aligned exact selector training。

500 场景验证结果明显高于最终 1000 场景，说明当前 checkpoint selection 仍可能被小验证子集偏置。v46 应使用固定、不可变、场景不重叠的 calibration/validation/test split；至少用固定 1000 场景 validation，并报告 bootstrap CI，而不是用小子集峰值代表最终结果。

---

## 4. 路线图中哪些步骤应做

### 4.1 立即做，已纳入 v46

1. **Exact deployment-path training**：所有 local scenes、每个 step 使用 exact selector。
2. **Pair-full-interface exact tournament diagnostic**：用最终 runtime tournament 和全 Top-M decision atoms 计算上限。
3. **Online teacher-best vs hard-rival mining**：在已缓存 pair 中，根据当前模型最脆弱的 teacher–rival margin 动态选择 hard rivals。
4. **Triangle/cycle consistency**：约束 `M(a,b)+M(b,c)+M(c,a)=0`，减少 pair graph 内部不可传递与局部漂移。
5. **Uncertainty/adverse-bound calibration**：重新训练 uncertainty head，并在独立 calibration split 上得到 one-sided residual quantile。
6. **AOCC nested ordering**：替代 MARS swap/quota/beam 主路径。
7. **Stage latency instrumentation**：分别记录 predict、selector、tournament 和 internal total。
8. **严格 paired gate**：按 scenario token join candidate/control JSONL。

### 4.2 已有，不应重复实现

1. **Antisymmetric pair-head**：当前 pair head 已通过 `f(z_ab)-f(z_ba)` 强制反对称。
2. **静态 hard-rival / near-tie pair construction**：pair builder 已包含 teacher-best 对 top rivals、near-tie pairs 和 safe–unsafe crossing pairs。
3. **Pair-score union materialization**：selector 与 tournament 的 pair union 已尽量在一次模型调用中评分。
4. **Proposal 与 pair head 的共享编码**：当前 scene/action/evidence encoding 已共享。

这些项仍需实验验证，但不应以“新增算法贡献”重复包装。

### 4.3 暂不优先

- quota sweep；
- beam width/branch sweep；
- swap-pass sweep；
- 单纯扩大 Top-M；
- 在 gate 未过前运行 CL20；
- 仅靠双 GPU 分片宣称 latency 改善。

### 4.4 需要设计，但当前 cache 不能完整回答

**Candidate-bank oracle regret** 需要区分：

1. teacher-best within current bank；
2. logged/expert trajectory 投影到 bank 后的 coverage gap；
3. 更密候选 bank 或 continuous optimizer 的 oracle；
4. closed-loop objective 下的真正 bank regret。

当前 cache 主要给出同一有限 bank 内的 teacher ranking，不能仅用 teacher regret 冒充 candidate-bank oracle regret。需要增加更密候选 bank、logged trajectory projection 和 oracle candidate coverage experiment。

---

## 5. v46 AOCC 算法

对当前预测 target action `w` 和 rivals `r` 定义 pair contribution：

`d_i(w,r)`。

对未保留 atom 使用 one-sided adverse lower bound：

`ell_i(w,r) = min(0, d_i(w,r) - beta * sigma_i(w,r) - epsilon)`。

查询/保留 atom 后，将 adverse bound 替换为模型贡献，产生非负 improvement：

`Delta_i(w,r) = d_i(w,r) - ell_i(w,r) >= 0`。

空集 certificate：

`C_wr(empty) = base_margin(w,r) + sum_i ell_i(w,r)`。

目标 margin 为 `gamma_wr`，deficit：

`D_wr(S) = [gamma_wr - C_wr(S)]_+`。

总目标：

`F(S) = sum_wr q_wr * min(D_wr(empty), sum_{i in S} Delta_i(w,r))`。

这是一类 weighted capped coverage objective。只要每个 atom 的 improvement 非负，它对集合 S 单调且具有 diminishing returns；因此可以生成单个 cost-aware greedy ordering，各预算取前缀，天然满足 nestedness。经典单调子模最大化理论可为相应的 cardinality/knapsack greedy 变体提供近似保证，但论文必须准确匹配实际 cost-aware algorithm、枚举/部分枚举条件和所引用定理，不能笼统声称所有实现都自动获得同一常数保证。

需要区分两种保证：

- **确定性算法结构**：给定有效 lower bounds，deficit reduction 单调、预算前缀 nested；
- **统计 coverage**：`ell_i <= d_i^true` 的概率保证，需要独立 calibration set、exchangeability/分布条件和有限样本 quantile。未经校准的 epsilon=0.05 只能是 heuristic，不应写成 formal coverage guarantee。

---

## 6. 已完成的代码修改

### 6.1 Selector

`bdse/planner/selector.py`

- 新增 `_anytime_one_sided_adverse_certificate_from_pair_delta`；
- 新增 mode alias：`anytime_adverse_certificate`、`one_sided_adverse_certificate`、`aocc`、`aobcc`、`nested_certificate`；
- 从 full Top-M pair field 推导 target action；
- 构造 target-vs-rival one-sided adverse bounds；
- 采用 cost-aware capped-deficit greedy；
- 生成共享 nested ordering；
- 支持 certificate satisfied 后 early stop；
- 输出 initial/final deficit、deficit reduction、certified pair fraction、stop budget、nested length、bound violation 等诊断。

### 6.2 表示学习损失

`bdse/model/losses.py`

- 新增 online teacher-hard-rival loss；
- 新增 sparse triangle/cycle consistency loss；
- 将 AOCC 参数传给 exact selector；
- 保持原有 exact antisymmetric pair construction；
- 重新启用 uncertainty/calibration loss 配置。

### 6.3 Planner 与评估

`bdse/planner/nuplan_planner.py`

- 接入 AOCC selector 参数；
- 新增 `stage_predict_ms`、`stage_selector_ms`、`stage_tournament_ms`、`stage_total_internal_ms`。

`bdse/experiments/evaluate_open_loop.py`

- 新增 `pair_full_interface_action_match`；
- 使用与最终部署完全相同的 pair-conditioned tournament，但输入 full Top-M atoms；
- 新增 `budget_vs_pair_full_match`；
- JSONL 输出 `pair_full_action`。

`bdse/metrics/bdse_metrics.py`

- 聚合逐阶段 latency 指标。

### 6.4 Gate 与 calibration

`bdse/tools/check_v46_aocc_gate.py`

- 检查 train loss finite；
- 检查 exact fraction；
- 检查 pair-full 与 budgeted 指标；
- candidate/control JSONL 按 token/timestamp 严格 join；
- 检查重复、缺失与 summary/JSONL 不一致；
- 计算 paired median、p90、CVaR/regret 类统计。

`bdse/tools/calibrate_v46_adverse_bounds.py`

- 在 held-out cache 上计算 one-sided residual；
- 使用 finite-sample split-conformal quantile；
- 输出全局 epsilon 推荐值与 family-wise diagnostics；
- 明确要求 calibration 与 evaluation 场景不重叠。

### 6.5 配置与脚本

- `bdse/configs/v46_bdse_aocc_train_2gpu.yaml`
- `bdse/configs/v46_bdse_aocc_cl.yaml`
- `run_v46_aocc.sh`
- `bdse/tests/test_v46_aocc.py`

关键训练设置：

- `deployment_selector_backend: exact_cpu`
- `deployment_selector_scenes_per_rank: 0`（全部 local scenes）
- `deployment_selector_every_n_steps: 1`
- `min_deployment_exact_fraction: 0.99`
- `selector_cap_mode: anytime_adverse_certificate`
- `force_fill_budget: false`
- `compute_pair_uncertainty: true`
- online hard-rival loss weight 2.0
- cycle consistency loss weight 0.5

完整测试结果：`147 passed, 5 warnings`。

合成 CPU selector microbenchmark（E=64, P=48, B=16，300 次，仅比较 selector 本体）：

| Selector | Mean | p95 | 平均 selected |
|---|---:|---:|---:|
| MARS swap-2 | 31.18 ms | 32.05 ms | 20.0 |
| AOCC | 26.65 ms | 26.89 ms | 19.0 |

AOCC 在该合成测试中约快 15%，但这不能外推为端到端 901→500 ms。必须依据新 stage timer 判断真实瓶颈。

---

## 7. 推荐实验顺序

### Phase 0：先复算 v45，修正诊断

使用 v46 代码但保持 v45 原 selector/config，重新计算 `pair_full_interface_action_match` 与 stage latency。这一步不训练，目的是确定上一轮 0.296 是否低估/高估真正 pair-full deployment upper bound。

然后用同一个 v45 checkpoint，只切换到 AOCC config 做 selector-only replay。这样可以隔离：

- AOCC 本身的收益；
- 表示学习新增 loss 的收益；
- exact training 的收益。

推荐形成四组：

| 组 | Checkpoint | Selector | 用途 |
|---|---|---|---|
| A | v45 | MARS | 修正后的真实 baseline |
| B | v45 | AOCC | selector-only |
| C | v46 | MARS | representation/exact-training only |
| D | v46 | AOCC | 完整方法 |

其中 C 只需复制 v46 eval config 并把 selector mode 改回 MARS，不需要重训第二个 checkpoint。

### Phase 1：v46 exact training

从冻结 v30 checkpoint 重新训练，不从 v45 fast checkpoint warm-start。原因是 v45 的主要 action supervision 与 exact runtime 不一致，直接 warm-start 会保留 surrogate-induced bias；可把 v45 warm-start 作为单独消融，而不是主结果。

### Phase 2：独立 adverse-bound calibration

训练完成后，在独立 calibration scenarios 上估计 epsilon。不要用最终 test/open-loop gate 的 1000 场景同时调 epsilon 和报告指标。

### Phase 3：固定 test split open-loop

报告：

- teacher match；
- pair-full-interface match；
- budget-vs-pair-full；
- decisive margin coverage；
- certificate deficit；
- false certificate rate；
- B=4/8/16/24/32/full nested curve；
- paired median/p90/CVaR regret；
- action family / interaction type 分层；
- stage latency；
- selected evidence count 与 full prescore count。

### Phase 4：CL20/CL100

只有 strict open-loop gate PASS 后运行 CL20；CL20 安全和交互指标不劣后再运行 CL100。

---

## 8. 结果后的决策树

### 情况 A：`pair_full_interface_action_match < 0.30`

主瓶颈仍是 representation/candidate interface：

- 增大 online hard-rival K 或 margin temperature；
- 增强 cycle consistency；
- 分 action family / interaction type 调整 loss；
- 检查 base normalization，降低绝对 base regression 对总 loss 的支配；
- 做 pair-full oracle replacement 与 candidate-bank coverage；
- 暂不继续调 AOCC beta/epsilon 以外的 selector 结构。

### 情况 B：pair-full 上升，但 `budget_vs_pair_full`、certificate sufficiency 不升

主瓶颈是 AOCC bound/ordering：

- 检查 adverse-bound violation rate；
- 使用 calibration epsilon；
- 比较 global vs family-wise bound；
- 分析 target-rival graph recall；
- 调 rival cap、certificate margin、beta；
- 不回到 quota/beam/swap sweep。

### 情况 C：accuracy 达标，但 p95 >500 ms

按 stage timer 决策：

- predict 占主导：减少重复 feature build，缓存 rival graph/query features，控制 Top-M pair union，进一步共享 pair encoding；
- selector 占主导：把当前 NumPy AOCC 改为 batched Torch/GPU implementation，并做 incremental deficit update；
- tournament 占主导：缓存 rival graph、vectorize soft-min、安全 guard 与 utility；
- CPU/GPU transfer 占主导：保持一次 device materialization，禁止 pair-by-pair `.cpu().numpy()`；
- 只报告 single-device per-replan latency，双卡只作为 throughput 指标。

---

## 9. 投稿与 novelty 边界

v46 能形成更清晰的论文贡献链：

1. pair-conditioned decision evidence；
2. calibrated one-sided adverse bounds；
3. nested anytime certificate coreset；
4. fixed-budget/any-budget unified ordering；
5. deployment-identical tournament diagnostics；
6. interaction planning 中的 budget–certificate–latency curve。

但目前不能声称“理论 SOTA”或保证 CCF-A 接收。要达到可信的 top-tier 贡献，至少还需要：

- theorem 与实际 AOCC 实现逐行一致；
- 独立 calibration protocol；
- candidate coverage/oracle regret；
- 多 seed、paired CI；
- 统一 nuPlan protocol 下的强 baseline；
- closed-loop 结果；
- 证明增益不是来自更强 candidate bank、backbone 或 teacher；
- 明确 query budget 与 retained-interface budget 的区别。

论文的数据应表述为 `nuPlan-derived BDSE teacher/evidence supervision cache`。除非公开 schema、生成脚本、固定 split、统计、许可和可复现下载，不宜直接宣称一个独立新数据集。
