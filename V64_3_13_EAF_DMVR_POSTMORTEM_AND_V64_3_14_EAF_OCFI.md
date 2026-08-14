# V64.3.13 EAF-DMVR 结果复盘与 V64.3.14 EAF-OCFI 设计

日期：2026-08-14

## 0. 结论摘要

本轮 V64.3.13 screen **不应解释为 EAF value representation 失败**，也不应直接进入 action/evidence representation unfreeze。

修复 screen checker 后，当前上传结果中唯一满足 exact-scene training instrumentation 的 epoch 是 **epoch 3**。相对 epoch -1 anchor：

- complete-frontier pair-sign accuracy：`0.46012 -> 0.69246`，**+23.23pp**；
- complete-frontier wrong-anchor corrected：`0.22135 -> 0.25717`，**+3.58pp**；
- complete-frontier action match：`0.25781 -> 0.23398`，**-2.38pp**；
- correct-anchor preservation：`0.21875 -> 0.03750`，**-18.13pp**；
- teacher action match：`0.178 -> 0.152`，**-2.6pp**；
- teacher regret：`20133.34 -> 14756.02`，**改善 26.71%**；
- raw residual flip proposed：`0.422 -> 0.930`；
- deployed residual flip：`0.176 -> 0.586`；
- guard-allowed flip：`0.194 -> 0.606`；
- beneficial intervention：`0.018 -> 0.076`；
- harmful intervention：`0.008 -> 0.092`；
- pair-full / local-pair-full：均保持 `0.174`；
- proposal decisive recall、exact-critical Top-M、exact-critical selected、evidence certificate 全部保持不变。

因此当前最值得检验的瓶颈是：

`selected B=16 evidence -> EAF complete-frontier value` **已经产生显著信号**，但

`EAF complete-frontier value -> one-sided intervention / anchor preservation` **没有与新 estimator 校准**。

V64.3.14 设计为 **EAF-OCFI: Evidence-Attributed Frontier with One-Sided Calibrated Frontier Intervention**。它完全冻结 acquisition、B/M、DARM/DBR、EAF 参数，只给 EAF challenger 一个专属的、逐 evidence 归一化的 one-sided split-calibration gate。

---

## 1. 与论文主线的一致性

论文的真正理论对象不是 proposal score，而是：在固定 planner-interface budget 下，预算 evidence 能否保护 teacher decisive pair margin，并通过 one-sided margin preservation 保住最终 action。

当前算法主线继续保持：

`fixed planner-interface budget`
`-> auditable evidence atoms`
`-> budget-feasible decisive-margin marginal utility`
`-> budgeted acquisition [terminally frozen]`
`-> exact selected B=16 evidence`
`-> evidence-attributed complete decisive frontier value`
`-> one-sided calibrated intervention`
`-> final decision preservation`.

V64.3.14 不改变论文问题定义，不把目标切回 distribution reconstruction，也不把性能问题重新归因给 proposal/HAB。

---

## 2. V64.3.13 当前结果的正确解释

### 2.1 原 checker 有两个会改变算法分支的工程问题

**问题 A：runtime EAF instrumentation 没有进入最终 metric aggregation。**

`train.py`/`evaluate_open_loop.py` 会把 tournament 的 `decisive_frontier_value_*` 放入 query diagnostics，但 `bdse/metrics/bdse_metrics.py::compute_bdse_diagnostics()` 原先没有传播这些 prefix。因此训练日志里的：

- `val_decisive_frontier_value_active`；
- `val_decisive_frontier_value_complete_star_coverage`；
- `val_decisive_frontier_value_residual_rms`

为空/NaN。

这不是“runtime EAF 一定没运行”的证据，而是 instrumentation plumbing 缺失。本版已修复。

**问题 B：旧 screen checker 会让 instrumentation-invalid epoch 因 noisy endpoint 获胜。**

上传的旧报告选择 epoch 1，但该 epoch 的 `frontier_value_exact_scene_fraction=0.259375`。唯一 exact-scene fraction 为 `1.0` 的是 epoch 3。

新 checker 将：

1. exact-scene training instrumentation；
2. runtime instrumentation；
3. value-estimation gain；
4. preservation interface；
5. endpoint

分层审计，并把 training instrumentation validity 放在 epoch selection 的第一优先级。

对上传日志重新审计后：

- `selected_epoch = 3`；
- `training_instrumentation_valid = true`；
- `runtime_instrumentation_valid = false`（因为旧日志缺少上述 runtime prefix）；
- `acquisition_frozen = true`；
- `value_estimation_gain = true`；
- `preservation_interface_failure = true`；
- `next_action = replay_selected_epoch_with_runtime_frontier_instrumentation_then_eaf_ocfi`。

### 2.2 EAF value 本身并非“没有学到”

如果 frozen action/evidence representation 根本没有足够 pair-value capacity，最直接的预期应该是 complete-frontier pair-sign 仍接近 anchor，或者只获得很小提升。

实际 epoch 3 的 pair-sign 是 `0.69246`，相对 anchor `0.46012` 提升 **23.23pp**。epoch 0/1/2 的 pair-sign 也都在约 `0.696~0.735`。

这说明 frozen embedding 中存在可以被 EAF 读出的 decisive pair signal。当前数据不足以支持“先解冻 representation”的结论。

### 2.3 真正断裂发生在 value -> intervention

EAF 学习后：

- raw flip proposal 率从 `42.2%` 上升到约 `93%`；
- existing guard 仍允许约 `60.6%` 的 scenes 翻转 anchor；
- harmful intervention 从 `0.8%` 上升到 `9.2%`；
- correct-anchor preservation 严重下降；
- 但 teacher regret 大幅改善。

这说明 EAF 并非随机噪声：它能修掉一些代价很高的错误，所以 regret 明显降低；但它同时对大量本来正确的 anchor 过度干预，所以 match 和 preservation 下降。

这正对应：

`pair-value informativeness` **positive**

但

`pair-value calibration / intervention selectivity` **negative**。

---

## 3. 代码层面的因果根因

现有 `pair_action_anchor_guard` 在 EAF residual 加入 `M_B` 后，仍使用旧 pair residual pathway 的：

- `pair_atom_variance -> sigma`；
- `dual_certificate.residual_beta_uncertainty`；
- `dual_certificate.residual_epsilon_cal`

来判定新的 EAF challenger 是否可信。

EAF-DMVR 是一个新的 complete-frontier estimator，但它没有 EAF-specific one-sided calibration radius。当前配置中相关 residual beta/epsilon 事实上非常弱，因此新 residual 可以大规模越过旧 guard。

这就是“新 value estimator + 旧可信度接口”的训练/部署接口错位。

注意：这与 V64.3.12 RET 所修复的 selector train/runtime mismatch 不同。这里不是再做 acquisition semantics，而是 **value estimator 与 one-sided decision intervention 的 calibration semantics**。

---

## 4. V64.3.14 EAF-OCFI

全称：**Evidence-Attributed Frontier with One-Sided Calibrated Frontier Intervention**。

### 4.1 不改 EAF value，只改是否允许 intervention

对 selected-local / DARM anchor `a` 和 EAF raw challenger `b`，保留 V64.3.13 的完整 margin：

`M_hat(a,b) = M_DARM+DBR(a,b) + r_EAF,S(a,b)`.

V64.3.14 不重新训练 `r_EAF,S`。

### 4.2 从 EAF 的逐 evidence decomposition 得到 attribution energy

V64.3.13 本来就有 selected-atom additive decomposition。现在显式保留每个 atom 对 anchor-challenger edge 的贡献：

`c_i(a,b) = <tanh(z_i), c(a,b) * (u_b-u_a)> / sqrt(|S| d)`.

其和严格等于 EAF residual：

`r_EAF,S(a,b) = sum_i c_i(a,b)`.

定义 attribution scale：

`A_S(a,b) = sqrt(sum_i c_i(a,b)^2)`.

这里 **不把 `A_S` 宣称成 epistemic variance**。它只是一个由已查询 evidence contribution 得到的、可审计的异方差归一化尺度：当多个 evidence atom 对同一个翻转给出大幅但互相抵消/冲突的贡献时，`A_S` 会较大，使 intervention 更保守。

它不增加任何 evidence query，也不改变 B=16。

### 4.3 proposal-conditioned one-sided split calibration

在 group-disjoint validation calibration scenes 上，只看固定 EAF policy 真正提出的 `anchor -> challenger` intervention edge。

方向统一为“正值表示 challenger 比 anchor 更好”。对 calibration edge `j`：

`e_j = M_hat_j - M_teacher_j`.

主分支使用 evidence attribution normalization：

`s_j = e_j / max(A_j, A_floor)`.

取有限样本 one-sided split quantile：

`q = order_statistic_{ceil((n+1)(1-alpha))}(s_j)`，并强制 `q >= 0`。

因此 calibration **永远不能放松** legacy guard，只能增加保守半径。

runtime lower bound：

`M_LCB = M_hat - q * max(A_S, A_floor) - beta_old * sigma_old - epsilon_old`.

只有同时满足：

- `M_LCB >= flip_margin`；
- existing score-margin condition；
- existing evidence-certificate condition；
- EAF frontier active

才允许 challenger 替换 selected-local/DARM anchor。

### 4.4 为什么保留一个 constant-radius control

同时做一个完全相同的 split，只把：

`A_S(a,b) = 1`。

也就是普通 constant one-sided radius。

这不是另一个待调算法，而是 **novelty control**：

- 如果 constant 和 attribution 都改善，说明主要收益来自 generic post-hoc thresholding；不能把 attribution 当核心贡献。
- 如果 attribution 显著优于 constant，才能把“evidence attribution 参与 one-sided intervention calibration”作为论文机制贡献。
- 如果两者都失败，不调 alpha/threshold 榨性能，直接转 selective representation capacity。

---

## 5. 为什么这不是历史失败机制的重复

V64.3.14 明确不做：

- RET/CET/BTP-v2 或任何新 acquisition loss；
- proposal/HAB/family gate 解冻；
- B/M 扩大；
- broad pair field；
- Hodge/global action potential；
- generic evidence-action potential；
- generic selected-set potential；
- EAF same-embedding loss v2；
- 降低 evidence certificate gate；
- 把 alpha 当 performance hyperparameter sweep；
- 修改 pair-full/local-pair-full ceiling。

它唯一改变的是：**EAF complete-frontier challenger 什么时候有资格干预 frozen anchor**。

---

## 6. 下一轮 causal screen

### Phase A：runtime instrumentation replay

复用修复 checker 选中的 V64.3.13 checkpoint（上传结果对应 epoch 3 / 通常为 `epoch_0004.pt`），在 val cache 上跑 raw EAF replay。

必须验证：

- runtime EAF active >= 0.99；
- complete-star coverage >= 0.99；
- residual RMS > 0；
- attribution-scale RMS > 0；
- B=16/M=24 unchanged。

如果 runtime instrumentation 不成立，先修工程，不解释算法。

### Phase B：同一 raw replay 上 deterministic group split

默认：

- `VAL_SCENARIOS=500`；
- calibration fraction `0.40`；
- evaluation fraction `0.60`；
- `alpha=0.10`；
- scenario-token SHA256 group split；
- attribution/constant 两分支必须使用 byte-identical token lists。

### Phase C：两个 OCFI gate 在同一 held-out val groups 上比较

主要 preservation success 条件：

- harmful intervention absolute reduction >= `1pp`；
- beneficial intervention retention >= `50%`；
- deployed flip rate 必须下降；
- selected-local anchor、pair-full、local-pair-full、evidence certificate 不得漂移。

endpoint success 条件：

- teacher match >= `+0.5pp` 且 regret non-harm；或
- regret >= `2%` improvement 且 teacher match non-harm（容忍 `-0.4pp`）。

只有 attribution branch 同时满足 instrumentation + frozen interface + preservation gain + endpoint gain，才允许下一步 full-val calibration/reproduction。

**当前不要跑 test / closed-loop。**

---

## 7. 后续分支解释

1. **Attribution OCFI pass，且优于 constant control**
   - 当前最强路径；
   - full-val 重新 fit calibration；
   - 冻结后再一次性 test；
   - 最后 closed-loop；
   - 论文 novelty 可围绕 fixed-budget evidence-attributed decisive-frontier intervention 展开。

2. **Constant pass，attribution 不支持**
   - 说明 preservation calibration 是对的，但 attribution-specific mechanism 没有被数据支持；
   - 不应过度声明 attribution novelty；
   - 可以把 constant calibration 当工程/ablation，再寻找更强结构性 mechanism。

3. **两者都显著减少 harmful flip，但 endpoint 不改善**
   - preservation interface 已被清掉；
   - 当前 EAF value 虽然 pair-sign 有提升，但还不够 decisive；
   - 下一步才进入 small selective action/evidence representation capacity test。

4. **两者都不能减少 harmful flip**
   - calibration score/当前 EAF edge 不足以区分可信与不可信 intervention；
   - 不 sweep alpha 榨结果；
   - 直接转 representation capacity diagnosis。

所有分支 acquisition 继续 terminally frozen。

---

## 8. 本轮工程修改

1. 修复 `compute_bdse_diagnostics()` 对 `decisive_frontier_value_*`、`decisive_frontier_ocfi_*`、`decisive_anchor_margin_*` 的传播。
2. 修复 V64.3.13 screen checker：exact-scene-invalid epoch 不得因 noisy endpoint 被选中；training/runtime instrumentation 分离。
3. EAF runtime decomposition 显式导出 per-challenger attribution RSS scale，residual 数值保持不变。
4. 在 `pair_action_anchor_guard` 内增加 OCFI one-sided radius；`q=0` 时对 V64.3.13 是 exact no-op。
5. pair-full/local-pair-full 无 EAF 时 OCFI 自动 no-op，避免污染 ceiling。
6. `evaluate_open_loop.py` 增加 scenario-token filter，用于 calibration/evaluation group-disjoint replay。
7. 增加 proposal-conditioned teacher edge margin/over-estimation diagnostics。
8. 新增 calibration tool、contract checker、two-control screen checker。
9. V64.3.14 raw config 清除所有 stale training loss / trainable module：本版本严格 evaluation/calibration-only。
10. 新增 V64.3.14 unit tests，并给 V64.3.13 checker 增加 invalid-epoch selection regression test。

---

## 9. Novelty 边界

不要把“使用 conformal calibration”本身写成 novelty。当前相关研究已经在探索 conformal prediction 与 downstream risk-averse/action-conditional decisions 的结合。

如果 V64.3.14 得到正结果，论文可主张的更强、也更贴合 BDSE 的机制对象应是：

**under a fixed planner-interface evidence budget, use the same auditable selected-evidence decomposition that constructs the complete decisive frontier to heteroscedastically calibrate whether a frontier challenger is allowed to intervene on a one-sided preserved planning decision.**

也就是 novelty 放在：

`budgeted auditable evidence`
`-> evidence-attributed decisive-margin value`
`-> evidence-attributed one-sided intervention calibration`

这一整条 planner interface，而不是泛化的 conformal wrapper。

---

## 10. 当前状态

- 新算法与工具链已实现；
- 当前 sandbox 没有用户 `/data0/...` cache mount；
- 上传的 compact V64.3.13 outputs archive 没有 `.pt` checkpoint；
- 因此本地不能伪造 V64.3.14 数值结果；下一步 GPU screen 命令已经写成严格 stop protocol，要求服务器上的原始 V64.3.13 checkpoint。

---

## 9. 最终工程回归

在最终 attribution arithmetic / novelty-control 收紧后重新执行完整回归：

- V64.3.7--V64.3.14 targeted regression：**53/53 PASS**；
- full repository：**346/346 PASS，36 warnings**；
- `python -m compileall -q bdse`：**PASS**；
- V64.3.14 raw config contract：**15/15 PASS**；
- 根目录 launcher/bash syntax：**PASS**；
- 新增测试明确验证：OCFI 关闭时 EAF deployed residual 保持 V64.3.13 原始 reduction arithmetic；逐 atom attribution 只作为 calibration side information，不偷偷改变已有 value head；
- constant-radius 与 attribution-scaled 的 parity 不会被 checker 误判成 attribution-specific novelty。

36 个 warning 全部来自仓库既有 PyTorch Transformer nested-tensor warning，没有新增 warning 类型或工程失败。
