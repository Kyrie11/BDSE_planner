# V59 FSCIP-BFAR-DBAP 结果审计与 V60 DWAPC 优化报告

## 1. 最终结论

V59 原始 gate 结果为：

| Gate | 状态 | 结论 |
|---|---|---|
| Protocol gate | PASS | 训练、独立 calibration、三路 paired open-loop 的旧协议完整 |
| Minimum gate | PASS | selector recall、evidence certificate、fallback 均满足最低阈值 |
| Competitive gate | FAIL | candidate 未优于 local/foundation，pair-full 与 deployed residual gain 均为 0 |

Competitive FAIL 不是阈值边缘问题。candidate、local、foundation 的 teacher action match 均为 0.141，pair-full candidate/local 也均为 0.141，1000 个场景没有任何 deployed residual flip。

V59 最重要的新发现不是 residual 仍为零，而是上游 evidence proposal 存在更大的结构性瓶颈：

- dense full-interface teacher match：0.359；
- proposal Top-M 后 sparse-full match：0.141；
- B=16 对 sparse-full winner 的保持率：0.981；
- B=16 对 dense-full winner 的保持率：0.172；
- evidence sufficiency：0.074。

因此，当前主要 winner 信息损失发生在 `dense evidence -> Top-M proposal`，不是 `Top-M -> B=16 exact selector`。V59 的 selector 在已进入 proposal pool 的 evidence 上已经较有效，但 proposal pool 本身没有保留 dense winner 所需的 evidence。

---

## 2. 工程错误审计

### 2.1 Pair-full 诊断遗漏 V59 set-conditioned residual

`evaluate_open_loop.py` 和训练 validation 的 pair-full tournament 只传入了 additive `residual_action_potential`，没有传入：

- `residual_set_atom_factors`；
- `residual_set_action_factors`。

这造成两个影响：

1. gate 中 `pair_full residual gain=0` 没有完整评价 V59 新增的 set-conditioned head；
2. competitive checkpoint score 也没有依据 set-conditioned head 的真实 pair-full winner 选择 best checkpoint。

V60 已修复两个路径，并增加 set-conditioned residual 的运行时诊断输出。

### 2.2 Set-conditioned head 缺少可观测性

V59 的 query diagnostics 没有传递 `set_conditioned_residual_*` 字段，因此日志无法回答：

- set head 是否激活；
- set potential 平均幅度；
- set rank 与 scale 是否正确；
- pair-full 与 deployed 路径是否使用同一组 factors。

V60 已补齐这些指标。

### 2.3 Residual calibration 与部署决策规则过度错位

V59 对每个 calibration scene 取“所有 valid rivals 的最坏 nonconformity”，得到 residual epsilon = 3.3076；训练阶段 reserve 只有 0.15。该全 rival 上界虽然保守，但部署只会采用冻结模型确定的 top rival。用最坏的任意 rival 校准确定性 top-rival policy，会产生极大的不必要保守性。

V59 原始 residual proposal：

- 96/1000 场景提出 raw flip；
- beneficial 2；
- harmful 5；
- neutral 89；
- raw margin mean 0.00141；
- raw margin max 0.012998；
- sigma mean 0.0795；
- 最大 robust margin 仍为 -3.3639。

因此 V59 的零 deployed flip 同时来自两方面：

1. 模型修正幅度本身过小且净效果偏负；
2. all-rival epsilon 又进一步把所有 proposal 全部拒绝。

V60 改为每个 calibration scene 对冻结 policy 选择的 top rival 进行 split-conformal calibration，同时保留 all-rival max epsilon 作为诊断，不再作为部署阈值。

### 2.4 Learned uncertainty 未覆盖 set-conditioned head

V59 的 `residual_action_var` 只对应 additive per-evidence residual，set-conditioned potential 没有独立 uncertainty head。使用 `beta * sigma + epsilon` 时，epsilon 被迫吸收 set head 的全部误差。

V60 默认采用 conformal-only residual certificate：

- `residual_beta_uncertainty = 0`；
- policy-aligned split-conformal epsilon 作为正式 residual error bound；
- learned variance保留为诊断和未来异方差扩展依据。

### 2.5 External baseline 结果不完整

上传的 `external_baselines.zip` 只有四个训练日志，没有：

- best checkpoint；
- 1000-scene open-loop JSON/JSONL；
- 20-scene closed-loop metrics。

因此当前只能分析训练收敛，不能声称 BDSE 优于或劣于外部 baseline。

此外，代码中的 external baselines 是 budget-compatible adapters/reimplementations，共享 BDSE candidate bank、runtime inputs 和 evidence-budget accounting，不应在论文中表述为官方原作者实现。

---

## 3. V59 的正向信号

### 3.1 Evidence selector 路径继续改善

| 指标 | V59 |
|---|---:|
| Proposal decisive recall | 0.80274 |
| Selected decisive recall | 0.61074 |
| Effective decisive recall | 0.77640 |
| Interaction decisive recall | 0.57955 |
| Evidence certificate | 0.88803 |
| Fallback | 0.110 |

这些指标说明 boundary curriculum、proposal supervision、exact AOCC 和 B=16 selector 值得保留。

### 3.2 B=16 selector 对 sparse proposal winner 的保真较高

`budget_vs_sparse_full_match=0.981`，说明一旦关键 evidence 已进入 Top-M pool，B=16 exact selector 基本能够保持 sparse-full winner。继续只强化 B=16 selector 的边际收益有限。

### 3.3 Residual 并非严格零函数

V59 在 96 个场景提出 raw action change，说明 additive/set-conditioned heads 能够产生非零分数扰动。训练中的 winner losses 也在下降：

- pair-full winner margin：10.341 -> 9.008；
- residual winner correction：8.379 -> 5.827；
- residual uncertainty：2.538 -> 2.384。

但这只证明梯度链存在，不证明 winner correction 已有效。raw proposal 中 harmful 多于 beneficial，最终应判断为“有非零扰动能力，但尚未学会净有益 winner correction”。

---

## 4. 有效、需要深化和无效的设计

### 应保留

- immutable foundation anchor；
- fixed B planner-interface budget；
- boundary/hard/near-tie pair curriculum；
- sparse periodic exact AOCC；
- direct integrable action potential；
- evidence certificate 与 residual certificate 分离；
- same-checkpoint local control；
- paired foundation control；
- group-disjoint tune/calibration；
- paired scenario/timestamp hash；
-并发 open-loop。

### 值得继续深化

- proposal decisive-evidence supervision，但目标必须从 atom recall 提升为 dense-winner preservation；
- set-conditioned potential，但必须加入正确的 pair-full evaluation、checkpoint selection 和更好的初始化；
- certified residual winner loss，但应在 proposal bottleneck改善后再评估；
- policy-aligned conformal certificate。

### 当前无效或不足

- 仅依赖 BCE/listwise atom gain 的 proposal training；
- 纯 action-potential reconstruction；
- atomwise residual reconstruction；
- all-rival max residual calibration 作为实际部署阈值；
- 用 selector recall 较高来替代最终 winner gain；
- 继续降低 certificate 阈值以人为制造 flip。

---

## 5. V60 算法：Dense-Winner-Aligned Policy-Calibrated Set-Potential

### 5.1 Dense-winner-preserving proposal loss

V60 新增 `L_proposal_dense_winner`。训练时：

1. 用所有 active local evidence 计算 dense-local winner；
2. 对 proposal logits 执行 hard-forward straight-through Top-M；
3. 要求 Top-M sparse cost 保持 dense-local winner及其 strongest-rival margin；
4. 对 dense winner 同时等于 teacher winner 的场景提高权重；
5. 对 `g` detach，使该损失只训练 proposal ranking，不允许 local cost head移动目标。

部署仍使用离散 Top-M 和固定 B，没有放宽 evidence budget。

### 5.2 Proposal-first residual curriculum

前两个 epoch residual family 只使用 0.1 scale，之后四个 epoch 逐步提升到 1.0。目的：

- 先修复 dense -> Top-M winner 丢失；
- 再让 residual 在更正确的 sparse anchor 上学习；
- 避免 residual 被迫补偿 proposal 丢失的大量关键 evidence。

### 5.3 Set head 非退化安全初始化

V59 将 set atom factor 最后一层严格置零，初始阶段只有一侧双线性因子能获得有效梯度。V60 使用：

- atom factor std = 0.005；
- action factor std = 0.01；
- bias = 0。

初始 set potential 仍很小，但两侧 factor 从第一步均可学习。

### 5.4 Policy-selected-top-rival conformal calibration

每个 calibration scene：

1. 冻结 selected-local anchor；
2. 用冻结模型输出选择 corrected-cost 最低的 valid rival；
3. 对该 policy-selected rival 构造单侧 nonconformity；
4. 所有 calibration scenes 都贡献一个 score；
5. all-rival maximum 只作为诊断。

### 5.5 修复 pair-full 和 checkpoint selection

- pair-full evaluation 传入 set factors；
- validation 与 open-loop 采用同一 set-potential路径；
- checkpoint score加入 sparse-full、budget-vs-dense-full 和 dense-proposal drop；
- gate 检查 `L_proposal_dense_winner` 是否真正执行。

---

## 6. External baseline 训练日志分析

| Adapter | Best epoch | Best val action CE | Best val loss | Last val action CE |
|---|---:|---:|---:|---:|
| PlanTF | 24 | 2.2756 | 138.4657 | 2.3941 |
| DTPP | 29 | 2.3789 | 208.2714 | 2.4194 |
| GameFormer | 18 | 2.7369 | 141.9993 | 2.7868 |
| PLUTO | 11 | 2.3398 | 211.0580 | 4.9537 |

解释：

- PlanTF 在这组 adapter 训练中具有最低 val action CE；
- DTPP 到后期仍接近最佳状态；
- GameFormer 在 epoch 18 后轻度退化；
- PLUTO 在 epoch 11 后明显失稳/过拟合，应使用 best checkpoint，不能使用 final checkpoint；
- 不同 adapter 的 cost/pair loss 权重不同，总 val loss 不可直接横向排名。

正式优劣必须使用同一 1000-scene open-loop 和相同 20/100-scene paired closed-loop。

---

## 7. Budget 对比协议

V60 提供 `RUN_V60_BUDGET_BASELINE_SWEEP.sh`：

- budgets：8/16/24/32；
- 同一 val_tune 前 1000 场景；
- 校验 scenario/timestamp SHA-256；
- fallback 全部关闭；
- 相同 strict evidence budget；
- 输出 CSV、JSON、Markdown。

快速导师展示表中 BDSE 使用 selected-local/no-residual 路径。这是因为 B=8/24/32 不能复用 B=16 的 residual conformal certificate。论文级 candidate curve 应为每个 budget 独立冻结 calibration epsilon 后再测试。

重点展示指标：

- teacher action match；
- teacher regret median/p90/CVaR90；
- effective query atom count；
- p95 latency；
- selected/interaction decisive recall；
- fallback（strict curve应为0）；
- closed-loop score、collision、TTC、drivable、progress、comfort。

---

## 8. 下一步判据

V60 smoke test 必须满足：

- `L_proposal_dense_winner > 0`；
- `proposal_dense_topm_match` 被记录；
- exact selector fraction > 0；
- set factors 正确传入 pair-full；
- residual curriculum scale 正确变化；
- policy calibration覆盖率 >= 80%。

完整训练后优先判断：

1. sparse-full teacher match 是否明显高于 0.141；
2. budget-vs-dense-full match 是否从 0.172 上升；
3. B16 是否继续保持 sparse-full；
4. 修复后的 set-aware pair-full gain 是否转正；
5. policy-aligned epsilon 是否显著低于 3.3076；
6. raw beneficial 是否超过 harmful；
7. deployed candidate-local teacher gain 是否转正。

只有 pair-full gain和 deployed gain转正后，才值得运行大规模 closed-loop；否则应继续处理 proposal或 residual expressivity，而不是放宽 gate。
