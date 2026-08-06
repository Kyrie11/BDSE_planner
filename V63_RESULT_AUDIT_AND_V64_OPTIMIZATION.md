# V63 实验审计与 V64 SAQA-BCC 优化报告

## 0. 结论摘要

这次上传的结果不能证明 V63 算法已经完成训练后仍然失败。两条 V63 主路径都在正式训练、calibration、三路 paired open-loop 和 gate checker 之前停止：

- `runtime_recompute` 路径在 immutable foundation anchor replay 失败后终止；
- cached-query 路径在 query-cache audit 中把历史 12 维缓存错误地要求为当前 18 维后终止。

因此 V63 的真实 gate 状态是：

| Gate | 上传结果支持的状态 | 解释 |
|---|---|---|
| Protocol | **NOT EVALUATED** | 没有 V63 candidate/local/foundation 三路正式输出，也没有 V63 gate report |
| Minimum | **NOT OFFICIALLY EVALUATED** | 只有旧 checkpoint 审计中的 proposal/selector 代理指标 |
| Competitive | **NOT OFFICIALLY EVALUATED** | 只有同 checkpoint 诊断和强烈失败预警，不能当作训练后正式结果 |

清除工程误判后，当前最可信的算法结论是：

1. B=16 selector 对已经形成的 sparse interface 仍较忠实，不是第一瓶颈；
2. HAB proposal 对“广义 decisive atoms”已有较强覆盖，但对真正 teacher winner-flip critical atoms 覆盖不足；
3. 旧 runtime base prior 在同 checkpoint 对比中明显改变并恶化 winner；
4. residual 没有产生任何 raw action flip，现阶段不能靠放松 calibration 或放大 residual 解决；
5. V63 新增的 6 个 query channels 被直接送入旧 checkpoint 的未训练投影列，导致 foundation anchor 从 0.359 跌到 0.170，是最主要工程混杂。

据此落地 V64：**Support-Aware Query Adapter + Budgeted Critical Coverage BFAR-DBAP（SAQA-BCC）**。

V64 保持论文核心 novelty：固定 planner-interface evidence budget、可审计 evidence atoms、预算内确定性 selector、以及 leave-one-out winner-action flip criticality。新增部分服务于这四个核心定义，而不是绕过预算接口。

---

## 1. 上传结果的实际执行状态

### 1.1 Runtime-recompute 路径

V63 日志显示加载 V53 checkpoint 后，anchor replay 使用 1000 个相同场景完成，但 gate 失败：

```text
V62 anchor full-interface action match = 0.359
V63 recompute anchor action match       = 0.170
V62 dense winner-rival sign accuracy    = 0.802922
V63 dense winner-rival sign accuracy    = 0.655778
V62 anchor teacher regret               = 7024.30
V63 anchor teacher regret               = 30633.83
```

场景 row-key SHA-256 完全相同，因此下降不能归因于换了 validation scenes。流水线随后停止，没有进入 V63 训练。

### 1.2 Cached-query 路径

cache audit 抽样 512 个场景：

```text
sampled_scenarios  = 512
compared_scenarios = 0
shape_failure_count = 512
cached last dimension = 12
required last dimension = 18
```

这不是“缓存数值已经不一致”的证据，因为 audit 根本没有比较任何一对数值。它只是把历史缓存的 12 维接口错误地当成必须完整覆盖 V63 的 18 维接口。cached pipeline 因此也在训练前停止。

### 1.3 Gate 报告缺失的正确解释

上传目录没有完整 calibration、candidate/local/foundation 三路 V63 open-loop，也没有 V63 gate report。因此缺少 report 的根因不是“最后一步忘记执行”，而是两个上游 preflight gate 都提前终止。

不能手工补写旧 summary，也不能把 V62/V53 anchor replay 重新命名为 V63 candidate result。

---

## 2. 工程错误与指标误判

### 2.1 18-D query interface 激活了 checkpoint 未训练的 6 个通道

历史 cache 提供 12-D query feature。V62 配置虽然声明 18-D projection，但数据输入的第 12:18 通道实际一直为零。因此旧 checkpoint 的 `query_proj` 最后 6 列没有得到有效训练支持。

V63 `runtime_recompute` 直接生成非零 18-D feature 并送入旧 `query_proj`，等价于突然激活一组未被数据训练约束的旧权重列。即使 checkpoint shape 完全兼容，也不代表语义兼容。这正好解释同一 checkpoint、同一场景下 anchor 从 0.359 降到 0.170。

### 2.2 旧 checkpoint loader 会静默跳过核心 shape mismatch

V63 的部分 inference/calibration loader 只加载 shape-compatible tensors，其余缺失或 shape mismatch 后继续执行。该策略适合新增可选 residual heads，但不适合 foundation、query projection、scene/action encoder 等核心接口。核心层被静默随机初始化时，结果会伪装成算法失败。

V64 改为 strict core-state contract：仅显式允许新增 adapter/residual heads 缺失，任何核心 tensor 缺失或 shape mismatch 直接报错。

### 2.3 V63 query contract gate 比较错了对象

V63 nominal 的四个指标实际是：

```text
dense_runtime_base_contract_pass             = 1.0
dense_runtime_base_value_allclose_fraction   = 1.0
dense_runtime_query_contract_pass            = 0.0
query neural-score allclose fraction @1e-5   = 0.144588
query neural-score MAE                        = 1.59339e-4
query neural-score max abs                    = 7.92944e-4
HAB Top-M dense winner vs sparse-full winner = 1.0
```

因此 base 两项已经通过；失败的是旧 query gate。旧 gate 比较的是不同 CUDA batch shape 下的神经网络 `g(i,a)` 输出，并要求近似逐位 `1e-5` max-error，而不是比较原始 query feature contract。其最大误差不到 `8e-4`，同时 winner decision 一致率为 1.0。把这种数值核差异判为 planner-interface 语义失败，是工程性假失败。

V64 将其拆成三层：

1. **raw feature contract**：同一 queried atom/action 的有效 query features 是否一致；
2. **score numerical contract**：`atol + rtol` 下神经 contribution 是否一致；
3. **decision contract**：winner action 是否一致。

三者分别报告，不再用一个严格 max-error 指标代替所有含义。

### 2.4 旧 base/sparse 归因混入额外 runtime prior

V63 nominal 的 dense/model base 与 deployment base 不同：后者叠加 opaque base prior 和 structural residual prior。同 checkpoint ablation：

```text
nominal teacher action match           = 0.141
no_base_prior teacher action match     = 0.214
no_runtime_priors teacher action match = 0.212
no_structural_prior teacher match      = 0.139
base_prior_replaced_best rate          = 0.991
```

绝对数值受到 query support drift 污染，不能作为最终方法质量；但同 checkpoint、同场景的方向非常明确：base prior 几乎总是在替换 learned foundation winner，并且 removal 后 teacher match 大幅上升。structural prior 对最终 teacher match 几乎没有正贡献。

V64 nominal 因此关闭这两个连续 runtime priors。结构安全信息仍保留为 auditable evidence atoms、hard feasibility/filter 和独立 ablation，不通过未计入 evidence budget 的连续 cost 偷渡到 winner。

### 2.5 配置 provenance 残留

最初 V64 工作配置继承了 V63 的 `experiment.name/algorithm`。虽然不改变数学前向，但会污染产物命名、provenance 和后续论文归档。最终交付已将 8 份 V64 配置统一改为 V64 SAQA-BCC 元数据。

---

## 3. 去除工程混杂后的算法状态

### 3.1 值得保留：固定 B=16 的 planner-interface certificate

V62 有效诊断：

```text
budget-vs-sparse-full winner preservation ≈ 0.981
```

V63 nominal 在损坏的 support contract 下仍为 0.981；no-prior 变体也显示 HAB Top-M dense 与 runtime sparse-full winner match 约 0.992–1.0。虽然绝对 teacher quality 不可信，但这些同接口保持率共同说明：给定 sparse values 后，B=16 selector 不是当前首要损失来源。

因此不能通过增加 B 获得短期指标。增加预算既会掩盖 proposal/query 缺陷，也会削弱论文固定接口预算的 novelty。

### 3.2 值得保留并升级：HAB family-aware proposal

有效正向信号：

```text
V62 dense-full teacher match       = 0.359
V62 HAB Top-M dense teacher match  = 0.358
proposal decisive recall           = 0.803583
selected decisive recall           = 0.611728
```

HAB Top-M 几乎完整保留旧 dense winner，且 broad decisive recall 已达到较高水平。这说明 family-aware proposal 不是完全无效，不能退回 global Top-M 或无结构 logit ranking。

但 teacher exact criticality 显示：

```text
teacher-critical scene rate       = 0.465711
teacher-critical atom fraction    = 0.015191
critical Top-M recall             = 0.354762
critical selected recall          = 0.337619
```

约 46.6% 场景至少包含一个会翻转 teacher winner 的 atom，但 Top-M 只捕获约 35.5%。selected recall 与 Top-M recall 很接近，表明主要 critical loss 发生在 proposal/HAB acquisition，而不是 B16 selector。

因此 HAB 应被**升级为 critical-coverage-aware acquisition**，而不是废弃。

### 3.3 需要修改：critical loss 只做逐 atom 分类/排序，未直接优化预算覆盖

V63 teacher exact label 方向是正确的：critical 的定义严格保持“移除 atom 后 winner action 翻转”。但独立 BCE/ranking 并不直接优化固定 Top-M 接口是否覆盖 critical utility。稀有 positive atoms 即使 logit 上升，也可能仍被 family quota、其他高 logit atoms或场景内密集 negatives 挤出 M=24。

V64 增加 **Budgeted Critical Coverage (BCC)**：

- hard forward 仍执行原有 deterministic HAB Top-M；
- backward 使用 straight-through soft mask；
- target distribution 只由 teacher exact winner-flip critical atoms 的 utility/severity 构成；
- loss 直接最小化未被固定 Top-M interface 覆盖的 critical utility；
- 部署时不使用 teacher、不改变 M、不改变 B、不改变 selector。

这是对核心 idea 的深化：从“识别 critical atom”推进到“在固定、可审计接口预算前提下最大化 critical evidence coverage”。

### 3.4 需要修改：query feature 版本演进必须显式建模 support

V63 的新 6 个 query channels 可能有价值，但不能直接穿过旧 projection。V64 引入 **Support-Aware Query Adapter (SAQA)**：

```text
legacy supported prefix: q[0:12]
new extension:           q[12:18]
legacy path:             query_proj(pad(q[0:12], zeros=6))
new path:                zero-init low-rank adapter(q[12:18])
combined embedding:      legacy path + residual adapter
```

step zero 与旧 checkpoint 完全等价；训练后新特征只能通过显式 residual adapter 改变结果。该设计提供：

- checkpoint 语义兼容；
- 新 feature 的独立可审计增益；
- 可做 adapter-on/off 同 checkpoint ablation；
- 不必重建全部历史 cache。

### 3.5 需要修改：opaque runtime priors 不应作为 nominal winner path

固定 evidence budget 的论文主张要求影响 planner winner 的信息能够进入可审计接口或明确的 hard feasibility contract。一个不计入 evidence budget、且替换 99.1% learned winners 的连续 base prior 会带来两类问题：

- 算法层面：掩盖 learned evidence contribution；
- 论文层面：审稿人可质疑真实决策是否由固定 evidence certificate 驱动。

V64 nominal 去除 base/structural continuous prior；structural prior 只保留为 ablation。安全硬约束、候选有效性、evidence atoms 和 certificate audit 不删除。

### 3.6 需要保持保守：residual

当前上传诊断：

```text
residual_flip_proposed                = 0
beneficial residual intervention rate = 0
harmful residual intervention rate    = 0
```

没有 raw proposal 说明 residual 还未学会正确动作方向。此时放宽 conformal epsilon、降低 flip margin 或扩大 scale 只会制造不可验证 flips。V64 保留 residual curriculum 和保守 calibration。只有 raw beneficial proposals 出现，且 beneficial > harmful，才进入 calibration/aggressiveness 调优。

### 3.7 闭环 SOTA 的独立上限：candidate bank

此前数据诊断显示部分场景没有安全候选，candidate hard-violation rate 明显高于 teacher。Evidence selector 无法选择不存在的轨迹。V64 当前先修复 planner-interface 和 critical coverage，因为这是上传结果能明确定位的瓶颈；在 V64 open-loop/CL20 证明 evidence path 有效后，应独立升级 candidate generation：route-conditioned maneuver diversity、interaction-conditioned yield/creep、longitudinal stop envelope 和 safe repair。该工作必须作为 matched candidate-bank ablation，避免把候选库改善误记为 selector novelty。

---

## 4. V64 落地内容

### 4.1 Support-Aware Query Adapter

修改 `bdse/model/bdse_model.py`：

- `query_legacy_support_dim=12`；
- 旧 `query_proj` 始终只接收 12-D supported prefix，尾 6 维补零；
- 新 6-D extension 使用 `LayerNorm -> Linear(6,32) -> SiLU -> Linear(32,256)`；
- 最后一层零初始化，step-zero 与旧 checkpoint 相同；
- adapter 单独列入 trainable modules，学习率 multiplier=3；
- dense/sparse 输出都报告 support dimension、extension dimension、adapter enabled。

adapter 仅约 8,684 参数，相对于主模型开销极小。

### 4.2 Prefix cache + online extension

修改 tensorizer 和 cache audit：

- 历史 cache 仅负责 checkpoint-supported 12-D prefix；
- 新 6-D extension 在线计算；
- audit 比较 prefix 数值与 fingerprint，不再要求旧 cache 具备 18-D；
- audit PASS 使用 prefix-cache fast path；
- audit FAIL 自动回退 `runtime_recompute`，不再阻断整个 pipeline；
- full 18-D `cache_verified` 模式仍要求完整维度，避免语义混淆。

### 4.3 三层 query contract

open-loop 新增：

- `dense_runtime_raw_query_feature_contract_available/pass`；
- raw feature MAE/max/allclose fraction；
- `dense_runtime_query_score_contract_pass`；
- score MAE/max/allclose fraction，使用 `atol=2e-3, rtol=1e-4`；
- `dense_runtime_query_decision_match`。

Protocol 必须同时通过 raw、score、decision 三层，不能用 score tolerance 替代 raw feature equality。

### 4.4 Strict checkpoint core-state contract

新增 `bdse/model/checkpoint_contract.py` 并接入 inference/calibration：

- core tensor missing/shape mismatch 直接失败；
- 只允许已声明的新 adapter/residual heads 缺失；
- 输出加载审计信息；
- 不再静默运行 partially initialized planner foundation。

### 4.5 Budgeted Critical Coverage

修改 `bdse/model/losses.py`：

- teacher exact winner-flip 标签定义不变；
- 利用已有 straight-through HAB mask；
- hard forward 与部署 deterministic Top-M 完全相同；
- coverage loss 权重初始为 2.0；
- 与 BCE、severity 和 hardest-negative ranking 联合训练；
- 不新增 teacher forward，不增加部署计算。

### 4.6 Runtime prior 归位

V64 nominal：

```text
runtime.base_prior.enabled = false
runtime.structural_safety_residual.enabled = false
```

另提供 structural-prior ablation。安全 hard filter、feasibility、evidence atoms 不变。

### 4.7 V64 Gate

Protocol 新增硬条件：

- support dim=12，extension dim=6；
- adapter enabled；
- base contract；
- raw feature contract；
- score contract；
- winner decision contract；
- retained B=16 budget/provenance。

Minimum/Competitive 继续分开报告 metrics pass 与 official pass，避免 Protocol 串联阻断被误读为算法指标失败。

---

## 5. 下一步实验及判因逻辑

### 5.1 必做 step-0 support audit

使用同一 V62 checkpoint 比较：

1. legacy anchor；
2. V64 support-aware runtime recompute；
3. V64 prefix-cache；
4. structural-prior ablation。

训练前必须确认：

- legacy anchor 恢复到约 0.359，而不是 0.170；
- adapter step-zero 与 legacy dense/deployed winner 一致；
- raw/score/decision contracts 通过；
- prefix-cache 与 runtime path 一致。

失败时只修工程，不调 proposal、selector 或 residual。

### 5.2 训练后的首要判据

按因果顺序读取：

1. critical Top-M recall 是否超过 0.355；
2. selected critical recall 是否超过 0.338；
3. broad proposal decisive recall 不得明显低于 0.804；
4. selected decisive recall 不得明显低于 0.612；
5. Top-M dense vs runtime sparse winner match >=0.95；
6. candidate teacher match 必须脱离 0.141 平台；
7. raw residual proposals 是否出现，并且 beneficial > harmful。

判因规则：

- critical Top-M 不升且 broad recall 下降：coverage loss 过强或 teacher utility 不稳，先降低 coverage weight；
- Top-M critical 升、selected 不升：再升级 B16 selector 的 critical-aware tie-break/coverage，不增加 B；
- selected critical 升、teacher action 不升：问题位于 atom-action value、candidate bank 或 teacher scalar alignment，不继续堆 proposal loss；
- open-loop action match 升、CL 无改善：检查 candidate dynamics、reactive interaction、replan caching 和 simulator-specific safety；
- raw residual proposal仍为0：修 residual target/routing，不放宽 calibration。

### 5.3 闭环顺序

- Protocol PASS 后才做 paired CL20；
- Minimum/Competitive 未过时的 CL20只能标为 diagnostic；
- CL20 无 safety regression、且 beneficial flips > harmful 后做 CL100；
- incomplete test 仍不能用于调参；完成 readiness audit 后冻结 config/checkpoint/calibration，只运行一次 final test。

### 5.4 公平论文结果

V62 continuation 用于快速验证 V64 方向。最终论文表格必须从同一 immutable V53 foundation、相同训练预算重新训练，不能把 continuation 的计算优势写成公平 SOTA 对比。

---

## 6. 代码与效率验证

本地验证：

```text
Python compile                 PASS
V64 YAML parse                 8/8 PASS
V64 shell syntax               4/4 PASS
pytest                         250 passed, 0 failed
warnings                       16（PyTorch Transformer nested-tensor warning）
```

效率设计：

- adapter 约 8.7K 参数；
- prefix-cache 只在线计算 6 个新增 channels；
- BCC 重用已有 teacher criticality 和 ST-HAB mask，不新增部署计算；
- strict checkpoint loading 开销仅发生在加载阶段；
- 三层 contract 仅在评估中增加小规模审计数组；
- 不通过减小 B、减少 valid actions 或删除 safety candidates 换取速度。

当前环境没有 nuPlan cache、真实 checkpoint/GPU 和闭环 simulator，因此未运行 fresh V64 training、calibration、open-loop 或 closed-loop，不宣称 V64 已通过 gate 或达到 SOTA。

---

## 7. 不重复尝试清单

1. 不再把历史 12-D cache 错误要求为当前完整 18-D 才能使用；
2. 不再把新 query channels 直接送入旧 checkpoint 未训练的 projection columns；
3. 不再以 shape-compatible 为由静默跳过核心 checkpoint tensors；
4. 不再用跨 CUDA kernel 的 `1e-5` neural score max-error 代替 raw query contract；
5. 不再把未完成 pipeline 的缺失 gate report解释为正式 gate FAIL；
6. 不再把 V53/V62 anchor replay冒充 V63/V64 candidate result；
7. 不再用 opaque continuous base prior作为 nominal winner path；
8. 不在 raw beneficial residual proposal 为零时放宽 calibration；
9. 不通过增加 evidence budget B 解决 proposal/query defect；
10. 不使用 incomplete test 选版本、调权重或选 checkpoint。
