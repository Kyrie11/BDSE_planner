# V64.3.12 RET/CET 因果审计与 V64.3.13 EAF-DMVR 设计

## 0. 结论先行

V64.3.12 的结果足以把 acquisition 分支正式结束。

上一轮关于 V64.3.11 BTP-BDMU 的判断——**训练阶段 fast B-selector surrogate 与 promotion 阶段 exact runtime B=16 selector 语义不一致**——是正确的，而且 RET 的 exact-runtime instrumentation 证明这个工程/监督问题确实存在。但 RET 也同时证明：**修复该 mismatch 并不能转化为 C2-B gain**。因此它是必要的语义修复，不是性能根因。

CET 进一步放开了 exact-oracle-controlled current-B exchange，而且 controlled exchange 在训练和验证中都实际激活；C2-B 不但没有改善，反而下降。因此“blanket current-B protection 阻止真实 B-set 改善”也不是下一轮应该继续追的方向。

当前最直接的瓶颈已经迁移为：

> **exact selected B=16 evidence → decisive pair value → complete runtime decisive frontier → one-sided final decision preservation**

而不是：

> proposal score → Top-M → B=16 acquisition。

V64.3.13 因而设计为 **EAF-DMVR (Evidence-Attributed Frontier Decisive-Margin Value Residual)**。它冻结 foundation、proposal/HAB、Top-M M=24、exact runtime B=16 selector、DARM 和 DBR，只训练一个新的 selected-evidence-conditioned decisive value head；它不改变 evidence budget、不增加 evidence query、不创建新的 acquisition loss。

---

## 1. 上轮 BTP bottleneck 判断是否正确

### 1.1 正确的部分

V64.3.11 的 BTP 设计声称 training positive 是“能够穿过真实 B=16 planner interface 的 atom”，但代码审计发现训练实际使用 `_fast_pair_margin_surrogate_masks(...)`，validation/promotion 才使用 exact runtime pair-conditioned selector。

V64.3.11 screen 中 surrogate↔exact B-set Jaccard 约为：

- current: 0.7749；
- oracle: 0.7688。

这意味着约四分之一的 B-set membership 语义并不一致。V64.3.12 RET 把 training target 改成 sampled stop-gradient **exact runtime B=16 selector**，因此把这一因果变量真正控制住了。

所以“BTP train/runtime mediator mismatch 是真实问题”这一判断成立。

### 1.2 不正确、或更准确地说不完整的部分

如果 mismatch 是主要性能瓶颈，RET 在 exact training target 下应该至少显著推动 C2-B。实际并没有：

| RET 指标 | Anchor | Selected epoch 1 | Delta |
|---|---:|---:|---:|
| C1-B oracle capture | 0.415105 | 0.415105 | 0 |
| C2-B learned capture | 0.380525 | 0.380531 | **+0.000006** |
| C2-B gain | — | — | **+0.000606pp** |
| C2-M learned capture | 0.478576 | 0.480972 | **+0.2396pp** |
| C1-B/C2-B gap closure | — | — | **0.0175%** |
| teacher match | 17.8% | 18.2% | +0.4pp |
| teacher regret | 20133.34 | 20314.42 | **+181.08，变差** |
| pair-full match | 17.4% | 18.0% | +0.6pp |

RET 的 exact training instrumentation 是有效的：selected epoch 的训练 exact projection fraction 约 0.245，actionable exact-candidate fraction 约 0.398；validation exact projection 为 1.0，`B subset Top-M` violation 为 0。

因此 RET 给出的因果结论是：

> **train/runtime B-selector mismatch 的确需要修，但修完以后 C2-B 仍不动。它不是主要性能瓶颈。**

---

## 2. RET 是否学到了预期内容

要区分“loss 没运行”与“机制运行但没有价值”。RET 属于后者。

RET 学到了我们预期的**监督语义**：

1. training B target 确实来自 exact runtime selector；
2. 非 exact-sampled rows 不会偷用 surrogate target；
3. exact B-set 始终嵌套于 injected Top-M；
4. proposal residual 参数确实变化；
5. C2-M 仍可朝 oracle Top-M 方向移动。

但它没有学到我们真正需要的**决策传递内容**：

> exact target learning 并没有增加 exact runtime B=16 captured decisive utility。

也就是说，模型能对“哪些 proposal score 应变化”产生梯度，但这种 score 改变仍不能稳定地转化为 B-layer 决策价值。

因此 RET 的定位应是：

- **有效的语义/工程修复**；
- **有效的因果诊断实验**；
- **无效的性能优化方向**。

后续不应该 RET-v2、增加 exact sampled scene 数量、改 rank margin 或多跑 epochs 来继续榨 acquisition。

---

## 3. CET 是否有效

CET 的假设是：RET 失败可能因为 blanket current-B protection 把真正应该被 oracle 替换的当前 B evidence 全部保护住。

CET 只允许满足 exact oracle intervention 条件的 current-B evidence 被替换。这个机制实际被触发：

- controlled-exchange pair fraction 在 selected validation 中约 0.682；
- training controlled-exchange pair fraction 也明显非零；
- protected-negative fraction 被按设计释放，而不是实现没生效。

但结果是负的：

| CET 指标 | Anchor | Selected epoch 3 | Delta |
|---|---:|---:|---:|
| C2-B learned capture | 0.380525 | 0.376380 | **-0.4145pp** |
| C2-M learned capture | 0.478576 | 0.479467 | +0.0892pp |
| C1-B/C2-B gap closure | — | — | **-11.99%** |
| proposal decisive recall | 75.37% | 73.18% | **-2.18pp** |
| exact-critical Top-M recall | 23.73% | 23.10% | **-0.63pp** |
| exact-critical selected recall | 14.87% | 14.56% | **-0.32pp** |
| pair-full match | 17.4% | 17.2% | -0.2pp |
| teacher match | 17.8% | 18.6% | +0.8pp |
| teacher regret | 20133.34 | 20204.54 | **+71.21，变差** |

旧 checker 的 `deployment_gain=true` 只是因为 teacher match 单点上涨，不代表 CET 是可接受机制：C2-B、critical recall、pair-full 和 regret 都没有形成一致的 causal path。因此不能把 18.6% teacher match 当成继续 CET 的理由。

CET 的正确结论是：

> **controlled exchange 的机制成功激活，但其优化方向对真实 C2-B/critical support 是负增益。**

所以 CET 也是一个成功的诊断实验、失败的优化方向。

---

## 4. RET 与 CET 四种结果中的实际分支

V64.3.12 原先定义的逻辑可以现在正式落地：

1. RET pass，CET worse：promote RET；
2. RET fail，CET pass：blanket protection 是 remaining constraint，promote CET；
3. C2-B improve，C3 fail：acquisition cleared，pivot value/frontier；
4. **RET fail，CET fail：terminal acquisition stop，pivot value/frontier。**

实际是第 4 种。

因此：

- RET 不应继续优化；
- CET 不应继续优化；
- 不应创建 BTP-v2 / RET-v2 / CET-v2；
- 不应调 current-B protection ratio；
- 不应扩大 B/M；
- 不应增加 cross-family fallback；
- 不应再用新的 proposal ranking surrogate 解释 3.46pp oracle headroom。

---

## 5. 当前最根本的问题是什么

### 5.1 最直接的数值证据：pair value 在 selected interface 上崩塌

RET/CET 共用的 anchor validation 中：

- base teacher winner-vs-rival sign accuracy: **0.6279**；
- dense teacher winner-vs-rival sign accuracy: **0.6261**；
- selected-B pair/tournament winner-vs-rival sign accuracy: **0.0608**；
- selected pair winner-vs-rival margin MAE: **1.3320**；
- selected pair signed error: **-1.3315**；
- pair-full teacher action match: **0.174**；
- local pair-full action match: **0.174**；
- final teacher match: **0.178**；
- evidence certificate fraction: **0.928**。

这组指标非常关键：certificate 很高并没有带来 teacher value alignment；base/dense 表征能区分大约 63% 的 teacher winner-rival 符号，但通过当前 selected pair interface 后只剩约 6%。

所以当前断点不是“证据有没有传进去”，而是：

> **传进去的证据被当前 pair/value representation 如何组合成 decisive margin。**

### 5.2 第二个直接证据：当前 runtime frontier 覆盖不足

teacher winner 在 base frontier 中的覆盖：

- Top-2: 16.87%；
- Top-3: 21.81%；
- Top-5: 31.69%；
- Top-6: 34.57%；
- Top-9: 48.56%。

同时 literal exact-critical boundary 落在 base Top-9 的比例只有约 **21.92%**。

这意味着即使当前 sparse DARM/DBR pair graph 上的 value correction 完全正确，大量真正应该挑战 selected-local anchor 的 teacher winner / decisive boundary 仍没有进入已有 sparse frontier。

因此现在是两个耦合的 downstream 问题：

1. **value error**：selected evidence 的 pair margin contribution 错；
2. **frontier coverage error**：value head 只在一个稀疏 rival graph 上可用，teacher decisive challenger 常常根本没边。

---

## 6. 为什么不是重新做以前的 value 方法

历史日志里已经有多类无效或不应重复的 value 方向：

- V46/V49：broad arbitrary pair field；
- V55：Hodge/global action potential；
- V56：generic per-evidence action potential；
- V59：generic set-conditioned potential。

这些方法的问题是把 selected evidence 压缩成较宽泛的全局 action/set correction，容易再次丢失“哪个 evidence 保护哪个 decisive pair margin”的可审计性。

相反，V64.3.7 DARM+DBR 是有正结果的历史方向：在 acquisition 冻结时，teacher/pair-full 都能得到约 1--2pp 的真实改善。这说明 **literal/decisive pair-boundary value** 是值得延伸的，但 V64.3.7 的 runtime correction 仍受 sparse pair graph coverage 限制。

因此 V64.3.13 不是回到 generic potential，而是扩展 V64.3.7 的有效方向：

> 从 sparse DARM+DBR edges，升级到 **selected-evidence-attributed complete anchor frontier**。

---

## 7. V64.3.13 EAF-DMVR

全名：

**Evidence-Attributed Frontier Decisive-Margin Value Residual**

主线保持：

`fixed planner-interface budget`
→ `auditable evidence atoms`
→ `budget-feasible decisive-margin marginal utility`
→ `budgeted acquisition (terminally frozen)`
→ `exact runtime-selected B=16 evidence`
→ `evidence-attributed complete decisive frontier value`
→ `one-sided margin preservation`
→ `final decision preservation`

### 7.1 冻结什么

V64.3.13 冻结：

- foundation；
- legacy proposal / CCBR proposal adapter；
- HAB family slots；
- Top-M = 24；
- exact runtime B = 16 selector；
- DARM；
- DBR；
- calibration / evidence certificate policy；
- candidate bank。

唯一可训练模块：

`decisive_anchor_frontier_value_adapter`

这使 screen 具有很强的因果可解释性：如果 selected B/critical recall 漂移，直接判为 isolation failure。

### 7.2 完整 selected-local anchor frontier

先用冻结的 selected-local cost 得到 DARM anchor `a`。

和 V64.3.7 只在已有 sparse rival graph 上修正不同，V64.3.13 对 **所有 valid challenger b** 构造 anchor star：

`{(a,b) | b valid, b != a}`。

因此 teacher winner 不需要先进入 base Top-L / sparse rival graph 才有机会被 value correction 看见。

### 7.3 可审计的 selected-evidence value residual

对每个已经被 exact B=16 selector 选中的 evidence atom `e_i`，新 head 输出 bounded atom factor `z_i`；对每个 candidate 输出 signed factor `u_a` 和 context factor `c_a`。

对 pair `(a,b)`：

`c(a,b) = tanh(c_a + c_b + c_a * c_b)`

并定义：

`r_S(a,b) = sum_{i in S} <tanh(z_i) * c(a,b), u_b - u_a> / sqrt(|S| * d)`。

性质：

1. `c(a,b)=c(b,a)`；
2. `u_b-u_a=-(u_a-u_b)`；
3. 因而 `r_S(a,b)=-r_S(b,a)`，严格反对称；
4. 每个 atom 在非线性前独立 bounded，最终为 selected atoms 的显式和，可逐 atom attribution；
5. 只使用已经 selected 的 B evidence，不改变 selector score；
6. 不增加 evidence query；
7. final atom head zero-init，因此 warm start 时严格 no-op。

这和 V55/V59 的 global/set potential 有本质区别：它是 **pair-specific + selected-evidence-attributed + complete anchor frontier**。

### 7.4 训练目标

训练时 exact B=16 selection 是 stop-gradient 输入，只训练 value head。

冻结 V64.3.7 DARM+DBR 的完整 margin reconstruction 作为 step-zero baseline；EAF-DMVR residual additive 到该 baseline。

训练监督覆盖完整 anchor star，并包含：

- robust Smooth-L1 teacher margin regression；
- near-boundary 加权；
- teacher-winner 加权；
- pair sign loss；
- wrong-anchor teacher-winner flip loss；
- correct-anchor strongest-rival preservation loss。

这使目标直接从“选择哪个 evidence”迁移到：

> **给定已经固定选中的 evidence，它是否能恢复 teacher decisive margin frontier。**

### 7.5 runtime one-sided preservation

EAF-DMVR 先给 complete anchor star 增加 residual，再进入原有 DARM scoring 和 `pair_action_anchor_guard`。

因此新 head 不能任意翻转 action：已有 guard 仍要求 corrected challenger-vs-anchor margin 超过 configured flip margin，并继续使用现有 evidence certificate 条件。

没有重新定义 certificate，也没有降低 one-sided preservation gate。

---

## 8. 为什么这个方向比继续 acquisition 更有理论一致性

论文的核心理论对象是 decisive pair margin，而不是 proposal score。V64.3.12 已经把 `epsilon_prop/epsilon_select` 方向做到了 terminal controlled experiment；当前更大的实际误差表现在 selected evidence 到 pair margin 的映射。

EAF-DMVR 直接优化：

- selected evidence-conditioned pair value；
- decisive frontier coverage；
- antisymmetric margin structure；
- one-sided decision preservation。

因此它更直接对应论文理论链中的 `epsilon_model / preserved decisive margin`，而不是再通过一个 acquisition proxy 间接影响 endpoint。

同时 complete anchor star 只扩展 **value computation over existing candidate/action embeddings**，并没有扩展 planner evidence budget，所以 fixed planner-interface evidence budget 仍然成立。

---

## 9. V64.3.13 screen 的因果判定

screen 只回答一个问题：

> acquisition 完全冻结以后，EAF-DMVR 是否能提升 complete-frontier value mechanism，并把提升传到 C3 endpoint？

硬 instrumentation：

- new value adapter parameter delta > 0；
- residual RMS > 0；
- exact selected-B scene fraction = 1；
- complete anchor-star coverage = 1；
- runtime EAF-DMVR active；
- critical proposal/DBR parameter delta = 0；
- proposal decisive / critical Top-M / critical selected 基本不漂移。

mechanism promotion：

- frontier pair-sign accuracy 相对 anchor至少 +2pp；
- frontier action match至少 +1pp；
- already-correct anchor preservation至少 97%。

endpoint promotion：

- teacher match至少 +1pp 且 regret 不恶化超过 1%；或
- regret至少改善 2% 且 teacher match不恶化超过 0.4pp。

只有 mechanism + endpoint 同时过，才跑 full。

### 如果 screen 失败，禁止回 acquisition

若 value head 明显激活、complete frontier coverage=1，但 pair-sign/action mechanism 仍不改善：

> 下一假设是 **frozen action/evidence representation capacity**。

下一版只允许做小规模 selective action/evidence representation adapter/unfreeze test，而不是 proposal loss。

若 mechanism 明显改善但 C3 不改善：

> 下一假设是 **frontier-to-final preservation guard / calibration**。

应审计 guard 和 value calibration，而不是再改变 acquisition。

---

## 10. 工程审计与修复

本轮代码审计额外修复了两个容易造成“实验看起来跑了但语义不对”的工程问题。

### 10.1 自动补丁 literal `\\n` 问题

最初插入 `losses.py` 的新 loss block 时，自动补丁曾把换行写成字面量 `\\n`，整段可能被 Python 视为注释的一部分。`py_compile` 无法识别这种“语法合法但机制没执行”的问题。

已改成真实代码行，并增加 gradient-level test，确认 zero-init head 虽输出 0，但 final atom layer 收到非零 gradient。

### 10.2 pair-full diagnostic 不应使用 B=16-trained EAF-DMVR

EAF-DMVR 专门训练在 exact selected B=16 set 上。若 pair-full diagnostic 把全 evidence set 传给 EAF-DMVR，会把 full-evidence ceiling 污染为 OOD value correction。

已修复：

- pair-full/local-pair-full 保持冻结 V64.3.7 diagnostic 语义；
- EAF-DMVR 只作用于真实 selected-B deployment tournament；
- 新 `decisive_frontier_value_*` runtime diagnostics 显式导出到 open-loop/train validation。

---

## 11. 工程验证结果

最终版本完成：

- `python -m compileall -q bdse`: PASS；
- V64.3.13 新 tests: **8/8 PASS**；
- V64.3.7--V64.3.12 targeted regression: **37/37 PASS**；
- full repository: **336/336 PASS**；
- warnings: **36**，均为项目已有 PyTorch Transformer nested-tensor warnings；
- V64.3.13 train/eval config contract: PASS；
- EAF-DMVR exact interface contract: PASS；
- exact `B subset injected Top-M` adversarial fixture: PASS；
- launcher `bash -n`: PASS。

---

## 12. 最终建议

V64.3.12 已经给出了足够强的 terminal negative evidence，不能再把 acquisition oracle headroom 误读成“还应该继续写一个 selector loss”。

下一步最值得做的是当前已实现的 V64.3.13 EAF-DMVR，因为它：

1. 延续历史上真实有增益的 DARM/DBR decisive-boundary value 路线；
2. 解决当前最明显的 sparse frontier coverage 缺口；
3. 直接针对 6% selected pair-sign 崩塌；
4. 保持 fixed B=16 planner evidence budget；
5. 保持 evidence-level auditability；
6. 保持 exact antisymmetry；
7. 保持 one-sided final decision preservation；
8. acquisition 完全冻结，因此 screen 结果可以明确归因。

如果这一版完整 value mechanism 仍然学不动，下一步不要 EAF-DMVR-v2 无限调 loss，而应执行已经写进 checker 的 **selective action/evidence representation capacity test**。这会是一个新的、可证伪的模型容量假设，而不是重复已终止的 acquisition 分支。
