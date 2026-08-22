# V64.3.29 FCR 结果归因与 V64.3.30 EAF-ICER-PCWER 设计

## 0. 结论先说

本轮 V64.3.29 的信息量很高，但结论不是“FCR 稍微没调好”，而是一个更强的机制判断：

> **FCR 成功做到了同预算下的 global complete-frontier compression，却失败于 safe extremal recovery。由此可以 falsify：全局 action-frontier reconstruction fidelity 本身等价于 downstream recovery decision sufficiency。**

因此下一版不应该做 FCR-v2、调 L∞/RMS 权重、加 acceptance threshold、扫 B/M 或再次换 tail classifier。V29 已经把缺口定位得更窄：**support/scalar 的正向机会可见性没有因为 FCR 消失，真正的 fresh collapse 主要发生在 aggregate DRC positive gate；同时 DRC 在 representation shift 下仍可跨过零边界制造 catastrophe。**

截至 V29，dominant bottleneck 应从 V28 的泛化描述

> safe recovery coverage under fixed-budget decision-evidence transmission

进一步精炼为

> **safe recovery coverage under operator-/proposal-conditioned decision-evidence transmission, with a representation-sensitive outcome-risk confirmation stage under a fixed interface budget.**

coverage 仍是 endpoint gap 的主瓶颈；tail 不是当前“唯一 bottleneck”，但仍然是**binding safety constraint**。二者不能再被拆成“先扩大 coverage、以后再看 tail”。

V30 因此设计为 **Proposal-Conditioned Witness Evidence Rebinding (PCWER)**：先在原 AOCC B-set 上用冻结的、无 outcome-risk 的 support + incumbent-dominance operator 产生唯一 tentative proposal q；随后仅在同一个已查询 Top-M=24 中、保持 B=16 不变，重绑定对 q 真正有因果意义的 `q↔anchor` 与 `q↔incumbent` margin/attribution witnesses。接受 rebind 必须保持 proposal、incumbent、anchor 完全不变；之后 DRC 只能确认或否决这个 q，失败直接回 incumbent，绝不 second-best fallback。

为了把“no-fallback operator 改动”和“evidence rebinding 改动”分离，V30 新增 **proposal-lock-only DRC control**。这使主臂相对该 control 的差异只来自 PCWER evidence，而不是来自 operator semantics。

---

## 1. 论文主线：哪些东西应该继续保留

当前论文的核心问题不是“用 16 个 evidence”本身，而是：

> **在固定/有界 planner-interface evidence budget 下，怎样保留足以支撑最终 action decision 的 evidence，而不是最大化 world-model reconstruction fidelity。**

当前 manuscript 的结构可概括为：

1. 候选轨迹银行给出离散 action set；
2. 固定上游 M=24 queried evidence pool、B=16 retained interface；
3. EAF 建立 selected-local anchor 到所有 valid challenger 的 complete action frontier，并提供 selected evidence 的 additive attribution；
4. deployment-admissible frontier 将算法作用域限制到冻结执行约束允许的 action；
5. incumbent-contrastive recovery 只允许 admissible incumbent → alternative 的受控干预；
6. no-fallback/monotone contract 防止新增 evidence view 重新排序产生新的 action path；
7. untouched double-fresh → independent full-val 的 fail-closed protocol 用来隔离算法设计与 generalization evidence。

这些仍然是有 CCF-A 潜力的主线。V28 PTMC 的具体 Gaussian tail model 已 fresh falsify，V29 FCR 的 global-compression implementation 也 fresh falsify，但它们反而帮助把真正贡献空间收缩到了更清楚的位置：

> **bounded interface → attributed decision frontier → operator-conditioned decision evidence → incumbent-contrastive extremal recovery → downside-aware same-proposal confirmation → monotone incumbent preservation.**

这里的 novelty 不应该是 KNN、Gaussian、greedy、某个 threshold，也不应该是 B=16。

---

## 2. V29 原始假设是什么

V29 的 FCR 不是另一个 classifier，而是在 V28 失败之后重新打开 fixed-interface observability/capacity，但严格不增加 query 或 budget：

- queried bank 仍为 M=24；
- retained interface 仍为 B≤16，实际 causal operating point 固定 B=16；
- full-M EAF/DARM complete anchor-star 只作为内部 reference；
- 从原 Top-M 内重新选择与 AOCC 同 cardinality 的 B-set；
- lexicographic 最小化 complete frontier `L∞`，再最小化 RMS；
- 只有 same budget、same Top-M、full-M local anchor 不变、exact full-M downstream target 不变、compression error 严格下降时才接受；
- 失败 fail-closed 回原 AOCC B-set；
- 不读 teacher、不增加 evidence query、不重新训练 acquisition、不 beam/swap repair。

这个假设本质上是：

> 如果 V28 的 dominant bottleneck 真的是 fixed-budget decision-evidence transmission，那么让 B-set 更忠实地逼近完整 full-M action frontier，应当提高 downstream safe recovery coverage。

这个假设现在已经被 fresh 结果否定。

---

## 3. V29 的机制本身不是工程失败

### TRAIN

3000 scene TRAIN 上：

- FCR accepted `2339 / 3000 = 77.97%`；
- 所有 accepted contract 都成立；
- accepted mean `L∞` reduction `0.04409`；
- accepted mean RMS reduction `0.03525`。

### Fresh A/B

Split A：

- `361 / 500 = 72.2%` accepted；
- mean `L∞` reduction ≈ `0.03075`；
- mean RMS reduction ≈ `0.02206`；
- contract 全部成立。

Split B：

- `367 / 500 = 73.4%` accepted；
- mean `L∞` reduction ≈ `0.03275`；
- mean RMS reduction ≈ `0.02359`；
- contract 全部成立。

所以 V29 可以正面确认一个机制事实：

> **在完全相同 B/M 和 query bank 下，可以大量激活一个 deterministic rebinding 并严格改善 complete-frontier compression。**

这本身也说明“B=16 纯粹因为容量太小，所以任何信息改善都不可能”目前没有证据支持。B=16 仍应作为固定 causal operating point。

但它没有转换成 recovery benefit，这就是 V29 最重要的科学结论。

---

## 4. V29 fresh endpoint 与 selected tail

### Split A

Direct positive-opportunity capture：

- V20 ≈ `34.97%`；
- FCR-V20 ≈ `36.20%`；
- V25 DRC ≈ `11.66%`；
- FCR+DRC ≈ `4.91%`。

FCR-only 对 V20 的 coverage 只带来约 `+1.23pp`，没有出现“global frontier compression → 大幅恢复 decision coverage”的证据。

V25 selected direct tail：

- count `28`；
- teacher improvement sum `−0.9147`；
- worst `−0.9292`；
- NegRMS `0.1756`；
- regret delta sum `+18,294.6`。

FCR+DRC selected direct tail：

- count `18`；
- teacher improvement sum `+2.3414`；
- worst `−0.02255`；
- NegRMS `0.00532`；
- regret delta sum `−41,646.6`。

所以 A 的 FCR 确实偶然修掉了严重 tail，但它付出的代价是 capture 从 11.66% 掉到 4.91%，并且 structural all-flagged identity 只有 `0.90`（20 个 all-flagged scene 中 2 个最终 action 与 raw 不一致）。A 因 preservation contract 直接 FAIL。

### Split B

Direct positive-opportunity capture：

- V20 ≈ `31.68%`；
- FCR-V20 ≈ `31.88%`；
- V25 DRC ≈ `9.32%`；
- FCR+DRC = `5.00%`。

V25 selected tail：

- count `25`；
- teacher sum `+0.3832`；
- worst `−0.6051`；
- NegRMS `0.1210`；
- regret delta sum `−7,663.4`。

FCR+DRC：

- count `21`；
- teacher sum `−2.5811`；
- worst `−0.9898`；
- NegRMS `0.3327`；
- regret delta sum `+51,696.2`；
- worst regret increase ≈ `19,796`。

B 同时 FAIL coverage、tail、endpoint。

因此双 fresh 的结论非常明确：

> **FCR structural contract reproduced；FCR safe-recovery mechanism did not reproduce。**

不能 pooled A+B rescue。

---

## 5. 为什么现在能更精确定位 dominant bottleneck

我按与 screen checker 一致的 direct admissible positive-opportunity 定义重新拆 gate。

### A

V25：

`163 opportunity → support 115 → scalar 77 → support+scalar 68 → DRC-positive 20 → selected-positive 19`

FCR：

`163 → support 122 → scalar 83 → support+scalar 78 → DRC-positive 9 → selected-positive 8`

### B

V25：

`161 → support 117 → scalar 66 → support+scalar 57 → DRC-positive 15 → selected-positive 15`

FCR：

`160 → support 120 → scalar 71 → support+scalar 66 → DRC-positive 10 → selected-positive 8`

这组结果极其关键：

1. FCR **没有把 positive support/scalar opportunity 藏起来**；两 split 的 support、scalar、support+scalar 反而大体增加。
2. 真正的新增 collapse 在 **DRC-positive gate**。
3. 所以不能把 V29 简单总结成“evidence reallocation 让候选生成变差”。更准确地说：**global frontier fidelity 改善后，DRC 所使用的 aggregate evidence-space neighborhood semantics 发生了不可靠的 representation shift。**

matched candidate 统计进一步支持这一点。

A teacher-positive candidate：

- DRC mean `−0.4812 → −0.5230`；
- `DRC>0` rate `3.82% → 1.53%`；
- DRC AUC 约 `0.6097 → 0.5855`。

B teacher-positive candidate：

- DRC mean `−0.4986 → −0.5031`；
- `DRC>0` rate `4.33% → 4.00%`；
- DRC AUC 约 `0.6026 → 0.6019`。

support / dominance 的 AUC 变化远小于 DRC，说明当前真正脆弱的是 outcome-risk view，不是 support/dominance 头本身。

---

## 6. 逐场景因果归因

### A: `eca13b6114895ee4` —— FCR 能修个例，但不能形成机制泛化

- incumbent = 7；anchor = 18；
- V25 DRC 选 action 3；regret ≈ `18,586.9`；
- action 3 的 DRC `+0.08386`；
- FCR 后 action 3 DRC 变为 `−0.1583`，不再通过；
- FCR+DRC 最终 action 16，regret ≈ `0.276`。

这说明同预算 rebinding 的确可以把一个坏 proposal 从 DRC positive 侧推回 negative 侧。但 B 中反方向跨界同样发生，因此它不是可泛化安全机制。

### B: `66c3346eaa795dd1` —— FCR contract 通过，但制造 catastrophe

- incumbent = 24；anchor = 28；teacher action = 24；
- raw regret = `0`；
- V25 action 1，regret ≈ `0.828`；
- FCR+DRC action 23，regret ≈ `19,796`；
- action 23 DRC `−0.00103 → +0.01860`；
- FCR full-M exact target = 28，即 anchor；
- changed atoms = 12；
- compression/anchor/target contracts 全部通过。

这里直接 falsify 了：

> “full-M target preservation + lower global frontier error 足以保护 final incumbent-contrastive recovery semantics。”

因为 full-M exact target 约束的是 pre-recovery/anchor-side decision，而最终 ICER recovery 可以变成另一个 action。

### B: `444554532a375e54` —— 更强的 contract gap 反例

- incumbent = 13；anchor = 18；
- V25 保留 incumbent，regret ≈ `1.743`；
- FCR+DRC 改成 action 3，regret ≈ `19,789`；
- action 3 DRC `−0.32068 → +0.10016`；
- support 与 scalar dominance 同时增加；
- FCR full-M exact target = 18，仍是 anchor；
- changed atoms = 16；
- FCR contract 全部通过。

这是最干净的证明：**全局 frontier 更准，不代表 proposal-conditioned risk evidence 更准。**

### B: `dec739cd22e05639` —— V25 的旧 catastrophe 仍存在

- incumbent = 7；anchor = 16；
- V25 与 FCR 都选 action 0；
- regret ≈ `51,706`；
- DRC `+0.0881 → +0.1007`。

所以不能说“FCR 只是 B 新制造了两个 bad cases，但总体 tail detector 已经 solved”。旧的 DRC latent failure 仍在。

### A all-flagged: `8fc79d...` 与 `1b1a8a...`

两场 FCR 都 accepted，ICER structural delegation 仍显示 preserve legacy，但最终 action 与 raw 不同。结果甚至可能更好，但这是**contract violation**：all-flagged domain 的语义是整个 learned recovery/interface intervention 应该 identity-preserving，然后交给冻结 structural guard；不能因为 outcome 恰巧变好就接受这种路径。

因此 V30 对 all-flagged domain **hard bypass evidence rebinding**。

---

## 7. V29 TRAIN gate 还暴露了一个实验协议问题

V29 TRAIN fit 最终选了 107 个 direct replacement，teacher improvement sum `+11.7636`，所以旧 gate 5/5 path-safe。

但五个 fold 的 worst teacher improvement 是：

- fold 0: `−0.99063`；
- fold 1: `−0.98053`；
- fold 2: `−0.97861`；
- fold 3: `−0.000455`；
- fold 4: `−0.00823`。

也就是说，V29 在花 fresh GPU 之前已经在 TRAIN cross-fit 暴露出三折接近 −1 的 catastrophe，只是旧 gate 没把 catastrophe-free 当 hard condition。

V30 已修正：**TRAIN 必须同时 all-fold path-safe AND all-fold worst > −0.5**。若不过，直接停止，不允许 fresh，也不允许调 B/M、DRC K、zero boundary 或 witness weight 救 gate。

---

## 8. 截止 V29，命题状态应该怎样冻结

### 得到支持、继续保留

- fixed/bounded planner-interface evidence budget 是有意义的研究设定；
- EAF complete selected-local frontier 有价值；
- deployment-admissible complete frontier 是必要作用域；
- incumbent-contrastive recovery framing 继续成立；
- support/generation 与 confirmation/risk 应分解，而不是一个黑箱 reranker；
- no-fallback monotone intervention 必须保留；
- admissible incumbent preservation 必须冻结；
- B=16 是合理的 causal operating point；
- decision sufficiency 应该是 **downstream-operator conditional**。

### 已 falsify / 不能继续当 novelty 主机制

- V27 type-local KNN confirmation；
- V28 PTMC global type-tail confirmation 的 fresh generalization；
- V29 complete-frontier global compression 作为 safe recovery sufficiency 定义；
- “strictly lower global L∞/RMS → safer/better recovery”；
- “preserve anchor/full-M target 就足以 preserve final ICER recovery semantics”；
- classifier/threshold tuning 能解决当前 endpoint gap；
- tail detector 已经 solved。

### 尚未证明

- B=16 intrinsically capacity insufficient；
- 增大 B 会解决问题；
- B≤16 看不到 catastrophe；
- DRC 的 KNN risk model 在 proposal-conditioned evidence 下必然不可救；
- PCWER 一定有效。

因此不应该现在扩大 B。V29 只证明了“global allocation target 错”，没有证明“16 的容量绝对不够”。

---

## 9. 明确禁止下一轮再尝试的方向

继续继承 changelog 的 terminal constraints，并新增 V29 hard stop：

- 不做 PTMC-v2/v3、type weight、95% coverage、catastrophic cutoff、Gaussian variance tuning；
- 不做 type-KNN K/threshold/weight tuning；
- 不调 V25 DRC K={32,64}、downside multiplier=1、zero boundary；
- 不做 FCR-v2，不调 global frontier L∞/RMS objective 权重，不调 acceptance threshold；
- 不做 broad B/M sweep 来“救”V30；
- 不恢复 learned incumbent→anchor；
- 不重复 V40–V43 DACC/beam/swap repair；
- 不重复 V64.3.8–.12 HAP/BTP/RET/CET learned acquisition branches；
- 不用 support/dominance threshold rescue 当 missing evidence proxy；
- 不用 action/maneuver blacklist；
- 不用 KNN radius/OOD、transition geometry、signed profile、failed-view AND stacking、naive concat classifier 重新包装旧路线；
- 不 pooled A+B rescue；
- 不因为某个 catastrophe scene 被修掉就 claim mechanism solved。

---

# 10. V64.3.30：Proposal-Conditioned Witness Evidence Rebinding (PCWER)

## 10.1 设计动机

V29 的关键失败不是“B-set 没变好”，而是“B-set 按错误的 sufficiency target 变好了”。

完整 action frontier 对 downstream planner 是有意义的 reference，但 direct extremal recovery 不是一个 global reconstruction operator。对已经固定的 recovery proposal q，真正决定 intervention 的主要关系是：

1. q 相对 selected-local anchor 是否有足够 support；
2. q 相对 frozen incumbent 是否值得 replacement；
3. 在同一 evidence representation 下，q 的 outcome-risk certificate 是否支持执行。

因此 V30 将“decision sufficient”从全局 frontier compression 改成**operator-conditioned witness preservation**。

## 10.2 Stage 0：唯一 risk-free proposal

先在原 AOCC B-set 上运行冻结的 ICER support + scalar incumbent-dominance operator，但关闭 outcome/DRC risk：

\[
q = \arg\max_{b\in C_{sup,dom}} s^{dom}_b
\]

其中 eligibility 仍由原 deployment admissibility、support>0、scalar dominance>0 决定。

这里只生成一个 tentative proposal q。没有 q 时不做 PCWER。

这个顺序非常重要：**evidence rebinding 不允许先改变 proposal identity，再说自己改善 proposal evidence。**

## 10.3 Stage 1：同预算 proposal-conditioned witnesses

reference 仍只来自已经 query 的 Top-M=24，retain exactly B=16。

对 q、anchor a 和 incumbent i，定义 margin witness：

\[
w_m(S)=\big[F_q(S)-F_a(S),\; (F_q(S)-F_a(S))-(F_i(S)-F_a(S))\big].
\]

等价地，第二项就是 q↔i contrast。

同时保留与这两个 contrast 对应的 EAF attribution-energy witness，因为 attribution scale 是冻结 ICER/DRC evidence representation 的组成部分。

PCWER lexicographic objective 不引入 validation-tuned weights：

\[
\min_S \left(
\|w_m(S)-w_m(S_M)\|_\infty,
\|w_a(S)-w_a(S_M)\|_\infty,
RMS_m,
RMS_a
\right).
\]

这与 V29 最大区别是：**不是逼近所有 challenger，而是逼近 downstream recovery operator 当前真正要判断的唯一 proposal 的 counterfactual witnesses。**

## 10.4 Hard acceptance contracts

candidate B-set 只有同时满足以下条件才被接受：

1. cardinality 与原 AOCC 完全相同；
2. cost/budget 不增加，仍为 B=16；
3. atom 只能来自原 queried Top-M；
4. witness lexicographic error 严格改善；
5. risk-free proposal 仍然是完全相同 q；
6. incumbent identity 与 admissibility 不变；
7. selected-local recovery anchor 不变；
8. all-flagged structural domain 不允许 rebinding。

任何失败均返回原 AOCC evidence。

## 10.5 Stage 2：same-proposal DRC confirmation

接受 PCWER 后，最终 DRC **只能**判断 q：

- q 仍 support>0、dominance>0、DRC>0 → confirm q；
- 任一失败 → incumbent；
- 不允许重新从其它 alternative 选 second best；
- 不允许 q 被 veto 后 fall through。

这把 V28 保留下来的“generation vs confirmation”结构判断真正放进了 V29 暴露的 evidence-sufficiency 问题里，而不是再加一个 classifier。

---

## 11. 为什么 V30 不是历史失败路线的重复

它不是 DACC/beam/swap：没有以 action preservation 为搜索 objective 去做 generic subset repair。

它不是 V64.3.8–.12 acquisition：不训练 acquisition head，不用 teacher-shaped selector loss，不改 queried bank。

它不是 PTMC-v2：没有新增 tail classifier/type view。

它不是 V25 threshold rescue：DRC K、downside multiplier、zero boundary 都冻结。

它不是 FCR-v2：global full-star 不再作为优化 sufficiency target。

它也不是简单“把 V20 恢复”：admissible incumbent preservation、no incumbent→anchor、structural guard 全部继续冻结。

真正的新机制命题是：

> **Under a fixed bounded interface, evidence compression should preserve the counterfactual witnesses consumed by the downstream recovery operator, and any risk view may only monotonically confirm the proposal generated under that evidence.**

如果 fresh 能成立，这比“我们找到一个更好的 subset objective”更有 CCF-A 级别的论文表达空间。

---

# 12. V30 必要实验设计

## 12.1 TRAIN

仍使用完全相同 frozen 3000 TRAIN scenes，只做 representation-consistent DRC refit。

固定：

- B=16；
- M=24；
- K={32,64}；
- downside multiplier=1；
- DRC boundary=0；
- support/dominance heads 不变；
- no validation tuning。

新的 TRAIN gate：

- PCWER mechanism 必须实际激活；
- accepted contracts 100% valid；
- 5/5 proposal-locked DRC path-safe；
- **5/5 fold catastrophe-free: worst Δ > −0.5**；
- selected count / teacher sum 达到原预注册基础要求。

如果 TRAIN 不过，launcher STOP，不会选择 fresh。

## 12.2 Fresh isolation

V29 的 fresh 1000 已经被看过，因此设计 exclusion 从 7700 扩展为 **8700**，精确等于旧 7700 ∪ V29 1000。

新的 1000 untouched scenes 使用新的 hash seed，再分 A/B=500/500。A/B 独立判定，禁止 pooled rescue。

## 12.3 六个 causal arms

1. **raw**：endpoint reference；
2. **V20**：historical high-coverage / unsafe context；
3. **V25 DRC**：历史 aggregate DRC control；
4. **proposal-lock-only DRC**：原 AOCC evidence，不 rebind，只改变“唯一 q → same-proposal DRC” operator；
5. **PCWER-V20**：只验证 evidence rebinding + proposal preservation，不带 DRC；
6. **PCWER+proposal-locked DRC**：V30 main。

第 4 臂非常关键。主臂相对 V25 的变化同时包含“no-fallback operator”和“PCWER representation”两个因素；只有相对 proposal-lock-only DRC 的增量，才能归因给 PCWER evidence。

## 12.4 Fresh promotion gate

每个 split 都必须：

- PCWER active，hard contracts 全成立；
- accepted scene 的 final proposal-lock integrity=100%；
- all-flagged PCWER accepted count=0；
- proposal-lock-only control integrity=100%；
- main direct positive capture 至少比 lock-only control `+3pp`；
- main 至少多 `+5` 个 positive direct recoveries；
- main coverage 不低于 historical V25；
- selected worst teacher Δ > −0.5；
- selected tail 相对 lock-only 与 V25 均 noninferior；
- selected direct path regret delta sum ≤0；
- learned admissible incumbent→anchor =0；
- all-flagged final identity vs raw=1.0；
- endpoint 对 V25 noninferior；
- 至少一 split 有 strict endpoint signal；
- 两 split 均通过才允许 independent full-val。

这比 V29 更强，因为既拆了 operator causal effect，又把 TRAIN catastrophe gate 前置。

---

# 13. V30 失败后如何解释，不再无限迭代

### 情形 A：PCWER witness error 明显改善，但 main 相对 lock-only coverage 不提高

结论：当前 `q↔anchor / q↔incumbent` witness 定义仍不足以解释 DRC outcome risk。**停止 PCWER-v2 权重调参。** 下一步应做 controlled capacity/observability diagnostic 或重构 outcome-risk sufficient statistic，而不是 global compression 回滚。

### 情形 B：PCWER 相对 lock-only coverage 提高，但产生 catastrophe

结论：proposal-conditioned transmission 有效，但当前 aggregate DRC risk representation 不是 tail-sufficient。此时才有依据单独研究“proposal-conditioned risk sufficient representation”，同时保持 same-proposal monotonicity；不能重新打开 generic classifier sweep。

### 情形 C：TRAIN catastrophe gate 直接失败

不花 fresh。说明当前 PCWER representation 在已知 TRAIN 上已经无法使 same-proposal DRC 稳定，机制本身失败。

### 情形 D：PCWER 与 lock-only 都安全，但都显著低于 V25 coverage

说明 no-fallback decomposition 本身牺牲了太多从 second-best fallback 得到的 recovery。需要重新定义 proposal generation，而不是松开 confirmation fallback；否则会推翻 monotone structural claim。

### 情形 E：两 split 都成功

冻结 V30；只允许一次 independent full-validation reproduction。之后才讨论 test/closed-loop、外部 planner baselines、budget sensitivity、ablation 和论文主 claim。

---

# 14. 对 CCF-A 投稿主线的建议

当前论文题目/abstract 还是 PTMC-era，不能继续把 PTMC 当 validated headline。V30 之前最稳妥的 candidate headline 是：

> **Decision-Sufficient Evidence for Monotone Extremal Recovery under a Bounded Planner Interface**

若 V30 成功，机制组合可表述为：

> **Evidence-attributed action frontiers expose the downstream decision geometry; proposal-conditioned witness compression preserves only the counterfactual evidence consumed by extremal recovery; downside evidence can then confirm only that same proposal, yielding a monotone intervention whose risk view cannot create a new action path.**

CCF-A 需要的不是更多 mechanism names，而是一个清晰可证伪的 general principle：

> **Decision sufficiency is operator-conditioned. A fixed-budget planner interface should preserve evidence relative to the exact counterfactual decision operator that consumes it, rather than minimize global representation error.**

如果 V30 double-fresh + independent full-val 都成立，再配合：

- external strong planner baselines；
- B sensitivity (`8/12/16/20...`) 作为 secondary ablation，而不是主算法调参；
- global FCR vs operator-conditioned PCWER ablation；
- proposal-lock-only operator control；
- DRC/no-DRC ablation；
- all-flagged structural identity；
- per-scene tail audit / tail CDF；
- runtime/query/B accounting；

这条主线才有机会达到 CCF-A standard。

---

## 15. 本轮代码落地

已实现 V64.3.30 EAF-ICER-PCWER：

- `bdse/planner/proposal_conditioned_witness_rebinding.py`
- `bdse/planner/nuplan_planner.py`：risk-free proposal evaluator、PCWER integration、all-flagged bypass、FCR mutual exclusion；
- `bdse/planner/tournament.py`：same-proposal forced confirmation，无 second-best rerank；
- `bdse/configs/v64_3_30_eaf_icer_pcwer_v20.yaml`
- `bdse/configs/v64_3_30_eaf_icer_proposal_lock_v20.yaml`
- `bdse/tools/fit_v64_3_30_eaf_icer_pcwer.py`：proposal-locked TRAIN crossfit + catastrophe-free gate；
- `bdse/tools/make_v64_3_30_proposal_lock_control.py`
- `bdse/tools/check_v64_3_30_eaf_icer_pcwer_contract.py`
- `bdse/tools/check_v64_3_30_eaf_icer_pcwer_split.py`
- `bdse/tools/check_v64_3_30_eaf_icer_pcwer_screen.py`
- `RUN_V64_3_30_EAF_ICER_PCWER_SCREEN_2GPU.sh`
- exclusion 更新为 8700 inspected validation tokens。

工程验证：V13–V30 targeted tests 114/114 PASS；Python compile PASS；launcher `bash -n` PASS。完整 repository 运行得到 443 PASS / 1 FAIL，唯一失败是**上传的 V29 `bdse.zip` 本身缺少历史 root `V64_SAQA_BCC_NEXT_COMMANDS.sh`**，而历史测试 `test_v64_2_gatefix.py` 仍硬读取该文件。该缺失与 V30 改动无关，且 changelog 自己记录过此前版本曾因 archive omission 出现过同一类问题。为避免伪造一个未知的历史 32,559-byte script，本轮没有用 stub 冒充原文件；V30 launcher 与 targeted regression 不依赖它。

由于本地没有服务器 nuPlan GPU cache/checkpoint，本轮没有伪造 V30 算法效果。真正机制判断必须由下一条服务器命令产生。
