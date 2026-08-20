# V64.3.25 EAF-ICER-DRC 结果审计、逐场景归因与 V64.3.26 EAF-ICER-SARC 设计

## 0. 结论摘要

V64.3.25 **没有 TRAIN STOP**。TRAIN 的预注册 main `aggregate_downside` 正确通过：固定 5-fold scene cross-fit 为 **5/5 path-safe，71 个 selected replacements，teacher-improvement sum = +5.527642**。服务器随后确实完成了新的 fresh 1000 token 选择，以及 A/B 两个 500-scene block 的 raw / V20 / aggregate-meanSE / aggregate-downside 共 8 个 arm。

V25 没有生成官方 split/double-fresh checker 文件的原因不是算法或数据本身，而是 launcher 的一个 **paired-identity 工程/协议 bug**：launcher 错误要求 evaluator 输出 JSONL 的 scene 顺序必须和 hash token manifest 顺序完全一致。实际 evaluator 在 `scenario_token_file` 预过滤后按 cache 遍历顺序输出。独立审计确认：

- A/B manifest 均为 500 unique token；
- A∩B=0；A∪B= fresh 1000；
- TRAIN 3000 与 fresh 1000 overlap=0；
- 8/8 arms 每个都是 500 unique scene；
- 每个 arm 的 token **集合**与对应 manifest 完全相同；
- 同一 split 的 4 个 arm 输出顺序彼此完全一致；
- 8/8 `scenario_token_prefilter_active=true`。

因此这个 bug **只阻止 checker 被调用，不改变已经生成的 action / metric / frontier edge**。按原 V25 预注册 checker 离线恢复后得到正式科学结论：

- `train_gate_pass = true`
- `split_A_pass = false`
- `split_B_pass = false`
- `full_promotion = false`

所以 V25 是 **TRAIN PASS → double-fresh mechanism FAIL**，不能进入 full-val/test/closed-loop。

---

## 1. 上传代码包审计

本轮上传的代码 zip `OC-RAP-v48.56-DCP-DRFC-BCDE-DRAC(2).zip` 实际顶层目录为 `OC-RAP-v48.56-DCP-DRFC-BCDE-DRAC/`，内容是 OC-RAP v48.56 系列，不包含 V64.3.25 EAF-ICER-DRC 代码。因此不能把它作为本轮 V25 source truth 修改。

本交付以会话中上一轮已经经过 **412/412 tests**、并与本次 V25 result config/provenance 一致的 `bdse_v64_3_25_eaf_icer_drc` 作为 verified V25 base。V26 patch 和完整代码均基于该 verified base 生成。服务器下一轮应使用本交付 V26 包，不应在本轮误上传的 OC-RAP zip 上应用 patch。

---

## 2. V25 TRAIN 是否 STOP？

没有。

`v64_3_25_train_drc_fit.json` 和 fitter stdout 一致：

| TRAIN arm | fixed folds path-safe | selected | teacher-improvement sum |
|---|---:|---:|---:|
| aggregate mean-SE | 4/5 | 111 | +9.096630 |
| **aggregate downside (V25 main)** | **5/5** | **71** | **+5.527642** |

因此上一轮提出的第一个问题——“downside-sensitive certificate 是否比 mean-confidence 更接近 selected-path reliability？”——在 TRAIN 上仍得到肯定证据：downside 是唯一 5/5 branch。

但 V25 的真正价值是：它第一次把这个 surviving branch 放到了两个完全 untouched fresh blocks 上。结果表明 **TRAIN 方向性成立并不等于 fresh selected-tail certification 成立**。

---

## 3. V25 的工程错误：为什么官方输出提前停在 paired identity

V25 launcher 原逻辑：

```python
want = manifest_order
got = emitted_jsonl_order
if got != want:
    STOP DATA
```

这把两个不同契约混为一谈：

1. 必须执行 manifest 中的准确 scene 集合；
2. evaluator 内部输出顺序必须等于 manifest hash 排序。

科学上只需要第 1 条，以及为了 index-paired checker 安全，同一 split 的所有 arm 必须有相同输出顺序。并不需要第 2 条。

V26 已修成：

- `len(got)==500`
- `len(set(got))==500`
- `set(got)==set(manifest)`
- 同一 split 四臂 `got_order` 两两完全一致
- pre-load filter active

并同步修复 V26 包中的历史 V25 launcher，防止以后再次出现同类 false STOP。

---

## 4. 恢复后的 Split A：endpoint 看似可接受，但 selected path 被一个 catastrophe 支配

V25 aggregate-downside：

- direct incumbent→alternative replacements: **24**
- replacement regret delta sum: **+19,786.41**（FAIL）
- normalized teacher-improvement sum: **−0.98932**（FAIL）
- teacher-positive precision: 62.5%
- worst teacher improvement: **−0.98677**

endpoint：

| arm | match | regret |
|---|---:|---:|
| raw | 16.2% | 13407.17 |
| V20 | 21.2% | 14138.47 |
| mean-SE | 16.0% | 13370.09 |
| DRC | 15.8% | 13446.74 |
| DARM anchor | 15.2% | 22793.88 |

因此 V25 原 checker 的 `endpoint_gain=true`，但这不能救 mechanism：selected direct replacement path 本身净 harmful，必须 STOP。

最坏 scene：

`85b3a4ed30b65780`，incumbent 2 → candidate 19：

- true teacher improvement = **−0.986769**
- actual regret delta = **+19,735.39**
- DRC score = **+0.031587**
- support logit = +1.151
- scalar dominance = +0.728

它并不是 OOD：在 TRAIN memory 中其 r32/r64 约位于 **49.5% / 38.3% percentile**。也就是说，这是一个处于相当稠密区域、但局部 outcome mode 错配的 catastrophe。

K=32 neighborhood：

- mean = +0.09593
- downside RMS = 0.02158
- DRC = **+0.07435**
- neighborhood worst = **−0.12288**
- fresh truth = **−0.98677**

K=64：

- mean = +0.05312
- downside RMS = 0.02153
- DRC = **+0.03159**
- neighborhood worst = **−0.12331**
- fresh truth = **−0.98677**

因此问题不是 DRC 计算错误，而是当前 representation 下的邻居根本没有包含当前 candidate 所属的 catastrophic latent mode。

---

## 5. 恢复后的 Split B：V25 对 downside 的 standalone claim 被更直接地否定

V25 aggregate-downside：

- direct replacements: **24**
- regret delta sum: **+43,170.21**
- teacher-improvement sum: **−2.15851**
- positive precision: 58.3%
- worst: **−0.99038**

endpoint：

| arm | match | regret |
|---|---:|---:|
| raw | 16.2% | 12763.74 |
| V20 | 23.2% | 13261.07 |
| mean-SE | 16.6% | 12770.90 |
| DRC | 16.4% | 12850.08 |
| DARM anchor | 16.8% | 22169.28 |

所以 B 同时 FAIL：selected path、recovery、DRC incrementality、preservation、endpoint。

两个决定性 catastrophe：

### `8b18a09c5743529a`: 0 → 2

- true improvement = **−0.990379**
- regret delta = **+19,807.58**
- DRC = **+0.014208**
- K32 neighborhood worst 只有 **−0.00127**
- K64 neighborhood worst 只有 **−0.12151**
- r32/r64 support percentile = 84.6% / 78.6%

### `f2469e5f4c2853c1`: 19 → 10

- true improvement = **−0.989746**
- regret delta = **+19,794.91**
- DRC = **+0.010882**
- support = +1.941
- scalar dominance = +1.061

它尤其重要，因为 K32：

- mean = +0.02714
- downside RMS = **0.000595**
- DRC = **+0.02654**
- mean-SE = **−0.000719**

也就是说，DRC 并不是 mean-SE 的单调收紧。局部负尾看起来极小的时候，`mean - negative-RMS` 可以允许一个 mean-SE 会拒绝的 candidate；但真实 fresh outcome 恰好是 −0.99 catastrophe。

---

## 6. V25 对“downside-sensitive certificate 有效”给出的进一步答案

上一轮正确结论应该保留一半、收回一半。

### 可以继续保留的结论

**Downside magnitude 是正确的风险对象。**

原因：

- V24 TRAIN 上 aggregate-downside 从 mean-SE 的 4/5 提升到 5/5；
- V25 A 中 DRC 的 positive-regret RMS 确实低于 mean-SE，因此原 checker 的 `downside_incremental=true`；
- catastrophic failure 明显由少数极端 negative outcome magnitude 主导，而不是普通 binary accuracy。

所以不能退回只看 AUC / probability / neighborhood mean。

### V25 否定的更强说法

**`mean - local negative-outcome RMS` 作为 standalone local certificate 并不具有 fresh robustness。**

B 中它不仅没稳定改善 mean-SE，还新增一个 mean-SE 没选的 −0.9897 catastrophe。

因此论文不能再把“downside-RMS certificate 本身”作为已经解决问题的 headline novelty。它应该降级为 **tail-sensitive risk objective/statistic**，而不是完整 solution。

---

## 7. mean-SE vs DRC 的 selection partition：为什么不能做两者阈值混合

A：

| subset | count | teacher sum | regret delta | worst |
|---|---:|---:|---:|---:|
| shared | 14 | −0.9928 | +19,856 | −0.9868 |
| DRC-only | 10 | +0.0035 | −70 | −0.00037 |
| meanSE-only | 23 | **+1.9198** | **−38,396** | −0.9901 |

DRC 的确过滤了 mean-SE 的一个 catastrophe，但 shared population 中仍保留另一个 catastrophe，同时丢掉大量净有益 meanSE-only replacement。

B：

| subset | count | teacher sum | regret delta | worst |
|---|---:|---:|---:|---:|
| shared | 14 | −1.1701 | +23,402 | −0.9904 |
| **DRC-only** | **10** | **−0.9884** | **+19,768** | **−0.9897** |
| meanSE-only | 23 | **+0.9911** | **−19,821** | −0.00810 |

这说明下一步不应该：

- 调 meanSE/DRC mixing weight；
- 做 `meanSE AND DRC` 双阈值作为 main；
- 调 DRC multiplier；
- 调 zero boundary；
- 调 K。

这些都是 certificate scalar 层的修补，而当前失败是 **邻域 outcome semantics 不对**。

---

## 8. dominant bottleneck 再次收紧

V24 的 bottleneck：

> selected-path downside certification under risk-geometry distortion：不要让 abs-sort/L1-normalized attribution shape 把 negative mode 隔离掉。

V25 已经移除 V24 的 full-spectrum geometry distortion，但 catastrophe 仍然出现，而且 A catastrophe 位于 dense TRAIN support。

因此当前 dominant bottleneck 应进一步收紧为：

> **selected-path tail certification under semantic outcome aliasing in the regret representation**：当前 18-D aggregate evidence features 对局部距离是“有支持”的，但不是 outcome-sufficient；不同 selected-evidence semantic states 在聚合统计后变得近似，导致 catastrophic candidate 的真实 latent mode 在其 TRAIN neighborhood 中不可见。

中文直观表述：

> **现在不是“邻居太少”，而是“找错了邻居”。**

这是比 V24 更窄、更可证伪的机制问题。

---

## 9. fixed B≤16 evidence interface 现在能否判定有用/没用？

**仍不能判定 interface capacity 不够，也不能宣称它已经被证明 sufficient。**

V25 证明的是：

- 18-D aggregate representation 不足以稳定区分 catastrophic mode；
- 简单 TRAIN density/coverage gate 不能解释全部失败。

但它没有证明 fixed B≤16 interface 本身缺少信息，因为：

1. B≤16 中每个 selected atom 的 family/type identity 真实存在；
2. exact EAF candidate attribution 真实存在；
3. V24 full-spectrum 把 atom identity 用 abs-sort 抹掉，又用 L1 normalization 丢了 scale，并独立排序 candidate/delta，破坏 candidate-incumbent correspondence；
4. V25 aggregate 进一步把这些 semantic identities 压成 18-D summary。

也就是说，**我们还没有测试过一个真正 identity-preserving 的 certificate representation**。

因此 V26 应继续冻结 B≤16，专门测试“interface 中已经存在但此前被 representation 丢掉的语义信息”。如果这一步仍不能在 TRAIN/fresh 稳定消除 catastrophe，才有更强理由重新打开 evidence interface capacity / acquisition。

---

## 10. V25 后明确禁止重复的方向

基于 V19–V25 changelog 与本轮结果，下一阶段不要再做：

- V24 的 abs-sorted + L1-normalized full attribution spectrum；
- attribution group-weight sweep；
- transition-conditioned main geometry / transition group weights；
- signed-profile equal-mean ranking / profile mixing weight；
- K=32/64 sweep、downside multiplier sweep、zero-threshold sweep；
- raw action/maneuver blacklist；
- scalar dominance threshold sweep（catastrophes 的 dominance 从 0.075 到 1.06 都存在）；
- support threshold sweep（catastrophes 的 support 从 0.62 到 1.94 都存在）；
- 单独增加 KNN radius/OOD guard 作为 main（A catastrophe 是 dense support）；
- 直接上 learned embedding / 更大网络，在 semantic identity 这个更简单可审计假设没被验证前；
- broad-unfreeze acquisition / selector / EAF / B/M / safety guard；
- 用 pooled A/B、endpoint 偶然 non-inferiority 或 AUC 去 rescue harmful selected path。

---

## 11. 论文主线：V25 后应该保留/舍弃什么

后续论文不以旧 TeX 的算法实现为主线，以当前代码机制为准。

### 保留为主体

1. **fixed planner-interface evidence budget B≤16**：控制可部署信息边界和审计复杂度。
2. frozen EAF complete DARM-anchor frontier。
3. exact selected-evidence attribution 作为 upstream auditable structure。
4. complete final-guard-admissible candidate population。
5. frozen support + scalar incumbent dominance。
6. asymmetric incumbent-default extremal replacement。
7. selected-path counterfactual teacher-regret audit，而不是 generic edge AUC。
8. unchanged evidence/one-sided certificate、structural guard、final decision preservation。
9. independent A/B fresh protocol。

### 从主 claim 中舍弃/降级

- “attribution-resolved regret certification”：V24 已否定当前 full-spectrum 实现；
- “downside-RMS alone solves reliability”：V25 fresh 已否定；
- transition geometry 作为 headline；
- signed-profile ranking 作为 headline；
- generic binary reliability/AUC 作为最终 reliability 定义；
- old RAER-style learned incumbent→anchor abstention 作为 main；
- old preservation `harmful -5pp` 作为 asymmetric main 的机制目标；
- TeX 中与当前代码不一致的 selector/tournament 细节不再作为核心实现 claim，除非后续同步成代码事实。

---

## 12. Novelty 是否需要调整

V24 后候选 novelty：

> evidence-attributed incumbent-contrastive downside-regret certification for deployment-admissible extremal recovery under a fixed planner-interface evidence budget.

V25 之后这句话的 **“downside-regret certification”过强**，因为 standalone aggregate DRC 没有 fresh path safety。

V26 若成功，建议收紧为：

> **semantically aligned evidence-attributed incumbent-contrastive tail-regret certification for deployment-admissible extremal recovery under a fixed planner-interface evidence budget.**

其机制含义是：

- evidence-attributed：risk representation 确实来自 exact selected-evidence contribution；
- semantically aligned：不再按 magnitude sort，而按固定 evidence family identity 对齐；
- incumbent-contrastive：candidate 与当前部署 incumbent 的同一 selected atoms correspondence 被保留；
- tail-regret：downside magnitude 仍是目标，但不再宣称一个 scalar downside statistic 本身足够；
- deployment-admissible extremal recovery：operator 与 final guard 语义保持一致；
- fixed evidence budget：没有通过增加 evidence query 容量来赢结果。

这个 novelty 必须以 V26 TRAIN gate + 双 fresh + independent full-val 后再最终定型；现在只能作为候选 headline。

---

# 13. V64.3.26 EAF-ICER-SARC

全称：

**Evidence-Attributed Incumbent-Contrastive Extremal Recovery with Semantic-Aligned Regret Certification (EAF-ICER-SARC)**

V26 只改变 **一个东西：replacement regret certificate 的 representation**。

### 13.1 固定 semantic family

代码中已有 5 个 frozen evidence certificate families：

1. feasibility
2. reachability_interaction
3. precedence
4. decision_boundary
5. dynamic_regularity

不新增 family，不根据 validation 合并/拆分。

### 13.2 Identity-preserving attribution representation

对 selected evidence atom 集合 S、candidate b、incumbent i，EAF 已提供每个 selected atom 对 action 的 signed contribution `a_e(action)`。

每个固定 family f 计算：

\[
s_f(b)=\sum_{e\in S,\; family(e)=f}a_e(b)
\]

以及同一 selected atoms 上的 incumbent contrast：

\[
d_f(b,i)=\sum_{e\in S,\; family(e)=f}\big(a_e(b)-a_e(i)\big).
\]

得到 10-D semantic vector：

\[
\psi(b,i)=[s_1,\ldots,s_5,d_1,\ldots,d_5].
\]

关键约束：

- **不按 contribution magnitude 排序**；
- **不做 candidate-specific L1 normalization**；
- candidate 与 incumbent 使用同一 atom/family correspondence；
- 不学习 embedding；
- 不调 family/group weight。

### 13.3 Regret representation

V25 的 18-D aggregate evidence vector `x_agg` 完整保留。

V26 main：

\[
x_{SARC}=[x_{agg};\psi]\in\mathbb{R}^{28}.
\]

所有 28 个固定坐标仅使用 TRAIN mean/std 标准化，并使用统一的：

\[
w_j=1/28.
\]

这里没有一个新的 semantic-group hyperparameter。

V25 control 仍为 18-D aggregate，权重 1/18。

### 13.4 Downside objective 保持不变

仍然固定：

\[
K\in\{32,64\},
\]

inverse-distance local weights，且：

\[
C_K=\mu_K-\sqrt{\sum_j w_j\min(\Delta_j,0)^2}.
\]

最终：

\[
C=\min(C_{32},C_{64}).
\]

只有 C>0 才允许 replacement。

multiplier=1，boundary=0，不 sweep。

### 13.5 Action operator 完全冻结

final-guard-admissible incumbent 默认保留。

alternative 必须：

\[
support>0
\land scalar\ dominance>0
\land C>0.
\]

survivors 继续按 **frozen scalar dominance** extremal ranking。

没有：

- learned incumbent→anchor veto；
- signed-profile rank；
- transition risk；
- density gate；
- attribution spectrum gate。

---

## 14. V26 TRAIN 实验：只回答 semantic identity 是否真的有增量

因为历史 V24/V25 frontier 没有序列化 family-aligned coordinates，V26 必须对冻结 V20/EAF pipeline 做一次 3000-scene TRAIN instrumentation replay。它不是 EAF/selector retraining。

TRAIN 只比较两个风险 arm：

1. `aggregate_downside`：V25 control，18-D；
2. `semantic_family_downside`：V26 main，28-D。

共同冻结：

- replacement population；
- support/scalar dominance；
- scene fold seed；
- K32/K64；
- downside formula；
- boundary；
- ranking。

V26 main 在 fresh 前必须：

- 5/5 fixed scene folds selected-path non-harmful；
- total selected >=64；
- total teacher-improvement >=0；
- selected negative RMS 不劣于 V25 control；
- selected worst outcome 不劣于 V25 control；
- 上述两项至少一项严格改善。

如果失败：TRAIN STOP，不花 fresh GPU；不调 family weights / K / multiplier / zero / density threshold。

---

## 15. V26 fresh 实验

V25 fresh 1000 已经用于本轮 diagnosis，因此永久加入 exclusion。

V26 design exclusion = **6700 unique validation tokens**：

- 原 5700；
- + V25 fresh A/B 1000；
- overlap=0。

使用新 hash seed 选择 1000 untouched validation scenes：A500/B500。

每个 block 只运行四臂：

1. raw EAF；
2. frozen V20；
3. V25 aggregate-downside control；
4. V26 semantic-family-downside main。

这使唯一 causal difference 清楚可解释。

每个 block independently 必须证明：

- instrumentation / frozen interface identity；
- structural domain delegation；
- candidate support + frozen support/dominance signal；
- zero learned admissible incumbent→anchor changes；
- main direct replacement count >=8 且 regret delta sum <=0；
- recovery precision/capture；
- semantic main 的 selected teacher negative-RMS 与 worst outcome 均不劣于 V25 DRC，且至少一项严格改善；
- raw harmful/flip asymmetric non-degradation；
- match >= DARM anchor +0.5pp；
- regret <=1.02×raw 且 <=1.02×V20。

A/B 任一失败即 STOP，不 pooled rescue。

---

## 16. V26 后如何解释不同结果

### A. TRAIN fail

结论：identity-preserving coarse semantic-family representation 在冻结 population 上没有足够增量。

不要 fresh，不调 family weight。

下一步应判断 5-family aggregation是否仍太粗，或者 fixed B interface 的 atom-level type/state 才是必要分辨率；但仍不能直接 broad-unfreeze。

### B. TRAIN pass，fresh A/B selected-tail fail

这是一个非常重要的 negative result：即使保留 coarse semantic identity，TRAIN local memory 仍无法辨识 fresh catastrophe。

此时对 fixed interface capacity 的怀疑显著增强。下一步应先审计 **within-interface atom-type/relational state** 是否仍被 family aggregation丢失，再决定是否真正 reopen evidence acquisition。

禁止再调 continuous KNN 超参。

### C. path-safe，但 semantic tail 不增量

不能 claim SARC novelty。保留更简单 control；不通过 weight tuning 制造增量。

### D. A/B 都 pass

冻结 V26；只允许一次 independent full-validation reproduction。

full-val 再 pass 后才能考虑 test/closed-loop 和论文最终 headline。

---

## 17. 工程实现与审查

V26 已落地：

- `tournament.py`：新增固定 5-family identity-preserving 10-D attribution coordinates；新增 `semantic_family_aligned` regret-risk mode；旧 mode backward-compatible。
- `nuplan_planner.py`：把 selected atoms 对应的 frozen family IDs 精确传入 tournament。
- `evaluate_open_loop.py`：只为 deployment-admissible challengers 序列化 10-D semantic-family provenance。
- 新 V26 fitter：streaming loader、fixed crossfit、aggregate DRC control vs SARC main、fail report lifecycle。
- 新 V26 contract checker：28-D exact schema、equal 1/28 metric weights、memory SHA、K/multiplier/boundary、frozen heads。
- 新 split/double checker：selected-tail semantic incrementality，不使用 pooled rescue。
- launcher：6700 exclusion、TRAIN instrumentation replay、4-arm double fresh、修正 paired identity contract。
- 历史 V25 launcher 同步修复 paired identity false STOP。

验证：

- V26 unit tests: **6/6 PASS**；
- V64.3.13–V64.3.26 targeted stack: **88/88 PASS**；
- full repository 分三批：**95 + 153 + 170 = 418/418 PASS**；
- warnings: **36**，均为历史 PyTorch Transformer `nested_tensor/norm_first`；无新 warning class；
- Python compileall: PASS；
- V25/V26 launcher bash syntax: PASS；
- synthetic aggregate 18-D memory/config contract: PASS；
- synthetic semantic-family 28-D memory/config contract: PASS。

注意：当前本地环境没有服务器 nuPlan GPU/cache，因此没有伪造 V26 算法效果。V26 的下一条科学结果必须来自服务器按照 launcher 的 TRAIN gate → untouched A/B 流程得到。
