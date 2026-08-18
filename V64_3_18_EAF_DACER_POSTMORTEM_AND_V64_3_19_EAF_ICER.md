# V64.3.18 EAF-DACER 结果归因与 V64.3.19 EAF-ICER 设计

## 结论摘要

V64.3.18 的 `STOP SCREEN` 应当保留：不能进入 full/test/closed-loop。但这轮并不是整体机制失败。按预注册核心链条逐段审计：

`candidate semantic recovery -> genuine multi-challenger support -> incumbent-dominance generalization -> signed-evidence attribution gain -> alternative/counterfactual recovery -> preservation -> endpoint`

得到：

1. **candidate semantic recovery：成立。** V64.3.17 的 near-singleton candidate collapse 已被修复；fresh 有 4933 条 final-guard-admissible edges，raw-proposal 中 56.81% 场景具有至少两个 admissible challengers。
2. **genuine multi-challenger support：成立。** 470 个 raw proposal scene 中 267 个存在多 challenger 竞争，mean 10.50 admissible candidates/proposal。
3. **incumbent-dominance generalization：有真实信号但仍不够强。** profile fresh AUC=0.6883，train internal holdout AUC=0.7006；明显高于随机但不足以保护 extremal top selection。
4. **signed-evidence attribution gain：成立但幅度小。** profile 相比 scalar 的 dominance AUC +1.09pp；alternative recovery 5.96% -> 11.28%；counterfactual opportunity capture 7.55% -> 13.21%；同时 endpoint 从 scalar 的 22.2% match / 15252.56 regret 改善到 23.2% / 14680.41。
5. **alternative recovery over anchor：基本成立。** profile alternative recovery=11.28%，这些 alternative 中 84.91% teacher-better-than-anchor，mean teacher margin=1.131。
6. **真正的 incumbent-relative counterfactual recovery：未成立，是当前唯一主要断点。** selected alternative > actual frozen incumbent 的 precision 只有 39.62%，低于预注册 60%。
7. **preservation：成立。** harmful 9.4% -> 0.2%，beneficial retention=80%，post-selection final-guard block=0。
8. **endpoint：screen 上成立。** profile match=23.2%，高于 anchor 16.2% 和 RAER 21.6%；regret=14680.41，在 `<=1.02*raw` 且 `<=1.02*RAER` 的约束内。

因此当前模型不是“EAF/reliability 不工作”，而是已经收缩到一个很窄的问题：**anchor-support reliability 与 incumbent-relative dominance reliability 被一个 shared pointwise score 混在一起，extremal argmax 仍会放大少数 dominance false positives。** 这可以视为 winner's-curse 的第二阶版本：V64.3.13–V64.3.16 处理的是 value extremum 的 winner's curse；V64.3.18 现在暴露的是 reliability extremum 的 winner's curse。

---

## 1. V64.3.18 fresh screen 的端点状态

同一组 fresh 500 scenes：

| Arm | Teacher match | Regret | Flip | Beneficial | Harmful | Final guard block |
|---|---:|---:|---:|---:|---:|---:|
| DARM anchor | 16.20% | 23785.04 | - | - | - | - |
| raw EAF | 15.80% | **14453.06** | 57.20% | 9.00% | 9.40% | 35.20% |
| frozen RAER | 21.60% | 15013.09 | 38.20% | 7.00% | 1.60% | 1.80% |
| frozen V17 DALER | 22.00% | **14400.88** | 38.80% | 7.60% | 1.80% | 0 |
| G-DALER | **23.20%** | 14722.83 | 35.60% | 7.20% | **0.20%** | 0 |
| DACER-scalar | 22.20% | 15252.56 | 33.40% | 6.20% | **0.20%** | 0 |
| DACER-profile | **23.20%** | 14680.41 | 36.80% | 7.20% | **0.20%** | 0 |

raw EAF regret 仍比 anchor 低约 39.2%，因此 EAF value information 继续被结果支持；禁止重新打开 acquisition / B / M / BTP / RET / CET / AF / HAP / broad EAF unfreeze。

G-DALER 和 DACER-profile 的 preservation 已非常强。下一版不应为了追 match 放松 evidence certificate、one-sided guard 或 structural safety，也不应做 probability/score threshold sweep。

---

## 2. V64.3.18 解决了 V64.3.17 的 candidate semantics 错误

V64.3.17 把 upstream utility-refinement pool 当成 hard deployment admissibility，导致所谓 listwise frontier 几乎成为 singleton。V64.3.18 去掉这个错误 hard mask，只保留真正 final-guard prerequisites：valid/safety、frozen EAF DARM-anchor margin、score gain、evidence certificate，并让 all-flagged scene 继续交给 frozen structural-risk guard。

fresh V64.3.18 profile：

- admissible edges：4933；
- mean admissible candidates / scene：9.866；
- raw proposal scenes：470；
- multi-admissible proposal scenes：267；
- multi-admissible proposal rate：56.81%。

所以 candidate semantic recovery 和 genuine multi-challenger support 已经通过。后续不能再恢复 V17 utility-equivalence hard mask。

---

## 3. incumbent-dominance 有泛化，但 shared score 的 extremal semantics 不够

V64.3.18 主 arm 用一个 standardized linear score 同时承担三种训练压力：anchor-augmented listwise CE、candidate-vs-anchor support BCE、candidate-vs-incumbent dominance ranking。runtime 再直接对同一个 score 做 argmax。

结果表明平均 ranking 信号存在：

- G-DALER fresh incumbent-dominance AUC：0.6567；
- DACER-scalar：0.6774；
- DACER-profile：0.6883；
- DACER-profile train internal holdout：0.7006。

但 AUC≈0.69 并不足以保证 extremal top 的 precision。profile 在 470 proposal scenes 中恢复 53 个 alternative；45/53 比 anchor 好（84.91%），但只有 21/53 比当前 incumbent 真正好，precision=39.62%。

因此 shared pointwise score 的主要逻辑缺陷是：

- `candidate better than anchor` 与 `candidate better than incumbent` 是两个不同 conditional statement；
- V18 让它们共享一个绝对 score，最终 argmax 不知道某个高分是因为“绝对 support 很强”，还是因为“相对 incumbent 真有 dominance”；
- profile 特征提高了平均 dominance ranking，但最终选取的是最大 reliability score，少数 false-positive alternative 被 extremal selection 放大。

这不是阈值问题。继续 sweep anchor logit / p threshold / dominance weight 会重复历史上 OCFI/EAIR/RAER threshold-style 无效路线，也破坏预注册数据纪律。

---

## 4. signed selected-evidence attribution 值得继续做，但不能把它单独当最终 novelty

DACER-profile 相比 DACER-scalar：

- dominance AUC：0.6774 -> 0.6883；
- alternative recovery：5.96% -> 11.28%；
- counterfactual opportunity capture：7.55% -> 13.21%；
- match：22.2% -> 23.2%；
- regret：15252.56 -> 14680.41；
- harmful 都保持 0.2%。

因此 exact signed selected-atom profile 得到了 causal ablation support，不能因为最终 counterfactual precision 不过就删除 attribution structure。

但它的作用应表述为：**为 incumbent-relative reliability 提供 structured evidence contrast**，而不是“用了 per-atom feature 就是 novelty”。V19 仍然保留 exact attribution / signed candidate-minus-incumbent profile，但把它放入一个直接的 incumbent-contrastive reliability operator。

---

## 5. novelty / 论文主线判断

V64.3.18 之前的 umbrella wording：

> evidence-attributed, deployment-aligned listwise reliability for extremal decision selection under a fixed planner-interface evidence budget

仍可作为高层描述，但不建议继续把 **listwise** 当核心 novelty。V18 已证明 listwise+shared score 本身不足以实现 reliable extremal recovery。

推荐 V64.3.19 的更精确 novelty：

> **evidence-attributed incumbent-contrastive reliability for deployment-admissible extremal recovery under a fixed planner-interface evidence budget**

它没有改变论文主线，而是把 V18 已被实验支持的部分收紧成可证伪机制：

`fixed planner-interface evidence cap B<=16`
`-> auditable evidence atoms`
`-> terminally frozen M=24 acquisition`
`-> B<=16 selected evidence`
`-> frozen EAF complete DARM-anchor frontier`
`-> exact selected-evidence attribution`
`-> complete final-guard-admissible challenger frontier`
`-> frozen anchor-support reliability + direct incumbent-contrastive evidence reliability`
`-> admissible incumbent preservation / evidence-supported extremal replacement / anchor abstention`
`-> unchanged final evidence/one-sided guard`
`-> unchanged structural-risk guard`
`-> final decision preservation`.

“counterfactual”如果继续在论文中出现，应明确限定为 same-scene operational alternative comparison，不声称 causal identification。

---

## 6. V64.3.19 EAF-ICER

全称：**Evidence-Attributed Incumbent-Contrastive Extremal Recovery**。

### 6.1 冻结 V18 已经证明有效的 anchor-support 头

V19 不重新拟合 support。它直接复用 V18 TRAIN-only G-DALER scalar support head：

\[
s_{sup}(b)=w_{sup}^{T}\phi_{scalar}(b)+\beta_{sup}.
\]

这有两个好处：

1. 速度更快；
2. 更重要的是 causal isolation：V18 profile -> V19 的变化只来自 incumbent-contrastive representation/operator，不会因为 support head 重新训练而混杂。

注意这个 frozen support head 已经见过全部 V18 TRAIN，因此 V19 fit report 明确标记 `support_holdout_independent=false`。其在 dominance holdout partition 上的 AUC 只叫 replay diagnostic；真正 generalization gate 只看新 fresh validation。

### 6.2 deployment incumbent 定义与 metric/operator 对齐

V19 明确定义：

- 如果 raw EAF incumbent 本身满足 final-guard admissibility，则 deployment incumbent = raw incumbent；
- 如果 raw incumbent 会被 frozen final guard 拒绝，则 deployment incumbent = DARM anchor。

因此 runtime 有两种不同语义：

1. **direct incumbent replacement**：raw incumbent admissible，alternative 必须同时 support-positive 且 incumbent-dominance-positive；
2. **anchor recovery**：raw incumbent inadmissible，此时不强迫 alternative 去击败一个部署时根本不会执行的 raw action，只使用 frozen support head 在 admissible frontier 中恢复。

V19 checker 把两类指标分开。promotion 的核心 novelty gate 使用 `direct_incumbent_replacement_precision` / `direct_incumbent_opportunity_capture_rate`；anchor recovery 单独报告，不能用来“冲高” incumbent-contrastive claim。

### 6.3 独立 direct dominance head

对 raw incumbent admissible 的 scene，TRAIN-only label 为：

\[
y_{dom}(b)=\mathbb{1}[M_T(b,a)>\max(0,M_T(b_{inc},a))].
\]

也就是说，positive alternative 必须同时优于 anchor 和 incumbent。

dominance head 使用 **unweighted direct BCE**，使 logit=0 对应模型自身的 fixed 0.5 direct conditional boundary，而不是再由 listwise pseudo-item / class-balanced loss 任意平移。没有 validation threshold sweep。

runtime：

- admissible incumbent 若 support-positive，默认保留 incumbent；
- 只有 support-positive alternative 且 direct dominance logit > 0 才具备 replacement 资格；
- 如果多个 alternative 通过，按 direct dominance logit extremal ranking；support/margin/legacy utility membership/action id 只做固定 tie-break；
- 如果没有通过者，不进行 replacement。

### 6.4 fixed quadratic evidence-interaction map

V18 的 dominance AUC 已有信号但 extremal precision 低，说明不是“完全没 representation”，而是简单线性 readout 无法表达某些 evidence reliability interactions。

V19 没有 broad-unfreeze EAF，也没有引入 generic deep network，而是使用预注册的 fixed degree-2 interaction map，随后仍然是线性 logistic readout：

\[
\psi(\phi)=[\phi_1,\ldots,\phi_d,\{\phi_i\phi_j\}_{i\le j}].
\]

- scalar view：25 个已审计 scalar runtime features -> 350 dims；
- profile view：30 个 pre-registered evidence/incumbent-contrast features（含 top-4 signed candidate-minus-incumbent selected-atom contributions）-> 495 dims。

它不增加 evidence query，不改变 EAF value，不新增 hidden learned representation；新能力来自固定的 evidence-interaction basis + direct incumbent-contrastive readout，仍保持可审计性。

### 6.5 dual-view main arm 与 scalar ablation

V19 同时产生：

- `ICER-scalar`：只用 scalar direct dominance logits；
- `ICER-dual`：固定 equal mean：

\[
s_{dom}(b)=\frac{1}{2}(s_{scalar}(b)+s_{profile}(b)).
\]

0.5 权重固定，禁止 validation tuning。dual 的作用不是卖 ensemble architecture，而是对 signed selected-evidence attribution 是否给 direct incumbent replacement 增益做 causal ablation。

TRAIN-only design diagnostics（不可用于 promotion / paper table）：

| arm | direct dominance AUC | direct replacement rate | direct replacement precision | direct opportunity capture | alt precision |
|---|---:|---:|---:|---:|---:|
| ICER-scalar | 0.7544 | 38.68% | 64.44% | 39.19% | 96.30% |
| ICER-dual | **0.7647** | 41.55% | **64.83%** | **42.34%** | 95.86% |

这些数字只说明 V19 值得跑 fresh causal screen。它们不能作为 V19 泛化结果，因为算法已经基于 V18 validation 结果设计。

---

## 7. V64.3.19 fresh causal screen

下一轮仍使用完全新、hash-selected、未看过的 500 validation scenes，但不再重复六个 full replay。

四臂即可回答当前所有问题：

| arm | 作用 |
|---|---|
| V19 raw EAF | frozen endpoint + interface control |
| frozen V18 DACER-profile | 上一版最强 control |
| ICER-scalar | direct incumbent-contrastive semantics without signed profile dual view |
| ICER-dual | V19 main arm |

V18 的 RAER/G-DALER/V17-DALER 已经在 fresh V18 screen 完成因果定位；下一版当前唯一问题是 V18 profile -> ICER-scalar -> ICER-dual。因此不需要再次付 GPU 成本重跑全部历史 arms。

promotion 首先要求 frozen interface 和 candidate support 不回归；随后要求：

- fresh support AUC >= 0.65；
- fresh direct incumbent-dominance AUC >= 0.70；
- overall alternative recovery >= 3%，alternative precision >= 80%；
- **direct incumbent replacement rate >= 2%**；
- **direct incumbent replacement precision >= 60%**；
- **direct incumbent opportunity capture >= 8%**；
- direct incumbent replacement precision 至少比 frozen V18 profile 高 10pp；
- signed profile dual view 相比 scalar 在 dominance AUC / direct precision / direct capture 至少一项达到预注册 causal gain，并保持 endpoint non-harm；
- final guard block <= 0.1%；
- harmful 比 raw 至少下降 5pp，beneficial retention >=35%，beneficial>harmful；
- match >= anchor +0.5pp；
- regret <=1.02*raw 且 <=1.02*V18 profile，并且 match 不比 V18 profile 低超过0.5pp。

若 fresh screen 通过，只允许 independent full-val reproduction。仍不运行 test/closed-loop。

---

## 8. V64.3.18 为什么很慢，以及 V64.3.19 的速度修复

V18 变慢有一部分是正常的：它进行了 train raw replay、4000-scene val discovery、多个 fitter，以及 6 个 fresh paired arms。

但结果日志证明存在大量可消除的工程耗时。

### 8.1 V18 实际耗时结构

progress 日志：

- 3000 train raw replay：34m23s；
- 4000 validation discovery raw replay：47m41s；
- fresh raw/RAER wave：每个进程都走到 `58418/58418`，约 1h46m43s；
- fresh V17/G-DALER wave：同样 `58418/58418`，约 38m42s；
- fresh scalar/profile wave：同样 `58418/58418`，约 27m06–27m07s。

然而所有 fresh arm 实际都只 evaluation 500 scenes，而且 planner latency 只有：

- raw 548ms/scene；
- RAER 553ms；
- V17 476ms；
- G-DALER 480ms；
- scalar 482ms；
- profile 489ms。

500 scenes 的纯 planner 核心大约只需要 4.0–4.6 分钟/arm。fresh log 却遍历 58,418 cache entries，说明大量时间花在 token filter 之前的 cache scan / NPZ deserialization；第一 wave 慢、后续 wave 快还表现出强烈 cache warming 特征。

### 8.2 根因

旧 evaluator 是：

`dataset.iter_samples()` -> NPZ already loaded -> check scenario_token -> skip if not fresh token`。

fresh 500 tokens 来自较大的 val cache，因此每个 arm 都加载大量根本不会评估的 sample。

同时，六个 arm 的 frozen acquisition / selected evidence / EAF frontier 完全相同，但每个 arm 都重新完整 replay。

### 8.3 V19 已落地的 speed changes

1. `PreprocessedBDSEDataset` 支持 `scenario_tokens`，先从 manifest/cache filename identity 解析 requested tokens，再加载 NPZ。
2. 当 manifest 可以解析全部 requested tokens 时，直接走 manifest fast path，跳过递归 full-split scan；legacy/resume cache 无完整 manifest 时仍有 fail-safe disk-union fallback。
3. evaluator 在 dataset 构造时就传入 requested tokens；原 post-load token check 仍保留作为 correctness guard。
4. fresh token 直接从 **cache identity + fixed SHA256** 选择，不再为了选择 500 scene 跑 4000-scene GPU val-discovery replay。
5. V19 重用 V18 已冻结的 TRAIN raw frontier edges，不再跑 3000-scene GPU train raw replay。
6. V19 fresh full replay 从 6 arms 降到 4 arms，只删除已经回答过的历史对照，不删除当前 causal identification 所需对照。
7. V19 scalar/dual dominance heads **一次拟合、同时产出两个 config**，不再重复拟合相同 scalar/profile heads。
8. launcher 新增 stage wall-time provenance，下一轮可以定量确认加速。

以 planner-evaluated scenario 数计算，V18 screen 是至少 `3000 + 4000 + 6*500 = 10000` 次 planner scene evaluations；V19 current screen 只需 `4*500 = 2000`，减少约 80%。更大的加速来自 fresh NPZ prefilter：V18 每个 fresh arm progress 遍历 58,418 cache entries，V19 目标是只 materialize requested 500 samples。实际 wall-time 仍取决于服务器 cache/manifest/文件系统，不能在本地承诺固定倍数，但代码层面已删除主要冗余路径。

没有实现“一个 GPU replay 同时 shadow 4 个 arm”的更激进优化，因为那会把 evaluator 内部逻辑复杂化，增加 paired metric / diagnostic 不一致风险。当前优化优先保持各 arm 独立、数值审计简单。

---

## 9. 数据泄漏与工程审计

V19 runtime tournament 里没有 teacher/J_T/future/ground-truth 输入。teacher margin 只存在于 TRAIN-only fitter target 和 evaluation diagnostics。

数据纪律：

- V18 fresh inspected tokens：500 unique；
- V19 design exclusion：2700 unique；
- V18 fresh 500/500 全部包含在 V19 exclusion；
- audited train tokens：3000 unique；
- train 与 V19 2700 validation design exclusion overlap=0。

fresh V19 token selection只使用 cache scenario-token identity 和固定 SHA256；不加载 NPZ label，不读取 teacher/match/regret/reliability。

还修正了一个 metric 语义隐患：V19 将 `anchor recovery` 与 `direct incumbent replacement` 分开，promotion gate 不允许 anchor-recovery 的好结果掩盖真正 incumbent-contrastive replacement 的失败。

最终 code contract 继续强制：B<=16、M=24、EAF frontier frozen、utility refinement frozen、one-sided/evidence certificate frozen、robust residual corrections frozen zero；如果这些 final-guard semantics 改动而 ICER preselection 没同步，contract fail closed。

---

## 10. 历史 no-repeat 与下一步失败分支

继续禁止：

- BTP / RET / CET / AF / HAP / acquisition / family allocation 重开；
- 增加 B 或 M；
- 放松 evidence / one-sided / safety / structural guard；
- OCFI radius/alpha、EAIR/RAER probability threshold、DACER/ICER threshold、anchor-logit、loss-weight sweep；
- 恢复 V17 utility-equivalence hard mask；
- 在 V19 causal screen 之前 broad-unfreeze EAF；
- 重用任何 2700 个 validation design token；
- 把 V18/V19 design-only diagnostics 放进论文结果表。

V19 失败后只沿断点继续：

- candidate support 回归：工程/candidate semantics audit；
- support AUC 低：support representation generalization 问题，但不改 acquisition；
- direct dominance AUC 低：继续设计 incumbent-contrast evidence representation；
- AUC 高但 direct replacement precision 低：继续处理 reliability extremal false positives / scene-relative robustness，不做 threshold tuning；
- direct recovery 成功但 dual 不优于 scalar：保留 scalar ICER，当前 signed-profile interaction view 被 falsify，不强行 claim per-atom causal gain；
- recovery/preservation 成功但 regret endpoint 失败：此时才允许在同一 frozen frontier 上研究 teacher-improvement magnitude / robust ordering objective；
- 全部通过：冻结 exact config，只进入 independent full-val reproduction。

