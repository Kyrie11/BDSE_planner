# V64.3.48.2 结果复盘与 V64.3.49 SIIR 设计

## 0. 证据范围与论文主线边界

本轮分析严格使用上传的 `bdse.zip`、`logs.zip` 与 `part1.zip`–`part5.zip` 组合结果，以及历史 `大模型建议.md`/`ALGORITHM_CHANGELOG.md`。当前会话上传列表中**没有 `.tex` 文件**，`bdse.zip` 内也没有 `.tex`；因此我没有把未看到的论文正文当作证据。论文主线只按仓库 `PAPER_SYNC_NOTES_V64.md`、历史分析和 changelog 中已经明确的 fixed bounded planner interface / auditable evidence / frozen extremal selector / no-fallback containment 来约束 V49。若后续补传主稿 `.tex`，应再做一次逐节 paper-method consistency audit。

## 1. V48.2 工程可靠性：PASS

V48.2 是 V48 的 provenance-repair rerun，不改变 OCRR 科学机制。此次上传满足：

- 当前包 `V64_3_48_2_SOURCE_MANIFEST.sha256`：900 项，**900/900 PASS**；
- `V64_3_48_OCRR_SCIENCE_LOCK.sha256`：5 个 V48 科学核心文件 **5/5 PASS**；
- server targeted regression：**232/232 PASS**；我对上传代码独立重跑同一 targeted suite：**232/232 PASS**；
- 新 fresh1000 是 label-free 选择，A/B 各 500、互不重叠，并与旧 V48 已消费 fresh1000 无重叠；
- fresh 失败后 launcher 按预注册 STOP，未进入 official closed-loop；
- combined screen 明确记录 `split_A_pass=false`, `split_B_pass=false`, `pass=false`, `next_action=STOP_no_promotion_do_not_pool_A_B_or_tune`。

**判决：V48.2 engineering-valid，可以进行算法归因。** 上一轮 source identity 缺口已经关闭，不需要 V48.3 工程修复。

## 2. 严格按 V47→V48 预注册条件：V48 OCRR promotion failure

### TRAIN design replay（已消费，只用于机制设计，不是 fresh 证据）

| Arm | selected | positive | capture | ΣΔT | catastrophe | no-op false | NegRMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| RSMR | 502 | 221 | 38.50% | +43.294 | 28 | 107 | .3557 |
| EGO-REF | 251 | 136 | 23.69% | +59.533 | 9 | 45 | .2233 |
| SIGN-NOMULT | 411 | 187 | 32.58% | +53.496 | 18 | 78 | .2707 |
| SIGN-MULT/OCRR | 439 | 204 | 35.54% | +62.634 | 14 | 74 | .2299 |

TRAIN 中 SIGN-MULT 只以约 0.038pp 超过 capture floor，因此本来就必须由双 fresh 决定。

### untouched A500

| Arm | selected | positive | capture | ΣΔT | catastrophe | no-op false | NegRMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| RSMR | 108 | 34 | 28.10% | +1.015 | 6 | 18 | .2519 |
| EGO-REF | 48 | 16 | 13.22% | +2.023 | 2 | 8 | .2273 |
| SIGN-NOMULT | 95 | 30 | 24.79% | +2.073 | 4 | 15 | .2103 |
| **SIGN-MULT** | **86** | **27** | **22.31%** | **−1.468** | **5** | **10** | **.2590** |

SIGN-MULT：`existence_and_capture=false`，`hard_tail=false`；direct incumbent opportunity capture 从 preserve 的 32.54% 降到 14.79%，约 **−17.75pp**；endpoint non-inferiority 失败。

### untouched B500

| Arm | selected | positive | capture | ΣΔT | catastrophe | no-op false | NegRMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| RSMR | 109 | 44 | 34.65% | +0.076 | 8 | 23 | .3247 |
| EGO-REF | 57 | 29 | 22.83% | +5.513 | 4 | 8 | .2454 |
| SIGN-NOMULT | 99 | 41 | 32.28% | +3.379 | 6 | 18 | .2417 |
| **SIGN-MULT** | **87** | **39** | **30.71%** | **+3.376** | **6** | **12** | **.2578** |

SIGN-MULT 同样 `existence_and_capture=false`、`hard_tail=false`；direct opportunity capture 比 preserve 低约 **13.25pp**；endpoint non-inferiority 失败。

V48 fresh hard-tail 预注册要求 catastrophe=0。A/B 分别 5/6，因此并非边界性失败。正式判决：

> **V64.3.48 OCRR = double-fresh promotion failure；SIGN-MULT/multiplicity 核心机制被 falsify。**

## 3. V48 真正成功与失败的机制

### 3.1 应保留的不是 `log K`，而是“post-selection functional”问题定义

V46 已证明 ordinary prediction improvement 可以伤害 extremal deployment；V47 又证明 identifiable representation 仍不足以闭合 execution gate。V48 把学习对象从 all-edge signed mean 转到 frozen selected proposal 的 sign-risk，这个**问题重定义仍然成立**。失败的是它的具体可观测 conditioning：`log K`。

### 3.2 `log K` 的 TRAIN 相关性不具 fresh transportability

TRAIN full-selected population 中，good proposal 平均 K≈13.84，bad≈12.08，模型因此学到“小 K 更危险”。但：

- fresh A：good/bad K≈22.50/22.53，几乎无差；`logK` 单独 AUC≈0.498；
- fresh B：good/bad K≈20.84/21.98，方向反转；`logK` AUC≈0.555；
- SIGN-MULT nonpositive-risk AUC：TRAIN 0.6374 → A 0.5378 / B 0.6182；A/B 均没有稳定优于已有 consequence baseline。

因此 in-domain 5-fold 的稳定关联不是 selector-conditioned law 的跨 selection-regime 不变量。

### 3.3 A split 直接给出 multiplicity 的有害 selected-set rotation

A 上 NOMULT 与 MULT：

- common 82：ΣΔT −0.209，4 catastrophes；
- **NOMULT-only 13：+2.283，4 positive，2 material positive，0 catastrophe**；
- **MULT-only 4：−1.258，1 positive，0 material positive，1 catastrophe**。

即本轮最干净的 causal ablation 反而说明 `log K` 在 fresh A 上把 policy 往坏方向旋转。

### 3.4 不是 threshold 可以救

以 fresh 标签做**只用于 postmortem、绝不用于调参**的 oracle 检查：

- A：若把 SIGN-MULT threshold 收紧到 0 catastrophe，只剩 4 selected / 3 positive / capture 3.33%，而 prereg gate 需要至少 32 个 RSMR positives；没有任何 threshold 同时满足机制 gate；
- B：0-cat threshold 只剩 1 selected / 1 positive / capture 1.16%，需要至少 42 positives；同样无可行 threshold。

说明 catastrophe 已被风险模型排在“安全”区域，问题在 **risk ordering / identification**，不是 calibration offset。

## 4. 当前 dominant bottleneck

V48 后，dominant bottleneck 应从泛化的 “operator-conditioned functional” 再收紧为：

> **selection-regime transportability / post-selection outcome-law identification**。

形式上，问题不再是继续找一个 `g(Z,K)`，而是识别在 extremal selector 改变所见竞争集合/selection regime 时仍保持稳定的：

`P(Y_selected <= 0 | consequence state, selection event)`。

V48 证明：直接在 observational full-set selected population 上学习，再把 realized `K` 当 conditioning variable，能得到 TRAIN cross-fit signal，却不产生 fresh-invariant selected-risk ordering。

这比“再加一个 feature”更本质：selected population 本身由 RSMR extremal operator 诱导，**训练 measure 与部署/新数据中的 selection regime 可以变化**。当前要学的是对 selection measure 稳健的 outcome law，而不是更多 future-state representation。

### Secondary residual

fresh 中仍存在 EGO-REF 自身也把 catastrophe 估成正值的样本，说明 representation 并非数学上已经完美。但按照 V47/V48 已预注册的停止纪律，当前不能借此重新开启 future feature expansion。若 V49 也失败，应该关闭当前 offline selected-risk family，转向真实 on-policy/closed-loop/interventional selected-outcome evidence，而不是 V50 再加 observable。

## 5. 模型层成熟度（V48.2 后）

| 层 | 状态 | V49 策略 |
|---|---|---|
| B16/M24 bounded interface | 成熟 | 永久冻结 |
| EAF complete frontier / exact attribution | 成熟、paper backbone | 冻结 |
| support/admissibility/evidence visibility | 成熟 | 冻结 |
| RSMR ordinal extremal selection | 最成熟 learned layer | **永久冻结** |
| incumbent / no fallback containment | 成熟 | 永久冻结 |
| EPV / QUALITY | 真实 partial mediator | 保留 |
| V44 ungated full-horizon occupancy | 强成功 | 保留 |
| V45 agent-local longitudinal response | 已识别且有 selected-value 证据 | 保留 |
| V47 EGO-REF | 强 supporting consequence mediator | 保留，不 headline promotion |
| V46 variance / handcrafted temporal profile | 已关闭 | 禁止恢复 |
| V47 AGENT-2D constant drift | 已关闭 | 禁止恢复 |
| V48 multiplicity/logK | **fresh falsified** | **关闭** |
| observational selected sign-risk | 有 TRAIN signal、fresh 不充分 | 作为 V49 control |
| selection-regime invariant risk identification | 未成熟 | **V49 唯一主目标** |
| absolute zero / material recall | 未成熟 | 最终 deployment gate |

## 6. V64.3.49：EAF-ICER-SIIR

**Selection-Interventional Invariant Retention**。

### 6.1 核心思想

V49 不改变 runtime selector、不新增 future observable，并**移除风险律对 K 的依赖**。为保持 V48 science-locked runtime byte-identical，旧 API 仍会收到 selection multiplicity，但 `use_extremal_multiplicity=False` 会在风险计算前把第 4 坐标强制置 0；测试已验证 K=2 与 K=999 输出完全相同。因此 K 在 V49 retention law 中没有任何数值作用。V49 真正改变的只有风险律的**识别分布**。

TRAIN 时，对每个 scene 的 admissible candidate IDs 做固定 seed 的 label-free hash permutation，并从 `1..K` 中由 hash 固定得到一个 prefix 长度；只把这个 prefix 暴露给**已经冻结的 RSMR**，得到一个 selection-interventional winner。整个 selector API 只接收 action id、RSMR score/support/margin/utility prior，不接收 teacher outcome。

在这些 interventional selected events 上仍拟合 V48 完全同构的：

- 状态：`[Q, P-Q, E-P]`；
- zero-bias pairwise sign-risk；
- `lambda=1`；
- 不 class-weight、不 focal、不 catastrophe reweight；
- threshold 仍由原 capture tolerance 推导，并在 full-set selected positives calibration fold 上 split-conformal 得到。

部署时完全回到 **full candidate set + frozen RSMR**，V49 只能 retain 同一个 winner 或 incumbent。

### 6.2 为什么这不是 top-K / candidate-count sweep

prefix intervention **只存在于 TRAIN identification**：每个 scene 只有一个由 frozen seed 决定的 prefix，既不看标签，也不根据结果选择 prefix；runtime candidate bank 一项不删。其作用是主动改变 selection measure，打破 observational full-set winner 与 realized competition regime 的偶然关联，而不是寻找一个更好的 candidate budget。

### 6.3 唯一 causal ablation

1. **OBS-SIGN**：exact replay V48 SIGN-NOMULT；
2. **SIIR**：状态、loss、lambda、calibration、runtime path 全同；**唯一差异是 TRAIN risk-fit population 来自 label-free selection intervention**。

代码强制 OBS-SIGN aggregate 和 AUC 精确 replay V48，否则 engineering STOP。

### 6.4 独立 identification gate

SIIR 必须在**真实 full-set OOF RSMR winners**上：

- aggregate nonpositive-risk AUC > OBS-SIGN；
- aggregate AUC > `-EGO-REF value` baseline；
- 对 OBS-SIGN 至少 4/5 folds 更优；
- 对 EGO baseline 至少 4/5 folds 更优；
- 同时 selected-policy deployment gate 必须通过。

held-out intervention seed 的 AUC 只作诊断，不作为调参/晋级 gate。

### 6.5 fresh preregistration

只有 nested TRAIN 全部通过才生成新 untouched A500+B500，fresh seed：

`v64.3.49-eaf-icer-siir-double-fresh-v1`

fresh exclusion 同时包含 design/TRAIN、原 V48 已消费 1000、V48.2 已消费 1000。A/B 禁止 pooling。

每个 split 继续使用原 hard gate：same-winner containment、capture/no-op、**0 catastrophe**、NegRMS/aggregate、direct incumbent opportunity capture `>= preserve +3pp`、endpoint non-inferiority。

### 6.6 preregistered STOP

- nested TRAIN fail：**不消费 fresh，关闭当前 offline selected-risk family**；
- TRAIN pass 但 A 或 B 任一 fail：scientific STOP，关闭 family；
- 禁止从失败结果调 intervention seed、prefix law、pairwise loss、lambda、threshold、class/catastrophe weight、Q/P/E feature、K transform；
- 禁止恢复 V46/V47 closed branches、CVaR、translation、catastrophe veto、rerank、second-best、candidate-count/top-K sweep；
- 如果 SIIR fail，下一步需要 on-policy/closed-loop/interventional selected-outcome evidence，而不是 V50 offline trick。

## 7. CCF-A 论文主线

本轮后更强的主线不是 “OCRR/SIIR 又一个分类器”，而是：

> **Selection–Valuation–Transport Sufficiency under a Bounded Auditable Planner Interface**

证据链可以形成：

1. bounded EAF + RSMR 解决 ordinal extremal selection；
2. endpoint/current/prospective response 逐步提高 absolute consequence sufficiency；
3. V46 给出 `prediction sufficiency != decision sufficiency` 的直接反例；
4. V47 给出 `representation identifiability != deployment sufficiency`；
5. **V48 给出 `in-domain post-selection identification != cross-regime transport sufficiency`**；
6. V49 用对算法 selection operator 的 label-free intervention，检验能否识别 transportable retention law，同时保持 runtime interface 和 winner completely frozen。

如果 V49 fresh 成功，这可以形成论文级机制贡献；如果失败，也应把负结果用于证明 offline observational selected-risk 的边界，并转向真实 interventional evaluation，而不是继续堆 feature。

## 8. 工程交付状态

V49 当前只做了**代码/协议工程验证**，没有伪造 scientific TRAIN 结果，因为当前上传不含运行 V49 所需的完整服务器 V44/V47 历史输出/cache/checkpoint。

已验证：

- Python compile：PASS；
- launcher `bash -n`：PASS；
- V48 + V48.2 provenance + V49 focused：17/17 PASS；
- V13→V49 targeted：242/242 PASS；
- full repository：125 test files，四分片共 **579/579 PASS**；
- V48 science-locked `tournament.py` 保持 byte-identical；V49 runtime 复用 locked V48 NOMULT path，不修改成熟 selector/runtime composition。

因此下一步应直接运行 V49 launcher；它会先重放 V48.2 失败签名、source manifest、历史测试和 nested TRAIN gate。若 TRAIN fail，脚本在 fresh 选择前终止。
