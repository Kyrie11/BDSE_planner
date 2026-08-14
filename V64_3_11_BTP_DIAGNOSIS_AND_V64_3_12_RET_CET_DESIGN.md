# V64.3.11 BTP-BDMU 因果诊断与 V64.3.12 RET/CET-BDMU 设计

## 结论摘要

本轮不应把 V64.3.11 的 STOP 解释为“acquisition 已经没有空间”，也不应继续 HAP/BTP 的 surrogate rank 微调。当前最直接的瓶颈是 **BTP 训练目标与 promotion 时的真实 B=16 runtime selector 不同构**：训练优化 `fast pair-margin surrogate`，而 C1-B/C2-B 的因果判断使用 exact runtime pair-conditioned selector。上传 screen 上 current/oracle surrogate-to-exact B-set Jaccard 只有约 0.77，因此训练梯度对 promotion mediator 存在约 23% 的离散集合语义错位。

与此同时，BTP 的 one-sided current-B protection 在当前数据上非常强：selected epoch validation 的 raw negative 中约 74.7% 被保护，train epoch 0--3 更约 84.6%--89.2%。这使得 BTP 即使识别到 exact-oracle 应该进入 B 的 evidence，也经常不能直接形成把当前 transmitted evidence 换出去的可执行 ranking pair。

但是 exact C1-B 仍有明显 headroom：

- anchor C1-B exact oracle capture = 0.415105
- anchor C2-B learned exact capture = 0.380525
- exact B-layer headroom = **0.034580 = 3.458pp**

这远高于 V64.3.11 自己规定的 0.5pp “capacity not binding” 阈值。因此，最值得做的下一步不是 value/frontier 直接改模，而是**最后一次语义严格的 acquisition mediation experiment**：先修正 exact transmission target mismatch（RET control），再仅在 exact oracle-B 明确淘汰当前 B atom 时允许受控交换（CET）。如果 CET 仍不能推动 C2-B，acquisition branch 即可被因果上终止，下一版正式 pivot decisive value/frontier。

---

## 1. 与论文主线的一致性

论文的核心不是“预测得更像完整世界”，而是在 fixed planner-interface budget 下保留能够维持 candidate action ranking 的 auditable evidence。论文把 teacher margin 分解为 dense base + atom-conditioned pair-margin residual，并以预算选择保护 decisive one-sided margins；理论 regret 又分别包含 proposal miss (`epsilon_prop`) 与 budget selection (`epsilon_select`)。

因此当前算法演进应该继续服务：

`fixed planner-interface budget`
→ `auditable evidence atoms`
→ `budget-feasible decisive-margin marginal utility`
→ `budgeted acquisition`
→ `one-sided margin preservation`
→ `final decision preservation`

V64.3.12 不改变 B=16、M、evidence bank、foundation、DARM、DBR 或 runtime planner。它只让 acquisition 的训练 intervention 与真实 B mediator 一致，并把“保护现有 B evidence”从 blanket rule 收紧成可审计的 exact intervention criterion。

---

## 2. V64.3.10 HAP 与 V64.3.11 BTP 对比

### V64.3.10 HAP-BDMU（上一轮）

- C2-M learned Top-M utility capture: 0.47858 → 0.48255，**+0.40pp**
- C1-M/C2-M gap closure: **15.25%**
- proposal decisive recall: 75.37% → 67.37%，**-8.00pp**
- exact-critical B recall: 14.87% → 14.87%，不动
- teacher match: 17.8% → 17.2%，**-0.6pp**
- teacher regret: 20133 → 20378，**+1.22% worse**
- pair-full teacher match: 17.4%，不动

结论：HAP 部分解决 `continuous utility -> hard HAB admission`，但新 admission 没有稳定传到 B=16 / teacher endpoint。

### V64.3.11 BTP-BDMU（当前上传 screen）

Anchor:

- C1-M exact-HAB oracle capture = 0.504656
- C2-M learned capture = 0.478576
- C1-B exact runtime oracle capture = 0.415105
- C2-B exact runtime learned capture = 0.380525
- exact C1-B/C2-B headroom = **3.458pp**
- teacher match = 17.8%
- teacher regret = 20133.34

Selected epoch 1:

- C2-M = 0.481656，**+0.308pp**
- C2-B = 0.378587，**-0.194pp**
- C1-B/C2-B gap closure = **-5.60%**
- teacher match = 18.2%，**+0.4pp**（低于 +0.5pp promotion threshold）
- teacher regret = 20272.51，**0.691% worse**
- proposal decisive recall = 75.325%，基本持平（-0.040pp）
- exact-critical selected recall = 15.506%，+0.633pp
- exact-critical Top-M recall = 23.734%，不变
- pair-full teacher match = 17.4%，不变
- B→Top-M certificate = 91.6%，-1.2pp

更关键的是四个 epoch 的 **exact C2-B 全部下降**：约 -0.250pp、-0.194pp、-0.478pp、-0.288pp。与此同时 C2-M 在前三个 epoch 小幅上升。这说明 BTP 相比 HAP 的确更好地控制了 decisive-support harm，但仍然没有让 learned proposal 的变化沿真实 B=16 interface 传递。

---

## 3. 主要瓶颈：不是“BTP weight 不够”，而是 train/eval interface semantics mismatch

V64.3.11 的算法叙事要求：utility-oracle Top-M 通过 frozen B=16 selector 后，只有实际 transmitted 的 positive 才参与 proposal supervision。

但实际代码的 V64.3.11 主训练路径是：

- `current/oracle budget mask <- _fast_pair_margin_surrogate_masks(...)`
- 只有 `budget_exact_eval and not torch.is_grad_enabled()` 时才调用 `_predicted_pair_certificate_masks(...)`

也就是：

- **train:** fast surrogate B-set
- **validation/promotion:** exact runtime B-set

当前 screen 自己量到：

- anchor current surrogate/exact Jaccard = 0.77490
- anchor oracle surrogate/exact Jaccard = 0.76881
- selected current surrogate/exact Jaccard = 0.77257

这不是微小 numerical approximation；它足以改变离散 B-set 的 transmitted positive/negative identity。因此训练 loss 下降并不保证 exact C2-B 上升。

这也解释当前最反常的结果：**C2-M +0.308pp，但 exact C2-B -0.194pp。**

### 第二瓶颈：blanket current-B protection 过强

BTP 用 one-sided protection 防止 HAP 式 proxy improvement 破坏已有 transmitted support，这个方向本身合理；但实现是 blanket protection：凡 current B 的 displaced negative 都不能被 proposal rank 直接压低。

当前数据中：

- validation anchor protected-negative fraction = 76.73%
- selected epoch = 74.70%
- training epoch 0--3 = 89.21%、87.12%、85.23%、84.61%

因此真正能形成 BTP pair 的 scene/atom 非常稀疏。当前 BTP 的 validation positive fraction 约 3.5%，scene fraction约 6.1%，selected epoch 只有约 2.22 pairs/scene aggregate diagnostic。

关键观察：若一个 `raw_neg = current Top-M \ oracle Top-M` atom 同时属于 exact current B，那么在 exact oracle Top-M intervention 下它必然不属于 oracle B（因为 oracle B 必须是 oracle Top-M 子集）。因此可以用**exact current-B / oracle-B membership change**作为一条严格的预算交换条件，而不是 blanket unprotect。

---

## 4. V64.3.12 实验 A：RET-BDMU

**RET = Runtime-Exact Transmission BDMU**。

它是因果 control，不额外引入交换机制，只修正一个变量：

`fast surrogate B target -> sampled stop-gradient exact runtime B target during training`

### 训练链

1. 保持 V64.3.11 同一个 fixed-reference decisive-margin BDMU utility C0。
2. 通过 exact frozen-family HAB 得到 utility-oracle Top-M。
3. 对有 actionable missed oracle positive 的 scene，训练阶段每 rank 每 step 轮转抽取最多 4 个 scene。
4. 在这些 scene 上调用**与 validation 相同的 exact runtime pair-conditioned B=16 selector**，分别得到 current-B 与 oracle-B。
5. 仅 exact-sampled rows 允许形成 B-transmission ranking target；未采样 row **禁止 silent fallback 到 fast surrogate target**。
6. fast surrogate 只保留做 diagnostic/Jaccard，不参与 RET gradient target。
7. 保持 V64.3.11 的 current-B blanket protection、same-family only、cross-family off、old AF/HAP/listwise off。

### 为什么 sampled exact 而非全部 exact

Exact selector 是离散 CPU/runtime-style mediator，直接对每 batch 全场景运行会显著增加训练开销。RET 只在“oracle Top-M 存在 missed positive”的 actionable scene 上消费 exact calls，并按 optimizer step/rank deterministic rotation。该 exact selection 本身 stop-gradient；梯度仍只穿过 proposal score gap，因此不会出现对离散 selector 求导的伪梯度。

RET 可以单独回答：**V64.3.11 失败是否主要由 surrogate-to-exact target mismatch 导致。**

---

## 5. V64.3.12 实验 B：CET-BDMU（推荐主候选）

**CET = Controlled Exact Transmission BDMU**。

CET 完全继承 RET 的 exact-runtime training target，只改 current-B protection 的语义：

### Slack negative

`current Top-M \ oracle Top-M` 且 **不在 exact current B**：仍可像 BTP 一样被替换。

### Controlled exchange negative

一个当前 transmitted atom 只有同时满足：

1. `e_j ∈ current Top-M \ oracle Top-M`；
2. `e_j ∈ exact current B`；
3. `e_j ∉ exact oracle B`；
4. 与 exact-oracle transmitted positive 属于同一 frozen HAB family；
5. positive 的 decisive-margin utility 更大；

才允许作为 negative 被压低。

因此 CET 不是：

- binary critical bonus；
- certificate maximization；
- broad current-B unfreeze；
- cross-family rank；
- beam/swap/bruteforce selector；
- 增大 B/M。

它是一次 **controlled interface intervention**：只有真实 runtime selector 在 oracle Top-M intervention 下已经明确表明“这个 current-B evidence 应被移出”时，proposal ranking 才允许执行该交换。Controlled exchange pair 的权重默认 0.5，保留 one-sided minimum-intervention bias。

### Novelty 位置

如果 CET 被实验支持，建议把论文算法 novelty 从“新 selector”进一步提升为：

**Controlled Exact Budget Transmission for Auditable Decision Preservation**

它和已有的 controlled interface causal attribution 组合为：

`fixed margin target -> exact hierarchical proposal intervention -> exact budget mediator -> controlled transmitted-evidence exchange -> one-sided margin preservation -> endpoint`

这比重新叠一个 proposal proxy 更容易形成 CCF-A 风格的完整方法论贡献。

---

## 6. 两臂 screen 及强 stop rule

同时跑 RET 与 CET，其他所有变量一致。

### Arm A: RET

只测试 `surrogate -> exact training target alignment`。

### Arm B: CET

相对 RET 只多测试 `blanket current-B protection -> controlled exact B exchange`。

### Promotion / pivot

1. **Instrumentation invalid**：修工程，不解释算法。
2. **C1-B - C2-B < 0.5pp**：acquisition capacity no longer binding，pivot value/frontier。
3. **RET C2-B improves + endpoint improves**：RET 可作为 full 候选；CET 若更差，不为 novelty 强行引入它。
4. **RET fails but CET succeeds**：说明 blanket protection 是剩余 transmission bottleneck；promote CET。
5. **C2-B improves but C3 does not**：acquisition causal mediator 已打通但 endpoint 不响应，立即 pivot decisive value/frontier。
6. **CET 在 valid exact training 下仍无法推动 C2-B**：`exact_acquisition_exhausted=true`，**终止 acquisition branch**，不再设计 V64.3.13 proposal loss。

这条 terminal rule 比 V64.3.11 更强，因为 CET 已经消除了 train/eval selector mismatch，并允许 exact-controlled B-set exchange。失败后再做 acquisition surrogate 缺乏新的可证伪假设。

---

## 7. 历史修改去重

本轮明确没有重复：

- V64.3.4 FPCCA/LBA / AP-WRCCA/LCV frontier representation；
- V64.3.5 CCBR/LEA（CCBR 仅继续作为 representation primitive）；
- V64.3.6 BCHA family-capacity intervention（历史 oracle 已证明 family capacity 非瓶颈）；
- V64.3.7 DARM/DBR value path（本轮完全冻结）；
- V64.3.8/3.9 broad BDMU/AF ranking；
- V64.3.10 HAP feasible-admission/cross-family fallback；
- binary literal/certificate/AOCC bonus；
- selector beam/swap/bruteforce；
- B/M 扩容；
- global proposal/family unfreeze；
- V55/V59 generic action/set potential。

RET/CET 的新增变量是 **exact runtime training mediator alignment + controlled exact B-set exchange**，历史日志中没有做过这两个组合。

---

## 8. 工程落地

主要修改：

- `bdse/model/losses.py`
  - sampled exact-runtime B target during grad-enabled training；
  - exact rows 只来自 actionable scenes；
  - non-sampled rows 不会用 surrogate 冒充 exact target；
  - CET exact-controlled exchange；
  - training/validation exact fraction、candidate fraction、current/oracle B Jaccard、controlled negative/pair fraction diagnostics。
- RET/CET screen/train/CL configs。
- `check_v64_3_12_cet_bdmu_contract.py`：exact HAB/runtime B/`B subset injected Top-M` adversarial contract。
- `check_v64_3_12_ret_cet_bdmu_screen.py`：要求 **train exact target 确实激活**，而不仅是 validation exact。
- `compare_v64_3_12_ret_cet_screens.py`：两臂因果判定和 terminal pivot。
- `test_v64_3_12_cet_bdmu.py`：RET/CET exact exchange 的梯度级 fixture。
- 删除代码包中两个 `.bak` 源文件副本，避免工程歧义；最终打包前清理 `__pycache__/*.pyc`。

工程验证：

- `python -m compileall -q bdse`: PASS
- 新脚本 `bash -n`: PASS
- RET/CET config validator: PASS
- RET/CET exact contract: PASS
- V64.3.9--V64.3.12 targeted regression: 17 PASS
- repository full regression: **328 PASS, 34 warnings**
- warning 均为原有 PyTorch Transformer nested-tensor warning，没有新增 test failure。

---

## 9. 下一轮最应该关注的指标

不要只看 teacher match。优先顺序：

1. `bdmu_budget_projection_exact_fraction`（train 必须 >0；val =1）
2. `bdmu_budget_projection_topm_violation_fraction`（必须 =0）
3. train/val `bdmu_budget_selector_surrogate_jaccard_*`（仅用于量化 V64.3.11 mismatch，不作为优化目标）
4. `anchor_budget_oracle_gap`
5. exact C2-B gain 与 `budget_oracle_gap_closure`
6. `bdmu_budget_current_oracle_jaccard`
7. CET 的 `controlled_exchange_negative_fraction` / `controlled_exchange_pair_fraction`
8. decisive support non-harm
9. C3 teacher match/regret
10. pair-full teacher match（用于判断后续 value/frontier ceiling）

若 CET 成功提高 exact C2-B 但 pair-full / teacher endpoint 不动，下一步 value/frontier 应针对 **decisive rival value representation / pair boundary frontier**, 而不是 generic global potential；这与历史 no-repeat 约束一致。
