# V64.3 结果审计与 V64.3.1 优化报告

## 结论摘要

这轮最重要的结论不是“V64.3 的 AP-WCCA 失败”，而是：**上传的实验没有真正训练 V64.3 AP-WCCA。** 输出目录虽然名为 `outputs_v64_3_cc_aocc_apwcca_fast_2gpu_v1`，但 query-path provenance 实际选择了 `v64_2_saqa_bcc_hcbe_train_2gpu.yaml`；训练日志也是 V64.2 的 10 epoch/trainable modules，V64.3 的 `critical_proposal_adapter` 在全部 epoch 中严格为 0。因此该结果只能解释为“V64.2 权重经过 V64.3 calibration/evaluation 接口后的表现”，不能用于评价 AP-WCCA 的有效性。

历史 gate report 确实给出 Protocol PASS / Minimum FAIL / Competitive FAIL；但增加算法激活审计以后，该 run 对“V64.3 AP-WCCA 是否有效”应视为 protocol-invalid，因为声明为 trainable 且 zero-init 的 AP-WCCA 分支从未离开 0。

Minimum 的低证书率已经不能再归因于 V64.2 的 calibration beta mismatch：本轮 beta 合同一致。真正异常是 **B16 对 full-TopM exact downstream winner 的保留率约 0.901，而旧 AOCC pairwise certificate 只有 0.066**。这表明当前 Minimum 的主要障碍是 certificate semantics 与真实 deployment decision operator 错配，而不是 B=16 本身容量不足。

Competitive 仍有真实算法风险：teacher match candidate 0.224 vs foundation 0.225，proposal decisive recall 0.751 vs foundation ~0.804，teacher-critical Top-M 0.3548 没有突破历史平台，selected critical 0.2726，residual calibration epsilon 1.2259 且 deployed flip=0。但由于 AP-WCCA 根本没训练，这些结果不能用来否定 AP-WCCA；最优先实验应该先验证 AP-WCCA 是否真正激活并改善 acquisition。

V64.3.1 因此只做两条彼此正交、由现有证据直接支持的修改：

1. **Provenance-correct AP-WCCA**：严格阻止 V64.2 config/checkpoint 串线，先用 12k/4-epoch activation screen 低成本验证 AP-WCCA；
2. **DA-EPC (Decision-Aligned Exact Preservation Certificate)**：AOCC/HAB 仍按原规则选 B=16，不做组合搜索/repair；证书改为对已经选出的 B16 运行同一个 residual-disabled exact downstream evaluator，并检查 winner 是否与 full-TopM winner 完全相同。旧 AOCC one-sided pair bound保留为 robustness diagnostic。

这两个修改都不增加 B/M，不改变 evidence atom，不改变 literal winner-flip criticality，也不重新走 V40--V43 的高代价 DACC/beam/swap 搜索路线。

---

## 1. 上传 V64.3 的实验完整性审计

### 1.1 实际训练配置不是 V64.3

上传输出中的 `provenance/v64_query_path_selection.json` 显示 selected train config 为：

```text
bdse/configs/v64_2_saqa_bcc_hcbe_train_2gpu.yaml
```

训练日志也与 V64.2 完全一致：

- 10 epochs，而 intended V64.3 为 8 epochs；
- trainable modules 包含 `query_extension_proj`、`proposal_head`、family modules 等 V64.2 分支；
- intended AP-WCCA 应冻结这些旧模块，只训练 `critical_proposal_adapter` + residual heads；
- `critical_proposal_residual_abs_mean == 0`、`critical_proposal_residual_rms == 0` throughout training。

因此 AP-WCCA 没有实际参与优化。

### 1.2 为什么后续看起来又像 V64.3

旧 `training_complete()` 只检查 checkpoint/log 是否存在及大致 mtime freshness，没有绑定：

- train-config SHA-256；
- foundation checkpoint SHA-256；
- best checkpoint SHA-256；
- algorithm family。

所以同一个 OUT_ROOT 里第一次错误训练出 V64.2 checkpoint 后，后续 wrapper 即使已经换成 V64.3 config，也可能直接把旧 checkpoint 当作“training already complete”。与此同时新的 config-contract JSON 会被重新写成 V64.3 PASS，造成“provenance 文件看起来新，但权重是旧算法”的假象。

### 1.3 open-loop 本身是不是旧文件？

不是。上传 artifacts 的 timestamp 显示 candidate/local/foundation shards 在 22:32--22:41 重新生成，suite report 在 22:41 完成。因此当前 0.184/0.224 等指标来自**fresh V64.3 eval config + stale V64.2-trained candidate checkpoint**，而不是直接复用了旧 metrics.json。

V64.3.1 仍进一步把 calibration/open-loop reuse 也升级为 SHA provenance，避免未来只靠 mtime 判断。

---

## 2. Gate 结果的正确解释

### 2.1 历史 report

上传正式 report：

```text
Protocol     PASS
Minimum      FAIL
Competitive  FAIL
```

Minimum failures：

```text
evidence_certified_fraction = 0.066 < 0.40
fallback_rate               = 0.934 > 0.60
```

Competitive failures核心为：

```text
candidate teacher match       = 0.224
foundation teacher match      = 0.225
teacher-match gain            = -0.001
proposal decisive recall      = 0.75084 < 0.80
critical Top-M recall         = 0.35476 < 0.80
critical selected recall      = 0.27260 < 0.50
residual deployed flips       = 0
residual calibration epsilon  = 1.22588
certificate                   = 0.066
fallback                      = 0.934
```

### 2.2 Corrected algorithm-integrity Protocol

V64.3.1 gate checker新增一个训练健康条件：

> 如果配置声明 zero-init `critical_proposal_adapter` 是 trainable，但完整训练日志里 adapter 始终 exact zero，则不能把该 run 当作 AP-WCCA 的有效算法实验。

因此重新审计上传结果时：

```text
Protocol = FAIL for AP-WCCA attribution
reason   = critical_proposal_adapter never moved from exact zero
```

这不是说场景 row identity/query contract 等旧 Protocol 子项失败；而是说该结果**无法建立“V64.3 算法被实际执行”的因果归因**。

---

## 3. Minimum gate 的根本原因

### 3.1 这轮不是 beta mismatch

V64.2 曾存在 calibration `beta=0`、deployment `beta=1` 的 contract error。V64.3 已经把 calibration/deployment beta 对齐，因此本轮 0.066 certificate 不应继续归因于旧 bug。

### 3.2 B16 实际保留 winner 的能力远高于 certificate

上传 1000-scene candidate rows：

```text
B16 vs pair-full exact winner preservation = 0.901
AOCC evidence certificate                 = 0.066
certificate/preservation gap              = 0.835
```

旧 certificate 与 exact preservation 的交叉表：

```text
certified + preserved       = 39
certified + not preserved   = 27
uncertified + preserved     = 862
uncertified + not preserved = 72
```

也就是说：

- 92.3% 的“旧证书不通过”场景实际上仍保留 full-TopM winner；
- 旧 pairwise-certified 场景中只有 59.1% 与 exact winner preservation 一致；
- teacher-critical scenes 上 B16 exact preservation 仍约 90.5%。

因此 Minimum 当前首先暴露的是**certificate formulation 与实际 deployment operator 不一致**，不是“16 个 evidence atom 不够”。

### 3.3 为什么旧 AOCC certificate 会错配

旧 AOCC certificate 基于 capped target-rival one-sided pair-margin deficits；最终 deployed planner 则使用：

- evidence-action-potential aggregation；
- final rival graph；
- normalized pair margins；
- hard safety/all-flagged guard；
- utility refinement。

更关键的是，本轮 `selector_aocc_full_target_certified_pair_fraction = 0.0`：连 full Top-M target 本身都几乎从不满足这个 pairwise surrogate。这说明它不适合作为当前 downstream winner sufficiency 的主 certificate。

AOCC 的平均：

```text
initial deficit   = 0.019389
deficit reduction = 0.000744
final deficit     = 0.018695
```

B16 只能减少很小一部分 surrogate deficit，但这并不等价于 winner 丢失，因为 exact winner preservation 已经达到 0.901。

### 3.4 V64.3.1：DA-EPC

DA-EPC 做的是：

1. AOCC/HAB 按原算法得到 B16；
2. 用 already-queried Top-M outputs，通过 residual-disabled exact downstream evaluator 得到 full-TopM winner；
3. 对 B16 运行**同一个** evaluator；
4. winner 完全相同才 certificate=1。

它是 exact post-selection audit，不是搜索算法：

- 不删除/交换/beam search；
- 不增加 B；
- 不增加 M；
- 不额外进行 neural evidence query；
- 不改变 selected atom identity；
- 旧 AOCC pair bound仍单独导出用于 robustness analysis。

从论文逻辑上，它和 literal winner-flip criticality 使用同一种 decision semantics：一个回答“移除 atom 是否翻转 winner”，另一个回答“压缩到 B 后 winner 是否仍与 full interface 相同”。

需要注意：由于 certificate semantics 发生了升级，论文和实验报告中必须明确同时报告 DA-EPC 与 legacy AOCC robustness bound，不能把二者混为同一个历史指标。

---

## 4. Competitive gate：哪些是真算法问题，哪些目前不能下结论

### 4.1 AP-WCCA 的效果：**本轮无法评价**

上一版 V64.3 的主要 acquisition 修改是：冻结 legacy HAB proposal，用 zero-init winner-conditioned critical residual adapter学习 literal critical evidence。

这次 adapter 从未训练。因此：

- 不能说 AP-WCCA 没提升 critical recall；
- 也不能说它成功；
- 继续立刻换 proposal 算法会把工程串线误判成算法失败，重复无效版本迭代。

### 4.2 当前 stale candidate 暴露的 acquisition 风险仍然真实

虽然不能归因到 AP-WCCA，但当前权重状态确实不好：

```text
foundation proposal decisive recall ≈ 0.8036
candidate  proposal decisive recall ≈ 0.7508
foundation critical Top-M recall    ≈ 0.3548
candidate  critical Top-M recall    ≈ 0.3548
candidate  critical selected recall ≈ 0.2726
```

也就是说旧 V64.2-style fine-tuning破坏了 broad proposal coverage，却没有突破 critical acquisition 平台。这正是“冻结强 legacy proposal + 只学 critical residual”设计 AP-WCCA 的原因。

### 4.3 当前 candidate 没有提供 teacher-action gain

```text
candidate teacher match  = 0.224
local teacher match      = 0.224
foundation teacher match = 0.225
pair-full candidate      = 0.236
pair-full local          = 0.236
```

所以当前部署 residual 没有贡献，selector/压缩之后也没有把 pair-full 的潜在信息转成最终 teacher gain。

### 4.4 Residual 仍没有学到可部署 correction

```text
raw residual proposal rate   = 0.089
deployed residual flip rate  = 0
calibrated residual epsilon  = 1.225876
training epsilon reserve     = 0.15
beneficial deployed flips    = 0
harmful deployed flips       = 0
```

这里不能通过手工降低 epsilon 来“制造 flip”。epsilon 很大是 residual raw error 本身大的结果。后续只有在 provenance-correct AP-WCCA/full training 后重新看 residual MAE、winner margin、sigma 与 proposal quality，才能判断是 target、head capacity 还是 routing 问题。

### 4.5 当前最可能的算法因果链

建议继续严格按以下顺序定位：

1. **Acquisition**：literal critical atom 能不能进入 Top-M？
2. **Budget selection**：进入 Top-M 后 B16 是否继续保留？
3. **Atom→action value**：critical evidence 被保留后，是否改变 action ranking 的正确方向？
4. **Residual**：是否有 calibrated、net-beneficial winner correction？
5. **Closed-loop dynamics**：open-loop 变好后是否能转成真实 replanning/safety/progress 改善？

当前第一步 AP-WCCA 尚未被有效实验，因此不应跳到第四、第五步大改算法。

---

## 5. 当前模型“学到了什么 / 没学到什么”

严格说，本轮 candidate 是旧 V64.2-style weights，不是 intended V64.3 model。但从现有可信指标可以看到：

### 学到/保住的内容

- query/base protocol contract 基本稳定：raw query/score/base contract 通过，decision match约 0.999；
- fixed B contract严格满足：retained budget pass=1；
- HAB Top-M 对 candidate 自己 dense winner 的保存仍高：dense→HAB Top-M ≈0.969；
- B16 对 candidate dense winner preservation ≈0.943；
- B16 对 exact pair-full downstream winner preservation ≈0.901；
- broad effective decisive recall仍有约0.755。

### 没学到/没有证据证明学到的内容

- teacher literal critical acquisition没有突破 0.355 plateau；
- AP-WCCA winner-conditioned residual根本没训练；
- deployed residual correction没有一次动作翻转；
- candidate teacher-match没有优于 foundation；
- paired regret没有建立稳定优势；
- 当前 certificate surrogate没有表达真实 winner sufficiency。

---

## 6. 最应该继续修的算法方向

### 第一优先级：先真正测试 AP-WCCA，而不是马上换算法

这是目前信息价值最高、成本最低的动作。

V64.3.1 新增 activation screen：

- train 12k scenes；
- 4 epochs；
- 2 GPUs；
- batch 16/GPU；
- val_tune 500 every epoch；
- 不跑 calibration/open-loop/CL/test。

至少要求：

```text
critical adapter RMS > 0
critical Top-M recall > ~0.355
proposal decisive recall >= ~0.78
```

这些只是筛选条件，不是 publication gate。

如果 adapter 激活但 critical Top-M 不涨，才说明 winner-only conditioning/adapter capacity 可能不足。此时最合理的下一方向是把 critical acquisition 条件从“winner-conditioned”进一步提升为 **winner–rival boundary conditioned**：显式编码当前 winner 与最危险 rival/near-boundary rival，而不是继续堆 BCE/HCBE 权重。因为 literal criticality天然是 decision-boundary-relative 的。

### 第二优先级：DA-EPC 修正 certificate semantics

这直接解决 Minimum 的主要错配，同时强化论文叙事：

> under fixed budget，证书审计的是“压缩后的 exact downstream winner 是否与 full planner-interface winner 一致”；criticality审计的是“移除单个 evidence 是否翻转 winner”。

两者都基于 literal winner semantics，而不是 margin surrogate。

### 第三优先级：只有 AP-WCCA 真正改善 Top-M/selected critical 后，才转 atom→action value

如果：

```text
critical Top-M ↑
critical selected ↑
teacher match / regret 不升
```

则 evidence 找对了，但 value mapping不对。下一版应该研究 atom-action potential / pair margin value learning，而不是再改 proposal。

### 第四优先级：open-loop 好而 CL 不好时再转 dynamics/replanning

不要在当前阶段提前改 candidate dynamics；否则 acquisition/value/dynamics 三个变量同时改变，无法归因。

---

## 7. 是否应该 gate 不通过也跑 CL20？

建议，但规则仍是：

- **Corrected Protocol FAIL：不跑。** 当前上传 run 因算法分支未训练，不值得作为 AP-WCCA diagnostic CL。
- **Corrected Protocol PASS，Minimum/Competitive FAIL：建议跑 paired diagnostic CL20。**

原因：如果真正的 V64.3.1 open-loop acquisition/value有所提升，但 formal gate某项仍差，CL20可以判断瓶颈是否已经转移到：

- candidate availability/dynamics；
- reactive interaction；
- replan state/cache；
- structural post-processing；
- action changes是否只发生在不重要场景。

CL20 必须标记 diagnostic only，不能绕过 gate或用于最终 SOTA claim。

---

## 8. 为什么实验这么慢？test set 是不是原因？

### 8.1 这轮 test set 根本没有参与主流程

主 pipeline实际使用：

- training：`bdse_train_v2`；
- gate/tuning：`val_tune` 1000 scenes；
- calibration：`val_calib`；
- `bdse_test_2` 只被 export/reserved，当前正式主指令没有跑 test。

因此本轮慢**不是 test set 大造成的**。最终模型冻结以后跑完整 test 时，test 数量当然会决定最终评测时间，但不应该影响现在的训练/调参迭代。

### 8.2 真正瓶颈是训练输入供给

上传训练 10 epochs累计：

```text
train wall        19702.8 s = 5.47 h
DataLoader wait   15195.8 s = 77.1%
loss build         1775.3 s =  9.0%
forward             435.5 s =  2.2%
backward            457.4 s =  2.3%
pair sampling       525.9 s =  2.7%
H2D                   10.7 s =  0.05%
```

平均每 step DataLoader wait约972 ms。GPU forward/backward不是主要瓶颈；单纯增加 GPU算力不会解决。

### 8.3 其他阶段

从上传 timestamps估计：

- foundation 1000-scene quality replay：约55 min，原实现基本串行；
- calibration：约30 min；
- 三路并发 open-loop：约12 min；
- training：约5.47 h，绝对主项。

### 8.4 V64.3.1 加速措施

1. **2 GPU DDP + batch 16/GPU** 保留；
2. 新增真实 storage microbenchmark，在服务器上比较 8/12/16 workers，而不是盲目加 worker；
3. 默认 12 workers/GPU、prefetch 2，但最终以 benchmark 为准；
4. intended AP-WCCA full run只有8 epochs，不再误跑 V64.2 的10 epochs；
5. 先跑12k/4-epoch activation screen，避免错误算法直接花5小时训练；
6. foundation 1000-scene replay改为2 GPUs × 2 workers/GPU；
7. calibration改成4 deterministic shards（2 workers/GPU）后 exact merge；
8. open-loop维持2 workers/GPU，因为它只有约12 min，不值得为了少量时间改变评测语义；
9. query-prefix cache audit当前严重失败，因此不强行走错误 cache 快路径；正确性优先。之后若希望继续压缩训练时间，应单独重建/验证与当前 runtime完全一致的12-D prefix cache，而不是复用旧 cache。

---

## 9. 工程问题修复清单

### 已修

- V64.3 wrapper不再接受历史 generic `MAIN_CONFIG/SPEED_CONFIG` 注入；
- train/eval config使用严格 V64.3.1 algorithm-family validator；
- validator拒绝 V64.2 proposal/family/query-extension trainables；
- checkpoint reuse绑定 train-config SHA、foundation SHA、best checkpoint SHA；
- gate检查 zero-init trainable AP-WCCA adapter必须实际离开0；
- calibration raw shard写入 config/checkpoint SHA；
- calibration merge拒绝跨 checkpoint/config混合 shard；
- merged calibration写入 source SHA；
- open-loop suite写入 candidate/local/foundation config/checkpoint SHA；
- open-loop reuse同时检查 val_tune manifest freshness；
- test set不进入当前选择/调参流程；
- DA-EPC不改变 selection，仅改变 exact certificate audit；
- legacy AOCC certificate作为独立 robustness diagnostic保留。

### 仍需服务器上实测的工程项

- 8/12/16 DataLoader workers在 `/data0` 实际 I/O拓扑上的最佳点；
- 2 workers/GPU calibration是否受显存/磁盘并发限制；
- 2 workers/GPU foundation replay是否保持期望线性加速；
- V64.3.1 fresh checkpoint的 query-prefix audit是否仍只能 runtime recompute。

---

## 10. 推荐下一步实验顺序

### Phase 0 — Input benchmark

只测 train cache I/O + tensorization，自动选择 workers。

### Phase 1 — **最有价值：AP-WCCA activation screen**

只回答一个问题：**真正的 V64.3 AP-WCCA 有没有把 critical evidence 拉进 Top-M，同时保住 legacy proposal？**

如果失败，停止，不做 calibration/CL。

### Phase 2 — Full V64.3.1

前提是 screen通过；用50k train、8 epoch、2GPU、batch16/GPU。

重点依次看：

1. adapter activation；
2. proposal decisive；
3. critical Top-M；
4. critical selected；
5. DA-EPC certificate/fallback；
6. teacher-match / regret；
7. residual MAE/epsilon/flip。

### Phase 3 — Diagnostic paired CL20

corrected Protocol PASS 后，即使 Minimum/Competitive仍失败也可以跑 diagnostic CL20。

### Phase 4 — 最终 test

只有算法、checkpoint、hyperparameters、CL policy全部冻结以后才运行完整 `bdse_test_2`。不要用未完成的大 test set参与当前版本选择。

---

## 11. CCF-A 投稿方向

V64.3.1 比继续堆经验性 loss 更有利于论文主线收敛，因为它把核心概念统一到同一个可审计 decision semantics：

- fixed planner-interface evidence budget B；
- auditable evidence atoms；
- deterministic/exact execution of the budgeted selector；
- literal winner-flip criticality；
- literal full-interface → B-interface winner preservation certificate。

下一步最重要的论文级证据不是再造一个版本名，而是得到一条干净因果链：

> AP-WCCA确实提升 winner-critical acquisition → B16 exact preservation仍高 → teacher/open-loop benefit出现 → paired CL benefit出现。

如果这条链在某一步断掉，再只修改对应模块。这样既避免重复历史失败，也最有利于建立 CCF-A reviewer 能理解和验证的机制性贡献。
