# V64.3.9 AF-BDMU 审计、因果结论与下一轮设计

## 1. 论文主线审计

论文当前最有价值的命题不是“decision-aware selection”，而是：**在固定 planner-interface query budget 下，只暴露可审计的 evidence atoms，并通过保留 full-information teacher 的 decisive action margins 来保留最终决策。** 论文中的 evidence atom、HAB、pair-conditioned margin、one-sided lower-confidence margin 与 candidate-set preservation theorem 已经形成一条可以持续收紧的理论链。

建议论文主 claim 固定为 **Auditable Budgeted Decision Preservation**，算法链固定为：

`fixed planner-interface budget -> auditable evidence atoms -> budget-feasible decisive-margin marginal utility -> budgeted acquisition -> DARM one-sided decisive-margin preservation -> final decision preservation`。

后续算法可以在这条链内部升级，但不要重新扩张成泛化的“decision-aware representation/selection”。

## 2. 当前 full pipeline 是否完成

### 2.1 DARM+DBR full training 本身完成了

本轮上传的 V64.3.7 DARM+DBR-LITERAL full 训练日志存在 epoch -1 与 epoch 0--5；训练日志明确记录 early stopping，并记录 final model save。上传 zip 没有包含 `.pt` 权重，因此无法从压缩包本身恢复 checkpoint，但不能据此认定原始训练没有保存 checkpoint。

### 2.2 整个 V64.3.8 BDMU promotion sequence 没有完成

旧 `NEXT_COMMANDS_V64_3_8_BDMU.txt` 在 Phase 0 后调用旧 full audit，并要求 `full_promotion=true` 才能启动 BDMU。旧 checker 把 `anchor teacher match >= 0.24` 当成 instrumentation gate。full-val 1000 scene 的 anchor 只有 0.180，因此 sequence 在 BDMU 之前停止，test/closed-loop 也没有进入。

这不是算法训练失败，而是 **promotion protocol 把旧 first-500 screen 的绝对性能阈值误当成跨验证组成的工程/接口 contract**。

### 2.3 full gate 修复后的结论

V64.3.7.2 对 FULL variant 使用 zero-residual interface consistency：epoch -1 的 pair-full 与 selected-local 必须一致，而不是要求绝对 teacher score 达到旧 screen 的 0.24。

在上传 full log 上重审计：

| 指标 | anchor | selected epoch 1 | 变化 |
|---|---:|---:|---:|
| teacher match | 0.180 | 0.198 | +1.8pp |
| pair-full match | 0.181 | 0.198 | +1.7pp |
| selected-local match | 0.181 | 0.181 | 0 |
| teacher regret | 19759.44 | 16496.54 | -3262.91 |
| pair-full regret | 19545.80 | 16461.69 | -3084.11 |
| exact critical Top-M micro | 0.4279 | 0.4279 | 0 |
| exact critical selected micro | 0.3076 | 0.3076 | 0 |
| proposal decisive recall | 0.8005 | 0.8005 | 0 |

residual intervention beneficial/harmful = 0.019/0.002，净 +1.7pp。修复后的 full audit 为 `meaningful_value_gain=true, deployment_gain=true, full_promotion=true`。

因此 DARM+DBR 在 **acquisition 完全冻结** 时单独提高 teacher decision 的结论，在 full 50k/1k 规模上得到进一步支持。

## 3. 为什么旧的 full gate 会误判

selected checkpoint 的 validation 前 500 与后 500 差异极大：

- first 500: teacher/pair-full/local ≈ 0.284/0.286/0.264；
- second 500: ≈ 0.112/0.110/0.098。

而 second 500 的 exact critical Top-M recall 反而更高。这说明 validation order/composition 存在明显难度漂移，绝对 teacher-match floor 不能承担 instrumentation contract。

V64.3.9 的 500-scene screen 因此改用 `uniform_blocks`：覆盖完整 validation order，同时保留短连续块以维持 NPZ I/O locality。

## 4. 当前主要瓶颈到底在哪里

不能再用“只有 acquisition 一个瓶颈”描述当前系统。应分成两个层级。

### 4.1 全局 end-to-end ceiling：value / foundation action ranking

selected checkpoint 的 pair-full teacher match 只有 0.198。即使 acquisition 给 pair pathway 完整 evidence，teacher action 仍有约 80% 不一致。因此从系统绝对 ceiling 看，**value/model alignment 是更大的长期瓶颈**。

### 4.2 当前冻结 value 的可归因下一步：Top-M acquisition ranking

在“不改 DARM/DBR，只测试 acquisition”这一轮里，最明确的可控 bottleneck 仍在 acquisition，但位置已经收紧到 **proposal Top-M decisive-utility ranking**，不是 family allocation，也不是 B=16 内部 certificate formation。

关键证据：

- exact winner-flip critical Top-M recall: 0.4279 micro / 0.3838 macro；
- selected critical recall: 0.3076 micro / 0.2777 macro；
- proposal decisive recall: 0.8005；
- frozen-family-slot oracle Top-M critical recall: 1.000；
- global oracle Top-M critical recall: 1.000。

也就是说，现有 family slots 足以容纳 decisive atoms，候选 proposal 也并非完全看不到它们，但 learned proposal score 没把它们可靠地排入 Top-M。

### 4.3 certificate 不是当前第一瓶颈

必须区分：

- raw pairwise AOCC certified fraction = 0.031；
- exact DA-EPC evidence certificate = 0.946。

代码中 exact DA-EPC 是在同一个 downstream operator 上比较 **B=16 selected set 的 winner 与 full Top-M target winner 是否完全相同**；raw AOCC 是保守的一侧 pair-margin surrogate。

因此不能因为 3.1% 就得出“certificate utilization 很差”的结论。当前 exact downstream preservation 已经很高，主要损失发生在 Top-M 之前。

## 5. 当前结果能提供哪些因果证据

### 证据 A：DARM+DBR downstream value/aggregation 有独立因果作用

acquisition exact metrics 全部严格不变，但 teacher match +1.8pp、pair-full +1.7pp、teacher regret 显著下降；selected-local 保持不变，DBR residual intervention 净收益为正。这是当前最强的机制归因。

### 证据 B：旧 full promotion failure 是 protocol failure，不是算法 failure

把 FULL 的绝对 0.24 anchor floor 替换为 zero-residual interface consistency 后，同一份训练日志在其他因果 gate 不变的条件下通过。这证明旧 stop 不能作为 DARM 失败证据。

### 证据 C：family admission 已基本排除为当前 acquisition bottleneck

frozen-family oracle 与 global oracle 都达到 1.0，而 learned Top-M 只有 0.4279 micro；因此继续做 BCHA / family quota intervention 没有依据。

### 证据 D：当前 selected-B certificate 已非主要瓶颈

exact DA-EPC 为 0.946。应优先提高进入 Top-M 的 decisive utility，而不是围绕 raw AOCC 3.1% 加 binary certificate bonus。

### 证据 E：acquisition 仍不是全局唯一 ceiling

pair-full teacher match 仍只有 0.198，且 validation second-half teacher match 很低但 critical recall 更高。因此如果 acquisition target 被明显优化而 teacher endpoint 不再响应，下一轮必须转 value/frontier，而不是继续调 selector。

## 6. V64.3.8 BDMU 哪些值得保留，哪些需要改

### 保留

- fixed B=16；
- auditable evidence atoms；
- frozen DARM+DBR / foundation causal isolation；
- frozen-foundation reference B-set；
- budget-feasible single exchange，禁止 B+1 teacher target；
- cost normalization；
- continuous one-sided decisive-margin deficit；
- literal winner flip 只作为 limiting-case diagnostic。

### 修改

1. fixed R=4 -> adaptive decisive frontier；
2. 单纯 weighted-mean deficit -> mean + weakest-rival deficit；
3. generic listwise utility -> listwise + deployment-boundary Top-M swap ranking。

### 暂不做

- 不加入大权重 `Delta certified_pair_fraction` binary bonus；
- 不把 LITERAL winner-flip BCE 重新设为主 acquisition objective；
- 不改 DARM/DBR；
- 不增大 B/M；
- 不做 family gate/BCHA；
- 不做 beam/swap/bruteforce selector search；
- 不 scratch 全 foundation。

## 7. V64.3.9 AF-BDMU 设计

### 7.1 Adaptive decisive frontier

默认至少包含 teacher winner 最近的 4 个正 margin rivals；再加入满足

`m_T(w,b) <= max(0.05, 2 * m_nearest)`

的 rivals，上限 8。这样既不把所有 K candidates 拉进 acquisition target，也不会因为固定 R=4 截断真实 near-boundary frontier。

### 7.2 One-sided mean/worst deficit

对于 reference set S：

`D(S) = (1-lambda) * sum_b pi_b [gamma_b - m_S(w,b)]_+ + lambda * max_b [gamma_b - m_S(w,b)]_+`

V64.3.9 取 `lambda=0.35`。utility 仍定义为预算可行 add/exchange/removal 对 D(S) 的局部下降，并除以 query cost。

这比额外加一个 certificate BCE 更贴合论文主线：它直接优化 one-sided decisive-margin preservation 的必要局部对象。

### 7.3 Top-M swap ranking

只有在以下条件同时成立时构造 pair：

- positive atom 有正 teacher marginal utility；
- positive 当前被 Top-M 漏掉；
- negative 当前占据 Top-M；
- positive utility > negative utility。

训练 `logit_pos > logit_neg`，并按 teacher utility gap 加权。这样 loss 直接对应实际 deployment proposal boundary，而不是泛化 hardest-negative ranking。

### 7.4 严格 causal isolation

- only train `critical_proposal_adapter`；
- DARM、DBR、legacy proposal/family、foundation 全冻结；
- B=16、selector、final tournament 不变；
- main loss only AF-BDMU；
- step-0 adapter zero-init，确保与 promoted DARM checkpoint 完全一致。

## 8. V64.3.9 的关键“退出条件”

下一轮不应无限优化 acquisition。screen/full audit 加入：

- 如果 Top-M utility capture / critical recall 明显提高，teacher match/regret 也提高：acquisition 仍 binding，AF-BDMU 有效；
- 如果 Top-M utility capture / critical recall 明显提高，但 teacher match/regret 没有提高：**停止 acquisition 方向，pivot 到 value/frontier**；
- 如果 acquisition mechanism 本身都不提高：AF-BDMU target/representation 失败，不应该跑 full/test/CL。

这个退出条件对于论文因果叙事比“不断追更高 open-loop score”更重要。

## 9. CCF-A motivation / novelty 判断

### Motivation

足够强，前提是坚持固定 interface budget：真实 planner 不能无限查询预测世界模型，因此“压缩什么”必须由最终决策边界定义，而不是 reconstruction likelihood 定义。这是 planning problem，而不是普通 feature pruning。

### Novelty

当前组合机制有潜力，但 novelty 应来自 **问题定义 + interface object + budget-feasible decision-margin optimization + one-sided preservation bridge + causal validation protocol** 的闭合，而不是任意一个模块单独新颖。

最应该强化的组合是：

1. hard planner-interface query budget；
2. auditable evidence atoms；
3. budget-feasible decisive-margin marginal utility；
4. one-sided decisive-action margin preservation；
5. frozen value/acquisition interface 的因果 attribution；
6. same-interface closed-loop validation。

如果后续 AF-BDMU 实验成功，这条链比“certificate-aware selection + DARM + DBR”更干净，因为每一环都对应论文 theorem/diagnostic 中一个可测 error term。

## 10. 训练 / open-loop / closed-loop 工程性能

### Training

上传 DARM full epochs 的总 wall time 约 963--1482 s/epoch；data wait 约 452--702 s/epoch，loss construction 约 282--345 s/epoch。I/O 是明确的大头之一。V64.3.9 保留 BDMU-only fast path，并扩展 input benchmark 同时 sweep workers 与 prefetch factor。

### Open-loop

selected checkpoint 1000-scene open-loop：

- total internal ≈ 592.51 ms；
- prediction ≈ 458.57 ms；
- selector ≈ 66.98 ms；
- tournament ≈ 8.95 ms。

prediction 约占 77%，因此下一阶段真正的 runtime speed 优化对象应该是 batched/cached prediction，而不是继续微调 selector。为避免污染本轮 causal experiment，V64.3.9 不改变预测语义，只优化 screen sampling 与输入 pipeline。

### Closed-loop

当前上传结果中没有完成的 closed-loop artifact，因此不能根据这轮结果声称闭环提升。下一条命令链只在 AF-BDMU screen + full validation 都通过后，用同一个 frozen checkpoint 跑 CL20 integration、CL100 non-reactive、CL100 reactive。

## 11. 本轮代码实现

已实现：

- `bdse/model/decisive_margin_utility.py`：adaptive frontier + mean/worst deficit；
- `bdse/model/losses.py`：Top-M swap ranking；
- `bdse/experiments/evaluate_open_loop.py`：AF-BDMU diagnostics；
- `bdse/data/nuplan_dataset.py` / `train.py` / `run_v64_saqa_bcc.sh`：representative capped sampling；
- `bdse/tools/check_v64_3_7_darm_dbr_screen.py`：FULL interface-consistency gate；
- `bdse/tools/check_v64_3_9_af_bdmu_screen.py`：mechanism/deployment/pivot audit；
- `bdse/tools/validate_training_artifacts.py`：training artifact contract；
- `bdse/tools/benchmark_training_input_pipeline.py`：workers x prefetch sweep；
- V64.3.9 configs + 2-GPU screen/full launchers；
- `NEXT_COMMANDS_V64_3_9_AF_BDMU.txt`；
- `ALGORITHM_UPDATE_LOG.md` 同步更新。

回归测试：V64.3.7/V64.3.8/V64.3.9 相关 23 个测试全部通过；V64.3.9 train/eval config contract 通过。
