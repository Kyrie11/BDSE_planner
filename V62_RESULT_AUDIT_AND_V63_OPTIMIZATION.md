# V62 上传结果审计与 V63 DCQC-TFCR 优化报告

## 0. 结论先行

上传的 `outputs_v62_dcab_ewfc_fast_2gpu_v1.zip` **不能用于判定 V62 的 Protocol / Minimum / Competitive gate 已通过或已失败**。它包含训练日志、两个未完成的 calibration shard 进度、以及 V53 immutable foundation anchor 的 1000-scene replay；不包含 calibration raw `.npz`、合并后的 dual-calibration JSON、V62 candidate/local/foundation 三路 open-loop 指标、V62 gate report，也不包含 checkpoint。流水线在正式 calibration 完成前中断，因此缺失 `open_loop/v62_dcab_ewfc_gate_report.json` 只是下游症状。

在隔离工程误判之后，V62 的训练代理显示：proposal/HAB 的大部分基础机制稳定，固定 B=16 selector 对“它实际拿到的 sparse-full interface”保持得很好；但部署端 winner 仍停留在 0.141，residual 没有产生有效 action flip。旧的 `budget_vs_full_match=0.172` 不能再被解释为纯算法损失，因为它混合了缓存 query feature 漂移、model base 与 deployment base/prior 漂移、以及真正的 evidence sparse-value 损失。

本次落地版本命名为：

**V63 Deployment-Consistent Query Contract + Teacher-Flip Criticality Ranking BFAR-DBAP (DCQC-TFCR)**

它保留论文核心 novelty：

1. 固定 planner-interface evidence atom budget（主配置 B=16）；
2. 可审计 evidence atoms；
3. 预算内确定性 AOCC selector 的 exact execution/audit；
4. criticality 由“移除 atom 是否翻转 winner action”定义；
5. evidence/residual 双证书。

V63 不增加 B，不使用 dense oracle 绕过 selector，不用降低 calibration 门槛制造 action flip。它首先把 dense diagnostic 与真实 deployment 的 base/query contract 变成可数值审计的同一接口，然后把 criticality 监督改为 teacher-interface 的 exact leave-one-out winner flip，并加入 hardest-negative ranking。

---

## 1. 上传材料与审计边界

交叉审计范围：

- 论文：`iclr2026_conference.tex`；
- 上一轮分析：`大模型建议.md`；
- 代码：`bdse.zip`；
- 数据分析：`dataset.zip`；
- 实验输出：`outputs_v62_dcab_ewfc_fast_2gpu_v1.zip`；
- 算法历史：`ALGORITHM_UPDATE_LOG.md`；
- V62 指令：`NEXT_COMMANDS_V62_DCAB_EWFC.txt` / `V62_DCAB_EWFC_NEXT_COMMANDS.sh`。

当前环境没有 nuPlan cache、GPU checkpoint 和可运行的 closed-loop simulator，所以本次完成的是：结果审计、工程归因、算法/代码修改、单元测试与静态验证。**没有 fresh V63 training、calibration、open-loop 或 closed-loop，不能提前声称 V63 gate PASS、闭环提升或 SOTA。**

---

## 2. 三个 Gate 的真实状态

| Gate | V62 上传包状态 | 可用代理证据 | 当前判断 |
|---|---|---|---|
| Protocol | 未生成三路 calibrated open-loop 与 gate report | V53 foundation anchor replay PASS；V62 train val 有 all-valid/query 指标 | **NOT EVALUATED**。不能把 V53 replay 当 V62 Protocol PASS，也不能因 report 缺失判 FAIL |
| Minimum | 未生成正式 candidate/local/foundation suite | train-time val proxy：proposal recall 0.800–0.804、selected recall 0.612–0.613、effective recall 0.777–0.779，`val_minimum_gate_feasible=1` | **NOT OFFICIALLY EVALUATED**。代理信号正向，但缺 calibration 与正式 paired suite |
| Competitive | 未生成正式 suite | teacher match / sparse-full / pair-full 均约 0.141；residual beneficial/harmful 都为 0；无部署 gain | **NOT OFFICIALLY EVALUATED，但强烈预警会失败** |

### 2.1 为什么不是“Protocol gate 失败”

上传输出目录只有 15 个文件。两个 pipeline log 都停在：

```text
[v62] stage 1 already complete: reuse final/best checkpoints
```

两个 calibration worker 只保留了进度与 manifest，约在 46%–47% 处终止；没有 raw `.npz`，也没有 traceback 或 worker status 文件。V62 脚本在 `set -e` 下顺序 `wait` 两个 PID，失败时没有可靠的 shard 状态、失败尾日志、原子 merge 或可复用完成 shard。它还会在重新进入 calibration 时删除 shard 工作目录，导致昂贵工作不可恢复。

因此真正的问题是：**pipeline observability/resumability 不足，且执行在 gate 前中断**。缺失 gate report 不是 gate 检测出的失败。

### 2.2 V53 anchor replay 能说明什么

上传包中的 `factorized_anchor/quality/open_loop.json` 使用 V53 foundation checkpoint 做 immutable anchor replay，结果为：

- full-interface action match = 0.359；
- base winner-rival sign accuracy = 0.6711；
- dense winner-rival sign accuracy = 0.8029；
- dense near-tie sign accuracy = 0.7057；
- dense all-pair sign accuracy = 0.7185。

这证明 V53 anchor 与当前数据/代码的固定质量门能通过；它明确排除了 budgeted regret、direct pair head、selector/AOCC。因此它**不能**说明 V62 candidate、local control、residual 或 fixed-budget interface 通过。

---

## 3. 去除工程错误后的根因分析

### 3.1 旧的 0.172 bridge 指标混合了三类不同误差

V62 训练日志显示：

```text
full interface teacher match             0.359
runtime sparse-full teacher match        0.141
B16 vs runtime sparse-full match         0.981
B16 vs old dense-full match              0.172
queried valid action fraction            1.000
```

表面上看，V62 已把 action query 扩展为 all-valid，但 dense→sparse 仍未恢复。代码审计发现旧比较并非同一 planner interface：

1. `tensorizer.evidence_arrays()` 在 dense/training 路径只要缓存 query tensor 形状可用就直接读取；
2. sparse runtime 通过当前代码现场 `compute_query_features_for_pairs()` 重算 canonical query features；
3. 缓存没有代码版本、query-relevant config fingerprint 或数值一致性校验；
4. dense 路径以原始 learned `J0` 为 base；
5. sparse deployment 在 scoring 前对 `J0` 应用了 runtime base prior 与 structural safety residual prior。

因此旧 `budget_vs_full_match=0.172` 同时包含：

- cached-query 与 runtime-query 的 feature/value drift；
- learned model base 与 deployment base/prior 的 winner drift；
- Top-M atom sparse scoring 本身的 value drift；
- B=16 selection loss。

其中最后一项其实已有强证据不是主要瓶颈：B16 对 runtime sparse-full 的 winner preservation 达到 0.981。V62 不能在 contract 未数值通过前，把 0.172 归因给 selector 或 proposal。

### 3.2 `all_valid` 生效，但旧诊断不能证明 bridge 算法失败

`val_action_query_mode_all_valid=1` 且 `val_queried_valid_action_fraction=1.0`，说明 V62 已不再只查询 rival graph actions。这个修改应保留。

但“all-valid action IDs 被查询”只说明 coverage 完整，不说明 dense 和 sparse 路径对同一 atom-action pair 得到了相同 `g(i,a)`，也不说明两条路径使用同一 base。必须先检查：

```text
max |J0_deployment_dense - J0_sparse| <= tolerance
max |g_dense[i,a] - g_sparse[i,a]| <= tolerance
```

只有这两个 numerical contract 通过后，`dense Top-M winner -> runtime sparse-full winner` 才是纯算法指标。

### 3.3 V62 criticality target 极度稀疏且与 teacher action 弱对齐

V62 的 model-self leave-one-out criticality 统计为：

- critical scene fraction：1.46%–1.59%；
- critical atom fraction：0.238%–0.276%；
- Top-M critical recall：6.05%–7.14%；
- teacher-aligned critical scene fraction：0.168%–0.220%。

这说明 loss 大多数 batch 上几乎没有正样本，而且 model 自己的 dense winner 与 teacher winner 只在极少 critical scenes 上一致。继续单纯放大 positive weight，容易得到高方差梯度或只学到 model 自洽性，不能直接改善 teacher action。

V63 改为 teacher-interface exact LOO：

```text
critical_T(i) = 1[ argmin_a J_T(a) != argmin_a (J_T(a) - g_T(i,a)) ]
```

仅在 scalar teacher 与 lexicographic teacher winner 一致的场景监督，并用 hardest-negative pairwise ranking 强制 critical atom 的 proposal logit 高于最危险 non-critical atom。severity 只在已 critical atoms 内排序，不替代二值 winner-flip 定义。

### 3.4 Residual 当前不是“calibration 太严格”这么简单

V62 train-time validation 中：

- candidate/local/pair-full teacher match 都约 0.141；
- beneficial residual intervention rate = 0；
- harmful residual intervention rate = 0；
- residual proposal/allowed/confident flip 相关率基本为 0；
- `L_residual_winner_correction` 从 11.95 降到 6.59。

loss 下降说明 residual head 在拟合训练目标，但部署决策上没有可见干预。不能通过关闭 conformal epsilon、扩大 residual scale 或降低 flip margin来强行制造 flips。正确顺序是：

1. numerical base/query contract 必须通过；
2. teacher-critical proposal recall 提升；
3. sparse-full / selected-local interface 明显改善；
4. raw proposed action 中 teacher-directed beneficial > harmful；
5. 最后再判断 calibration 是否过保守。

### 3.5 Proposal/HAB 仍有 family allocation / surrogate gap

V62 的 global/dense Top-M winner preservation 约 0.972–0.973，global Top-M match 约 0.955；但 exact runtime HAB Top-M match 只有 0.484–0.488，尽管 fast/exact HAB mask Jaccard=1。

这不是 fast 与 exact executor 不一致，而是训练 surrogate/目标更容易保住 global/dense winner，却没有把真正 teacher-critical atoms稳定排到 HAB family-constrained proposal 中。V63 的 teacher-flip ranking直接攻击这个 gap，但后续实验还应按 family/type/city 分解 critical recall，避免总 recall 掩盖 family starvation。

---

## 4. V62 中值得保留的正向信号

### 4.1 必须保留

- **固定 B=16 evidence certificate budget**：V62 每次 validation 都使用 16 个 decision atoms；B16 对 sparse-full winner preservation=0.981。
- **可审计 evidence atoms + unique hard-event ownership**：与论文 novelty 一致，且 dataset oracle 中 B16 decision sufficiency 在 val 达 0.912。
- **HAB family-aware proposal**：不退回 global Top-M 作为部署主路径；fast/exact HAB mask Jaccard=1 表明实现稳定。
- **确定性 AOCC selector / exact execution audit**：selector 本身暂不是主要信息损失源，继续作为固定预算证书执行器。
- **all-valid action bridge**：query coverage=1.0，修复了 V61 action-axis 缺失；应保留并做 contract audit。
- **foundation/local/candidate 三路 paired control**：是区分 base、selector 与 residual gain 的必要实验设计。
- **group-disjoint val_tune / val_calib**：继续用于 checkpoint/hyperparameter 与 calibration 分离。
- **structural safety bypass / residual prior**：hard decisive structural coverage=1.0，必须保留为安全结构，但要单独报告其 action influence，不能混入 evidence selector 的 B。

### 4.2 值得升级

- Proposal supervision：从 model-self dense winner / generic decisive atom，升级为 teacher-interface exact winner-flip criticality + hardest-negative ranking。
- Dense bridge diagnostics：升级为 model base → deployment base → dense Top-M → runtime sparse Top-M → B16 selected dense → deployed action 的逐层分解。
- Query cache：从“shape 可用就信任”升级为随机数值 audit + code/config fingerprint；主配置默认 runtime recompute。
- Gate checker：把 metric pass 与 official pass 分开；Protocol 数值 contract 失败时禁止把后续结果解释为算法失败。
- Pipeline：calibration shard 可复用、worker failure 可定位、merge 原子化、完整 pipeline detach、同 OUT_ROOT 锁。

### 4.3 需要修改或暂缓

- **model-self criticality**：不再作为主监督，可保留为 ablation/辅助统计。
- **residual aggressiveness**：暂不提高。只有 raw beneficial direction 出现后才升级。
- **pair residual head**：当前主路径 skip scoring，继续保持；其前代多次未产生净 gain，不应在 interface contract 未稳定时恢复昂贵 head。
- **增加 B**：禁止用更大 evidence budget掩盖 proposal/bridge错误。B24/B32仅作为 oracle/ablation。
- **用 incomplete test 调参**：禁止。

---

## 5. 数据集诊断对模型上限的影响

### 5.1 Validation（58,418 scenes）

E0 13 项检查中 10 PASS / 3 FAIL：

- safe candidate exists = 0.7173（低于 0.75）；
- candidate hard violation rate = 0.5668；
- teacher hard violation rate = 0.2827；
- candidate-log ADE to teacher p50 = 5.480 m；
- p90 = 12.915 m；
- quality keep = 0.9175；
- full-interface oracle action match = 0.9657；
- runtime decision sufficiency = 0.7490；
- evidence sufficiency = 0.5922；
- selector value ratio = 0.6603；
- B16 oracle decision sufficiency = 0.9120；
- route-distance p95 p90 = 3.263 m。

约 **28.3%** 的 val scenes 没有 safe candidate。即使 evidence/selector 完美，候选库也无法在这些场景输出安全轨迹。要冲闭环 SOTA，候选生成/route conditioning 是与 BDSE interface并行的必要升级，但应在 V63 contract 稳定后单独做，避免把 candidate ceiling 与 evidence算法混在一起。

### 5.2 Incomplete test（67,042 scenes）

E0 13 项中 6 PASS / 7 FAIL：

- safe candidate exists = 0.5779；
- candidate hard violation rate = 0.6700；
- teacher hard violation rate = 0.4221；
- ADE p50 = 4.956 m，p90 = 17.141 m；
- route-distance p95 p90 = 17.55 m；
- quality keep = 0.6791；
- full-interface match = 0.9343；
- runtime decision sufficiency = 0.6407；
- evidence sufficiency = 0.5395；
- selector value ratio = 0.6411；
- B16 oracle decision sufficiency = 0.8399。

当前 test 有明显构建未完成/route distribution 异常或分布偏移。它可以作为**冻结 checkpoint 的一次性 stress/数据 QA**，不能用于 threshold、checkpoint、版本或算法方向选择。完成构建、identity/leakage/parity/readiness audit 后，再作为冻结最终 testing。

### 5.3 城市差异

| 城市（各 1000） | safe candidate | runtime decision suff. | evidence suff. | selector ratio | B16 oracle suff. |
|---|---:|---:|---:|---:|---:|
| Boston | 0.646 | 0.740 | 0.591 | 0.659 | 0.913 |
| Pittsburgh | 0.840 | 0.865 | 0.624 | 0.656 | 0.922 |
| Singapore | 0.930 | 0.846 | 0.678 | 0.701 | 0.950 |
| Las Vegas | 0.742 | 0.625 | 0.555 | 0.617 | 0.917 |

Las Vegas 的 B16 oracle sufficiency 很高，但 runtime decision sufficiency 最低，说明预算本身足够，实际 proposal/interface quality 仍有大幅优化空间。建议训练采样和 gate report增加 city-balanced macro 指标，并按 city/family 报 teacher-critical recall。

---

## 6. V63 已落地的算法与代码修改

### 6.1 Deployment-consistent query contract

文件：`bdse/data/tensorizer.py`、`bdse/model/bdse_model.py`、`bdse/experiments/evaluate_open_loop.py`

新增 `runtime.dense_query_feature_source`：

- `runtime_recompute`：主配置，dense/training 与 sparse runtime 都用当前 canonical query实现；
- `cache_verified`：逐样本重算并验证 cache，适合 debug；
- `cache`：只有在独立 cache audit 报告 PASS 且 fingerprints 匹配时允许；
- `cache_or_recompute`：legacy compatibility，不用于 V63 main claim。

`predict_dense_numpy()` 现在显式返回：

- `J0_model`：immutable learned foundation base；
- `J0_deployment`：应用与 sparse runtime完全相同的 base prior + structural safety residual prior；
- `g` / `g_var`；
- base/structural prior诊断。

新增 numerical contract：

- `dense_runtime_base_value_mae/max_abs/allclose_fraction/pass`；
- `dense_runtime_query_value_mae/max_abs/allclose_fraction/pass`。

Protocol gate要求 base/query contract pass≈1 且 allclose fraction≥0.999。未通过时，不允许解释 bridge/selector/competitive 指标。

### 6.2 Teacher-interface exact winner-flip criticality ranking

文件：`bdse/model/losses.py`

`exact_winner_flip_criticality.target_source` 支持：

- `model_dense`；
- `teacher_interface`（V63 main）；
- `hybrid_union`。

V63 main 使用 teacher cost与 teacher atom contribution做 vectorized LOO，排除 scalar/lexicographic winner不一致样本。损失由：

- critical vs non-critical weighted classification；
- critical severity ranking；
- deployment Top-M recall surrogate；
- hardest-negative pairwise logit margin；

组成。主配置 positive weight 16、teacher-aligned weight 6、pairwise rank weight 1、rank margin 1；没有增加总 evidence budget。

### 6.3 分层 bridge 与 criticality 指标

文件：`bdse/experiments/evaluate_open_loop.py`、`bdse/experiments/train.py`

新增并在 train validation/open-loop 共用同一 helper：

- model dense vs teacher；
- deployment dense vs teacher；
- model base → deployment base winner match；
- deployment dense → HAB Top-M dense-value match；
- HAB Top-M dense-value → runtime sparse-value match；
- deployment dense → B16 selected dense-value match；
- B16 selected dense-value → deployed winner match；
- teacher exact critical Top-M/selected recall、scene/atom fraction、scalar alignment。

这使每次失败都能落到 base、query value、proposal、selector、potential、residual 中的具体层，不再用一个 0.172 猜根因。

### 6.4 诚实的 query accounting

文件：`bdse/planner/nuplan_planner.py`、`bdse/metrics/bdse_metrics.py`

论文 novelty 的预算是**保留到 planner-interface certificate 的 atom budget B**。V63 分开报告：

- proposal/acquisition atom count M；
- action-conditioned acquisition scores M×K；
- retained certificate atoms B；
- retained certificate action queries B×K；
- pair-conditioned scores单独报告。

当前 M=24、B=16、K≤32：

- acquisition action scores上界 768；
- retained certificate payload上界 512；
- M/B=1.5。

不再声称“总内部 query 上界是 B×K”；B×K只描述固定 retained interface。

### 6.5 V63 gate checker

文件：`bdse/tools/check_v63_dcqc_tfcr_gate.py`

- Protocol、Minimum metrics、Minimum official、Competitive metrics、Competitive official分开；
- Protocol增加 numerical base/query contract、all-valid coverage、B budget、teacher scalar alignment；
- Competitive关键新增：dense-HAB/runtime sparse bridge≥0.95、teacher-critical Top-M recall≥0.80、selected recall≥0.50；
- critical scene估计少于20时警告高方差，要求正式论文结论扩大冻结 open-loop样本；
- paired regret、beneficial/harmful flip、candidate/local/foundation gain仍保留。

### 6.6 可恢复的两 GPU pipeline

文件：`V63_DCQC_TFCR_NEXT_COMMANDS.sh`、`run_v63_dcqc_tfcr.sh`

- 完整 pipeline detach，而不是只 detach training child；
- OUT_ROOT lock防止并发覆盖；
- final/best checkpoint freshness检查；
- calibration shard独立 manifest/raw freshness；
- 只运行缺失 shard；
- worker失败保留 `.failed` 与 log tail；
- merge先写临时文件再 atomic rename；
- `PIPELINE_FORCE=0` rerun复用已完成阶段；
- candidate/local/foundation共享 bounded worker pool；
- CL20 token list hash严格一致；
- Competitive PASS才允许自动升级 CL100。

另有 `RUN_V63_CONTRACT_AUDIT_FROM_V62_CHECKPOINT_2GPU.sh`：使用同一个 V62 checkpoint运行 nominal/no-base/no-structural/no-runtime-priors四路 same-checkpoint attribution，并支持 freshness复用。

---

## 7. 当前模型状态与优化方向

### 7.1 当前状态

当前模型不是“selector容量不足”，而是：

1. foundation/full interface只有 0.359 teacher match，base上限本身偏低；
2. B16 selector对已有 sparse interface很忠实（0.981）；
3. proposal generic decisive recall已经约0.80，但 exact teacher/winner-flip evidence没有被稳定提到 Top-M；
4. old dense/sparse diagnostic不可信，必须先跑 V63 numerical contract；
5. residual训练 loss下降但部署方向未形成；
6. candidate bank在约28.3% val场景没有 safe option，构成独立闭环上限。

### 7.2 优先级

**P0：同 checkpoint contract audit**

先用 V62 best checkpoint跑 V63四路 audit，回答：

- query/base numerical contract是否恢复到≈1；
- old 0.172中多少来自 stale query cache；
- base prior、structural prior分别改变多少 winner；
- 修复后 HAB dense-value→runtime sparse-value match是否接近1。

若 numerical contract仍不通过，禁止训练 V63；继续修工程。

**P1：V63 teacher-critical proposal training**

contract通过后训练 V63。主要看 teacher-critical recall、HAB exact match、sparse-full teacher match，不先看 residual。

**P2：residual方向性**

只有 raw proposal rate≥0.001、beneficial>harmful、candidate>local后，才讨论 calibration或 residual expressivity。否则要修改 residual target/routing，而不是放宽门槛。

**P3：candidate generator/route conditioning**

V63 fixed-interface gain稳定后，单独升级候选生成：提高 safe candidate exists、减少 candidate hard violation、修复route-corrupted场景。此改动不应改变 evidence budget/criticality定义，可作为 orthogonal candidate-bank module。

**P4：closed-loop**

Protocol PASS后即使 Minimum失败，也可运行**明确标注 diagnostic**的 paired CL20帮助定位；正式 CL100只在 Minimum+Competitive PASS且CL20无安全退化后运行。

---

## 8. 止损条件与下一版决策树

1. **Contract audit FAIL**：不训练；修 query/base/provenance。
2. **Contract PASS，但 teacher-critical Top-M recall <0.50**：优先改 proposal family allocation / ranking sampler；不改 selector、不加B。
3. **Top-M recall≥0.80，selected recall<0.50，且 Top-M→sparse match≈1**：这时才归因 selector目标/预算分配；检查critical family quota与AOCC utility。
4. **selected recall≥0.50，但 sparse-full teacher match仍≈0.141**：检查teacher contribution scale、base/evidence partition、action potential integrability。
5. **local/sparse改善，但 candidate=local**：residual无效；检查raw teacher-directed proposal，而非calibration阈值。
6. **candidate>local且beneficial>harmful，但official flip少**：再审 calibration reserve/epsilon。
7. **open-loop gain转正但CL20退化**：先看candidate coverage、route、hard safety、latency，不直接扩大算法容量。
8. **candidate bank safe-exists仍<0.75**：启动独立 candidate-generation版本；BDSE固定B接口不变。

---

## 9. 论文需要同步的表述（暂不填结果）

上传 TeX 的核心方法叙述是合理的，但与当前V63实现有以下不一致：

- 正文仍称 runtime只查询小 rival graph中的 actions；V63 main是 Top-M atoms × all valid actions，rival graph只定义pair tournament。
- 正文称典型 M∈[2B,4B]；当前 M=24、B=16，即 M=1.5B。
- 必须区分 retained-interface budget B 与内部 acquisition compute M×K。
- “exact selector”应限定为configured deterministic AOCC operator 的 exact execution/audit，不声称 greedy acquisition求全局组合最优。
- 数据协议应写清 group-disjoint `val_tune`（调参/选checkpoint）、`val_calib`（校准）、完成并冻结的test（一次性最终测试）。
- Appendix calibration公式中 `\text{Quantile}` 的反斜杠缺失（源文件第614行显示为 tab + `ext{Quantile}`）。
- 当前结果表是占位，不能把train-time proxy或V53 replay写成V62/V63正式结果。

建议等V63 gate与paired CL20完成后再同步正文和表格，避免方法版本与结果再次错位。

---

## 10. 验证与效率边界

代码验证：

- Python compile：PASS；
- V63 YAML：9/9 PASS；
- Shell syntax：3/3 PASS；
- full unit tests：245 passed，0 failed（13个既有Transformer warning）；
- targeted contract/criticality/query-accounting测试已包含在full suite；
- 未执行GPU numerical contract、fresh training/open-loop/closed-loop。

效率分析：

- V62 train throughput 从50.86提升并稳定在约65.27 samples/s；
- val内部延迟最后一次约666.3 ms：predict 499.9 ms、selector 87.4 ms、tournament 7.7 ms，优化优先级应在predict/query path，不是继续微调selector；
- V63 `runtime_recompute`是正确性默认，会增加dense training/diagnostic query feature构造成本；
- 若独立 cache audit PASS，使用 `v63_dcqc_tfcr_train_2gpu_verified_query_cache.yaml` 可恢复cache速度，但pipeline强制核验code/config fingerprints；
- exact teacher LOO完全张量化，无Python per-atom循环；
- pair head保持skip；local variance关闭时不计算；顶层action loss为0时不构造无用mask；
- calibration/open-loop支持shard/阶段复用，避免V62中断后全量重跑。

---

## 11. 不再重复的尝试

- 不用global Top-M指标代替runtime HAB。
- 不在numerical query/base contract失败时归因selector或算法。
- 不通过增加B隐藏proposal/bridge缺陷。
- 不把B×K写成全部内部acquisition compute。
- 不把model-self criticality作为主监督。
- 不通过放松epsilon/flip margin制造未经teacher方向验证的residual flips。
- 不恢复昂贵pair head来补偿上游接口错误。
- 不用incomplete test调参、选checkpoint或选版本。
- 不把V53 foundation replay写成V62/V63 candidate gate。
- 不在calibration/open-loop未完成时依据缺失report判断gate失败。
- 不允许同一OUT_ROOT中不同closed-loop challenge共享缓存结果。
