# V60 结果复核、根因分析与 V61 DE-HWPP 优化报告

## 结论摘要

V60 的官方 gate report 为：

| Gate | 官方结果 | 复核后的结论 |
|---|---:|---|
| Protocol | PASS | 现有 checker 下通过；但论文级严格可归因审计不充分，set-conditioned residual 的实际结果字段完全缺失，且 gate 未拦截 proposal loss runaway。应记为“官方 PASS / strict attribution incomplete”。 |
| Minimum | **FAIL** | 真实失败。proposal decisive recall `0.70011 < 0.72`。 |
| Competitive | **FAIL** | 真实失败。candidate、same-checkpoint local、foundation 的 teacher action match 均为 `0.141`，pair-full 与 deployed residual gain 均为 0。 |

V60 没有解决 V59 的核心问题“Proposal Top-M 没有保存 dense local winner 信息”。它训练出的 `proposal_dense_topm_match≈0.965` 是 global Top-M 的训练代理指标，而实际部署使用 HAB family slots、interaction reserve、group diversity 与 structural safety bypass。真实验证始终为 dense full `0.359`、sparse-full `0.141`、B16 `0.141`、budget-vs-dense `0.172`。

更严重的是，V60 proposal 分支在训练中失稳：`L_prop` 从 `15.09` 增长到 `1452.33`，`L_deploy_select` 从 `6.45` 增长到 `833.63`，而 validation proposal recall 从 epoch 1 的 `0.7260` 连续降到 epoch 7 的 `0.6439`。因此当前最大的上游问题已经可以更精确地表述为：

> **dense local representation 中存在可用 winner 信息，但 V60 的 proposal 监督与真实 HAB 部署路径不一致，并存在 proposal-logit 的无效优化方向，导致 dense→runtime HAB Top-M 的桥接被错误训练；B16 exact selector 对已进入 proposal pool 的信息仍基本有效。**

V61 已将这一问题直接落地到代码：使用 deployment-HAB hard forward、抽样 exact runtime HAB、translation-invariant straight-through、proposal logit稳定正则、formal Minimum-feasible checkpoint selection，以及 proposal/residual stage routing。

---

## 1. 分析证据与复核范围

本报告基于：

- 论文：`iclr2026_conference.tex`；
- V60 代码与日志：`bdse.zip`；
- V60 大模型建议：`大模型建议(1).md`；
- V60 输出：`outputs_v60_dwapc_bfar_dbap_fast_2gpu_v1.zip`；
- 数据集诊断：`dataset.zip`；
- `ALGORITHM_UPDATE_LOG.md` 与 `NEXT_COMMANDS_V60_DWAPC_BFAR.txt`。

复核包含：gate report、train log、candidate/local/foundation paired JSONL、calibration JSON、calibrated configs、dataset diagnostics、论文方法与代码部署路径。当前上传输出不包含可重新执行 V60 inference 的 best checkpoint，因此对缺失的 set-conditioned residual 不能进行补跑，只能做结果可归因审计。

---

## 2. 三个 Gate 是否真实通过

### 2.1 Protocol gate

V60 原 checker 的 `protocol_pass=true`，以下协议确实成立：

- 训练日志为 8 个唯一 epoch，无重复 epoch；
- exact selector supervision 非零，最后 exact fraction `0.03124 >= 0.015`；
- val_tune 与 val_calib 为 log-group-disjoint，manifest hash完整；
- candidate/local/foundation 为同一 1000 个 scenario/timestamp paired evaluation；
- controls 关闭 residual；
- calibration epsilon、beta 与运行配置一致；
- candidate/local/foundation anchor row drift 为 0。

但严格审计发现两项会影响论文归因的工程问题：

#### A. Set-conditioned residual 的实际结果可观测性缺失

candidate calibrated config 启用了：

```text
set_residual_rank = 8
pair_tournament_aggregation_mode = evidence_action_potential
```

当前代码与测试也声明会导出：

```text
set_conditioned_residual_active
set_conditioned_residual_rank
set_conditioned_residual_abs_mean
set_conditioned_residual_scale
```

但 V60 candidate 的 1000 条 JSONL 中，上述字段覆盖率全部为 0。可能原因包括：实际评测运行了旧代码、合并结果来自旧 shard、某条 export 路径未进入，或上传代码与生成结果的 commit 不一致。

这不会改变“candidate 最终 action 与 local 完全相同”的事实，但会阻止以下算法归因：

- set head 是否实际 active；
- set correction 幅度是否为零/过小；
- pair-full 是否确实包含 set factors；
- checkpoint 是否按实际 set-aware pair-full 选择。

#### B. Protocol checker 未识别 proposal 分支的训练失稳

V60 gate 只验证 dense-winner loss非零，没有检查：

- `L_prop` 96.2x runaway；
- proposal logits 的绝对值/RMS；
- global Top-M 与 runtime HAB Top-M 的一致性；
- exact runtime HAB 是否真正进入 dense-winner proposal objective。

因此，对论文主实验建议使用两层表述：

> V60 在原有协议定义下 Protocol PASS；在新增的 paper-grade provenance/optimization-integrity 审计下 Protocol attribution incomplete。不能把 V60 的 global Top-M train metric 当作已验证的部署一致证据。

### 2.2 Minimum gate

Minimum 真实 FAIL，唯一正式失败项：

```text
proposal decisive recall = 0.7001098 < 0.72
```

其余最低指标达到原门槛：

- selected decisive recall `0.53735 >= 0.50`；
- effective decisive recall `0.70301 >= 0.62`；
- selected interaction decisive recall `0.58986 >= 0.40`；
- evidence certificate `0.88753 >= 0.40`；
- fallback `0.11 <= 0.60`；
- budget fill `1.0`。

这个失败不是统计边缘误差，也不是 aggregation bug。训练验证清楚显示 proposal recall 持续下降：

| Validation epoch | Proposal recall | Selected recall | Minimum proposal threshold |
|---:|---:|---:|---:|
| 1 | **0.72604** | 0.56691 | PASS |
| 3 | 0.70011 | 0.53724 | FAIL |
| 5 | 0.66335 | 0.51404 | FAIL |
| 7 | 0.64393 | 0.50439 | FAIL |

V60 best checkpoint 选择了 epoch 3，而 epoch 1 已满足 Minimum。原因是 competitive score 中微小的辅助项变化盖过了 formal gate feasibility。这是 checkpoint-selection 工程错误，V61 已改为 gate-feasible lexicographic selection。

### 2.3 Competitive gate

Competitive 真实 FAIL：

```text
candidate teacher match      = 0.141
same-checkpoint local match  = 0.141
foundation control match     = 0.141
candidate-local gain         = 0.000
candidate-foundation gain    = 0.000
pair-full residual gain      = 0.000
beneficial/harmful deployed  = 0 / 0
```

此外：

- proposal decisive recall `0.7001 < 0.80`；
- selected decisive recall `0.53735 < 0.55`；
- p95 latency `778.0 ms > 500 ms`，虽然 latency 未设为 hard gate。

即使完全排除 calibration 的影响，residual 仍没有 teacher-directed修正：1000 个场景只有 9 个 raw proposal，9 个 proposed action 全部不是 teacher action；raw margin均值约 `0.00140`，最大 `0.00295`，低于 flip margin `0.015`。因此不存在“只是 epsilon 太大把本来正确的修正挡住”的解释。

---

## 3. V60 是否解决 dense winner 保存问题

### 3.1 真实部署分层

V60 关键结果：

| 路径 | Teacher action match |
|---|---:|
| Dense full-interface | **0.359** |
| Proposal Top-M sparse-full | **0.141** |
| B=16 deployed local | **0.141** |

以及：

| 指标 | 值 |
|---|---:|
| B16 vs sparse-full winner match | **0.981** |
| B16 vs dense-full winner match | **0.172** |
| evidence certificate | **0.8875** |
| fallback | **0.11** |

结论仍与 V59 大方向一致：B16 exact selector 不是首要瓶颈，dense→Top-M proposal bridge 才是上游瓶颈。

### 3.2 V60 新 loss 的实现目标与部署目标不一致

V60 `L_proposal_dense_winner` 的 hard-forward 选择是 proposal logit 上的 global Top-M；真实 runtime proposal 是 HAB：

1. family gate 分配 slots；
2. family-conditioned atom proposal；
3. minimum family slots；
4. soft interaction reservation；
5. agent group diversity；
6. structural safety bypass/exclusion/refill。

因此训练日志中的 `proposal_dense_topm_match≈0.965` 只说明“训练 batch 上 global Top-M proxy 与 dense winner一致”，不说明 runtime HAB Top-M 保存 dense winner。它与验证的 `budget_vs_full_match=0.172` 不是同一指标。

### 3.3 Straight-through 存在无效 logit 优化方向

V60 使用 detached M-th threshold，并在未中心化的 proposal logits 上构造 soft mask。所有 active proposal logits 同时上移时，hard Top-M集合不变，但 soft surrogate 仍可产生有利梯度。结果是：

| Epoch | L_prop | L_deploy_select | L_cf_critical_proposal | Train proxy match |
|---:|---:|---:|---:|---:|
| 0 | 15.09 | 6.45 | 10.91 | 0.9665 |
| 1 | 57.20 | 31.73 | 15.39 | 0.9645 |
| 3 | 274.20 | 158.17 | 62.73 | 0.9698 |
| 5 | 719.85 | 412.06 | 242.88 | 0.9652 |
| 7 | **1452.33** | **833.63** | **545.99** | 0.9658 |

proxy match不变、loss爆炸、真实 recall下降，是典型的 surrogate/部署错位，而不是模型“正常收敛但泛化差”。

### 3.4 当前最大的上游问题

> **当前最大上游问题不是 B16 selector，也不是 calibration，而是 deployment-inconsistent、optimization-unstable 的 proposal bridge。**

更精确的因果链：

```text
Dense local g 已包含部分 teacher winner 信息
    ↓
V60 global Top-M/ST surrogate 优化了错误接口并产生 logit drift
    ↓
Runtime HAB Top-M 丢失 winner-critical evidence
    ↓
B16 exact selector 基本忠实保持 sparse pool winner
    ↓
Residual 被迫面对缺 evidence 的 anchor，且只输出极小、非 teacher-directed扰动
```

---

## 4. 排除工程错误后，对 V60 模型状态的判断

V59 的状态描述是：

> 当前模型不是从底层完全没有学到 teacher decision，而是部署前的 evidence proposal 截断破坏了已有信息。

V60 结果支持并强化了这一描述，但需要增加一层：

> **dense local representation 仍保留可用 teacher decision 信息；V60 不仅没有修复 proposal 截断，而且其 global proxy/ST 实现主动把 proposal 分支训练失稳。当前 residual 分支存在梯度，但学到的是极小的连续扰动，没有形成 teacher-directed winner correction。**

证据：

- dense full-interface match `0.359`，远高于 final `0.141`；
- B16 vs sparse-full `0.981`，说明进入 sparse pool 后 winner 基本被保持；
- residual winner/certificate/boundary losses下降，说明训练链不是严格零梯度；
- 但 9 个 raw proposal 全非 teacher action，且 margin最高 `0.00295`；
- pair-full与deployed均无 gain。

因此，下一阶段不能继续加大 residual LR、降低 certificate 或继续单独增强 B16 selector。必须先恢复 dense→runtime HAB Top-M winner preservation，再评估 residual。

---

## 5. V60 中的正向信号与设计取舍

### 5.1 值得保留

1. **Immutable foundation anchor**：dense full `0.359` 是当前最重要的可用上游信号。
2. **Fixed planner-interface budget**：这是论文 novelty 的中心，不应通过增加 B 或隐式查询全部 evidence 解决问题。
3. **HAB family hierarchy**：论文方法明确提出 family gate与within-family proposal；问题是训练没有对齐它，而不是 hierarchy本身无效。
4. **Exact AOCC / B16 selector**：`budget_vs_sparse_full=0.981` 表明 selector 对已有 proposal pool 高度稳定。
5. **Interaction evidence pathway**：proposal interaction recall `0.7833`、selected interaction recall `0.5899`，说明 interaction family并非完全失效。
6. **Dual certificates**：evidence certificate `0.8875`，fallback `0.11`，安全审计链稳定。
7. **Group-disjoint calibration 与 paired controls**：协议设计正确，应继续保留。
8. **Direct integrable action potential**：保持 antisymmetry/cycle consistency，与论文可解释性一致。

### 5.2 值得升级

1. **Proposal supervision**：从 global Top-M/atom recall升级为 runtime HAB winner preservation。
2. **Checkpoint selection**：formal Minimum gate feasibility优先于加权综合分。
3. **Set-conditioned potential**：保留低秩 set interaction思路，但必须强制 end-to-end observability，不能只靠代码测试。
4. **Residual training**：只让 residual 主攻 dense local 本身仍错的 intrinsic correction；dense正确/sparse错误应归 proposal stage。
5. **Policy-aligned conformal calibration**：epsilon从 all-rival约 3.3降到 policy epsilon `0.535` 是方向正确的，但当前不是瓶颈。

### 5.3 需要修改或暂缓

1. **V60 global ST Top-M dense-winner loss**：需要替换，而非继续加权。
2. **Proposal BCE/listwise组合**：在logit失稳时会共同爆炸；V61降低其权重并增加稳定正则。
3. **Residual certificate reserve**：`0.15` 小于calibrated epsilon `0.535`，但不能仅扩大 reserve；raw proposal本身不够好。
4. **Set head的性能结论**：现有结果不可归因，必须重跑带真实诊断的 V61。
5. **Closed-loop大规模实验**：当前不值得运行 CL100。open-loop上游没有 winner gain，闭环只会消耗时间。

---

## 6. V61 算法与代码优化

V61 名称：

## Deployment-Exact Hierarchical Winner-Preserving Policy-Calibrated Set-Potential BFAR-DBAP（DE-HWPP）

### 6.1 Deployment-HAB winner preservation

V61 在所有训练场景执行 GPU HAB hard forward，包含：

- family slot allocation；
- family logits；
- within-family selection；
- soft interaction slot reservation；
- structural safety exclusion/refill。

每 2 step、每 rank 抽样 2 个场景，使用与 runtime 相同的 NumPy HAB实现替换 hard mask。最后 32 step可全量 exact。这样：

- forward目标真实对应 deployed proposal pool；
- fast GPU路径维持训练效率；
- exact path用于纠正近似偏差并导出 fast/exact Jaccard；
- 不增加部署 B、M 或 action-evidence query budget。

### 6.2 Translation-invariant straight-through

V61 对 active logits先中心化，再计算soft mask；soft总质量归一到 M；forward为hard HAB mask，backward为family-conditioned surrogate。新增：

```text
proposal_logit_abs_mean
proposal_logit_rms_mean
L_proposal_logit_stability
```

默认 center limit 2、RMS limit 8；strict gate RMS > 20 直接失败。

### 6.3 Global 与 runtime 指标分离

新增：

```text
proposal_dense_topm_match       # 实际 deployment-HAB hard forward
proposal_fast_hab_topm_match
proposal_global_topm_match      # 仅诊断
proposal_exact_hab_topm_match
proposal_exact_hab_fraction
proposal_fast_exact_mask_jaccard
```

以后不得再用高 `proposal_global_topm_match` 代替实际 HAB改进。

### 6.4 Proposal-first curriculum

V61 前 3 epoch residual scale为 0.05，随后 3 epoch ramp到1.0。proposal分支先恢复可用信息，再训练 residual。proposal相关旧loss权重同步降低：

- proposal `3 → 2`；
- deployment selection `6 → 5`；
- counterfactual proposal `4 → 2`；
- deployment-HAB dense winner loss保留高权重 24。

### 6.5 Stage-decoupled residual routing

对 selected sparse anchor选错的场景：

```text
if dense_local_winner == teacher:
    proposal_failure_residual_weight = 0.1
else:
    intrinsic_correction_weight = 1.0
```

该权重同时进入 residual winner correction、boundary margin distillation与 certified winner loss。新增两类场景比例，便于判断问题究竟在 proposal还是residual。

### 6.6 Gate-feasible checkpoint selection

V61 对以下 Minimum shortfall加入高权重惩罚：

- proposal recall 0.72；
- selected recall 0.50；
- effective recall 0.62；
- interaction recall 0.40；
- evidence certificate 0.40；
- fallback <= 0.60。

并记录 `val_minimum_gate_feasible`。这会保留类似 V60 epoch 1 的可行checkpoint，而不是被epoch 3的微小综合分差覆盖。

### 6.7 Strict provenance gate

V61 gate 强制检查：

- set-conditioned diagnostics覆盖率 >= 99%；
- configured rank与实际rank一致；
- exact runtime HAB实际执行；
- fast/exact/global metrics完整；
- proposal loss没有严重runaway；
- proposal logit RMS稳定；
- residual stage routing diagnostics存在。

### 6.8 Warm start策略

主实验必须从 immutable V51 foundation anchor warm start：

```text
/home/senzeyu2/code/BDSE_planner/outputs_v51_far_dbap_2gpu_v1/
foundation_anchor/train/bdse_v51_foundation_anchor.best.pt
```

不得从 V60 best checkpoint继续，因为 V60 proposal/residual heads已处于失稳/错误目标状态。V61 config会重新初始化 residual与set heads；proposal/family heads应从foundation或fresh状态训练。

---

## 7. 工程错误与误判防护

### 已确认并修复

1. **Checkpoint选择违反Minimum gate**：修复为可行性优先。
2. **Global Top-M冒充runtime HAB**：新增分离指标与exact sampled forward。
3. **Uncentered ST logit shortcut**：中心化、固定soft mass、logit稳定正则。
4. **Set head结果不可观测**：strict JSONL coverage gate。
5. **Protocol checker不拦截proposal runaway**：新增loss growth/RMS检查。
6. **Residual补偿proposal错误**：新增stage routing。
7. **Gate fail后自动跑CL20浪费算力**：默认 `RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL=0`。

### 已排除的伪问题

- `L_sel == L_prop` 是日志alias，不是total loss中重复相加；
- calibration epsilon不是零gain的唯一原因：raw proposed action本身均非teacher，且raw margin低于flip threshold；
- B16 selector不是首要瓶颈：它与sparse-full winner match为0.981。

### 仍需通过fresh run验证

- fast GPU HAB与exact runtime HAB的Jaccard；
- V61 proposal logits是否真正稳定；
- set head在actual JSONL中的激活与幅度；
- residual stage routing后，intrinsic correction是否产生teacher-directed proposal；
- latency是否因新training路径变化。部署inference本身未增加exact训练路径，因此理论上不增加runtime query budget，但需重新测p95 latency。

---

## 8. 训练与测试效率

V60每epoch约16.1–16.8分钟（epoch 2–7），主要耗时：

- loss computation约 `227–234 ms/step`；
- pair sampling约 `28–32 ms/step`；
- forward约 `20–21 ms/step`；
- backward约 `18 ms/step`；
- data wait约 `1.1–1.3 ms/step`。

因此优化重点应放在loss/selector路径，而非继续增加DataLoader workers。

V61采用：

- all-scene GPU fast HAB；
- 2 scenes/rank、每2 step一次exact runtime HAB；
- 最后少量step全exact；
- CPU exact selector process backend；
- no gate-fail CL by default；
- fail-fast training health tool。

本地CPU synthetic microbenchmark（仅用于相对开销估计，不代表目标GPU）：

| 路径 | 时间 |
|---|---:|
| global Top-M, batch 8 | 0.069 ms |
| fast HAB, batch 8 | 2.72 ms |
| exact HAB, 2 scenes | 1.19 ms |
| 按每2 step抽样后的exact平均 | 0.59 ms/step |

相对 V60 `~230 ms/step` 的loss computation，抽样exact路径预计是低个位数百分比量级；真实GPU/DDP环境仍需从新日志确认。若fast/exact Jaccard高于0.9，可进一步降低exact frequency；若低于0.7，不能为了速度减少exact supervision。

新增 `check_v61_training_health.py`，在epoch 1/3后可立即停止以下失败run：

- L_prop >100且增长>8x；
- logit RMS >20；
- fast/exact Jaccard <0.70；
- proposal recall <0.72；
- latest checkpoint不满足Minimum gate。

这比训练完整10 epoch再calibration/open-loop节省更多时间，且不牺牲性能。

---

## 9. Dataset 诊断与test工程问题

### Validation set

val共 `58,418` 样本、无identity重复。上游oracle ceiling较高：

- full-interface oracle action match `0.96566`；
- B16 oracle decision sufficiency `0.91201`；
- runtime decision sufficiency `0.74903`；
- evidence sufficiency `0.59216`；
- selector value ratio `0.66035`。

说明val上的主要模型差距不是数据上限不足。需要注意：safe candidate exists `0.7173 < 0.75`，teacher candidate ADE p50/p90也未过原数据gate，论文应单独披露candidate coverage限制。

### Test set

当前test为 `67,042` 个已构建样本，但用户明确说明尚未完成。现有诊断明显差于val：

- full-interface action match `0.93428`；
- B16 oracle `0.83992`；
- runtime sufficiency `0.64073`；
- safe candidate exists `0.57786`；
- route distance p95-p90 `17.55 m`。

此外 `full_interface_teacher_regret=-1961.54`。如果teacher action定义为同一 `J_T` 上的argmin，则teacher regret理论上不应为负；该字段可能存在teacher/action index、cost sign、mask或aggregation错误。修复前：

- 禁止使用test选择checkpoint或调超参；
- 禁止把test现有结果写为论文主结果；
- 应先冻结完整test manifest、重算teacher argmin一致性，并添加 `regret >= -tol` assertion。

---

## 10. 下一轮成功判据

按以下顺序判断，不允许跳级：

1. **Training integrity**：exact HAB fraction >0；proposal RMS <20；L_prop不爆炸；fast/exact Jaccard>=0.70，目标>=0.90。
2. **Minimum gate**：proposal recall>=0.72，且epoch 1→3不再单调下降；best checkpoint `val_minimum_gate_feasible=1`。
3. **Proposal winner bridge**：sparse-full match >0.141；budget-vs-full >0.172；global match不作为通过依据。
4. **Selector preservation**：budget-vs-sparse-full保持接近0.98，避免修proposal时破坏B16 selector。
5. **Residual directionality**：raw proposed action中beneficial > harmful，并且至少部分 proposed action等于teacher。
6. **Set-head provenance**：四个set diagnostics覆盖率约100%，rank=8，abs mean/scale非异常。
7. **Pair-full gain**：pair-full residual gain >0，再看certificate是否阻挡。
8. **Deployed gain**：candidate-local teacher match gain >0；达到正式阈值后再运行CL20。
9. **Closed loop**：paired NR-CL20与frozen reactive-CL20无安全退化后，才运行CL100。

若第3步仍不改善，但fast/exact Jaccard高：问题转向proposal features/family allocation表达力，应增加winner-critical family/atom supervision，而不是继续改selector。若第3步改善而pair-full仍为0：再升级set potential或teacher-directed residual。若pair-full改善但deployed为0：再分析calibration/certificate，而不是提前放宽阈值。

---

## 11. 代码验证结果

本地完成：

```text
Python compile: PASS
V61 YAML: 4/4 PASS
Shell syntax: 3/3 PASS
Targeted V60+V61 tests: 8 passed
Full unit tests: 231 passed, 12 warnings
Strict V60 re-audit: Protocol/Minimum/Competitive = FAIL/FAIL/FAIL under V61 checker
Patch dry-run: PASS
ZIP integrity: PASS
```

警告来自PyTorch Transformer nested tensor配置，不是本次修改引入的测试失败。

当前环境未执行fresh V61训练、calibration、open-loop或nuPlan closed-loop，因此不能预先声明V61通过gate或闭环提升。
