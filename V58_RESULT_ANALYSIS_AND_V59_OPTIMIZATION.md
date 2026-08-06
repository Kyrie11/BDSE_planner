# V58 结果诊断与 V59 FSCIP-BFAR-DBAP 优化报告

## 0. 结论摘要

V58 的正式 1000-scene paired open-loop gate 报告为：

| Gate | V58 报告状态 | 审计结论 |
|---|---|---|
| Protocol gate | PASS | 训练、独立 calibration、paired open-loop 在旧 gate 范围内完整；但该 gate 不覆盖后续 closed-loop 成功性 |
| Minimum gate | PASS | evidence path 已稳定达到最低完整性要求 |
| Competitive gate | FAIL | 真实算法失败：最终 winner gain、pair-full residual gain、beneficial residual 均为零 |

V58 的 closed-loop 不是“只是尚未全部跑完”。Candidate 的两个 10-scene shard 均经历数小时仿真，但在每个 scenario 结束时因 planner 内 `threading.RLock` 无法被 `SimulationLogCallback` pickle 而失败。两个 shard 最终都是 `successful=0, failed=10`。旧脚本仍生成 combined summary 和 `.closed_loop_complete.json`，所以这些文件无效，不能用于算法判断。

V58 的 selector/evidence path 明显有效；residual winner path 仍无效。V59 因此保留 fixed-budget selector、AOCC 和双证书主线，新增低秩 selected-set interaction potential、boundary-margin distillation、scene-uniform all-rival residual calibration，并修复闭环序列化、重复计算、线程锁队列和无效完成标记。

---

## 1. V58 gate 状态与根本原因

### 1.1 Protocol gate：PASS，但范围有限

V58 gate report 中：

```text
protocol_pass = true
protocol_failures = []
```

它证明以下链路在 V58 gate 定义内运行完成：

- 8 epoch 训练；
- winner/certificate loss family 已接通；
- group-disjoint calibration；
- candidate/local/foundation 1000-scene paired open-loop；
- 三路 scenario/timestamp hash 一致；
- frozen foundation row drift 为零。

但该 protocol gate 没有把“closed-loop shard 的成功数和失败数”作为前置审计项，因此没有发现后续 20/20 simulation 全部失败。准确表述应为：

> V58 open-loop protocol gate PASS；publication-complete train→calibrate→open-loop→closed-loop protocol 尚未完成且当前 closed-loop 结果无效。

### 1.2 Minimum gate：PASS

关键指标：

| 指标 | V58 |
|---|---:|
| Evidence certificate fraction | 0.88803 |
| Fallback rate | 0.110 |
| Proposal decisive recall | 0.80247 |
| Selected decisive recall | 0.61065 |
| Effective decisive recall | 0.77631 |
| Interaction decisive recall | 0.57934 |
| Frontier retained | 0.78142 |

`minimum_failures=[]`。这说明 V56 开始的 evidence/residual certificate 分离仍然有效，且 V58 selector 已把 proposal recall 推过 0.80。

Minimum PASS 不代表最终算法有竞争力。它只说明在 B=16 下，证据检索、证据证书和 fallback 已达到既定最低完整性。

### 1.3 Competitive gate：FAIL

V58 的失败项：

```text
total teacher-match gain        = +0.000 < +0.015
residual teacher-match gain     = +0.000 < +0.005
pair-full residual gain         = +0.000 < +0.005
beneficial / harmful residual   = 0 / 0
```

三路最终 teacher match：

```text
candidate  = 0.141
local      = 0.141
foundation = 0.141
```

Pair-full：

```text
candidate pair-full = 0.141
local pair-full     = 0.141
```

最终 deployed residual flip 为零。Competitive FAIL 不是 proposal recall 差一点，也不是证书阈值设置问题，而是：

> V58 residual 没有改变任何最终 action winner；即使使用 full selected evidence，residual 也没有带来 teacher-match gain。

因此不能通过降低 gate 阈值或直接放宽 residual certificate 来解决。

---

## 2. 工程错误优先审计

### 2.1 Closed-loop RLock 序列化错误

两个 candidate shard 日志都反复出现：

```text
TypeError: cannot pickle '_thread.RLock' object
```

最终：

```text
GPU0: successful=0, failed=10
GPU1: successful=0, failed=10
```

错误发生在 `SimulationLogCallback` 的 scene-end serialization。仿真主循环先消耗了大量时间，最后才在保存 simulation log 时失败，导致全部计算无效。

V59 修复：

1. `BDSEPlannerCore.__getstate__/__setstate__` 明确移除 process-local `inference_lock`；
2. 正式大规模评估删除 `callback.simulation_log_callback`，但保留 metric main callbacks；
3. 每个 process 只使用一个 simulation worker，不再把共享 RLock 用作多线程 GPU 队列；
4. 汇总前从每个 shard 日志提取 `successful/failed`，要求 success 等于该 shard token 数且 failure 为零；
5. 只有严格验证后才创建 `.closed_loop_complete.json`。

### 2.2 旧脚本错误生成“完成”标记

V58 在 0 success / 20 failure 的情况下仍生成：

- `closed_loop_combined_summary.json`；
- `closed_loop_shard_and_combined_summary.csv`；
- `.closed_loop_complete.json`。

这是独立的工程错误。存在这些文件不能证明仿真有效。V59 将完成标记从“文件存在”改成“日志成功数、失败数、token 数、aggregator parquet 全部一致”。

### 2.3 Residual certificate 聚合指标语义容易误导

V58 aggregate 中：

```text
residual_flip_certificate_pass = 0.988
dual_certificate_deployment_certified = 0.988
```

但 proposal-conditional pass 实际为：

```text
dual_certificate_pass_conditional = 0.0
```

原因是 98.8% 场景根本没有 residual proposal，旧聚合把“无需认证”算入通过率。V59 gate 将 unconditional、proposal rate、proposal-conditional margin pass、evidence pass、dual pass 和 deployed flip 分开报告。

### 2.4 V58 residual calibration 样本稀疏

V58 calibration：

- calibration scenes：5000；
- actual residual proposals：90，仅 1.8%；
- recommended residual epsilon：0.46431；
- residual raw error MAE：0.12540；
- residual sigma mean：0.09089。

训练阶段 residual epsilon reserve 只有 0.05。部署时 0.464 的 conformal epsilon 远大于训练预留，导致本来就很小的 residual margin 全部被拒绝。

这不属于“代码崩溃”，但属于训练—校准协议错配：训练目标没有为实际校准误差留出足够 robust margin；校准又只依赖极少数已经发生 proposal 的场景。

### 2.5 V59 二次审计发现并修复的新键名风险

V59 初版校准器曾读取不存在的：

```text
residual_action_variance
```

而模型实际输出为：

```text
residual_action_var
```

如果保留宽泛异常捕获，校准会静默使用零不确定度。最终 V59 已改为严格 contract：

- 缺少 `residual_action_var` 立即报错；
- shape 不是 `[E,K]` 立即报错；
- 无法构造完整 `[K,K]` pair sigma 立即报错；
- 不再用 broad `except` 将工程错误替换为零方差。

新增回归测试专门阻止错误键名再次出现。

---

## 3. 时间剖析

### 3.1 主流水线

从日志时间戳还原：

| 阶段 | 近似时间 |
|---|---:|
| Anchor quality gate | 13 分钟 |
| 8 epoch training | 2 小时 40 分钟 |
| 2-GPU dual calibration | 25 分钟 |
| Candidate/local/foundation 并发 open-loop | 586 秒，约 9.8 分钟 |
| Gate | < 1 分钟 |

所以总体慢并不是因为 open-loop 过多。V58 已经把三路 open-loop 并发运行，正式 1000-scene × 3 系统只占约 10 分钟。

训练仍是主流水线中的最大固定成本；closed-loop 是后续最大的异常成本。

### 3.2 Candidate closed-loop

Candidate 两个 shard 从约 19:30 开始，分别到约 22:37 和 23:05 才结束，且全部失败。Local control 随后才刚启动。

Profile：

| 指标 | GPU0 shard | GPU1 shard |
|---|---:|---:|
| Planner calls | 1642 | 1592 |
| Plan cache hit | 0.7990 | 0.7990 |
| Mean core plan | 29.39 s | 25.85 s |
| Mean certificate stages | 16.67 s | 14.48 s |
| Mean final safety flags | 4.76 s | 4.80 s |
| Mean rule rerank | 1.81 s | 1.18 s |
| Max core plan | 348.61 s | 321.06 s |

由于约 80% call 命中 plan cache，按均值反推，非缓存 replanning 的平均成本约为 129–147 秒。单次 certificate path 隐含成本约为 72–83 秒。

### 3.3 慢在哪里

1. **安全几何重复计算**：同一 candidate bank 的 route distance、agent envelope、TTC、red-light/hard/soft risk 被多个函数和 fallback stage 重复调用。
2. **Scene encoder 重复执行**：B/M/L fallback 只改变 budget、proposal top-M 和 rival graph，却重复 `_make_batch` 与 `encode_context`。
3. **线程共享模型形成锁队列**：4 simulation workers 共用一份 GPU model 和 RLock；GPU forward 串行，且部分 CPU 工作位于锁作用域中。
4. **失败发生得太晚**：scene 完成后才 pickle 失败，浪费整个 rollout。
5. **Summary PDF 不是主因**：关闭 PDF 只能节省较小的后处理时间。

### 3.4 V59 性能优化

- `runtime_safety_cache_scope`：一个 planner call 内，按 runtime/candidate/safety config 缓存昂贵安全几何；跨 scene 自动清空。
- `runtime_prediction_cache_scope`：一个 planner call 内复用 batch、scene encoder context、set factors；不同 fallback stage 只重算 stage-dependent Top-M/pair query。
- 闭环改为 `2 GPUs × P processes/GPU × 1 worker/process`，默认 P=2；不再使用多线程共享 RLock。
- 提供 CL4 benchmark 比较 P=1/2/3，以目标机器实际 wall time 选择并发度。
- 默认训练/calibration/open-loop gate 完成后不自动运行闭环，先避免在 competitive 仍为零时浪费数小时。
- Open-loop 保留已有的三路并发任务池；不再增加无意义的重复 open-loop。

V59 未在当前环境跑 nuPlan，因此不能预先给出加速倍数。性能验收应看：

- `runtime_safety_cache_hits/misses`；
- `runtime_prediction_cache_hits/misses`；
- noncached core-plan mean/p95；
- CL4 suite wall time；
- 每个 shard success/failure。

---

## 4. V58 的正向信号

### 4.1 Selector 已有效

V58 继续提高或稳定：

- proposal recall 已超过 0.80；
- selected recall 超过 0.61；
- interaction recall 接近 0.58；
- certificate 维持 0.888；
- fallback 维持 0.11；
- B16 budget 与 pair-full winner match 为 1.0。

最后一项非常重要：V58 中 pair-full winner 与 B16 winner 完全一致，说明当前 competitive failure 不是 B16 又丢掉了 residual 已经学到的 winner。Residual 在 pair-full 本身就没有 gain。

### 4.2 Winner loss 训练链已接通

V58 训练指标：

| Loss | Epoch 0 | Epoch 7 | 判断 |
|---|---:|---:|---|
| Residual winner correction | 9.5447 | 7.2470 | 有梯度、明显下降 |
| Residual uncertainty | 2.5344 | 2.2538 | 有下降 |
| Certified residual winner | 3.0916 | 2.8699 | 早期下降后停滞 |
| Action-potential teacher | 0.3720 | 0.3826 | 无效/恶化 |
| Atomwise residual | 0.01203 | 0.01204 | 基本不学习 |

这说明 V57 的训练断链修复仍然有效；V58 不是“loss 没跑”。问题是 loss 下降没有转化为 final winner change。

### 4.3 双证书分离仍值得保留

Evidence certificate 和 fallback 没有再被 residual variance 污染。Residual certificate 虽然过于保守，但“只有 proposal 时单独检查”的结构是正确方向，不应退回 mixed certificate。

---

## 5. 哪些设计保留、深化或替换

### 5.1 保留并继续深化

- immutable base + dense-local foundation anchor；
- fixed B=16 planner-interface budget；
- proposal/selected/effective/interaction recall 分层审计；
- boundary/hard/near-tie pair curriculum；
- sparse periodic exact AOCC；
- direct action potential 的 integrability；
- evidence certificate 与 residual flip certificate 分离；
- selected-local zero-residual control；
- matched foundation control；
- group-disjoint tune/calibration；
- paired open-loop hashes；
- candidate/local/foundation 并发 open-loop。

### 5.2 当前无效或不足

1. **纯 additive per-evidence residual**：无法表达 evidence set 中的组合交互，pair-full 仍无 winner gain。
2. **全局 action-potential reconstruction**：平均重构目标与最终 winner 边界不一致，且 V58 loss 恶化。
3. **atomwise residual reconstruction**：loss 平坦，无法识别改变 anchor winner 所需的联合贡献。
4. **proposal-conditional residual calibration**：只有 90 个样本，epsilon 不稳定且与训练 reserve 严重错配。
5. **selector 主导的 checkpoint score**：所有 residual gain 为零时仍能选出“最佳”epoch。
6. **通过放宽证书获得 flip**：不应采用；这会把无法校准的 residual 直接部署，破坏安全归因和论文可信度。

---

## 6. V59 算法优化

V59 名称：

> **Focused Set-Conditioned Certified Integrable-Potential BFAR-DBAP（FSCIP-BFAR-DBAP）**

### 6.1 Low-rank selected-set interaction potential

V59 保留每个 evidence 的 additive potential，并新增：

```text
set summary z_S = tanh(sum_{i in S} phi_i / sqrt(|S|))
set potential h_set(a) = <psi_a, z_S> / sqrt(r)
final residual h_S(a) = sum_{i in S} h_i(a) + h_set(a)
```

默认 rank `r=8`。

优势：

- 仍只使用已选 B=16 evidence，不增加 query budget；
- 直接输出 action potential，pair margin 自动 antisymmetric、cycle-consistent；
- 可以表达“两个 evidence 单独不足、组合后改变 winner”的交互；
- 不恢复 arbitrary pair field 或 Hodge projection；
- 便于做 rank=0/4/8/16 ablation。

### 6.2 Boundary-margin distillation

新增 `L_residual_boundary_margin_distill`：

- 仅训练 selected-local anchor 错误；
- teacher winner 相对 anchor 必须有最小真实 margin；
- 直接回归 predicted corrected margin 到 teacher margin；
- 对大 teacher margin 与 correctable scenes 加权；
- 不再让所有 action/evidence 平均重构支配训练。

V59 权重调整：

- action-potential teacher reconstruction：14 → 4；
- atomwise residual：4 → 1；
- certified residual winner：提高到 16；
- boundary-margin distillation：24；
- residual winner correction：12。

### 6.3 Scene-uniform all-rival conformal calibration

每个 calibration scene 固定 selected-local anchor `a0`，对所有 valid rival `r` 计算：

```text
score(a0,r) = predicted_margin(a0,r)
              - teacher_margin(a0,r)
              - beta * sigma(a0,r)
scene_score = max_r score(a0,r)
```

然后对每场一个 `scene_score` 做 split-conformal quantile。

这使 5000 scenes 理论上可贡献接近 5000 个 residual calibration scores，而不是只依赖 90 个实际 proposal。它对部署时任意被模型选中的 rival 提供统一上界。

该方法更保守，因此必须与更强的 set-conditioned robust margin 学习配套。V59 保留 actual-proposal epsilon 作为 diagnostic，不作为正式部署 epsilon。

### 6.4 Competitive checkpoint selection

V59 score 显式奖励：

- candidate-local teacher gain；
- pair-full residual gain；
- beneficial-harmful；
- robust margin；
- raw proposal rate；
- selected/interaction recall；
- low fallback。

当 residual gain 与 pair-full gain 同时非正时施加大额 penalty，防止 selector-only checkpoint 被当成论文主 checkpoint。

---

## 7. V59 工程防错设计

- 严格 residual variance key contract；
- calibration 不再 broad-except；
- control configs 在运行时同时清零 additive residual、variance、set factors；
- train/runtime set scale 保持一致；
- RLock pickle 回归测试；
- safety cache 不跨 planner call 回归测试；
- set potential winner flip 与 cycle integrability 回归测试；
- gate 要求新 boundary loss 非零；
- gate 要求 scene-uniform calibration method、独立 split、样本数和覆盖率；
- closed-loop shard 成功数和失败数强校验；
- NR 与 reactive 输出根目录分离；
- fresh V59 output root，禁止复用 V58 calibration 或 invalid CL marker。

---

## 8. 下一轮判断树

### 8.1 Smoke test 必须满足

```text
action_family_enabled > 0
selector_exact_fraction > 0
L_deploy_select > 0
L_pair_full_action > 0
L_residual_winner_correction > 0
L_certified_residual_winner > 0
L_residual_boundary_margin_distill > 0
L_residual_action_uncertainty > 0
certified_correctable_fraction > 0
set residual heads use 5x LR
variance head uses 2x LR
```

### 8.2 Open-loop 结果解释

- **pair-full gain 仍为 0**：set-conditioned residual 仍未学会 winner；提高 hard-correctable mining或 rank，不能放宽 certificate。
- **pair-full gain > 0，B16 gain = 0**：问题转为 selector-set coupling；强化 exact selector target 对 corrected winner 的保存。
- **raw proposal 有 beneficial，但 conditional cert pass=0**：检查 uniform epsilon、raw error、sigma 与训练 reserve；先改善 calibration/uncertainty，不得直接降低 epsilon。
- **beneficial 与 harmful 同时增多**：加强 do-no-harm correct-anchor loss与 safety-conditioned gating。
- **open-loop gain 转正但 CL 退化**：做 rollout distribution failure taxonomy，优先分析 collision/TTC/progress，而不是继续放大 residual。

### 8.3 Competitive 投稿前最低要求

- 三个 gate 全 PASS；
- pair-full 与 B16 residual gain 均稳定为正；
- beneficial 显著高于 harmful；
- paired bootstrap CI；
- NR 与 R-CL20 均完成且 success=20/failure=0；
- CL100；
- B=8/16/24 与 rank=0/4/8/16；
- random/score-only/greedy/no-residual/full-budget baselines；
- fixed-budget latency与完整 profile；
- complete held-out test 仅在冻结后一次使用。

---

## 9. 交付与验证边界

代码静态与单元验证通过：

- Python compile：PASS；
- 4 个 V59 YAML：PASS；
- 6 个 V59 Shell：PASS；
- Unit tests：219 passed，8 warnings；
- RLock serialization regression：PASS；
- safety memo isolation：PASS；
- set-conditioned integrability：PASS；
- residual variance key contract：PASS。

当前环境没有 nuPlan、GPU 数据集和 fresh V59 输出，因此没有执行训练、calibration、open-loop、NR/R closed-loop。本文不预先声称 V59 会通过 competitive gate或达到 CCF-A 竞争结果。
