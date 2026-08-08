# V64.3.1 AP-WCCA Activation Screen 审计与 V64.3.2 优化

## 结论

本次 `RUN_V64_3_1_APWCCA_ACTIVATION_SCREEN_2GPU.sh` 的 `continue_to_full_run=false` **不能被解释为 AP-WCCA 已经算法失败**。至少有三个工程问题污染了 screen 的判定，其中两个直接使 screen summary 无法回答它声称要回答的问题：

1. `val_critical_topm_recall_last` / `val_critical_selected_recall_last` 为 NaN，是因为训练期 validation 根本没有写出 screen 脚本读取的 formal criticality keys；不是 critical recall 的真实 NaN。
2. `apwcca_activated=false` 只依赖 forward diagnostic scalar；但当前上传源码与运行日志对该 scalar 的数值语义不一致，因此这个 flag 不能可靠证明 adapter 参数没有更新。
3. 通用 launcher 无条件用 CLI 默认值覆盖了 screen YAML 的稀疏 exact-selector supervision：YAML 要求 `scenes_per_rank=1, every_n_steps=4`，实际运行被覆盖为 `0,1`，即全 batch、每一步都执行 exact CPU selector。这既改变训练目标占比，又造成严重性能浪费。

因此最合理的下一步不是换掉 AP-WCCA，而是先跑一个 **instrumentation-correct、same-subset step-zero controlled screen**。

---

## 1. 本次 screen 中哪些结果可信

训练确实使用 V64.3.1 config，`critical_proposal_adapter` 也在 trainable prefixes 中；没有再出现上一轮 V64.2/V64.3 train-config 串线。

训练侧 literal teacher winner-flip critical Top-M recall 是有限值：

| epoch | train exact critical Top-M recall |
|---:|---:|
| 0 | 0.364829 |
| 1 | 0.367799 |
| 2 | 0.368384 |
| 3 | 0.367973 |

所以 screen summary 中的 NaN 不是数据本身产生的 NaN。

500-scene validation 上可观察到：

| metric | epoch 0 | epoch 3 |
|---|---:|---:|
| proposal decisive recall | 0.771838 | 0.766953 |
| proposal interaction decisive recall | 0.766310 | 0.761787 |
| selected decisive recall | 0.567085 | 0.568271 |
| selected interaction recall | 0.519455 | 0.521958 |
| teacher action match | 0.260 | 0.262 |
| DA-EPC certificate fraction | 0.970 | 0.962 |
| fallback | 0.000 | 0.000 |
| B16 vs pair-full winner match | 0.956 | 0.956 |

这些结果说明：

- fixed-budget B=16 / DA-EPC 在本 screen 中不是主要瓶颈；
- broad proposal recall 有约 0.5 pp 的轻微下降，值得监控；
- selected recall 和 teacher match 没有明显恶化；
- 由于没有 step-zero 同一 validation subset baseline，不能把 `0.76695 < 0.78` 直接当成算法失败。历史 0.78/0.80 数字来自不同场景数、不同 checkpoint/protocol，绝对阈值会把 subset variance 与训练效果混在一起。

---

## 2. NaN 的直接工程根因

旧 screen 读取：

- `val_teacher_exact_winner_flip_critical_recall_topm`
- `val_teacher_exact_winner_flip_critical_recall_selected`

但 V64.3.1 的训练期 `_run_validation_open_loop` 没有调用 formal open-loop 使用的 `_criticality_metrics`，所以这两个 key 根本不存在。旧脚本的 `last()` 在没有 key 时返回 NaN。

V64.3.2 已让 training-time validation 复用 formal open-loop 的 literal teacher winner-flip criticality 实现，避免训练 screen 与正式 gate 使用不同定义。

---

## 3. 为什么旧 `apwcca_activated=false` 不可信

当前上传源码中 `critical_proposal_residual_rms` 的实现包含数值稳定项；即使 adapter 输出严格为零，也不应与运行日志里的严格 `0.0` 完全一致。与此同时，validation proposal metrics 在 epoch 间发生变化，而 nominal V64.3.1 已冻结 legacy proposal/family stack；这与“AP-WCCA 完全没有改变 proposal path”相矛盾。

因此 V64.3.2 不再使用 forward output scalar 判断 activation，而直接记录：

- critical adapter 参数初始 snapshot；
- `critical_adapter_parameter_delta_rms`；
- `critical_adapter_parameter_delta_max_abs`；
- 当前 adapter parameter RMS；
- 关键源码 SHA-256。

Activation 定义改为参数真实变化，而不是某一个输出统计量是否大于阈值。

---

## 4. 冻结 foundation 仍存在 dropout 语义问题

PyTorch 中 `requires_grad=False` 并不会让模块自动进入 eval mode。旧训练代码在每个 epoch 调用 `model.train()`，会重新打开 frozen scene/action/evidence encoders 内的 dropout。

这会使所谓 immutable foundation anchor 在训练 forward 中仍带随机性，并污染 zero-init residual adapter 的学习目标。

V64.3.2 新增 `training.eval_frozen_modules=true`：每次 `model.train()` 后，所有完全冻结的 top-level module 被递归设为 eval；trainable adapter/residual heads 保持 train mode。

---

## 5. screen 很慢的真实根因

本次 4 epoch 累计训练 wall time约 1394.7 s，其中：

- loss construction: 1163.7 s，**83.4%**；
- pair sampling: 58.4 s，4.2%；
- forward: 43.8 s，3.1%；
- data wait: 41.5 s，3.0%；
- backward: 31.4 s，2.3%；
- H2D: 1.1 s，0.08%。

这次 screen 与上次 full run 的瓶颈不同：不是 DataLoader，而是 loss/exact-selector construction。

Screen YAML 原本写：

- `deployment_selector_scenes_per_rank=1`
- `deployment_selector_every_n_steps=4`

但旧 launcher 默认：

- `SELECTOR_SCENES_PER_RANK=0`
- `SELECTOR_EVERY_N_STEPS=1`

并无条件将 CLI 传给 train.py，覆盖 YAML。运行日志也显示 `selector_exact_fraction=1.0`。

V64.3.2 改为：只有用户明确设置 selector env override 时才传 CLI，否则严格服从 YAML。预期会显著降低 screen loss-stage 时间；实际加速比必须由服务器 fresh run 测量，不能提前宣称。

---

## 6. 算法层面目前能得出的结论

### 不能得出的结论

不能说 AP-WCCA 已经失败，因为：

- val literal critical acquisition 没有被正确记录；
- activation 判定不可靠；
- exact-selector training cadence 被 launcher 改写；
- frozen backbone 在 train mode 下仍可能带 dropout。

### 仍然值得关注的算法信号

在污染条件下，broad proposal decisive / interaction recall 有轻微下降，而 train critical Top-M 在约 0.368 附近平台化。这说明即使 corrected AP-WCCA 激活后，也必须严格监控“提高 rare critical acquisition 是否破坏 broad evidence coverage”。

因此 V64.3.2 **不解冻 legacy proposal/family stack、不增加 B/M、不修改 hard selector、不重新定义 criticality**。

---

## 7. 算法日志约束与本轮避免的重复尝试

根据历史修改日志，本轮明确不重复：

- V40–V43 的昂贵组合搜索 / repair / beam 路线；
- 增加 B 或 M；
- 用 margin deficit 重新定义 critical；
- 再次整体解冻 legacy proposal/family stack；
- 单纯继续放大 global hardest-negative ranking，V64.2 已显示它可能伤害 broad recall；
- 在没有 raw/calibrated evidence 支持时手工放宽 residual calibration。

保留论文核心：fixed planner-interface evidence budget、auditable atoms、budget 内 deterministic/exact selector、literal winner-flip criticality。

---

## 8. V64.3.2 的最小算法补强：ACRA

为了避免 zero-init AP-WCCA residual 在多个组合 loss 中被淹没，新增 **Anchor-Centered Residual Alignment (ACRA)**：

- target 仍只来自 literal teacher winner-flip critical mask；
- 只监督新增 critical residual adapter，不改 legacy proposal logits；
- residual 在 scene 内中心化，避免学习一个无意义的全局 proposal shift；
- target 也按 scene critical rate 中心化；
- critical positive 使用较高权重；
- Smooth-L1 提供直接且稳定的 zero-init gradient。

默认参数：

- alignment weight = 0.25
- target scale = 1.0
- huber delta = 0.25
- positive weight = 8.0

这是对 AP-WCCA 的局部稳定化，不是新一轮 full proposal rewrite。

---

## 9. 可选第二阶段算法：AP-WRCCA

只有当 corrected AP-WCCA screen 满足：

- instrumentation valid；
- adapter parameter delta > 0；
- 但 same-subset critical Top-M delta <= 0；

才建议运行 AP-WRCCA screen。

AP-WRCCA 在 AP-WCCA winner conditioning 基础上增加 strongest frozen-base rival action embedding，使 acquisition 显式看到 winner-rival boundary。它仍：

- zero-init residual；
- legacy proposal frozen；
- teacher 不参与部署；
- B/M/selector 不变。

这对应 literal criticality 的真正条件结构，比继续堆 BCE/ranking weight 更有机制意义。

---

## 10. V64.3.2 corrected screen 判定逻辑

新 screen 首先在 **同一个固定 500-scene validation subset、训练前** 跑 step-zero anchor，然后每个 epoch 使用同一 subset。

必须满足：

1. instrumentation 完整；
2. adapter parameter-delta RMS > 1e-9；
3. val literal critical Top-M recall 相对 step-zero anchor **严格上升**；
4. proposal decisive recall 相对 anchor下降不超过 0.02。

不再使用历史绝对 `proposal decisive >= 0.78` 作为 screen 硬门槛。

这是 activation/causal screen，不是正式 Minimum/Competitive gate。

---

## 11. 下一阶段决策树

- Corrected AP-WCCA screen PASS → 跑 fresh V64.3.2 AP-WCCA full pipeline。
- Adapter 有真实参数更新，但 critical Top-M delta <= 0 → 跑 AP-WRCCA screen；不要先跑 full。
- critical Top-M ↑，selected critical 不升 → 下一步才看 selector tie-break / budget allocation。
- critical Top-M ↑、selected critical ↑，teacher action / regret 不升 → atom→action value / pair-margin learning 是下一瓶颈。
- open-loop ↑、paired CL 不升 → candidate dynamics / reactive interaction / replanning/cache。

一次只改变一个因果模块，避免版本迭代无法归因。

---

## 12. 落地后的工程验证

- Python compile: PASS
- shell syntax: PASS
- AP-WCCA V64.3.2 train/eval config contract: PASS
- AP-WRCCA V64.3.2 train/eval config contract: PASS
- targeted V64.3.2 tests: 5 passed
- full `bdse/tests`: **265 passed, 0 failed**
- warnings: 21，均为 PyTorch nested-tensor warning，没有 test failure

本环境没有实际 GPU/nuPlan cache/foundation checkpoint，因此没有运行 fresh corrected screen，也不提前声称 AP-WCCA 或 AP-WRCCA 会通过下一阶段 screen。
