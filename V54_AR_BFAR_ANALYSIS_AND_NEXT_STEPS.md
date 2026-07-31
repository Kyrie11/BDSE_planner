# V53 实验诊断与 V54 AR-BFAR-DBAP 设计

日期：2026-07-31

## 1. 结论先行

V53 的流水线状态如下：

- immutable anchor gate：**PASS**；
- residual/selector fresh training：**已完成**；
- candidate、same-checkpoint local control、matched foundation control 独立 calibration：**已完成**；
- 三路严格配对的 1000 场景 open-loop：**已完成**；
- protocol-integrity：按修正后的同接口检查为 **PASS**；
- minimum-completeness gate：**FAIL**；
- competitive gate：**FAIL**；
- CL20/CL100：**没有运行**。

因此 V53 首次真正验证了 V52/V53 的 selector、boundary curriculum 和 residual 路径，但仍没有提供任何 closed-loop 证据。

V53 的主要问题不是 selector 完全不会挑 evidence，也不是所有 gate 阈值都过高。结果反而表明：

> selector 已经能以较高 coverage 和 certificate 保留其训练目标，但当前 pair tournament 的训练目标本身不是一个可靠、可积的 action winner。

V54 的主修改是把 pair tournament 从“由 `J0` 和稀疏边重新拼出 action graph”改成：

> **selected-local integrable action cost + learned residual correction**。

零 residual 时，V54 必须精确退化为同一 B=16 evidence set 下的 local planner；residual 只有通过完整 margin 认证后才允许改变 winner。

---

## 2. V53 gate 的准确结果

### 2.1 V53 minimum-completeness gate：FAIL

失败项：

1. candidate teacher action match `0.100`，低于 foundation `0.116 - 0.010`；
2. pair-full action match `0.100`，低于 local pair-full `0.118 - 0.010`；
3. gate 报告的 frozen anchor drift `0.018 > 0.005`；
4. residual beneficial/harmful 为 `0.015 / 0.033`，净有害。

其中第 3 项是工程 gate bug。V53 比较了：

- candidate 的 `local_pair_full_interface_action_match`；
- local control 的 `pair_full_interface_action_match`。

二者不是同一个接口。逐行比较同接口后：

- candidate/local 的 `local_pair_full_action` 完全相同，drift 为 `0`；
- candidate/local 的 dense `full_action` 完全相同，drift 为 `0`。

修正该 bug 后，minimum gate 仍会因为 action match 退化和 residual 净有害而 FAIL。因此最终失败不是仅由 gate bug 造成。

### 2.2 V53 competitive gate：FAIL

失败项：

- candidate 相对 foundation teacher-match gain：`-0.016 < +0.015`；
- residual 相对 same-checkpoint local control gain：`+0.000 < +0.005`；
- pair-full residual gain：`-0.018 < +0.005`；
- beneficial residual `0.015` 小于 harmful `0.033`；
- harmful residual `0.033 > 0.030`。

selector/evidence 指标本身并未导致 competitive gate 失败。

---

## 3. V53 pipeline 是否完整

### 已完整执行

1. log-disjoint `val_tune/val_calib` protocol；
2. matched immutable foundation replay；
3. anchor gate；
4. residual/selector 6-epoch fresh training；
5. candidate calibration；
6. same-checkpoint local control calibration；
7. matched foundation control calibration；
8. 三路严格配对 1000-scene open-loop；
9. minimum/competitive gate 与 paired regret 分解。

### 未执行

1. candidate/local/foundation paired CL20；
2. CL100；
3. Val14 NR/R；
4. Test14/Test14-Hard；
5. B=8/B=16/B=24 closed-loop budget curve；
6. equal-budget random/score-only/greedy baselines；
7. no-residual/no-certificate/full-budget closed-loop ablations。

结论：V53 已经足以分析 open-loop 算法结构，但仍不足以证明 fixed-budget closed-loop idea 有效。

---

## 4. V52/V53 三个核心问题的验证结果

## 4.1 Pairwise sign 好，但最终 action match 低

**没有解决。**

强 anchor：

- dense winner-rival sign：`0.803`；
- dense near-tie sign：`0.706`；
- dense all-pair sign：`0.718`；
- dense full-interface action match：`0.359`；
- B=16 selected-local sparse action match：`0.141`。

弱 pair interface：

- local pair-full action match：`0.118`；
- residual pair-full action match：`0.100`；
- V53 final action match：`0.100`；
- residual pair sign all/winner/near-tie：约 `0.519/0.565/0.354`。

这说明问题不是 dense/local evidence 完全没有 decision signal，而是 pair tournament 在把局部 margin 合成为全局 winner 时破坏了原有的可积 action cost。

### 根因

V53 pair tournament 从 `J0` margin 出发，只在被查询的稀疏 pair edge 上加入 local/residual pair delta。未被查询的边回退到 `J0`。这会丢掉 selected evidence 的 action-local cost：

\[
J_{B}(a)=J_0(a)+\sum_{i\in S_B}g_i(a)
\]

于是原本可积的 `J_B(a)` 被一个不完整、可能不满足 cycle/transitivity 的稀疏 pair graph 替代。

## 4.2 Selector 是否锁定 decisive evidence

**coverage 与 certificate 已明显成功，但 action target 没有成功。**

V53 candidate：

- proposal decisive recall：`0.801`；
- selected decisive recall：`0.588`；
- effective selected decisive recall：`0.754`；
- selected interaction-decisive recall：`0.555`；
- AOCC certificate：`0.821`；
- fallback：`0.157`；
- frontier retained：`0.760`；
- 预算填充：`16/16`；
- budget-vs-pair-full match：`0.987`。

这些指标已经达到上一轮设定的 competitive selector 目标。然而 `pair-full action match=0.100`。因此 selector 在 98.7% 场景中忠实保留的是一个错误或较弱的 pair-full winner。

正确结论不是“selector 仍完全筛不到 decisive evidence”，而是：

> 当前 decisive target 和 certificate 依赖于错误的 pair tournament reference。selector 搜索本身已经不是第一瓶颈。

## 4.3 训练是否减少无关 pair 计算

**已经显著改善。**

V53：

- 前 4 epochs：约 `20–21 min/epoch`，`39.6–41.2 samples/s`；
- 后 2 epochs：约 `28–30 min/epoch`，`28.0–29.1 samples/s`；
- training pair fraction：约 `0.779`；
- full-graph fraction：约 `0.25`；
- exact scene fraction：约 `3.12%`，末 epoch `5.11%`；
- data loading 不是瓶颈。

对比 V51 的约 `42–79 min/epoch`，V52/V53 的 boundary sampling 与稀疏 exact supervision 已证明有效。

后两轮变慢主要来自 final/full alignment 与 pair sampling/cycle mining，而不是 DataLoader。

---

## 5. Minimum 与 competitive gate 暴露的主要问题

### 5.1 Minimum gate 暴露的问题

Minimum gate 失败说明目前 candidate 甚至还不能稳定保持 matched control 的 action quality：

- selector 对 foundation 的总 gain 为 `-0.016`；
- pair residual 相对 local pair interface gain 为 `-0.018`；
- residual beneficial/harmful 为 `0.015/0.033`；
- candidate/local 最终 deployed action 只有 `0.4%` 不同，说明 residual 在最终部署几乎不起作用；
- pair-full candidate/local 只有 `0.5%` 不同，但训练后的 pair graph 相对 immutable foundation 大幅漂移。

这不是“因为阈值要求必须有大提升所以无法跑闭环”。即使把 gain 阈值降为 0，candidate 仍低于 foundation，residual 仍净有害。

### 5.2 Competitive gate 暴露的问题

Competitive selector 指标已基本达标，但 action gain 完全未达标。说明论文主线中的前半段：

```text
flip-critical evidence → fixed-budget coreset
```

已经出现有效信号；后半段：

```text
coreset → correct action margin → certified winner flip
```

仍未建立。

---

## 6. 工程问题及修复

## 6.1 Gate 比较不同接口

已修复。V54 使用逐场景、同接口 action 比较：

- candidate/local `local_pair_full_action`；
- candidate/local `full_action`。

V53 数据在修正后 protocol-integrity 为 PASS，local/dense anchor row drift 均为 `0`。

## 6.2 Minimum gate 导致 closed-loop 永久饥饿

V53 在 minimum gate FAIL 后立即终止，导致无法知道 open-loop 退化是否会在真实闭环中出现、放大或被 planner fallback 吸收。

V54 将 gate 拆为三层：

1. **protocol-integrity gate**：失败时闭环结果无效，阻断全部 CL；
2. **minimum-completeness gate**：决定 CL20 能否作为正式最小完整性结果；
3. **competitive gate**：决定是否升级到 CL100/论文主结果。

当 protocol PASS、minimum FAIL 时，默认仍运行严格配对的 candidate/local/foundation CL20，并写入：

```text
$OUT_ROOT/closed_loop/.diagnostic_cl20
```

该结果仅作算法诊断，不计作投稿 PASS。

## 6.3 Latency

V53 p95 planner latency 为约 `1139 ms`，超过 500 ms 警戒线。当前仍将其作为独立 deployment warning，不阻断算法闭环。原因是当前首要目标是验证算法闭环方向；但论文中不能声称 real-time，后续必须进一步 profile prediction、pair inference 和 selector。

---

# 7. V54 AR-BFAR-DBAP

全称：**Anchor-Relative Boundary-Focused Anchor-Residual Decision-Budget Action Preservation**。

## 7.1 保持不变的论文主线

\[
\text{complete decision boundary}
\rightarrow
\text{flip-critical evidence}
\rightarrow
\text{fixed }B=16\text{ coreset}
\rightarrow
\text{certified residual correction}
\rightarrow
\text{action winner}
\rightarrow
\text{closed loop}
\]

修改的是错误的 action aggregation 接口，而不是放弃 evidence-budget 核心 idea。

## 7.2 Selected-local integrable anchor

V54 在推理时先计算：

\[
J_{B}^{L}(a)=J_0(a)+\sum_{i\in S_B}g_i(a)
\]

pair margin anchor 为：

\[
m_B^{L}(a,b)=J_B^L(a)-J_B^L(b)
\]

pair head 只预测 residual：

\[
r_B(a,b)=\sum_{i\in S_B}
\left[\hat\Delta_i(a,b)-\Delta_i^L(a,b)\right]
\]

最终 margin：

\[
m_B^{AR}(a,b)=m_B^L(a,b)+r_B(a,b)
\]

因此，当 residual 为零时：

\[
m_B^{AR}(a,b)=m_B^L(a,b)
\]

即 pair tournament 精确退化为 selected-local planner，不再丢弃已保留 evidence 的 action cost。

## 7.3 Certified action-anchor guard

pair tournament 提议的新 action 只有同时满足以下条件才可覆盖 selected-local winner：

1. proposed action 对 anchor winner 的 uncertainty-shrunk robust pair margin 超过阈值；
2. proposed tournament score 比 anchor winner 有正 gain；
3. safety/valid mask 允许。

否则 final action 回退到 selected-local anchor。

该 guard 使“residual 可改变 winner”成为明确、可统计、可消融的事件，而不是 pair graph 的隐式副作用。

## 7.4 Anchor-relative training

### Full evidence target

使用：

\[
J_F^L=J_0+\sum_i g_i
\]

作为 full-local anchor，只训练 residual correction。

### B=16 target

使用 selected-local：

\[
J_B^L=J_0+\sum_{i\in S_B}g_i
\]

作为 budget anchor，再叠加 residual。

### Teacher-correct scene

若 anchor winner 已是 teacher winner：

- 强制 residual do-no-harm；
- `pair_full_anchor_preserve` 保持 winner；
- `budget_preserve_pair_full` 保持 B=16 winner。

### Anchor-wrong scene

若 anchor winner 错误：

- teacher action 获得 correction target；
- winner-vs-strongest-rival loss 直接作用于 residual tournament；
- correction 权重由 `anchor_wrong_action_weight` 提升。

## 7.5 训练速度进一步优化

V54：

- epochs：6 → 4；
- ordinary max pairs：64 → 48；
- full graph cadence：4 → 8 steps；
- exact selector cadence：4 → 8 steps；
- full-exact tail：64 → 32 steps；
- cycle cadence：4 → 8 steps；
- consistency triangles：48 → 32；
- B=16 仍是 exact primary deployment budget；
- B=8/B=24 仍仅作稀疏 robustness regularizer；
- 没有用 surrogate 替代 exact AOCC。

实际速度需服务器日志确认。

---

## 8. 投稿目标差距与分阶段目标

不存在官方的“CCF-A 分数录用线”。以下是工程化目标，而不是录用保证。

## 8.1 最低投稿完整性

下一轮必须先补齐：

- three-way paired CL20，即使是 diagnostic；
- protocol PASS；
- candidate 不产生新增 collision/TTC/drivable-area 灾难；
- minimum open-loop 非灾难退化；
- B=8/16/24 open-loop/closed-loop curve；
- random、score-only、greedy 等 equal-budget baselines；
- no-residual、no-certificate、selected-local-only、full-budget ablations；
- paired bootstrap confidence interval；
- hard/interaction scenario failure taxonomy。

V54 下一轮最低 open-loop 目标：

- selected-local anchor match 不低于 V53 sparse full `0.141`；
- candidate teacher match至少不低于 selected-local/local control超过 `0.005`；
- candidate pair-full 不低于 selected-local anchor超过 `0.005`；
- harmful residual `<=0.03`，且 beneficial >= harmful；
- selected decisive recall `>=0.55`；
- effective decisive recall `>=0.70`；
- interaction decisive recall `>=0.50`；
- certificate `>=0.55`；
- fallback `<=0.40`。

最重要的是拿到配对 CL20，确认 action-anchor guard 在真实闭环中是否减少新增失败。

## 8.2 有竞争力的 fixed-budget CCF-A

建议目标：

- B=16 显著优于所有 equal-budget baselines；
- relative to matched full-information planner，reactive CLS gap 控制在 `1–2` 分内；
- Val14 reactive CLS 先达到 `90+`，更强竞争区间约 `91–93`；
- Test14-Hard 需要在对应协议下达到强公开基线附近；不要混用 CLS 与 PDMScore；
- interaction-heavy/hard 场景显示明显、统计显著收益；
- safety 不退化；
- 有清晰 budget-quality-latency Pareto curve。

当前与该目标的差距不是简单的若干 closed-loop 分数，而是尚无任何 closed-loop 分数，且 open-loop candidate 仍低于 matched controls。

---

## 9. V54 可证伪预期

V54 应至少表现出以下结构性变化，否则 anchor-relative 假设不成立：

1. zero-residual local control 的 pair action 与 selected-local action严格一致；
2. local control 的 teacher match应从 V53 pair-local `0.118` 向 selected-local `0.141` 靠拢，而不是回落到 `0.100`；
3. candidate/local deployed flip rate应只来自 certified action-anchor guard；
4. beneficial certified flips应高于 harmful certified flips；
5. `budget_vs_pair_full_match` 仍高时，pair-full winner 本身必须比 V53 更正确；
6. diagnostic CL20 中 candidate不能产生 local/foundation没有的新安全灾难。

这些条件比单纯提高 certificate 或 selected recall 更直接地检验论文核心 idea。

---

## 10. 运行与结果解释

完整指令见：

- `V54_AR_BFAR_DBAP_NEXT_COMMANDS.sh`
- `NEXT_COMMANDS_V54_AR_BFAR.txt`

关键输出：

```text
$OUT_ROOT/open_loop/v54_ar_bfar_gate_report.json
$OUT_ROOT/logs/v54_ar_bfar_dbap_gate.out
$OUT_ROOT/closed_loop/.diagnostic_cl20
$OUT_ROOT/control_local_same_checkpoint/closed_loop/
$OUT_ROOT/control_foundation_matched/closed_loop/
```

若 minimum FAIL 但 protocol PASS，CL20 会被标记为 diagnostic。只有 competitive PASS 才允许自动升级 CL100。

---

## 11. Claim boundary

V54 已通过本地 compile、配置、shell、unit tests 和 V53 replay gate 语义验证，但没有在当前环境执行 fresh nuPlan training 或 simulation。不能提前声称：

- V54 gate 必然通过；
- closed-loop 有提升；
- fixed-budget SOTA；
- real-time deployment。

下一轮实验的首要价值，是首次同时得到正确 action anchor、fresh residual/selector、三路 open-loop 和三路 paired CL20。
