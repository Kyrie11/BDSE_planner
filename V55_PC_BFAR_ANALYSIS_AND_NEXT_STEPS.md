# V54 结果诊断与 V55 PC-BFAR-DBAP

日期：2026-07-31

## 1. V54 gate 结论

V54 的实际状态为：

- immutable anchor gate：PASS；
- fresh residual/selector training：已运行，共 4 epochs；
- candidate/local/foundation 三路独立 calibration：已运行；
- 三路配对 open-loop：已运行，各 1000 个场景；
- V54 protocol gate：FAIL；
- V54 minimum gate：因此显示 FAIL，但 `minimum_failures=[]`；
- competitive gate：FAIL；
- CL20/CL100：均未运行。

Protocol 的唯一失败项是：

```text
last selector_exact_fraction=0.02572 < 0.03
```

但 V54 train config 明确设置：

```yaml
training:
  min_deployment_exact_fraction: 0.015
```

在每 8 step、每 rank 1 scene 的稀疏 exact 调度下，常规理论覆盖约为 `1/(8*8)=0.015625`；实际最后一轮 `0.02572` 已高于配置要求。因此该失败是 gate 的硬编码阈值与训练配置不一致，属于工程 gate bug。

用 V55 的配置派生门控在相同 V54 结果上重放：

- protocol-integrity：PASS；
- minimum-completeness：PASS；
- competitive：FAIL。

因此，V54 并非连最低完整性都在算法上失败；真正未达到的是竞争性增益。

## 2. 五个关键观察项

### 2.1 selected-local control 是否从 0.118 向 0.141 靠拢

没有。

- local pair-full match：0.118；
- local control final match：0.118；
- `sparse_full_interface_action_match`：0.141；
- dense full-interface match：0.359。

关键是 V54 的 `sparse_full=0.141` 表示使用全部 sparse/local evidence 的 action，不是实际 B=16 selected-local argmin。V54 没有把直接的 B=16 selected-local action作为最终 action path 和独立指标暴露出来。因此不能声称 local control 已向 0.141 靠拢。

### 2.2 candidate/local certified flip rate

没有正向信号。

- candidate 与 same-checkpoint local control 的部署 action：1000/1000 完全相同；
- causal deployed residual flip rate：0；
- selector residual edge proposal/allowed：约 0.00836/0.00121；
- tournament residual edge proposal/allowed：约 0.01020/0.00163。

pair edge 层面有少量允许修正，但没有任何修正改变最终 action。

### 2.3 beneficial 是否高于 harmful

无法形成正向结论：两者均为 0。原因不是残差完美安全，而是 candidate 与 local 的部署 action 完全相同。

### 2.4 pair-full winner 是否真正改善

没有。

- candidate pair-full match：0.118；
- local pair-full match：0.118；
- final action match：0.118；
- budget-vs-pair-full match：0.992；
- budget-vs-sparse-full match：0.749。

这再次证明 exact selector 几乎完美保存了弱 pair-full target，而不是改善 target winner。

### 2.5 diagnostic CL20

未运行，因此上传结果不支持 collision、TTC、drivable-area 或 progress 的任何结论。Protocol gate 的错误 exact-fraction 阈值阻断了 diagnostic CL20。

## 3. 三个核心问题的状态

### 3.1 Pairwise sign 到 action winner

未解决。V54 虽将 selected-local cost 加入 margin，但最终仍通过 restricted-rival tournament 选 action，而不是直接对 selected-local global cost 取 argmin。零 residual 并不严格等价于 selected-local planner。

另外，restricted pair graph 允许 cycle/non-integrable residual 通过遍历方式影响 winner；这与“保留的 evidence 形成全局 action ordering”的论文主张不一致。

### 3.2 Selector 锁定 decisive evidence

该方向基本成立，但收益被错误的 downstream target 限制。

V54 candidate：

- proposal decisive recall：0.800；
- selected decisive recall：0.577；
- effective selected decisive recall：0.743；
- AOCC fully certified scene rate：约 0.715；
- gate 使用的 certificate fraction：0.822；
- fallback：0.156；
- frontier retained：0.759。

Selector 已不再是首要瓶颈。下一步应保持 exact AOCC 与 boundary curriculum，不继续堆叠更多 selector heuristic；重点应是使 certificate 对应一个可积、正确、可训练的 action target。

### 3.3 无关 pair 计算

已明显改善。

V54 每个 epoch 约 17.3–20.3 分钟，训练吞吐约 42.6–48.2 samples/s；相比 V51 的约 42–79 分钟/epoch 和 V53 的约 20–30 分钟/epoch继续改善。普通训练 pair fraction 约 0.589，exact fraction 约 0.0156，最后对齐轮为 0.0257。

该设计应保留，不应退回全图、每步 exact。

## 4. 新发现的工程/因果归因问题

### 4.1 exact-fraction gate 硬编码

Gate 使用 0.03，而训练配置使用 0.015。V55 从 train config 读取门槛，命令行仅可显式提高，不能静默覆盖配置。

### 4.2 residual-disabled local control 仍受 pair variance 影响

V54 local control 关闭了 residual mean，但没有同时关闭 residual variance。Pair variance 仍进入 AOCC/tournament uncertainty，导致 5/1000 个 `local_pair_full -> pair_full` 内部 flip，尽管 residual mean 为零。

V55 在 `disable_pair_residual_intervention=true` 时同时清零：

- selector pair residual mean；
- tournament pair residual mean；
- selector pair variance；
- tournament pair variance。

因此 local control 现在是真正的 pure selected-local control。

### 4.3 gate 的 beneficial/harmful 必须用三路配对 action

内部 pair flip 不能代表残差的因果部署贡献。V55 使用 candidate 与 same-checkpoint local control 的逐场景 deployed action 差异计算：

- flip rate；
- beneficial rate；
- harmful rate；
- beneficial/harmful given flip。

## 5. V55：Potential-Consistent BFAR-DBAP

### 5.1 Direct selected-local action anchor

先用实际选择的 B=16 evidence 构造可积 action cost：

```math
J_B^L(a)=J_0(a)+\sum_{i\in S_B}g_i(a).
```

这是真正的 local control。零 residual 时，最终 action 必须是：

```math
\arg\min_a J_B^L(a).
```

### 5.2 Residual edge field 的 Hodge projection

模型输出的 residual pair field 可能有 cycle：

```math
r_{ab}+r_{bc}+r_{ca}\neq 0.
```

V55 不允许 non-conservative cycle 通过 tournament traversal 改变 winner。它求解加权投影：

```math
\phi^*=\arg\min_\phi\sum_{(a,b)}w_{ab}
[(\phi_b-\phi_a)-r_{ab}]^2+\lambda\|\phi\|_2^2.
```

权重对 near-boundary pair 更高。最终 action cost 为：

```math
J_B^{PC}(a)=J_B^L(a)+s\phi(a).
```

由此所有 residual margin 都来自同一个全局 potential，天然满足 transitivity。

### 5.3 Global certified action flip

Residual potential 提议将 selected-local winner `a_L` 改为 `a_R` 时，必须满足：

- corrected global cost 确实使 `a_R` 优于 `a_L`；
- uncertainty-shrunk full cost margin 高于 `flip_margin`；
- safety/validity guard 通过。

Pair uncertainty只用于 flip certification，不再直接扰动 anchor action scores。

### 5.4 Action-potential distillation

V53/V54 的主要监督仍偏向独立 pair sign。V55 新增全局 potential target：

```math
\phi_T(a)\propto J_T(a)-J_B^L(a),
```

并在每个 scene 内去除常数平移。训练重点加权：

- teacher winner；
- 当前 strongest rival；
- teacher near-boundary actions；
- selected-local anchor 选错的 scenes。

这使“关键 evidence → residual correction → action winner”成为直接可微的全局路径。

### 5.5 Exact AOCC target 与部署路径一致

V55 的训练 exact selector target 使用完整 Top-M evidence 下的 integrable-potential action，而不是只保护旧 pair tournament target。部署 B=16 仍使用 exact AOCC；没有替换为 surrogate。

## 6. Gate 目标

### 最低完整性目标

下一轮至少应满足：

- protocol gate PASS；
- local control 与 direct selected-local anchor 逐场景完全一致；
- candidate 相对 local 无灾难性 action/regret 退化；
- causal harmful flip 不高于 5%；
- B=16 16/16 fill；
- selected/effective decisive recall 不低于约 0.45/0.58；
- 无论 minimum 是否通过，只要 protocol PASS，必须生成三路 paired diagnostic CL20。

### 竞争性 fixed-budget 目标

Open-loop 内部目标：

- candidate vs foundation teacher-match gain ≥ 0.015；
- candidate vs same-checkpoint local gain ≥ 0.005；
- beneficial causal flips > harmful flips；
- selected/effective/interaction decisive recall ≥ 0.55/0.70/0.50；
- paired median/p90 regret 不退化。

Closed-loop 论文目标应以 equal-budget comparison 为主：

- B=16 显著优于 random、score-only、greedy、no-residual；
- 与 matched full-information planner 的 reactive score gap 控制在 1–2 分以内；
- Val14 reactive 先达到 90+；
- Test14-Hard reactive 争取接近当前强公开方法约 80–82 的区间；
- collision、TTC、drivable-area 不得通过牺牲 safety 换 progress。

这些数值不是 CCF-A 官方门槛，而是根据 2025–2026 公开 nuPlan 结果设置的竞争性实验目标。

## 7. 下一步重点观察

V55 训练完成后优先检查：

1. `selected_local_anchor_action_match` 是否被正确记录；
2. local control 的 deployed action 是否与 selected-local anchor 100% 一致；
3. `pair_potential_cycle_fraction` 是否下降；
4. potential teacher loss 是否持续下降且无 NaN；
5. candidate-local causal flip 是否非零；
6. beneficial given flip 是否高于 harmful given flip；
7. candidate 是否至少超过 local control；
8. diagnostic CL20 是否产生新增 collision/TTC/drivable failures；
9. latency 中 prediction stage 是否仍为主瓶颈。

## 8. Claim boundary

V55 已通过代码、配置、shell、单元测试和 V54 replay gate 验证。本环境没有 nuPlan GPU 训练和闭环模拟，因此不预先声称 V55 会通过 competitive gate、提升 closed-loop 或达到 fixed-budget SOTA。
