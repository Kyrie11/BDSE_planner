# V64 结果审计、Gate 诊断与 V64.2 优化报告

## 1. 结论

当前缺失 `open_loop/v64_saqa_bcc_gate_report.json` **不是三个 gate 已经判负**。上传流水线在训练完成并写出 epoch 5 验证结果后，于 calibration shard 启动函数中因 Bash `set -u` 变量展开错误退出：

```text
V64_SAQA_BCC_NEXT_COMMANDS.sh: line 390: sid: unbound variable
```

因此正式状态是：

| Gate | 正式状态 | 结论 |
|---|---|---|
| Protocol | NOT EVALUATED | 缺少独立 calibration、三路同场景 open-loop rows 和 checker report |
| Minimum | NOT OFFICIALLY EVALUATED | 只能用训练期 val 代理指标预警 |
| Competitive | NOT OFFICIALLY EVALUATED | 缺少 candidate/local/foundation 的配对增益与 regret/flip attribution |

从现有 val 代理指标看，**protocol 的 query/base/budget 子合同大概率可过**；但 minimum 很可能被 evidence certificate 卡住，competitive 还会被 literal critical coverage 和净有益 deployed flip 卡住。

## 2. Pipeline 为什么没有跑完

### 2.1 直接中止点

原函数在同一条 `local` 命令中声明 `sid` 并使用它构造 `raw`：

```bash
local sid="$1" gpu="$2" raw="$OUT_ROOT/calibration/raw/gpu${sid}.npz"
```

在 `set -u` 下，右侧展开时 `sid` 尚未绑定，所以 calibration 尚未真正启动，后续 merge、三路 open-loop 和 gate checker 全部没有执行。相同模式也存在于 wait 函数和继承的 V63 脚本。

### 2.2 第二次执行的 preflight 阻断

第二条 pipeline log 显示：

```text
audit report is not PASS
query-relevant config fingerprint changed
max_abs_error=nan > tolerance=1e-05
audited_support_dim=12 != expected=18
```

其中有两个问题：

1. 历史 cache 只支持 12-D prefix，而当前 runtime contract 是 12-D supported prefix + 6-D online extension；不能要求旧 cache 直接提供完整 18-D。
2. `max_abs_error` 使用 `arr.max(initial=nan)`，即使数组全为有限值也会返回 NaN，这是工程 bug。

严格 prefix numerical audit 的实际误差很大，因此此次应当 **fallback 到 full runtime recompute**，而不是强行启用 cache；fallback 不应阻断训练。

### 2.3 Support-contract audit 的假失败

原 support report 中：

- legacy vs support **deployed action mismatch = 0.0**；
- runtime vs prefix deployed mismatch = 0.0；
- base/raw-query/score/decision 硬合同通过；
- 但内部 `full_action` diagnostic mismatch = 0.793。

`full_action` 的内部诊断语义在 support-aware path 中发生改变，不应覆盖部署动作和三层 query contract。修正 analyzer 后该 audit 为 PASS，同时保留内部 full-action drift 作为 warning。

## 3. 三个 Gate 的分析

### 3.1 Protocol gate

**正式：NOT EVALUATED。** 现有 best-primary checkpoint（epoch 1）代理指标：

| 子合同 | 值 |
|---|---:|
| base contract | 1.000 |
| raw query contract | 1.000 |
| score contract | 1.000 |
| query decision match | 0.999 |
| support dimensions | 12 + 6 |
| adapter enabled | 1.000 |
| retained B=16 budget pass | 1.000 |
| scalar/lexicographic winner alignment | 0.977 |
| dense-HAB vs runtime sparse bridge | 0.999 |

这些支持“模型接口部分大概率通过”，但 protocol checker 还要求：独立 calibration、candidate/local/foundation 同 row identity、冻结 anchor drift、有效 action-query coverage 等。只有补跑 calibration/open-loop 后才能给正式 PASS。

### 3.2 Minimum gate

**正式：NOT OFFICIALLY EVALUATED；代理判断：LIKELY FAIL。** epoch 1：

| 指标 | 当前 | minimum 阈值 | 代理判断 |
|---|---:|---:|---|
| evidence certificate fraction | 0.046 | 0.40 | FAIL |
| frontier retained | 0.729 | 0.45 | PASS |
| proposal decisive recall | 0.795 | 0.72 | PASS |
| selected decisive recall | 0.600 | 0.50 | PASS |
| effective selected recall | 0.765 | 0.62 | PASS |
| interaction selected recall | 0.568 | 0.40 | PASS |
| fallback | 0.000 | <=0.60 | PASS |

根本瓶颈不是 B=16 selector 的一般 decisive recall，而是 **one-sided evidence certificate 极度保守**。现实现按遗漏原子逐项累加最坏半径，平均每个被选原子只回收约 0.02 的 deficit，B=16 理论上最多回收约 0.32；许多场景初始 deficit 约 0.49，因此最终仍不能 certified。不能通过降低阈值或扩大 B 来“修”论文结果。

### 3.3 Competitive gate

**正式：NOT OFFICIALLY EVALUATED；代理判断：LIKELY FAIL。** best-primary checkpoint：

| 指标 | 当前 | competitive 阈值 |
|---|---:|---:|
| certificate | 0.046 | 0.55 |
| proposal decisive recall | 0.795 | 0.80 |
| teacher literal critical Top-M recall | 0.345 | 0.80 |
| teacher literal critical selected recall | 0.329 | 0.50 |
| raw residual proposal | 0.408 | >=0.001 |
| deployed residual flip | 0.000 | 需要 beneficial > harmful |
| sparse bridge | 0.999 | 0.95 |

这里最关键的信号是：teacher critical scene rate 约 0.466，但 Top-M 只覆盖约 0.345，selected 约 0.329，二者相近。这说明主要损失发生在 **HAB acquisition/proposal**，不是 B=16 之后的 selector；同时 residual 有大量 raw proposal，但部署保护条件使最终动作从不翻转，因此 competitive 的净有益 intervention 也无法成立。

## 4. 已落地的代码修复

1. 修复 V64/V63 calibration launch/wait 的 `set -u` unbound-variable bug。
2. 修复 cached query audit 的 NaN `max_abs_error`。
3. 修正 support audit：部署 action + raw/score/decision contracts 为硬条件，内部 full-action drift 仅 warning。
4. 新增 pipeline config contract，阻止继承旧 `TRAIN_CONFIG`、错误 qdim/B/adapter/critical target。
5. 新增 pipeline status inspector，识别 calibration launch crash、cache audit rejection、正式 gate 是否真正执行。
6. 新增 checkpoint-only evaluation：`SKIP_V64_TRAINING=1` + `V64_CANDIDATE_CHECKPOINT`，可比较多个已训练 checkpoint 而不重训。
7. 新增 `RUN_V64_RESUME_AFTER_GATEFIX_2GPU.sh`，直接从现有 best checkpoint 补跑 calibration/open-loop/gate。
8. 持久化 candidate checkpoint SHA-256、query path 和 config contract，避免“跑的是哪个模型/哪个路径”不可审计。
9. 训练 loader 的 workers/prefetch/val workers 改为可配置。
10. teacher-interface critical loss 不再计算随后被丢弃的 model leave-one-out labels。

## 5. V64.2 算法升级：HCBE

新增 **HAB-Consistent Critical Boundary Exchange (HCBE)**，目标是修复 BCC 与 hard HAB 前向之间的监督错位：

- hard forward 仍是原 deterministic HAB Top-M；
- B、M、evidence atom、部署 selector、literal winner-flip critical 定义均不变；
- 只对“literal critical 且未进入 Top-M”的 atom 施加交换监督；
- 优先要求它超过同 family 中最弱的已保留 non-critical atom；没有同-family 边界时再使用 family-conditioned global boundary；
- 旧的 strongest-negative ranking 从主目标降为弱正则；
- teacher 仅用于训练，不进入部署。

它比“让 critical atom 超过全场最强负样本”更符合 HAB 的真实交换规则，也比增加 M/B 更能保持论文核心 novelty。

## 6. 数据集诊断

Validation（58,418 场景）中，full-interface match=0.966、B16 oracle=0.912、runtime sufficiency=0.749，说明 fixed-budget evidence idea 有基础；但 safe-candidate-exists=0.717，teacher candidate ADE p50=5.48、p90=12.92，candidate bank 已是潜在闭环上限。

当前 partial test（67,042 场景）更难：full-interface=0.934、B16 oracle=0.840、runtime sufficiency=0.641、safe-candidate=0.578、ADE p90=17.14。它可以作为 **冻结 checkpoint 的 stress test**，但在 test 构建完成前不能用于模型选择、阈值调整或算法迭代。

优化顺序应是：先修 critical acquisition/证书；若 open-loop evidence 指标提高但闭环仍差，再升级 candidate dynamics、reactive interaction 和 replan cache，而不是增加 B。

## 7. 效率

训练日志显示 data wait 是最大且波动最大的耗时来源（不同 epoch 约占 20%--63%）；其次是 loss construction 与 pair sampling。已做的无性能损失优化包括：

- teacher target 分支去除无效 leave-one-out 计算；
- 多 checkpoint 只评估不重训；
- worker/prefetch 可调；
- 完成的 calibration shard/open-loop stage 自动复用；
- prefix cache 只有 audit PASS 才启用，避免以错误 cache 换速度。

建议先用 300--500 step benchmark 比较 workers=4/6/8、prefetch=2/4；不应减少 B/M、critical labels、calibration 样本或正式 open-loop 场景数来制造速度结果。

## 8. 论文同步风险

论文当前仍使用 “Greedy selector / lazy-greedy” 叙述，并在 fallback 中允许增加 B。若投稿主张“预算内 exact selector”，必须明确 exact 的数学含义：

- 若是组合优化全局最优，需要给出算法与复杂度/证明，代码也不能仍是 greedy 语义；
- 若只是“严格执行声明的确定性 selector、无近似漂移”，应改称 deterministic budget-feasible selector，避免 reviewer 将 exact 理解为全局最优；
- nominal/fallback 都应保持固定 B；扩 B 只能是独立 out-of-protocol diagnostic，不能混入主结果。

建议等 V64.2 的正式结果出来后，再把 SAQA、BCC、HCBE 写入方法与消融，不提前宣称收益。

## 9. 验证状态

```text
Python compile       PASS
YAML parse           6/6 PASS
Shell syntax         6/6 PASS
Config contract      PASS
Targeted tests       4 passed
Full pytest          254 passed, 0 failed
Support reanalysis   PASS
```

当前环境没有实际 checkpoint bytes、nuPlan cache、CUDA 和闭环 simulator，因此没有运行 fresh V64.2 training、正式 calibration/open-loop/closed-loop，也不声称 gate PASS 或闭环 SOTA。
