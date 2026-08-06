# V64 论文同步建议（在实验通过后再写入主稿）

## 1. 核心 claim 保持

论文核心保持为：

- 固定 planner-interface evidence budget B；
- 可审计 evidence atoms；
- 对论文定义的 deterministic budgeted selector 进行 exact execution/audit；
- evidence critical 当且仅当 leave-one-out 后 winner action 翻转；
- evidence/residual 双证书。

不要把内部 acquisition pool M 或 M×K 计算写成 B×K 证书预算。

## 2. 可新增的方法贡献

### Support-aware feature evolution

当新 query channels 超出 foundation checkpoint 的 empirical support 时，旧 projection 仅处理 supported prefix；新 channels 通过零初始化 residual adapter 加入。该机制保证 step-zero policy invariance，并提供 feature extension 的独立审计和 ablation。

建议正文用“checkpoint-supported subspace”而不是简单“旧/新维度”，并给出 adapter-on/off、step-zero winner invariance 和 cache-prefix parity 表格。

### Budgeted critical coverage

保留 literal winner-flip criticality 定义。新增训练目标不重新定义 critical，而是优化在固定 HAB Top-M acquisition interface 下保留的 critical utility。hard forward 与部署 deterministic selector一致，soft path只用于梯度。

建议给出：critical scene prevalence、critical atom sparsity、Top-M recall、B-selected recall、winner preservation，以及 coverage loss ablation。

## 3. Runtime prior 表述

主方法不要依赖未计入 evidence budget 的 opaque continuous prior 改变 winner。安全 hard feasibility、candidate validity和evidence atoms可以作为显式接口。若保留 structural prior，只能作为清楚标注的 ablation，并解释其是否属于 planner-interface budget。

## 4. 必须完成的论文实验

1. V62 checkpoint step-zero support audit；
2. V64 continuation 快速机制验证；
3. 从同一 immutable V53 foundation、matched training compute 重训；
4. BCC weight、adapter、base prior、structural prior、rival graph ablation；
5. paired open-loop candidate/local/foundation；
6. paired CL20/CL100，报告 safety、progress、comfort和真实 decision flips；
7. 完整 test readiness 后一次冻结 testing。

## 5. 禁止提前写入的 claim

目前不能声称 V64 gate PASS、闭环提升或 SOTA。只有 fresh GPU experiments 和冻结 test 完成后，才能把 V64 写入主结果表。
