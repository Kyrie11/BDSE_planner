# Paper Sync Notes — V64.2

## 必须保留的核心贡献

1. 固定 planner-interface evidence budget B；proposal budget M、candidate budget K 和实际 query count 分开报告。
2. 可审计、可追踪、可按 action 查询的 evidence atoms。
3. critical 的定义保持 literal leave-one-out winner-action flip，不用 margin deficit 代理替换定义。
4. selector/证书必须围绕同一个部署 winner 与同一个固定预算接口。

## 当前论文与代码/主张的关键不一致

### 1. “Exact selector” 与 Greedy 叙述

TeX 方法和附录当前明确写的是 Greedy/lazy-greedy。代码中也存在 runtime greedy selector 命名。投稿前必须选择并统一：

- 全局组合最优 exact：提供明确优化问题、算法、复杂度以及 exactness proof/test；或
- 确定性 budget-feasible selector：删除会被理解为组合全局最优的 “exact selector” 用语，把 exact 限定为 implementation/decision consistency。

不要在没有证明时用 “exact optimal selector”。

### 2. 固定 B 与 fallback 扩 B

附录 fallback 当前允许增加 B。主论文若以 fixed planner-interface budget 为核心，nominal 与 safety fallback 都应保持 B；扩 B 只能作为独立、明确标注的 out-of-protocol upper-bound/diagnostic，不能计入主方法闭环指标。

### 3. Hidden prior

任何不计入 evidence budget 却系统性改变 winner 的 runtime prior 都会削弱核心主张。主方法保持 opaque base/structural runtime prior 关闭；安全信息通过 candidate validity、hard feasibility、auditable atoms 和 certificate 进入。

## 结果后再加入的内容

- Support-Aware Query Adapter：12-D checkpoint-supported prefix + zero-initialized 6-D extension residual。
- Budgeted Critical Coverage：固定 HAB Top-M hard forward，优化 literal critical utility coverage。
- HCBE：针对 missed critical atom 的 HAB-consistent exchange-boundary ranking。
- 三层 query contract：raw feature、numerical score、winner decision。

这些内容只有在正式 paired open-loop 与 closed-loop 结果完成后再作为贡献/消融写入，不提前声称提升。

## 必须新增的表/消融

- B 固定下：critical Top-M recall、critical selected recall、certificate coverage、teacher match、closed-loop score。
- BCC vs BCC+HCBE；相同 K/M/B/训练预算。
- primary best、critical-score-best、teacher-match-best checkpoint 的预注册选择规则。
- candidate oracle/安全 candidate coverage，防止把 candidate-bank 上限误归因于 evidence selector。
- partial test 仅 frozen stress test；完整 test 一次性最终评估。
