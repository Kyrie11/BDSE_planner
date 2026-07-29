# 论文与 v49 DBAP 对齐建议

## 需要修改的核心术语

论文当前写明 atom-pair label 是 comparative label，不是 causal counterfactual label。v49 的 leave-one-out 发生在 teacher evidence decomposition 上，因此建议使用：

- `leave-one-out deployment-boundary criticality`；
- `teacher-decomposition intervention`；
- `boundary-critical evidence`。

不要直接写“causal evidence”，除非增加真实干预数据或因果识别假设。

## Contribution 建议

1. Fixed planner-interface budget 下的 action preservation，而不是 distribution reconstruction；
2. Pure LOO boundary-critical supervision；
3. Integrable local margin with bounded uncertainty-gated sparse residual；
4. Exact-tournament-aligned nested certificate frontier，同时报告 first-certificate 与 exact-B；
5. Candidate/interface/compression 分层错误分解。

## Theorem 边界

当前 theorem 是条件式 action-preservation guarantee。它不能证明：

- LOO target 的统计最优性；
- 模型会学到边界贡献；
- certificate premise 在 test/closed-loop 成立；
- candidate teacher 与真实 closed-loop objective 一致。

实验必须分别验证 decisive rival recall、pair margin calibration、pair-full action match、exact-B preservation 和 candidate coverage。

## 实验主表建议

主表至少分成：

- Candidate ceiling；
- Dense/local interface；
- Pair-full interface；
- Exact-B budgeted result；
- Certificate/fallback；
- Latency；
- Paired CL20/CL100。

不要只报告 budgeted action 与 teacher match，否则无法判断错误来自 candidate、interface 还是 compression。
