# V64.3.9 V2 causal audit and V64.3.10 HAP-BDMU design

## Executive result

The repaired V64.3.9 runtime-parity screen is valid for partial algorithm attribution. Runtime Top-M semantics, config isolation, training artifacts and optimizer activation are all valid. The screen still correctly blocks full/test/closed-loop because AF-BDMU does not demonstrate a meaningful acquisition mechanism shift before the endpoint gate.

The strongest result is not simply that validation teacher match gains only 0.4pp. The optimizer strongly reduces AF-BDMU surrogate losses while hard proposal admission barely moves. Train swap loss falls about 26%, but train hard Top-M utility capture rises only about 0.25pp; validation exact winner-flip Top-M recall remains exactly 0.23734. This localizes the failure to the utility-to-hard-HAB-admission link.

A secondary protocol defect is also present: the V64.3.9 promotion metric `val_teacher_bdmu_topm_utility_capture` is moving-reference, while the training target uses a frozen foundation B-set. It should not serve as the main causal mechanism gate. This does not explain away the independent recall failures, so the result is partially attributable rather than engineering-invalid.

## Why full/test/CL are not justified

The intended chain is:

`fixed budget -> BDMU utility -> proposal Top-M -> B selector -> frozen DARM/DBR -> teacher endpoint -> closed loop`.

In V64.3.9 V2 the proposal-admission mechanism has not shown a meaningful change. Running full would increase compute without evidence that the intervention reached its intended mediator. Running test/CL would then produce endpoint variation that could not distinguish acquisition effect from ordinary model/simulation variance. The correct response is to repair the mechanism, not relax the gate.

## HAP-BDMU

HAP-BDMU preserves the adaptive-frontier one-sided BDMU target but projects detached teacher utility through the exact deployed hierarchical atom builder. The feasible oracle Top-M becomes the structured target. Ranking is applied only to oracle-vs-current set differences, primarily within the same frozen HAB family stratum. This attacks the exact discrete mediator that failed in V64.3.9 while preserving the fixed interface and all downstream value machinery.

## Causal experiment protocol

Every validation epoch now records four levels on one fixed target/interface:

- C0: frozen-reference utility definition;
- C1: exact HAB utility oracle ceiling;
- C2: learned proposal capture;
- C3: teacher match/regret under frozen DARM/DBR.

The checker measures C1-C2 headroom and oracle-gap closure. If C1 has no headroom, acquisition is no longer binding. If C2 improves but C3 does not, value/frontier becomes the next bottleneck. Only C2+C3 improvement promotes full/test/CL.

## Expected scientific value

The algorithmic contribution is not merely another ranking loss. It enforces **interface-feasible marginal utility acquisition** under the exact same hierarchical query constraints used by deployment. The experimental contribution is an executable causal attribution protocol that separates interface capacity, learned acquisition and downstream value transmission while holding the planner interface fixed.
