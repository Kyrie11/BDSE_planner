# V64.3.10 HAP-BDMU Causal Audit and V64.3.11 BTP-BDMU Design

## Executive conclusion

The uploaded V64.3.10 screen is valid for algorithm attribution. Runtime Top-M parity, HAP semantic/config contracts, and training artifact contracts all passed. The stop is therefore not primarily an engineering failure.

HAP-BDMU partially solved the V64.3.9 bottleneck: it can move exact HAB hard admission. The anchor exact-HAB utility ceiling (C1-M) is 0.504656 and learned capture (C2-M) is 0.478576, leaving 2.608pp headroom. Epoch 3 increases C2-M to 0.482552, closing 15.25% of that gap, while validation feasible-admission rank loss drops from 1.6695 to 1.0195.

However, these admission changes are not decision-useful. Proposal decisive recall drops by 8.00pp, exact-critical Top-M recall decreases slightly, teacher match falls 0.6pp, and teacher regret worsens 1.22%. Therefore the bottleneck has shifted from **utility -> hard HAB admission** to **hard HAB admission -> fixed-B transmitted evidence -> downstream teacher decision**.

V64.3.11 BTP-BDMU is designed as a final acquisition mediation experiment. It trains only admissions that survive a frozen B=16 projection, protects currently B-selected evidence, and uses same-family replacements only. Crucially, validation C1-B/C2-B is recomputed through the exact runtime pair-conditioned B=16 selector rather than the training fast surrogate. If exact C1-B headroom is negligible, or exact C2-B moves without C3 endpoint gain, the next cycle must pivot to decisive value/frontier modeling.

## 1. Why HAP-BDMU did not pass

### 1.1 The intended HAP mechanism did learn

The screen is not a failed optimizer run. At validation:

- feasible-admission rank loss: 1.6695 -> 1.0195;
- feasible pair count: 156.97 -> 72.47;
- same-family pair fraction: 22.68% -> 41.09%;
- exact-HAB learned utility capture: 47.86% -> 48.26%;
- exact-HAB oracle capture remains 50.47%;
- oracle-gap closure reaches 15.25%.

Thus HAP's projection removed part of the V64.3.9 surrogate/hard-boundary mismatch. Hard membership can move in the intended realizable HAB interface.

### 1.2 The moved support is misaligned with decisive support

At selected epoch 3:

- proposal decisive recall: 75.37% -> 67.37%;
- exact winner-flip Top-M recall: 23.73% -> 23.42%;
- exact winner-flip selected recall: unchanged at 14.87%;
- teacher action match: 17.8% -> 17.2%;
- teacher regret: 20133.34 -> 20378.10;
- pair-full teacher match: unchanged at 17.4%.

The exact B->TopM winner-preservation certificate increases from 92.8% to 93.6%, yet teacher correctness worsens. This demonstrates that preserving the current learned Top-M decision is not equivalent to preserving the teacher decision; certificate fraction must remain a preservation diagnostic rather than a teacher objective.

### 1.3 HAP's remaining C1-M headroom is modest

The exact-HAB oracle can improve utility capture by only 2.61pp over the anchor. Even a perfect HAP utility ranking therefore has limited remaining proposal-layer upside under this fixed interface. Another stronger HAP ranking loss is not well motivated.

### 1.4 Cross-family fallback is empirically suspicious

HAP intended same-family competition, but at the selected epoch only 41.09% of feasible admission pairs are same-family. Most training pairs therefore came from cross-family fallback. With frozen HAB family allocation, those comparisons are often weakly actionable. The rise in HAP target agreement coincides with a severe decisive-recall decline. V64.3.11 removes cross-family fallback rather than strengthening it.

## 2. Bottleneck transition

The V64.3.9 diagnosis was:

`continuous decisive-margin utility -> realizable HAB hard Top-M admission`

V64.3.10 shows this mediator is **partly repaired**. A measurable C2-M shift is now possible.

The new unresolved chain is:

`C0 fixed decisive-margin utility`
`-> C1/C2 exact HAB support`
`-> fixed B=16 transmission`
`-> frozen DARM one-sided margin preservation`
`-> DBR/final decision`

The current screen has no controlled C1-B/C2-B decomposition, so it cannot tell whether:

1. HAP's newly admitted utility atoms never survive B=16;
2. they survive B=16 but displace more decisive evidence;
3. B=16 transmission improves but downstream DARM/DBR value cannot use it.

That missing mediator is the target of V64.3.11.

## 3. What to keep

Keep the paper's core mainline unchanged:

`fixed planner-interface budget -> auditable evidence atoms -> budget-feasible decisive-margin marginal utility -> budgeted acquisition -> DARM one-sided decisive-margin preservation -> final decision preservation`.

Specifically retain:

- fixed B=16 and fixed proposal M;
- auditable evidence atoms;
- adaptive decisive rival frontier;
- continuous one-sided margin-deficit utility and weakest-rival term;
- exact runtime HAB semantics;
- CCBR representation as a proposal representation primitive;
- frozen V64.3.7 DARM+DBR while acquisition is causally isolated;
- exact DA-EPC as a preservation diagnostic;
- representative validation and strict checkpoint/config contracts.

## 4. What to stop or modify

Do not continue:

- stronger broad HAP feasible-admission rank;
- cross-family HAP fallback;
- generic listwise BDMU ranking as the main loss;
- AF-BDMU arbitrary Top-M swap ranking;
- binary literal-critical BCE as the main acquisition target;
- binary certificate/AOCC bonus;
- larger B/M or global proposal unfreezing;
- AP-WCCA/AP-WRCCA/LCV/FPCCA/CCBR/LEA/BCHA objective retries;
- selector beam/swap/bruteforce;
- changing DARM/DBR in the same acquisition causal cycle.

The changelog already contains negative or non-binding evidence for these directions.

## 5. V64.3.11 BTP-BDMU

### 5.1 Core algorithm

BTP-BDMU moves the supervision target one mediator downstream.

1. Compute the immutable frozen-reference BDMU utility.
2. Project that utility through exact HAB to obtain a realizable utility Top-M oracle.
3. Pass current and oracle Top-M pools through the frozen B=16 selector projection.
4. Define proposal positives only when an atom is:
   - in the exact-HAB utility oracle,
   - selected under the oracle B=16 projection,
   - missing from current deployment Top-M.
5. Protect any currently B-selected evidence from becoming a negative.
6. Rank only same-family replacements; no cross-family fallback.
7. Disable broad listwise, AF swap, and HAP broad rank terms.

This is a proposal-training intervention only. Runtime M, B, selector, DARM, DBR and query accounting remain unchanged.

### 5.2 Why protection is important

V64.3.10 demonstrates that maximizing an upstream proxy can improve utility capture while destroying decisive recall. Protecting currently transmitted B evidence gives BTP a one-sided, minimum-intervention character: acquisition may add a budget-transmitted utility atom only by replacing proposal support that is not already used by the fixed B layer.

This is aligned with the paper's preservation framing: improve the missing support without discarding already effective evidence.

### 5.3 Train/eval selector separation

The training objective uses the existing vectorized pair-margin selector surrogate for tractability. The new BTP pair loss is fully vectorized and avoids HAP's Python per-scene/per-positive loop.

However, the paper-level causal metric must not inherit the surrogate. During validation loss (`torch.no_grad()`), C1-B and C2-B are recomputed with the exact runtime pair-conditioned B=16 selector on controlled exact-HAB Top-M masks. The code logs current/oracle surrogate-vs-exact B-mask Jaccard and requires exact projection fraction 1.0 for promotion.

This avoids creating a new V64.3.9-style semantic attribution error at the B layer.

## 6. Upgraded causal experiment protocol

The executable protocol is now:

- **C0:** fixed frozen-foundation decisive-margin utility target.
- **C1-M:** exact-HAB oracle Top-M utility capture.
- **C2-M:** learned exact-HAB Top-M utility capture.
- **C1-B:** C1-M passed through the exact runtime B=16 selector.
- **C2-B:** C2-M passed through the same exact runtime B=16 selector.
- **C3:** teacher match/regret with DARM, DBR and foundation frozen.

This is suitable for a stronger paper contribution because it separates interface capacity, learned acquisition, budget transmission and downstream decision transmission under deterministic controlled planner-interface interventions.

The claim should remain an interface-level causal attribution protocol, not a physical-world causal counterfactual claim.

## 7. Stop/pivot rules

V64.3.11 is intentionally designed to end the acquisition ambiguity.

- If anchor exact C1-B minus C2-B is below 0.5pp, acquisition transmission is not binding. Pivot to decisive value/frontier.
- If C1-B headroom exists but C2-B cannot gain at least 0.5pp and close 15% of the gap without harming decisive-support diagnostics, B-transmitted acquisition remains the bottleneck.
- If C2-B improves but C3 does not, set `pivot_to_value_frontier=true`; stop acquisition work.
- Only if C2-B and C3 both improve can the run proceed to full.

The checker reports separate best-mechanism and best-endpoint epochs and whether they are concordant.

## 8. Engineering audit and efficiency

V64.3.10 profiling shows loss construction is the largest training cost: about 99--116 s of a 189--203 s epoch. BTP removes the executed HAP Python pair-ranking loop and replaces it with a vectorized pair tensor over a small Top-M support. The B=16 projection is still computed, so no speedup is claimed before measurement.

Engineering changes in V64.3.11 include:

- exact Top-M override support in the exact pair-conditioned selector training helper;
- exact B-projection validation semantics;
- surrogate/exact Jaccard diagnostics;
- strict V64.3.11 configuration/semantic contract;
- BTP screen checker with explicit value-pivot stop rules;
- separate screen/full launchers and non-overlapping output roots;
- test coverage for same-family transmitted ranking, exact validation projection and capacity-not-binding pivot behavior.



## 9. Post-implementation exact-mediator contract fix

Before delivery, a targeted test of C1-B/C2-B found that the exact-evaluation adapter re-applied soft-interaction Top-M reservation after an already-finalized exact Top-M override was injected.  This could admit an atom outside the controlled Top-M pool and would invalidate the mediation interpretation.  The deployed selector itself was not leaking outside its active domain; the error was the override adapter rebuilding part of the Top-M policy.

The override is now terminal: exact injected Top-M membership is never post-processed again.  Validation additionally reports `bdmu_budget_projection_topm_violation_fraction`, the screen requires it to be zero, and the V64.3.11 preflight contract contains a synthetic adversarial nested-interface check.  This changes no algorithmic objective or runtime planner behavior; it only makes the C1-B/C2-B causal protocol executable and auditable.

## 10. Delivery validation

Final repository regression after BTP-BDMU and exact-mediator hardening: **323 passed, 0 failed, 34 warnings**.  The warnings are the pre-existing PyTorch Transformer nested-tensor warnings.  Both V64.3.11 screen/full config contracts pass, the BTP semantic contract passes, shell syntax for the screen/full launchers and `NEXT_COMMANDS_V64_3_11_BTP_BDMU.txt` passes, and the adversarial exact-budget fixture returns `selected_budget=[2,3]`, `injected_topm=[2,3]`, `outside_topm=[]`.
