# BDSE v37 runtime failure analysis and v38 MARS-BDSE plan

## 1. Executive conclusion

v37 did not fail because of insufficient hard-evidence coverage, interaction-evidence recall, runtime safety enforcement, or query budget.

The balanced v37 result already achieved:

- structural hard decisive coverage: 1.0;
- effective hard decisive recall: 1.0;
- selected soft-interaction decisive recall: about 0.545;
- effective interaction decisive recall: about 0.620;
- avoidable selected-action safety-flag rate: 0;
- effective query count: about 5.38k;
- total sparse query count: about 10.4k.

The remaining failure is a **full-interface to fixed-budget margin-preservation failure**:

- full-interface teacher action match: 0.265;
- B=16 teacher action match: 0.223;
- B=16 vs full-interface action match: about 0.169.

Thus the selector is retrieving many relevant atoms, but the selected subset does not preserve the signed aggregate action margin produced by the full decision interface.

## 2. Evidence from paired per-sample analysis

For the v37 balanced 1000-sample result:

| Slice | Count |
|---|---:|
| Full interface and B=16 both correct | 102 |
| Full interface correct, B=16 wrong | 163 |
| B=16 correct, full interface wrong | 121 |
| Both wrong | 614 |

There is a net recoverable headroom of 42 samples if B=16 better approximates the full-interface decision.

More importantly, the samples where B=16 is wrong do **not** have lower interaction recall. Their selected soft-interaction recall is often higher. By contrast, winner/rival pair-sign accuracy is sharply lower on wrong samples. This invalidates further attempts to solve the problem through family quotas, larger interaction coverage, or another hard/soft budget rebalance.

The four v37 configurations select the same action on about 92.3% of samples. Therefore changing structural residual weight, interaction quota, or pair-frontier weighting within the old objective cannot provide the missing improvement.

## 3. Root cause in the v37 selector

The existing selector optimizes atom-wise positive certificate support, uncertainty and action-rank utilities. This objective is not equivalent to preserving the full signed margin:

1. Negative evidence can be decisive, but receives little or no positive acquisition credit.
2. Two individually strong atoms can cancel; independent ranking cannot model the cancellation.
3. A collection of modest atoms can jointly flip a winner/rival margin, even if none ranks highly alone.
4. High recall only says decisive atoms were included, not that their signed sum reproduces the full-interface ordering.
5. The runtime pair path largely ignores the existing action-conditioned local head, although dense diagnostics show substantially better interaction and winner/rival sign accuracy from that interface.

This is a semantic mismatch between the paper's decision-sufficiency objective and the runtime selector's coverage-oriented surrogate.

## 4. v38 algorithm: MARS-BDSE

MARS means **Margin-Aligned Residual Sparsification**.

### 4.1 Adaptive local-head calibration

For every Top-M atom and queried action, v38 uses the existing action-conditioned local head. For pair `(a,b)`, the local delta is:

`delta_local(e,a,b) = g(e,b) - g(e,a)`.

The final runtime delta is an adaptive blend of the pair-conditioned prediction and local delta. The local weight increases when:

- pair variance is high;
- pair and local heads disagree in sign;
- the local margin has meaningful magnitude.

No new learned parameters are introduced and the v30 checkpoint remains structurally compatible.

### 4.2 Signed margin coreset

Let the Top-M decision atoms define the predicted target field:

`m_full(p) = m_base(p) + sum_e delta(e,p)`.

Instead of independently adding the highest-scoring atoms, v38 begins with all active Top-M atoms and removes atoms until the total cost is B=16. Each removal minimizes a coreset loss containing:

- robust signed-margin reconstruction error;
- pair-sign mismatch penalty;
- target winner/rival certificate penalty;
- target-action disagreement penalty;
- a minimum soft-interaction floor.

A deterministic swap refinement then replaces weak selected atoms with excluded atoms when this lowers the same loss.

The selector uses only runtime predictions, candidate validity, runtime safety flags and the fixed budget. It does not use teacher actions, logged futures or future labels.

### 4.3 Why this is aligned with the paper

The paper's central claim is not merely that the selector retrieves atoms belonging to a relevant family. It is that the selected evidence is decision-sufficient under a fixed budget. A margin coreset directly operationalizes this claim:

> the sparse certificate should preserve the action preference field induced by the complete decision interface.

This is a stronger and more defensible contribution than manually enforcing ever more family quotas.

## 5. What v38 deliberately does not repeat

v38 does not repeat the previously attempted changes:

- no larger B;
- no larger proposal M;
- no new hard/soft family quota sweep;
- no further structural safety partition change;
- no pair-frontier hard restriction;
- no simple structural-risk weight sweep;
- no lowering of the absolute teacher-match, safety, recall or query gates;
- no newly trained AFV head before runtime causality is established.

## 6. Runtime gate design

The absolute gates remain:

- teacher action match >= 0.215;
- B=16 vs full-interface action match >= 0.17;
- structural/effective hard coverage >= 0.98;
- selected soft-interaction recall >= 0.32;
- effective interaction recall >= 0.35;
- avoidable safety flag <= 0.005;
- fallback <= 0.02;
- effective queries <= 8500;
- total sparse queries <= 33000.

The gate also checks that structural safety partitioning, structural residual, local calibration and margin coreset are actually active in the intended configurations.

For coreset configurations, the gate additionally requires at least 0.90 target-action preservation and 0.90 target pair-sign agreement. When a baseline JSONL is available, v38 performs paired one-sided non-inferiority checks by scenario token and timestamp with a strict 0.005 practical margin. If paired data are unavailable, the original strict summary tolerances are retained. The absolute performance gates are unchanged.

## 7. Expected causal interpretation of the four runs

- **balanced passes, pair-only fails**: local-head calibration is necessary.
- **pair-only passes, action-rank control fails**: signed margin coreset is the key mechanism.
- **action-rank control improves but coreset improves more**: both calibration and coreset contribute.
- **all coreset configs fail but action-rank control improves**: pair calibration helps, but the coreset loss/implementation needs per-sample attribution before CL20.
- **all configurations fail without action changes**: verify v38 activation diagnostics and checkpoint/config wiring.

## 8. Training recommendation

Continue using the unchanged v30 checkpoint for this runtime-only gate. v38 adds no parameter tensors. Training now would mix selector causality with weight adaptation.

Only after a v38 main configuration passes open-loop and shows no CL20 safety/progress regression should a controlled v30-initialized finetune be run. A clean multi-seed training run is appropriate only after the runtime architecture, candidate bank, evidence taxonomy and gate are frozen.
