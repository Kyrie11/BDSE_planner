# BDSE v41 Runtime-Gate Failure Analysis and v42 CBL-DACC Design

## 1. Executive conclusion

The v41 runtime gate is no longer failing because of pair-sign quality, teacher alignment, budget violation, query count, or a missing deployment callback. It fails only because the decision coreset preserves the Top-M deployment action in 947/1000 scenarios, three below the required 950.

The attempted v41 repair mechanism is ineffective in the real run: it triggers on all 53 failures, reaches the final B=16 depth in every case, and records zero successes. The issue is not that it stops early. The issue is that it spends most of its exact evaluations on intermediate cardinalities and retains only 12 final subsets from a B=16 search layer containing 145,422,675 possible subsets.

v42 therefore replaces the deletion-tree repair with **Counterfactual Budget-Layer DACC (CBL-DACC)**: a bounded, exact, rival-directed exchange search operating only on executable B=16 subsets.

## 2. Audited v41 runtime result

### Failed gate

```text
selector_deployment_coreset_target_action_preserved = 0.947
required                                             = 0.950
shortfall                                            = 3 scenarios
```

### Passed quality and budget checks

```text
teacher_action_match             0.239
control teacher_action_match     0.238
paired mean difference           +0.001
one-sided 95% LCB                -0.0039372
allowed margin                   -0.005

budget_vs_full_match             0.209
control budget_vs_full_match     0.205
paired mean difference           +0.004
one-sided 95% LCB                -0.00120038
allowed margin                   -0.002

winner-rival sign accuracy       0.6395124
paired mean difference           +0.00114443
one-sided 95% LCB                -0.00181182
allowed margin                   -0.005

effective_query_count            5383.152
total_sparse_query_count         11252.15
```

The runtime experiment therefore isolates a search problem. Changing the checkpoint or neural query path now would destroy that isolation.

## 3. Per-scenario failure slice

There are 53 failed-preservation scenarios.

| Diagnostic | Failed scenes | Preserved scenes |
|---|---:|---:|
| Top-M decision atoms | 30.000 | 29.970 |
| Removed atoms | 14.000 | 13.970 |
| Forced action-flip deletion steps | 6.528 | 0.010 |
| Preservation scan evaluations | 306.887 | 1.515 |
| Beam evaluations | 1669.585 | 0 |
| Beam depth | 14 | 0 |
| Beam terminal count | 12 | 0 |
| Winner-rival sign accuracy | 0.5266 | 0.6459 |
| B=16 vs full match | 0 | 0.2207 |
| Teacher match | 0.0377 | 0.2503 |

Additional facts:

- one-swap repair succeeds on none of the 53 failures;
- the configured screened two-swap repair succeeds on none;
- beam search is attempted on exactly the 53 failures;
- beam success is 0/53;
- all 53 beams reach the expected 14 deletion levels;
- every failed beam ends with exactly 12 terminal states;
- only 5 of the 53 failure actions differ from the MARS control action.

The last point is important: v41 mostly reconstructs another path to the same MARS action on these failures. It does not discover new executable target-preserving subsets.

## 4. Why the v41 beam did not help

The v41 beam begins with approximately 30 active decision atoms and deletes one atom per level until 16 remain. With width 12 and branch 14, about 1670 exact subsets are evaluated per failed scene, but almost all of those subsets have more than 16 atoms and cannot be deployed under the fixed budget.

At the only layer that matters, the algorithm retains 12 subsets. The complete final layer contains:

```text
C(30, 16) = 145,422,675 subsets
```

Thus the retained terminal coverage is about `8.25e-8` of the layer before accounting for constraints. Zero observed success does not establish infeasibility. It establishes that the search allocation is poorly matched to the final decision constraint.

A second issue is the recovery ranking. v41 combines deployment distortion, full-target gap loss, and margin sign loss. For an action-mismatching state, the most useful local direction is often narrower: improve the exact target action relative to the action currently selected by the full tournament/utility/safety operator. v41 does not prioritize that counterfactual strongly enough at the terminal layer.

## 5. v42 algorithm: CBL-DACC

### 5.1 Search domain

CBL-DACC searches directly on the B=16 layer. A state is an executable evidence subset. An edge swaps one selected atom for one unselected atom:

```text
S' = S - {e_out} + {e_in}
```

The budget, cardinality, and soft-interaction floor are checked before exact evaluation.

### 5.2 Trigger

The search runs only when all prior mechanisms fail:

1. lexicographic deployment-aligned deletion;
2. normal deployment-preserving swap refinement;
3. exhaustive direct one-swap repair;
4. screened direct two-swap repair;
5. optional deletion beam.

The v42 main configuration disables step 5 because the real v41 experiment demonstrated zero utility.

### 5.3 Exact recovery state

For each B=16 subset, the unchanged deployment callback returns:

- final action after tournament;
- final action scores;
- final antisymmetric margin matrix;
- safety/utility/all-flagged effects already included in the chosen action.

A mismatching state is ranked by the tuple:

```text
(action mismatch,
 target score rank,
 target-vs-current/strongest-rival score deficit,
 target-vs-current-rival margin deficit,
 deployment distortion,
 surrogate distortion)
```

An exact action-preserving state always dominates a mismatching state. Mismatching states are retained only as paths toward later recovery.

### 5.4 Rival-directed mutations

For the current selected rival, CBL-DACC orients pair-atom contributions as target minus rival. A swap receives multiple cheap priorities:

- reduction in deployment-graph reconstruction loss;
- gain in target-vs-current-rival support;
- gain in global target-vs-rivals support;
- gain in proposal prior;
- deterministic outgoing/incoming diversity.

The first B=16 one-swap neighborhood is exhaustive in the common 30-to-16 case: 224 candidates. Later generations use a width-12, branch-18 bounded search for at most eight generations and 2400 new exact evaluations.

### 5.5 Seeds

The current v41 B=16 subset is always included. Up to four additional executable seeds are constructed from target support and proposal orderings. They are used only inside the recovery search.

### 5.6 Safe replacement

The deployed subset is changed only if the exact callback returns the Top-M target action. If no exact recovery is found, v42 returns the unchanged v41 subset. This avoids a new failure mode where a smoother target gap produces a different but still incorrect action.

## 6. Expected runtime behavior

CBL-DACC adds no model forward pass and no evidence query. It reallocates the existing style of CPU/Numpy deployment re-evaluations:

- v41: about 1670 evaluations spread over 14 cardinality levels on 53 scenes;
- v42: at most 2400 evaluations concentrated entirely on executable B=16 subsets on the same failure class.

The runtime gate requires only three successful recoveries, but success must not break paired teacher, B-vs-full, or sign non-inferiority.

## 7. New diagnostics and next branching decision

The new analyzer reports attempts, successes, unique B-layer states, best target rank, exact target deficit, recovered v41 tokens, lost v41-preserved tokens, and teacher/full transition counts.

Interpretation:

- `budget_layer_success >= 3`, `v41_preserved_lost = 0`: preservation mechanism worked;
- best target rank reaches 1 but action remains wrong: utility/safety refinement, not raw tournament rank, is the residual mechanism;
- unique states near evaluation cap and target rank stays >1: search needs a larger mutation radius or more targeted branch proposals;
- unique states saturate early: branch diversity, not evaluation cap, is limiting;
- paired teacher LCB fails after recovery: inspect only recovered transitions before changing the algorithm.

## 8. Validation completed

- all Python modules compile;
- shell syntax passes;
- complete test suite: 124 passed, 5 warnings;
- a synthetic test requires three sequential fixed-budget exchanges; one/two-swap repair fails and CBL-DACC succeeds;
- configuration loading and main/fallback/control separation pass.

No nuPlan runtime claim is made in the delivery environment.
