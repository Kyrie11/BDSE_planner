# v40 Runtime Gate Failure Analysis and v41 PR-DACC Design

## 1. Exact v40 gate result

The strict 1000-scenario gate failed only because:

```text
selector_deployment_coreset_target_action_preserved = 0.946 < 0.950
```

All main paired tests passed:

```text
teacher_action_match: diff +0.001, one-sided 95% LCB -0.003937 > -0.005
B=16 vs full:       diff +0.004, one-sided 95% LCB -0.001200 > -0.002
winner-rival sign:  diff +0.000102, LCB -0.003047 > -0.005
```

Consequently, the next experiment should not change the checkpoint or prediction heads. The failing component is the deployment coreset subset search.

## 2. What v40 fixed

v40 correctly removed the v39 finite-action-penalty failure. It also expanded exact deletion scans, added one/two-swap repair, and re-audited post-fill. Preservation improved from 0.919 to 0.946.

## 3. Why the remaining 54 samples fail

The failure slice is qualitatively different from v39:

| Diagnostic | v40 failed 54 | v40 preserved 946 |
|---|---:|---:|
| Forced action-flip deletion steps | 6.35 | 0.013 |
| Expanded exact deletion evaluations | 306.61 | 1.97 |
| Total deployment subset evaluations | 750.41 | 121.97 |
| One-swap evaluations | 224 | 0.95 average |
| Two-swap evaluations | 256 | 0.54 average |

For a failed state, v40 already evaluates every feasible one-step deletion when no preserving deletion exists. Therefore the problem is not simply an insufficient exact top-k at that state.

The actual limitation is single-path greedy deletion. Action preservation can be non-monotone over evidence-set cardinality: a locally best preserving subset can become impossible to shrink while another earlier branch remains feasible. In other cases, every subset at an intermediate cardinality can change the action even though a smaller subset restores it. A strict one-path algorithm discards both possibilities.

A second residual mismatch is that v40's exact evaluator uses the final rival graph while its cheap branch screen still uses the selector graph. This matters more once several branches must be ranked.

## 4. v41 PR-DACC

PR-DACC adds a failure-triggered, bounded, fixed-cardinality beam over the evidence deletion lattice.

### State dominance

The beam retains:

1. exact target-action states, ranked by deployment distortion;
2. a bounded fraction of temporary mismatches, ranked by target score-gap and margin recovery potential;
3. action diversity among mismatches, so all exploration slots do not select the same wrong winner.

### Branch generation

Each state proposes removals using a union of:

- deployment-aligned surrogate loss;
- low absolute target-action pair impact;
- negative oriented support that currently harms the target action;
- low learned proposal prior.

All retained children are evaluated using the exact deployment callback. The callback remains the final rival graph, pair-margin reconstruction, soft-min tournament, safety score guard, certificate/utility refinement and all-flagged guard.

### Trigger and bound

The beam runs only if v40 lexicographic search plus one/two-swap repair still fails. Defaults:

```yaml
deployment_coreset_beam_width: 12
deployment_coreset_beam_branch: 14
deployment_coreset_beam_max_evaluations: 2400
deployment_coreset_beam_mismatch_fraction: 0.42
```

For the common 30-to-16 deletion case, the theoretical expansion is about 2200 child evaluations before duplicate-cache savings, below the configured cap.

## 5. Expected outcome and falsification

The gate needs at least four of the 54 failures to be recovered without losing already preserved cases. PR-DACC never replaces an already preserved v40 set and only accepts a beam result that exactly matches the target action, so its direct preservation effect is monotone by construction.

The hypothesis is falsified if:

- beam is attempted on nearly all 54 failures;
- it reaches full depth with adequate width/branch;
- beam success remains zero or below four scenarios.

That result would suggest that the Top-M deployment target is genuinely infeasible under B=16 in most failures, rather than merely hidden by search. The next step would then be an explicit budget-feasibility diagnostic or a redefinition of the deployment target, not a larger soft penalty.

## 6. Required next experiment

Run v41 with the unchanged v30 checkpoint and frozen MARS control. Do not run closed-loop or finetuning unless the strict runtime gate passes.
