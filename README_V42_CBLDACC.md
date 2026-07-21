# v42 CBL-DACC

CBL-DACC is a runtime-only, query-free repair of v41 PR-DACC's remaining fixed-budget action-preservation failures.

## Why v41 failed

The strict 1000-scene runtime gate reached 0.947 target-action preservation. The v41 deletion-lattice beam was attempted on all 53 failures, completed 14 deletion levels, retained 12 final B=16 subsets per scene, and recovered none. Its exact evaluations were concentrated at larger, non-executable cardinalities.

## v42 change

CBL-DACC searches directly on the executable B=16 layer. It uses one-out/one-in counterfactual exchanges, exact target-vs-current-rival deficits, deterministic diversity and bounded multi-step search. It replaces the prior subset only after the unchanged deployment callback confirms exact target-action recovery.

## Main settings

```yaml
selector_cap_mode: counterfactual_budget_layer_coreset
deployment_coreset_use_deployment_pair_graph: true
deployment_coreset_beam_width: 0
deployment_coreset_budget_layer_width: 12
deployment_coreset_budget_layer_branch: 18
deployment_coreset_budget_layer_iterations: 8
deployment_coreset_budget_layer_max_evaluations: 2400
deployment_coreset_budget_layer_exhaustive_first: true
deployment_coreset_budget_layer_seed_count: 4
deployment_coreset_budget_layer_diversity_distance: 4
```

Run `run_v42_cbldacc.sh` with the frozen v30 checkpoint. Do not run closed loop or finetune until the strict runtime gate passes.
