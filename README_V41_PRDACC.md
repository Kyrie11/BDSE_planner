# v41 PR-DACC

PR-DACC is the controlled successor to v40 Lex-DACC for the strict runtime gate.

## Why v40 failed

v40 passed every paired non-inferiority check but reached only 0.946 deployment-target action preservation, below the 0.950 gate. The remaining failures are dominated by single-path deletion dead ends: exhaustive one-step scans and local one/two swaps do not retain alternative earlier subset paths.

## What v41 changes

- Uses the final rival pair graph for the coreset search screen.
- Keeps v40 lexicographic deletion and local repair.
- On a remaining action flip only, runs a bounded fixed-cardinality beam over evidence subsets.
- Retains exact target-action states plus action-diverse temporary mismatches.
- Accepts a beam result only when it restores the exact Top-M deployment action under B=16.
- Adds no neural query and does not change the checkpoint.

## Main configuration

```yaml
selector_cap_mode: path_relaxed_deployment_coreset
deployment_coreset_use_deployment_pair_graph: true
deployment_coreset_beam_width: 12
deployment_coreset_beam_branch: 14
deployment_coreset_beam_max_evaluations: 2400
deployment_coreset_beam_mismatch_fraction: 0.42
```

## Run

Use `run_v41_prdacc.sh` with the frozen v30 checkpoint. See `run_v41_prdacc_commands.txt`.

Do not run closed-loop or finetuning until `open_loop/runtime_gate.txt` reports PASS.
