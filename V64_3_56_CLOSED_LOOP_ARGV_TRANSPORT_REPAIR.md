# V64.3.56 benchmark engineering repair: file-backed scenario transport

## Failure
The converged closed-loop runner previously serialized all 66,671 ordered scenario tokens into one Hydra override (`scenario_filter.scenario_tokens=[...]`) and passed it through `execve(2)`.  Linux limits the length of an individual argv string; the single override was roughly 1.27 MB, so the child Python process could not even start and raised `OSError: [Errno 7] Argument list too long`.

This was an engineering transport failure. It does not change the frozen V56 science result or any planner algorithm.

## Repair
- `run_fixed_budget_closed_loop_suite.py` writes the ordered token list to `scenario_tokens.json` and passes only the manifest path + ordered SHA256 to `evaluate_closed_loop.py`.
- A potentially long resolved raw-DB list is also transported through a JSON manifest.
- `evaluate_closed_loop.py` validates the token manifest hash and uniqueness before reconstructing the exact Hydra override in memory.
- Because re-execing the reconstructed 1.27 MB single argument would fail again, manifest-backed nuPlan runs execute the metric-safe `python -m ...` target in the already isolated evaluation process via `runpy`, preserving its environment/GPU isolation without a second `execve`.
- Ordinary small invocations still use subprocess execution.
- The complete 66,671-token population and one official metric aggregation are preserved; this is not scenario subsampling or post-hoc shard averaging.

## Scientific invariants
- Ordered token list and SHA remain identical across all systems and budgets.
- Metric-safe serialization remains enabled; only the stateful metric callback is serialized, not planner/simulation workers.
- V64.3.56 science sources and the frozen EAF/RSMR artifacts are unchanged.
- Resume accepts only runs with the required metric-safety provenance.

## Budget interpretation
The paper-grade primary comparison is B=16. External trainable adapters are trained separately at B=8/16/24. The converged BDSE model remains the frozen B16 learned method; B=8/B=24 are cross-budget interface robustness tests unless a new, separately preregistered budget-specific EAF/RSMR refit is performed before looking at test results.
