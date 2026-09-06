# V64.3.56 post-convergence external benchmark protocol

## Primary comparison

- full public_set_test closed loop;
- exact same ordered scenario tokens for every system;
- challenge: closed_loop_nonreactive_agents;
- primary budget: B=16;
- proposal bank M=24 for the bounded-interface comparison;
- own method: frozen EAF-RSMR (V47 RSMR config + exact V13 EAF checkpoint);
- external trainable adapters: GameFormer-inspired, DTPP-inspired, PlanTF-inspired, PLUTO-inspired, each trained independently for B=8/B=16/B=24;
- PDM-Closed-style is diagnostic and must not be named as official PDM-Closed.

## Budget sweep

B=8/B=16/B=24 is run on exactly the same scenario manifest. External trainable adapters use their budget-specific checkpoint. BDSE B=8/B=24 use the frozen B16 policy with only the interface budget changed and are therefore **cross-budget robustness ablations**, not primary matched-training comparisons.

## Metric safety

All closed-loop systems use `bdse.tools.nuplan_metric_safe_run_simulation`. Only the stateful nuPlan metric callback is serialized; simulation/planner workers remain parallel (`CL_WORKERS_PER_JOB=4` by default). Resume requires an explicit metric-safety provenance marker, so legacy unsafe results are automatically rerun.

## Reporting

Report the B16 table as primary. Report the three-budget curves as a separate budget robustness/efficiency figure. Do not pool or relabel the B8/B24 BDSE ablation as budget-specific retraining.

## Secondary reactive check

After the full nonreactive B8/B16/B24 sweep, run a B16-only `closed_loop_reactive_agents` comparison on all systems. This is a robustness check rather than a replacement for the primary matched internal outcome setting; it is especially important because the V50-V56 paired mechanism-development evidence used the nonreactive challenge and therefore cannot by itself support claims about causal behavioral reactions of other traffic participants.
