# v33 CARB-BDSE

CARB-BDSE addresses the measured v32 regressions without changing the fixed evidence budget B=16:

- restores feasibility/hard evidence coverage with balanced quotas;
- adds an absolute TTC/agent-overlap barrier before relative Pareto recovery;
- evaluates off-route distance against all declared route-graph interior edges;
- activates speed-adaptive horizons at moderate-high speed;
- conditionally preserves the BDSE certificate inside the physically viable set;
- selects checkpoints with hard-recall, interaction-recall, fallback, and query-budget constraints;
- provides a reproducible closed-loop artifact packager.

See `V33_CARB_ANALYSIS_AND_PLAN.md` and `run_v33_two_gpu_commands.txt`.
