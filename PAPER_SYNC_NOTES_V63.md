# V63 Paper Synchronization Notes

1. Preserve the fixed retained evidence atom budget B as the primary interface novelty.
2. State separately that V63 acquisition scores M atoms on K valid actions (M×K) before selecting B; do not call B×K the total internal query cost.
3. Update the runtime rival-graph text: all valid actions receive action-conditioned values; the rival graph defines pair comparisons/tournament structure.
4. Replace “typically M in [2B,4B]” or change the implementation; current main configuration is M=24, B=16 (1.5B).
5. Define criticality literally using teacher-interface leave-one-out winner flips. Severity is a ranking score only among critical atoms.
6. Scope “exact selector” to deterministic exact execution/audit of the configured fixed-budget AOCC operator. Do not claim global combinatorial optimality of the greedy/anytime acquisition order.
7. State the split protocol: group-disjoint val_tune for model/algorithm selection, val_calib for one-sided calibration, completed frozen test exactly once.
8. Fix the malformed `\\text{Quantile}` command near line 614 of the uploaded TeX.
9. Do not populate V62/V63 result tables with V53 anchor replay or train-time proxy metrics.
10. Add a layered bridge ablation: model base, deployment base, dense Top-M, sparse Top-M, B16 selected, residual-deployed winner.
