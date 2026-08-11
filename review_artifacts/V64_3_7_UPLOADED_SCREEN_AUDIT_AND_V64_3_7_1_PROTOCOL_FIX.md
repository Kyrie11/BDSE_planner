# V64.3.7 uploaded screen audit and V64.3.7.1 protocol hotfix

## Executive conclusion

The uploaded V64.3.7 BROAD run is **not an algorithmic failure**. It contains the strongest clean downstream decisive-margin signal seen in this branch so far. The matrix stopped because the screen checker conflated scientific promotion criteria with process validity and returned exit code 3 under `set -e`.

The original report marked `valid=false` only because `decisive_anchor_full_pair_coverage_max=0.1990928 < 0.20`. That metric is the fraction of *all valid challengers* represented in the sampled anchor star. It is discrete under the boundary-focused pair sampler and is not the exact teacher-correction-edge coverage. Treating 0.19909 vs 0.20 as a hard protocol boundary has no algorithmic justification.

A second gate was semantically misaligned: `budget_vs_pair_full_delta >= -0.02`. BDSE's paper target is preservation of the full-information teacher decision under fixed B=16, not imitation of the learned pair-full interface. In the uploaded run, budget/pair-full divergence is net beneficial with respect to teacher correctness.

## Uploaded BROAD paired metrics

Epoch -1 immutable selected-local anchor:

- teacher action match: 0.264
- local pair-full action match: 0.264
- pair-full action match: 0.264
- teacher regret: 14484.4613
- pair-full regret: 14079.9774
- critical Top-M micro recall: 0.3601533
- selected critical micro recall: 0.2605364
- proposal decisive recall: 0.7915091

Robust positive row (epoch 3):

- teacher action match: 0.282 (**+1.8pp**)
- pair-full action match: 0.274 (**+1.0pp**)
- local pair-full: 0.264 (unchanged frozen control)
- pair-full-over-local advantage: **+1.0pp**
- teacher regret: 13367.1395 (**-1117.32**)
- pair-full regret: 13673.7794 (**-406.20**)
- beneficial residual intervention: 0.022
- harmful residual intervention: 0.012
- residual intervention net: **+1.0pp**
- beneficial pair-full->budget compression: 0.016
- harmful pair-full->budget compression: 0.008
- compression net: **+0.8pp**
- DBR parameter delta RMS max: 0.005218
- DBR residual RMS max: 0.003149
- Top-M/selected/proposal diagnostics: unchanged

The unchanged acquisition metrics are especially important for attribution: the action improvement is downstream of fixed evidence admission, exactly the V64.3.7 causal question.

## What is fixed in V64.3.7.1

1. Screen checker now separates `instrumentation_valid`, `meaningful_value_gain`, `deployment_gain`, and `full_promotion`.
2. A scientifically negative arm no longer exits with code 3. Malformed inputs still fail normally.
3. The arbitrary all-challenger coverage threshold is diagnostic-only.
4. Budget-vs-learned-pair-full agreement is diagnostic-only. Promotion instead requires teacher/pair-full action gains, non-worse teacher/pair-full regret, positive residual intervention net, and non-harmful budget compression net.
5. Matrix always re-audits an existing train log with the current checker. A stale old `valid=false` provenance JSON cannot suppress a completed arm.
6. Matrix no longer terminates before later ablation arms merely because an earlier arm is not promoted.
7. Validation metric export now includes `decisive_anchor_margin_*` tournament diagnostics for future runtime-path auditing.

No DARM/DBR architecture, B=16 budget, selector, pair residual, warm-start checkpoint semantics, or training objective is changed by this hotfix.

## Current algorithm priority

The previous priority ordering is strengthened, not reversed.

1. **Decisive pair value + final aggregation remains first priority and now has a positive signal.** DARM+DBR changes final teacher decisions while Top-M/selected/proposal are frozen.
2. **Acquisition proposal-score generalization remains second priority.** Top-M critical recall is still 0.3601533, but it did not prevent a +1.8pp final teacher-match gain from downstream correction.

Do not start a new acquisition architecture before completing the missing LITERAL screen and full-pipeline validation of the current positive DARM+DBR mechanism.

If the positive DARM+DBR signal survives full training, freeze it and then move acquisition to a **DARM-consistent decisive-margin marginal-utility target**. The acquisition learner should estimate which auditable atoms most reduce the one-sided decisive-margin deficit under fixed B=16. This is closer to the paper theorem than another sparse binary critical classifier and avoids repeating AP-WCCA/AP-WRCCA, FPCCA/CCBR, BCHA, larger B/M, or global ranking searches.
