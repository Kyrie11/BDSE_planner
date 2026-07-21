# BDSE v42 Runtime Result Analysis and v43 SAB-DACC Design

## 1. Executive conclusion

The uploaded v42 experiment does not indicate that the runtime selector failed its substantive gate. It achieved:

- target-action preservation: 0.951, above 0.950;
- teacher-match paired LCB: -0.0039372, above -0.005;
- B=16-vs-full paired LCB: -0.00120038, above -0.002;
- winner-rival sign paired LCB: -0.000345971, above -0.005;
- unchanged evidence-query counts.

The reported `FAIL` is caused solely by applying an activation threshold to the unconditional mean of a conditionally executed diagnostic. Recovery ran on 53 scenes; its conditional mean iteration count was 7.547 and its conditional minimum was 1. The value 0.4 is the mean after adding 947 legitimate zeros from scenes that required no recovery.

Therefore two independent corrections are required:

1. correct the gate checker semantics and execution cost;
2. improve the residual search objective without altering any parameter or threshold.

## 2. What v42 actually improved

v41 retained 947/1000 Top-M deployment actions. v42 fixed-budget layer search recovered four of those failures and retained 951/1000. This validates the v42 hypothesis that executable-layer multi-step exchange is more effective than a deletion-lattice beam.

The correct interpretation is not “v42 failed again”; it is:

- the algorithm crossed the substantive threshold;
- the checker was inconsistent with conditional execution;
- 49 residual failures remain available for a more principled search improvement.

## 3. Residual failure mechanism

Among the 49 remaining failures, 12 have:

- best raw target rank equal to 1;
- raw target score deficit equal to 0;
- no selected-action safety flag;
- no all-flagged risk-guard activation.

The deployment callback executes:

1. pair-conditioned score reconstruction;
2. safety score guard/hard filtering;
3. utility refinement inside a certificate-constrained score band;
4. all-flagged structural guard.

However, v42 returns only final action, scores and margins to the selector. The scores are post-safety scores, while the final action may already have been changed by utility refinement. Consequently, a state can have target rank 1 and zero raw deficit while still producing the wrong final action. v42's search key cannot distinguish this state from one that is arbitrarily close to recovery.

This is the root algorithm issue addressed in v43.

## 4. SAB-DACC

SAB-DACC preserves the full deployment diagnostics in the exact evaluation cache and classifies each fixed-budget subset by the stage at which it fails:

- stage 0: final target action preserved;
- stage 1: target is not the post-safety score winner;
- stage 2: target is the post-safety score winner but utility refinement selects another action.

For stage 2, trajectory utility is fixed by candidate geometry, so evidence selection should not attempt to optimize utility. It should alter certificate eligibility. The utility-selected rival can be excluded by either:

- moving its score below `best_score - score_slack`; or
- moving `M[rival,target]` below `-pair_margin_tolerance`.

SAB-DACC computes both exact boundary distances and uses the smaller one as the recovery certificate. The fixed-budget search then has a meaningful gradient even when raw target rank is already 1.

The search configuration is unchanged from v42. The improvement is informational and structural, not a parameter sweep.

## 5. Gate checker correction

### Semantic correction

The checker now separates:

- configured recovery capacity: `budget_layer_iteration_limit`;
- executed iterations per scene;
- conditional executed-iteration statistics over attempted scenes.

The same threshold of one iteration is applied to attempted scenes. No gate floor or non-inferiority margin is reduced.

### Performance correction

The original path loaded both complete JSONL files into dictionaries, allocated NumPy arrays, and then launched a second parser for the analysis report. The new path:

- streams aligned candidate/control rows;
- retains only online count, mean and M2 per metric;
- falls back to a compact keyed scalar map only when ordering differs;
- generates conditional analysis in the gate process;
- runs with `python -S` because it uses only the standard library.

On the uploaded files in the delivery environment:

- original checker: 2.34 s, 379 MB peak RSS;
- original separate analysis: 2.07 s, 334 MB peak RSS;
- v43 combined checker and analysis: 0.57 s, 18 MB peak RSS.

The paired means and one-sided 95% LCB values are numerically identical.

## 6. Controlled experiment

Frozen for v43:

- v30 checkpoint;
- 1000 validation scenarios and ordering;
- Top-M evidence set;
- B=16;
- pair/local model predictions;
- MARS control;
- tournament, safety, utility and structural guard configuration;
- width=12, branch=18, iterations=8, max evaluations=2400;
- all runtime gate thresholds.

The only algorithm change is deployment-stage-aware recovery feedback.

## 7. Acceptance criteria

The v43 run should satisfy:

- target-action preservation at least 0.950;
- zero loss of v42-preserved scenarios when aligned JSONLs are compared;
- all existing paired LCB gates;
- stage-aware recovery diagnostics present;
- conditional recovery iterations at least one whenever attempted.

Additional recovery beyond 0.951 is desirable but not required to correct the false v42 gate failure. If v43 exactly matches v42 actions and passes, it remains the preferred code because its checker semantics and search-state representation are correct.
