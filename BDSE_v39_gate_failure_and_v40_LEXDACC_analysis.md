# BDSE v39 Runtime Gate Failure Analysis and v40 Lex-DACC Design

## 1. Paper and implementation understanding

The paper frames autonomous planning as a **decision-preserving compression problem under a planner-interface budget**, rather than a future-distribution reconstruction problem. At each nuPlan timestep, the planner receives runtime-observable ego/agent history, map, traffic-control state, route and goal, constructs a finite candidate set, and queries only a bounded evidence subset.

The implemented pipeline is:

1. route-conditioned candidate generation, default K=32;
2. robust metric-aligned candidate teacher for offline supervision;
3. explicit evidence atoms for interaction, rules/map and dynamic regularity;
4. scene/action/evidence encoding and pair-conditioned atom-margin prediction;
5. Top-M evidence proposal, default M=64;
6. runtime rival-pair graph and bounded evidence selection, B=16;
7. normalized antisymmetric pair-margin reconstruction;
8. soft-min tournament, hard safety filtering, certificate utility refinement and all-flagged structural guard;
9. nuPlan trajectory output and open-/closed-loop evaluation.

The dataset protocol in the manuscript uses nuPlan DB train/validation samples with 2 s history and 8 s future horizon. Candidate rollouts are generated at 10 Hz (80 states), optionally scored at 5 Hz (40 states), and interpolated back for nuPlan output. Logged futures are label-only, not runtime features.

The experiment plan is correctly divided into candidate/teacher quality, evidence sufficiency, rival recall, frozen-support budget sweeps, fallback ablations and final closed-loop testing. The main metrics are nuPlan score, collision, DAC, route progress, speed, TTC and comfort, plus teacher match/regret, full-interface match, B-vs-full match, decisive-rival recall, preserved-margin error, fallback rate, latency and effective query count.

## 2. What exactly failed in v39

The strict runtime gate compared v39 DACC against the frozen v39 MARS control on 1000 aligned scenarios.

| Gate item | v39 DACC | Result |
|---|---:|---|
| Teacher match | 0.238 | absolute pass; paired pass |
| B=16 vs full | 0.206 | absolute pass; paired LCB fail |
| Teacher regret | 10221.999 | no gate failure |
| Winner-rival sign | 0.636914 | paired pass |
| Interaction sign | 0.619793 | paired pass |
| Target action preservation | 0.919 | fail, needs >=0.950 |
| Effective query count | 5383.152 | pass |
| Total sparse query count | 11252.15 | pass |

The paired B-vs-full mean difference was +0.001, but its one-sided 95% lower confidence bound was -0.003354, lower than the allowed -0.002.

This is a narrow and structurally interpretable failure: query budget, absolute action metrics, safety/recall gates and pair-sign non-inferiority are already acceptable; deployment-action preservation is not.

## 3. Per-scenario causal diagnosis

### 3.1 Soft penalty caused deliberate action flips

v39 optimized a scalar objective of the form

```text
continuous deployment distortion
+ 4.0 * I[selected deployment action != Top-M target action]
```

The 81 failed target-preservation samples have objective values concentrated just above 4. This is the signature of a finite action-change penalty being paid deliberately. The search is behaving as coded; the coding objective is inconsistent with a gate that treats action preservation as a primary requirement.

### 3.2 Surrogate top-k screening was incomplete

The exact deployment callback was only evaluated for the top 8 removals ranked by a MARS surrogate. Because the surrogate pair graph/objective and full deployment decision map are not identical, an action-preserving deletion may be ranked outside top 8. v39 then accepts a screened action-changing deletion without knowing that a preserving alternative exists.

### 3.3 Two final actions were inconsistent with selector diagnostics

Diagnostic target preservation was 0.919, while direct `target_action == final bdse_action` was 0.917. In two samples, the selector reported that its selected set preserved the DACC target, but the final planner action differed. The reason is that generic safety-aware quota/post-fill ran after DACC and could exchange atoms without re-running the exact deployment audit.

### 3.4 Counterfactual upper bound supports the fix

Using the already logged Top-M deployment target as the final budget action yields the following counterfactual on the same 1000 samples:

| Metric | Counterfactual | vs control mean diff | one-sided 95% LCB | Gate margin |
|---|---:|---:|---:|---:|
| Teacher match | 0.239 | +0.001 | -0.004934 | -0.005 |
| B-vs-full | 0.213 | +0.008 | +0.001856 | -0.002 |

Both paired conditions would pass. This is not proof that the new search will recover every target action, but it establishes that target preservation is the correct first intervention and that no checkpoint change is required to explain the current gate failure.

## 4. v40 algorithm: Lex-DACC

### 4.1 Lexicographic optimization

For a trial evidence subset S, let D(S) be the exact deployed action and Ldist(S) be continuous distortion of deployment scores, winner gap and target-vs-rival margins relative to the full Top-M interface.

v40 ranks candidate subsets lexicographically:

```text
(I[D(S) != D(U_M)], Ldist(S))
```

rather than collapsing the two terms into a finite weighted sum. Therefore an action-changing subset cannot beat an available action-preserving subset merely by reducing continuous RMSE.

### 4.2 Conditional exact-scan expansion

MARS remains a cheap screen. At each greedy deletion:

1. evaluate the top-k screened deletions exactly;
2. if any preserves the deployment target, choose the least-distorting preserving deletion;
3. otherwise expand exact evaluation beyond top-k; configuration value 0 means scan all feasible deletions only in this failure case;
4. record a forced-flip step only when the expanded scan finds no preserving deletion.

This adds CPU evaluations only on ambiguous steps and does not add neural evidence queries.

### 4.3 Repair neighborhood

If the budgeted greedy result still changes the target action:

- exhaustively check feasible one-out/one-in swaps;
- screen two-out/two-in swaps with MARS and exactly evaluate the best 256;
- accept the lowest-distortion action-preserving repair.

This handles cases in which preserving the target requires a non-monotone exchange rather than a sequence of individually preserving deletions.

### 4.4 Post-fill consistency audit

After generic quota/post-fill, v40 evaluates the actual returned subset with the same deployment callback. If post-fill breaks an action that the pre-fill Lex-DACC set preserved, v40 reverts to the pre-fill set. Diagnostics now correspond to the actual executed selection.

## 5. Why this is a controlled algorithm experiment

Frozen across v39 control and v40:

- v30 checkpoint;
- validation scenarios and ordering;
- candidate bank;
- teacher;
- Top-M=64;
- B=16;
- pair-query caps;
- pair/local calibration;
- tournament, hard filter, utility refinement and all-flagged guard.

Changed variable: the combinatorial selection/search policy and its post-fill consistency audit.

This makes any action-metric change attributable to deployment-aligned coreset search rather than retraining or data variation.

## 6. Strict acceptance criteria

Do not weaken the existing gate. Require:

- all v38/v39 absolute recall, safety and query-budget thresholds;
- deployment evaluator present;
- Lex-DACC active rate >=0.99;
- exact target-action preservation >=0.95;
- paired teacher-match LCB >= -0.005;
- paired B-vs-full LCB >= -0.002;
- paired winner-rival sign LCB >= -0.005.

Also inspect, without hiding behind aggregate means:

- `forced_action_flip_steps`;
- preservation scan evaluations;
- one/two-swap repair attempt and success;
- post-fill changed/reverted rates;
- selector deployment evaluations and wall-clock p50/p95/p99.

## 7. Closed-loop decision tree

1. **Strict gate fails because preservation <0.95**: inspect forced flips and repair failures. Next algorithm is action-preserving beam search with state dominance; do not increase the soft penalty.
2. **Preservation passes but B-vs-full paired gate fails**: the Top-M deployment target or full-interface calibration is the bottleneck. Improve target alignment/pair-local gating while freezing Lex-DACC.
3. **Open-loop passes, CL20 fails catastrophically**: inspect scenario traces and integration/config errors; do not tune on aggregate CL20 score.
4. **CL20 is healthy**: run CL100 and compare paired collision/TTC/DAC/progress transitions.
5. **CL100 safety regresses**: keep selector fixed and introduce a separately ablated safety fallback.
6. **Runtime and CL100 both improve**: only then start controlled head finetuning or learnable reliability gating.

## 8. Paper-positioning implications

Lex-DACC repairs an important theorem-algorithm mismatch: the coreset now explicitly prioritizes preserving the action of the actual deployment operator. For a CCF-A submission, this runtime fix alone is not enough. A stronger paper version should formalize a deployment-map decision-preservation guarantee or constrained regret result, report search complexity/latency, and add an ablation separating:

- finite soft penalty;
- lexicographic deletion;
- expanded exact scan;
- one-swap repair;
- two-swap repair;
- post-fill audit.

After runtime validation, the next novelty step should be a validation-calibrated or learnable reliability gate between pair and local margins, with calibration metrics such as pair-sign ECE, Brier score and selective risk, followed by sufficiently powered closed-loop evaluation.

## 9. Validation status

Completed locally:

- full source compilation;
- shell syntax validation;
- 120 unit/regression tests passed;
- targeted tests for preserving scan and post-fill revert passed.

Not completed locally because the delivery environment does not contain the user's nuPlan caches and v30 checkpoint:

- actual 1000-scene v40 open-loop run;
- actual strict v40 gate outcome;
- CL20/CL100 results;
- GPU/CPU runtime profile in the target machine.
