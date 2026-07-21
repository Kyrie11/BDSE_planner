# BDSE Algorithm Update Log

> 维护规则：后续每一轮算法修改都追加一条记录；不要覆盖旧记录。每条记录至少包含：版本、动机、修改文件、关键超参数、验证结果、失败样本结论、下一步，以及“禁止重复尝试”。

## Baseline context

- Paper: `Budgeted Decision-Sufficient Evidence for Interactive Autonomous Planning`
- Dataset/protocol: nuPlan DB train/validation; 2 s history, 8 s horizon; candidate bank default K=32; evidence budget B=16; proposal Top-M=64.
- Runtime checkpoint for controlled selector experiments: `outputs_v30/train/bdse_v30_pmvrbsr.best.pt`.
- v39 runtime outputs: 1000 aligned validation samples.
- Strict experimental principle: change the runtime selector first while freezing checkpoint, candidate bank, Top-M, B, pair queries, teacher and evaluation set.

---

## 2026-07-20 — v40 Lex-DACC

### Motivation

The v39 DACC strict runtime gate failed:

| Metric | v39 DACC | Gate / control condition |
|---|---:|---:|
| Teacher action match | 0.238 | absolute pass; paired pass |
| B=16 vs full-interface | 0.206 | absolute pass; paired LCB fail |
| Deployment target action preserved | 0.919 | required >= 0.950 |
| Winner-rival sign accuracy | 0.636914 | paired pass |
| Effective query count | 5383.152 | budget pass |

Paired `B=16 vs full` difference was +0.001, but the one-sided 95% LCB was -0.003354, below the strict -0.002 non-inferiority margin.

### Root-cause diagnosis

1. **Finite soft action penalty is incompatible with the gate.**
   - v39 adds `deployment_coreset_action_weight=4.0` to a continuous distortion loss.
   - On the 81 diagnostic action-flip samples, the deployment objective starts near 4, showing that the search accepts a deployment-action change whenever score/gap/margin distortion savings exceed the finite penalty.
   - Action preservation is therefore encouraged, not guaranteed.

2. **Top-8 exact screening can miss a preserving deletion.**
   - MARS is used to screen removal candidates, and only the top 8 receive exact deployment evaluation.
   - The surrogate and final deployment operator are not identical, so a preserving deletion can fall outside top 8.

3. **Generic post-fill can invalidate the DACC result.**
   - DACC diagnostics were computed before `_complete_safety_aware_selection()`.
   - The post-fill/quota logic can exchange atoms afterward.
   - Two observed samples reported target preservation in selector diagnostics but the actual final `bdse_action` differed.

4. **The failure is not caused by neural query budget or checkpoint capacity at this stage.**
   - v39 changed only 30/1000 final actions versus MARS control.
   - Query-count and all absolute safety/recall gates passed.
   - A counterfactual using the already logged DACC Top-M deployment target gives:
     - teacher match: 0.239; paired diff +0.001; LCB -0.004934, passing the -0.005 margin;
     - B-vs-full: 0.213; paired diff +0.008; LCB +0.001856, passing the -0.002 margin.
   - Therefore the first controlled intervention is to enforce target-action preservation, not to retrain the checkpoint.

### Algorithm modification

New runtime selector: **Lexicographic Deployment-Aligned Certificate Coreset (Lex-DACC)**.

Search priority:

1. preserve the exact full-Top-M deployment action whenever a feasible candidate exists;
2. among preserving subsets, minimize exact deployment score/gap/margin distortion;
3. accept an action-changing step only when no preserving step exists in the configured exact scan;
4. attempt exact one-swap repair and guided two-swap repair if the final greedy subset still changes the target action;
5. re-evaluate the actual post-fill set with the deployment callback, and revert post-fill when it breaks an action that DACC had preserved.

The deployment callback is unchanged and includes the actual downstream decision path: final rival graph, normalized antisymmetric pair margins, soft-min tournament, hard safety filtering, utility refinement, and all-flagged risk guard.

### Runtime/query properties

- Neural evidence queries: unchanged.
- Evidence budget: B=16, unchanged.
- Proposal pool: Top-M=64, unchanged.
- Pair-query budget: unchanged.
- Extra cost: deterministic CPU/Numpy deployment evaluations over already cached pair deltas.
- Exact-evaluation memoization remains active.

### Modified / added files

- `bdse/planner/selector.py`
  - lexicographic removal/exchange objective;
  - preserving-candidate scan expansion;
  - one/two-swap repair;
  - post-fill deployment audit and conditional revert;
  - additional diagnostics.
- `bdse/planner/nuplan_planner.py`
  - v40 selector aliases and config plumbing.
- `bdse/configs/v40_bdse_lexdacc_fast_cl.yaml`
- `bdse/configs/v40_bdse_lexdacc_fallback_fast_cl.yaml`
- `bdse/configs/v40_bdse_mars_control_fast_cl.yaml`
- `bdse/tools/check_v40_lexdacc_gate.py`
- `bdse/tests/test_v40_lexdacc.py`
- `run_v40_lexdacc.sh`

### New configuration

```yaml
selector_cap_mode: lexicographic_deployment_coreset
deployment_coreset_lexicographic_action_preservation: true
deployment_coreset_preservation_scan_candidates: 0  # scan all only if top-k has no preserving deletion
deployment_coreset_repair_one_swap: true
deployment_coreset_repair_two_swap_candidates: 256
```

### Verification completed in delivery environment

- `python -m py_compile`: passed.
- `bash -n run_v40_lexdacc.sh`: passed.
- full test suite: **120 passed, 5 warnings**.
- New regression tests verify:
  1. preserving deletion outside cheap exact top-1 is discovered;
  2. a destructive post-fill action flip is detected and reverted;
  3. v40 main/control/fallback configs are distinct and strict.

### Verification still required in nuPlan environment

- Strict 1000-scene open-loop runtime gate.
- CPU selector latency distribution, especially p95/p99.
- CL20 smoke test only after runtime gate passes.
- CL100 paired scenario-transition analysis.

### Do not repeat

- Do not increase the finite action penalty and call it a hard preservation guarantee.
- Do not use only Top-8 surrogate candidates without exact preserving-candidate expansion.
- Do not trust pre-post-fill DACC diagnostics as the final executed action.
- Do not change checkpoint and selector simultaneously before isolating the runtime effect.
- Do not launch closed-loop or finetuning when the strict runtime gate fails.
- Do not judge CL20 only by aggregate score; one binary scenario changes a mean by 0.05.

### Next decision rule

- If v40 passes strict open-loop gate and selector latency is acceptable: run CL20, then CL100.
- If target preservation remains below 0.95: inspect `forced_action_flip_steps`, repair success, and per-scene exact-evaluation counts; next candidate is beam-search deletion with action-preserving state dominance, not a larger soft penalty.
- If preservation passes but paired B-vs-full still fails: the Top-M target itself is not aligned enough with dense full-interface; next modification should redefine the deployment target or improve full-interface pair/local calibration.
- If open-loop passes but closed-loop safety regresses: keep Lex-DACC fixed and tune a separately identifiable safety fallback using paired collision/TTC transition counts.

---

## Append-only template

```markdown
## YYYY-MM-DD — version/name
### Motivation
### Hypothesis
### Changes
### Files/config
### Controlled variables
### Validation
### Result
### Failure slices
### Do not repeat
### Next step
```

---

## 2026-07-20 — v41 PR-DACC (Path-Relaxed Deployment-Aligned Certificate Coreset)

### Motivation

The v40 strict 1000-scenario runtime gate still failed, but only one hard condition failed:

| Metric | v40 PR predecessor | Gate / control condition | Status |
|---|---:|---:|---|
| Teacher action match | 0.239 | paired LCB >= -0.005 | pass, LCB -0.003937 |
| B=16 vs full-interface | 0.209 | paired LCB >= -0.002 | pass, LCB -0.001200 |
| Winner-rival sign accuracy | 0.638470 | paired LCB >= -0.005 | pass, LCB -0.003047 |
| Deployment target action preserved | **0.946** | **>= 0.950** | **fail by 4/1000 scenarios** |
| Effective query count | 5383.152 | unchanged budget | pass |

The v40 main output contained 54 target-action preservation failures.

### Failure slices and diagnosis

1. **The finite-penalty problem was fixed, but the remaining search is path dependent.**
   - Failed samples averaged **6.35 forced action-flip deletion steps** versus 0.013 on preserved samples.
   - Failed samples averaged 306.61 expanded exact deletion scans and 750.41 deployment evaluations.
   - Thus v40 already exhausts every feasible one-step deletion whenever the cheap top-k has no preserving candidate; the problem is no longer a missed top-k item at one state.

2. **The single greedy subset path reaches a dead end.**
   - Lex-DACC keeps only one subset after every deletion.
   - It can choose an action-preserving deletion that is locally best but later reaches a cardinality where every possible next deletion changes the target action.
   - A different earlier branch, or a branch that temporarily changes the action and later restores it, is never retained.

3. **The local repair radius is insufficient.**
   - Every failed sample exhaustively evaluated 224 one-swap candidates.
   - Every failed sample evaluated the configured top 256 two-swap candidates.
   - Only 4/1000 total samples were repaired by v40, and none of the final 54 failures were repairable by the retained one/two-swap search.
   - Increasing the same local-swap candidate count alone is therefore not the primary algorithmic fix.

4. **A residual graph mismatch remained in the cheap branch screen.**
   - v40 exact subset evaluation uses `rival_pair_atom_delta/rival_pair_indices`, which drive the final tournament.
   - The cheap deletion/swap screen still receives `pair_atom_delta/pair_indices`, the selector graph.
   - This does not change exact decisions, but it can prevent a useful branch from entering a bounded multi-path search.

5. **The failure is localized to coreset search rather than checkpoint/query capacity.**
   - All paired non-inferiority tests passed.
   - Query counts are exactly unchanged from the MARS control.
   - 49/54 failed v40 scenes selected the same final action as the MARS control.
   - Therefore v41 keeps the v30 checkpoint, B=16, Top-M, candidate bank, teacher, pair-query outputs and validation scenarios fixed.

### Hypothesis

Target-action preservation under a cardinality/budget constraint is not a monotone property along a greedy deletion path. A bounded subset-lattice beam that retains:

- exact target-preserving states; and
- a small, action-diverse set of temporarily mismatching states

can recover feasible final B=16 subsets that a single-path lexicographic greedy plus local exchange cannot reach.

### Algorithm changes

New runtime selector alias: **PR-DACC / Path-Relaxed DACC**.

1. **Deployment-graph-aligned screening**
   - When enabled, the coreset search receives the already queried final rival pair graph.
   - Exact evaluator and cheap screen now use the same pair index/delta family.
   - Rival graph weights are set to one because the deployed tournament is unweighted over its rival sets.

2. **Failure-triggered action-diverse beam search**
   - Triggered only when lexicographic deletion and one/two-swap repair still change the Top-M deployment action.
   - Starts from the complete active Top-M decision atom set.
   - Searches exactly the same final cardinality as the greedy B=16 set.
   - Retains both preserving states and a configured fraction of mismatching states.
   - Mismatching states are diversified by their exact deployed action so the beam does not collapse to one wrong winner.
   - Branch candidates combine four cheap rankings:
     - deployment distortion screen;
     - low target-action pair impact;
     - target-harming oriented contribution;
     - low proposal prior.
   - Every retained child is evaluated by the unchanged exact downstream deployment callback.
   - A beam result replaces v40 only when it exactly restores the target action and satisfies the budget.

3. **Bounded computation and unchanged model queries**
   - Beam is disabled on the 94.6% v40-preserved cases.
   - Default bound: width 12, branch 14, at most 2400 new exact subset evaluations per triggered scene.
   - No extra neural inference or evidence query is performed.

### Configuration

```yaml
selector_cap_mode: path_relaxed_deployment_coreset
deployment_coreset_use_deployment_pair_graph: true
deployment_coreset_beam_width: 12
deployment_coreset_beam_branch: 14
deployment_coreset_beam_max_evaluations: 2400
deployment_coreset_beam_mismatch_fraction: 0.42
```

The v40 lexicographic scan, one-swap repair, guided two-swap repair and post-fill audit remain enabled.

### Modified / added files

- `bdse/planner/selector.py`
  - bounded fixed-cardinality deletion-lattice beam;
  - action-diverse beam pruning;
  - target-oriented branch features;
  - beam diagnostics and config plumbing;
  - new PR-DACC aliases.
- `bdse/planner/nuplan_planner.py`
  - optional rival-graph-aligned coreset search;
  - v41 beam config plumbing and diagnostics.
- `bdse/configs/v41_bdse_prdacc_fast_cl.yaml`
- `bdse/configs/v41_bdse_prdacc_fallback_fast_cl.yaml`
- `bdse/configs/v41_bdse_mars_control_fast_cl.yaml`
- `bdse/tools/check_v41_prdacc_gate.py`
- `bdse/tests/test_v41_prdacc.py`
- `run_v41_prdacc.sh`

### Controlled variables

Unchanged:

- checkpoint: `outputs_v30/train/bdse_v30_pmvrbsr.best.pt`;
- validation scenarios/order;
- candidate trajectories and K;
- proposal Top-M and evidence bank;
- budget B=16;
- neural pair/local predictions;
- teacher labels;
- final tournament, safety guard, utility refinement and all-flagged guard;
- MARS control configuration.

### Validation completed in the delivery environment

- all Python files compile;
- `bash -n run_v41_prdacc.sh` passes;
- full test suite: **122 passed, 5 warnings**;
- new synthetic regression verifies a case where:
  - the target action disappears at an intermediate cardinality;
  - the v40 single path ends with the wrong action;
  - PR-DACC keeps a temporary mismatch branch and restores the target at the final budget;
- v41 configs load and require the rival graph plus bounded beam.

### Runtime validation still required

No nuPlan cache or v30 checkpoint is available in the delivery environment, so no claim is made that v41 already passes the 1000-scene gate.

Inspect after running:

```text
selector_deployment_coreset_target_action_preserved
selector_deployment_coreset_beam_attempted
selector_deployment_coreset_beam_success
selector_deployment_coreset_beam_evaluations
selector_deployment_coreset_beam_depth
selector_deployment_coreset_beam_peak_width
selector_deployment_coreset_search_uses_rival_graph
```

### Do not repeat

- Do not lower the 0.95 gate or relabel the v40 result as a pass.
- Do not increase `deployment_coreset_action_weight`; v40 already uses a true lexicographic comparison.
- Do not only increase the two-swap top-k and treat it as a path-dependence solution.
- Do not enable a full beam on every scenario; it wastes runtime on already-preserved samples.
- Do not change checkpoint or finetune while testing v41 runtime search.
- Do not compare v41 against a newly changed control; retain the frozen MARS control.
- Do not run closed loop until the strict open-loop gate passes.

### Next decision rule

- If preservation reaches >=0.95 and all paired gates pass: measure selector evaluation count/latency, then run CL20 and CL100.
- If beam success recovers at least 4 scenes but another paired metric fails: inspect the exact recovered/lost action transitions before changing beam width.
- If beam is attempted on the 54-like failures but success remains near zero: the target action may be infeasible at B=16 for those scenes; next step is a target-feasibility diagnostic or deployment-target redefinition, not a larger blind beam.
- If beam reaches the 2400-evaluation cap before full depth: increase the cap only after confirming branch diversity is useful; do not increase width, branch and cap simultaneously.

---

## v42 — CBL-DACC: Counterfactual Budget-Layer Deployment-Aligned Certificate Coreset

### Date / input

- Input code: user-provided `bdse.zip` containing v41 PR-DACC.
- Runtime result: user-provided `outputs_v41.zip`, 1000 validation scenarios, frozen v30 checkpoint.
- Only the strict runtime gate was run; no closed-loop or finetune result is used in this iteration.

### Observed v41 runtime result

The v41 gate failed only on:

```text
selector_deployment_coreset_target_action_preserved = 0.947 < 0.950
```

All other strict comparisons passed:

```text
teacher_action_match                  = 0.239
paired teacher LCB                    = -0.0039372  (margin -0.005)
budget_vs_full_match                  = 0.209
paired B-vs-full LCB                  = -0.00120038 (margin -0.002)
pair_sign_acc_winner_rival            = 0.6395124
paired winner-rival sign LCB          = -0.00181182 (margin -0.005)
effective_query_count                 = 5383.152, unchanged
```

The gate is three scenes short of the required 950/1000 preservation count.

### Failure analysis

1. **The v41 beam was triggered correctly but recovered no scene.**
   - attempted: 53/1000;
   - success: 0/53;
   - remaining failures: 53;
   - mean exact beam evaluations per attempted scene: 1669.585;
   - beam depth: 14/14 on every failure;
   - terminal subsets retained: exactly 12 per failure.

2. **Most beam computation was spent at non-executable cardinalities.**
   - Top-M decision atoms: 30;
   - executable budget atoms: 16;
   - the deletion beam evaluates 14 intermediate levels;
   - after about 1670 exact evaluations it retains only 12 B=16 terminal states;
   - the final layer contains `C(30,16)=145,422,675` possible subsets.
   - Therefore zero recovery is not evidence that the target action is infeasible at B=16; it mainly shows that the terminal coverage of the deletion-tree beam is negligible.

3. **The residual failures are multi-step exchange failures.**
   - mean forced action-flip deletion steps: 6.528;
   - mean preservation-scan evaluations: 306.887;
   - one-swap repair: 0/53;
   - screened two-swap repair: 0/53;
   - a direct executable-layer search must be able to move through several temporarily mismatching B=16 subsets.

4. **The failure is still localized to search, not network query capacity.**
   - all paired quality gates and query-count gates pass;
   - v41 changes only 63/1000 actions relative to MARS;
   - only 5 of the 53 failed-preservation scenes differ from the MARS final action;
   - changing checkpoint, Top-M, B, pair head, or teacher at this point would confound a nearly isolated search failure.

5. **Failure slices are harder in winner-rival sign quality, but this does not justify training yet.**
   - mean winner-rival sign accuracy on failures: 0.5266;
   - mean on preserved scenes: 0.6459;
   - nevertheless the aggregate paired sign gate passes, so first test whether a stronger query-free combinatorial repair can recover the required three scenes.

### Hypothesis

The target action can often be recovered by a sequence of one-out/one-in exchanges at the final B=16 layer even when:

- no direct one-swap restores it;
- no screened two-swap restores it; and
- a width-12 deletion-tree beam misses the necessary terminal subset.

A fixed-budget search should rank each mismatching state by the exact counterfactual deficit of the target action against the action currently selected by the full deployment operator. This gives a useful path signal while every visited state remains directly executable.

### Algorithm changes

New runtime selector alias:

> **CBL-DACC / Counterfactual Budget-Layer DACC**

1. **Disable the v41 deletion-lattice beam in the v42 main configuration.**
   - width = 0, branch = 0, max evaluations = 0;
   - avoids spending evaluations on cardinalities above B=16.

2. **Failure-triggered fixed-budget exchange search.**
   - triggered only after v40 lexicographic deletion, one/two-swap repair, and optional v41 beam all fail;
   - starts from the executable B=16 result;
   - every edge is one-out/one-in and preserves budget/cardinality;
   - performs an exhaustive first one-swap neighborhood (`16 x 14 = 224` for the common 30-to-16 case);
   - later generations use a bounded width/branch search;
   - permits several mismatching intermediate B=16 states before accepting only an exact target-action recovery.

3. **Rival-directed exact recovery key.**
   Mismatching subsets are ranked by:
   - exact deployment action mismatch;
   - target action rank under the exact tournament scores;
   - score deficit against the current selected rival and strongest rival;
   - target-vs-selected-rival margin deficit;
   - full deployment distortion as a tie-breaker.

   This differs from v41, whose recovery potential was dominated by full-score reconstruction and was evaluated mostly before the final budget layer.

4. **Rival-directed mutation proposals.**
   Candidate swaps combine:
   - cheap deployment-graph reconstruction loss;
   - target-vs-current-rival oriented pair contribution gain;
   - global target-vs-rivals contribution gain;
   - proposal-prior gain;
   - deterministic set/action diversity.

5. **Multiple executable seeds.**
   In addition to the v41 B=16 result, bounded seeds are constructed from:
   - global target-support ranking;
   - proposal ranking;
   - support/proposal hybrid;
   - low absolute target-impact ranking.

6. **No-regression replacement rule.**
   - v42 replaces the v41 selected subset only when the exact deployment callback returns the Top-M target action;
   - a merely “closer” but still mismatching subset is never deployed;
   - therefore unsuccessful repair leaves the prior runtime action unchanged.

7. **New diagnostics.**

```text
selector_deployment_coreset_budget_layer_attempted
selector_deployment_coreset_budget_layer_success
selector_deployment_coreset_budget_layer_evaluations
selector_deployment_coreset_budget_layer_iterations
selector_deployment_coreset_budget_layer_peak_width
selector_deployment_coreset_budget_layer_unique_states
selector_deployment_coreset_budget_layer_seed_states
selector_deployment_coreset_budget_layer_best_target_rank
selector_deployment_coreset_budget_layer_best_action_deficit
selector_deployment_coreset_budget_layer_best_margin_deficit
```

### v42 configuration

```yaml
selector_cap_mode: counterfactual_budget_layer_coreset
deployment_coreset_use_deployment_pair_graph: true

deployment_coreset_beam_width: 0
deployment_coreset_beam_branch: 0
deployment_coreset_beam_max_evaluations: 0

deployment_coreset_budget_layer_width: 12
deployment_coreset_budget_layer_branch: 18
deployment_coreset_budget_layer_iterations: 8
deployment_coreset_budget_layer_max_evaluations: 2400
deployment_coreset_budget_layer_exhaustive_first: true
deployment_coreset_budget_layer_seed_count: 4
deployment_coreset_budget_layer_diversity_distance: 4
```

### Modified / added files

- `bdse/planner/selector.py`
- `bdse/planner/nuplan_planner.py`
- `bdse/configs/v42_bdse_cbldacc_fast_cl.yaml`
- `bdse/configs/v42_bdse_cbldacc_fallback_fast_cl.yaml`
- `bdse/configs/v42_bdse_mars_control_fast_cl.yaml`
- `bdse/tools/check_v42_cbldacc_gate.py`
- `bdse/tools/analyze_v42_cbldacc.py`
- `bdse/tests/test_v42_cbldacc.py`
- `run_v42_cbldacc.sh`
- `README_V42_CBLDACC.md`

### Controlled variables

Unchanged:

- v30 checkpoint;
- validation scenarios/order;
- candidate trajectories;
- Top-M evidence universe;
- B=16 query budget;
- neural pair/local outputs;
- final tournament, safety guard, utility refinement and all-flagged guard;
- MARS control;
- gate thresholds.

### Validation completed in the delivery environment

- all Python files compile;
- `bash -n run_v42_cbldacc.sh` passes;
- complete test suite: **124 passed, 5 warnings**;
- new synthetic regression requires three consecutive B-layer swaps:
  - v41-style greedy + one/two-swap repair ends at the wrong action;
  - CBL-DACC traverses two still-mismatching executable subsets;
  - the third exchange restores the target action;
- main/fallback/control configs load and are byte-distinct.

### Runtime validation still required

The delivery environment does not contain the user's nuPlan cache or v30 checkpoint. No claim is made that v42 already passes the 1000-scene gate.

Run and inspect:

```text
selector_deployment_coreset_target_action_preserved
selector_deployment_coreset_budget_layer_attempted
selector_deployment_coreset_budget_layer_success
selector_deployment_coreset_budget_layer_evaluations
selector_deployment_coreset_budget_layer_best_target_rank
selector_deployment_coreset_budget_layer_best_action_deficit
paired teacher_action_match LCB
paired budget_vs_full_match LCB
paired pair_sign_acc_winner_rival LCB
```

### Do not repeat

- Do not enlarge the v41 deletion-tree beam; it already completed all levels and recovered 0/53 while keeping only 12 terminal subsets.
- Do not interpret zero v41 beam success as proof of B=16 infeasibility.
- Do not change checkpoint, Top-M or B in the v42 runtime experiment.
- Do not accept a mismatching subset merely because its target gap is smaller.
- Do not lower the 0.95 preservation threshold.
- Do not run closed loop or finetune before the strict runtime gate passes.
- Do not increase width, branch, iterations and evaluation cap simultaneously in the next iteration; use the new diagnostics to identify which bound is active.

### Next decision rule

- If v42 recovers at least 3 v41 failures, loses zero v41-preserved scenes, and all paired gates pass: proceed to latency measurement, CL20 and CL100.
- If the search succeeds but teacher LCB fails: inspect only the recovered action transitions; do not alter the search before identifying teacher-correct-to-target-wrong changes.
- If `best_target_rank` frequently reaches 1 but exact action remains wrong: the remaining mismatch is likely caused by utility refinement/safety guard rather than raw tournament ranking; next add a guard-aware utility counterfactual diagnostic.
- If `best_target_rank > 1` and the 2400 cap is reached: increase only iterations or branch according to unique-state growth, not all bounds.
- If unique-state growth saturates far below the cap with no recovery: add a larger destroy-repair mutation (3-out/3-in), not another deletion-tree beam.

---

## v43 — SAB-DACC: Stage-Aware Budget-Layer Deployment-Aligned Certificate Coreset

### Date / input

- Input code: user-provided `bdse.zip` containing v42 CBL-DACC.
- Runtime result: user-provided `outputs_v42.zip`, 1000 validation scenarios, frozen v30 checkpoint.
- Only the strict runtime gate was run. No closed-loop or finetuning result is used in this iteration.

### Observed v42 result

v42 improved the actual selector objective enough to satisfy the strict preservation and paired-quality requirements:

```text
selector_deployment_coreset_target_action_preserved = 0.951  (gate floor 0.950)
teacher_action_match                                 = 0.239
paired teacher LCB                                   = -0.0039372  (margin -0.005)
budget_vs_full_match                                 = 0.209
paired B-vs-full LCB                                 = -0.00120038 (margin -0.002)
pair_sign_acc_winner_rival                           = 0.64158287
paired winner-rival LCB                              = -0.000345971 (margin -0.005)
effective_query_count                                = 5383.152
```

CBL-DACC was attempted on 53 scenes and exactly recovered 4, reducing target-action failures from 53 to 49.

### Why the reported gate still failed

The only reported failure was:

```text
selector_deployment_coreset_budget_layer_iterations = 0.4 >= 1
```

This was a checker aggregation error, not an algorithm failure:

- budget-layer recovery is conditionally executed only when the normal selector fails;
- it was attempted on 53/1000 scenes;
- 947 correctly preserved scenes therefore report zero executed recovery iterations;
- among the 53 attempted scenes, conditional mean iterations were 7.547 and the minimum was 1;
- averaging executed iterations over all 1000 scenes produces 0.4 and is not a valid activation check.

The threshold is not changed. v43 checks the configured iteration limit and, when recovery is attempted, verifies the conditional minimum/mean over attempted scenes.

### Residual algorithm diagnosis

The 49 remaining failures reveal a downstream-stage observability problem:

1. In 12/49 failures, `best_target_rank == 1` and raw score deficit is zero.
2. These 12 scenes have no selected-action safety flag and no all-flagged guard activation.
3. The exact deployment evaluator returns final action, post-safety scores and margins, but v42 discards the tournament diagnostics that identify the action before and after utility refinement.
4. Therefore v42 knows that the final action is wrong but ranks the subset as if the target already has zero deficit. Its fixed-budget search loses a recovery gradient precisely when utility refinement overrides the raw tournament winner.
5. This is a theorem/implementation mismatch: the deployment callback is exact for acceptance, but the search-state certificate is not sufficient for the full deployment map.

### Hypothesis

When the target is already the post-safety score winner but utility refinement selects a lower-utility rival, evidence cannot change trajectory utility. It can only recover the target by excluding the rival from the certificate-constrained utility set through one of two exact boundaries:

1. push the rival below the configured score-slack band; or
2. push the rival below the target-relative pair-certificate tolerance.

The minimum distance to these two boundaries is a meaningful recovery certificate for fixed-budget search.

### Algorithm changes

New runtime selector alias:

> **SAB-DACC / Stage-Aware Budget-Layer DACC**

1. **Preserve full deployment diagnostics in the selector callback.**
   - `deployment_evaluator` may now return `(action, scores, margins, diagnostics)`;
   - backward-compatible 3-tuples and dictionary results remain supported;
   - exact evaluation cache stores stage diagnostics without additional model queries.

2. **Stage-aware recovery state.**
   Each fixed-budget state distinguishes:
   - stage 0: final target action already preserved;
   - stage 1: target is not the post-safety score winner;
   - stage 2: target is the post-safety score winner but utility refinement changes the final action.

3. **Utility-boundary recovery certificate.**
   For stage-2 states, the search computes:
   - score-band exclusion distance for the final utility-selected rival;
   - pair-certificate exclusion distance `M[rival,target] + tolerance`;
   - the smaller exact boundary distance as the stage violation.

   This replaces the v42 zero-gradient condition where raw target rank and raw score deficit are both already optimal.

4. **Stage-aware lexicographic beam key.**
   Fixed-budget states are ranked by:
   - exact final-action preservation;
   - exact downstream-stage violation;
   - raw target rank;
   - score and margin deficits;
   - deployment distortion and deterministic diversity.

5. **Rival-directed mutations remain unchanged.**
   The current final rival continues to define oriented target support, so the new stage certificate changes the search objective rather than tuning width, branch, iterations, evaluation cap or thresholds.

6. **No-regression deployment rule remains unchanged.**
   A new subset replaces the prior v42 subset only after the unchanged complete deployment callback returns the exact Top-M target action.

7. **New diagnostics.**

```text
selector_deployment_coreset_budget_layer_iteration_limit
selector_deployment_coreset_budget_layer_best_stage
selector_deployment_coreset_budget_layer_best_stage_violation
selector_deployment_coreset_budget_layer_best_raw_action
```

### Runtime gate acceleration

The old gate path performed:

1. full `read_text().splitlines()` parsing of both JSONL files into dictionaries of complete records;
2. NumPy allocation for each paired metric;
3. a second full candidate JSONL parse in `analyze_v42_cbldacc.py`;
4. two Python process startups.

v43 changes this to:

- aligned-order streaming fast path;
- Welford online mean/variance for the exact same one-sided 95% LCB formula;
- keyed fallback retaining only requested metric scalars when files are not aligned;
- candidate conditional diagnostics and report generation integrated into the gate command;
- `python -S` for the standard-library-only checker, avoiding unrelated site-package startup.

Delivery-environment benchmark on the uploaded 1000-scene files:

| Path | Wall time | Peak RSS |
|---|---:|---:|
| Original v42 gate checker | 2.34 s | 379 MB |
| Original v42 analysis pass | 2.07 s | 334 MB |
| v43 combined gate + analysis | 0.57 s | 18 MB |

The numerical paired results are identical to the original checker.

### Configuration

The search budget/configuration is intentionally unchanged from v42:

```yaml
selector_cap_mode: stage_aware_budget_layer_coreset
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

### Modified / added files

- `bdse/planner/selector.py`
- `bdse/planner/nuplan_planner.py`
- `bdse/planner/tournament.py`
- `bdse/tools/check_v38_runtime_gate.py`
- `bdse/tools/check_v42_cbldacc_gate.py` (backward-compatible conditional-statistics fix)
- `bdse/tools/check_v43_sabdacc_gate.py`
- `bdse/configs/v43_bdse_sabdacc_fast_cl.yaml`
- `bdse/configs/v43_bdse_sabdacc_fallback_fast_cl.yaml`
- `bdse/configs/v43_bdse_mars_control_fast_cl.yaml`
- `bdse/tests/test_v43_sabdacc.py`
- `run_v43_sabdacc.sh`
- `README_V43_SABDACC.md`

### Controlled variables

Unchanged:

- v30 checkpoint;
- validation scenario set and ordering;
- candidate trajectories;
- Top-M evidence universe;
- B=16 evidence/query budget;
- pair/local network output;
- MARS control;
- safety guard, utility refinement and all-flagged guard;
- all absolute and paired gate thresholds;
- v42 fixed-budget search width, branch, iteration and evaluation limits.

### Validation completed in the delivery environment

- all Python files compile;
- `bash -n run_v43_sabdacc.sh` passes;
- complete test suite: **126 passed, 5 warnings**;
- new synthetic regression reproduces the residual v42 failure mode:
  - target is raw rank 1 in every state;
  - final action is changed only by utility refinement;
  - 3-field deployment feedback gives zero recovery gradient and fails;
  - stage-aware 4-field feedback crosses multiple executable B-layer states and restores the exact target action;
- the fast checker reproduces all original paired means and LCBs on the uploaded results;
- the patched v42 checker directly re-evaluates the uploaded v42 files as PASS using conditional attempted-scene iterations;
- applying corrected conditional semantics to the uploaded v42 result yields a gate pass, because the actual v42 quality requirements were already satisfied.

### Runtime validation still required

Run v43 on the same frozen 1000-scene setup. Do not claim additional algorithm gain until observing:

```text
selector_deployment_coreset_target_action_preserved
selector_deployment_coreset_budget_layer_success
selector_deployment_coreset_budget_layer_best_stage
selector_deployment_coreset_budget_layer_best_stage_violation
previous_failures_recovered
previous_preserved_lost
paired teacher_action_match LCB
paired budget_vs_full_match LCB
paired pair_sign_acc_winner_rival LCB
```

### Do not repeat

- Do not lower the 0.95 preservation threshold; v42 already reaches 0.951.
- Do not require an all-scenario mean conditional-iteration count to exceed one.
- Do not increase beam width, branch, iterations or evaluation cap before testing stage-aware feedback.
- Do not treat raw target rank 1 as final deployment-action preservation when utility refinement is enabled.
- Do not optimize trajectory utility through evidence atoms; utility is fixed candidate geometry. Optimize certificate eligibility boundaries instead.
- Do not parse complete JSONL records into two large dictionaries when only five paired scalars are needed.
- Do not run the separate v42 analysis script after the new combined checker.

### Next decision rule

- If v43 preserves at least the v42 0.951 rate, loses zero v42-preserved scenes, and all paired gates pass: runtime gate is complete; measure selector latency and proceed to CL20/CL100.
- If stage-2 failures are recovered but stage-1 failures remain: keep stage-aware utility logic fixed and next study multi-rival raw tournament feasibility, not utility parameters.
- If `best_stage=2` and `best_stage_violation` reaches approximately zero without exact recovery: inspect top-k eligibility and all-flagged structural-guard diagnostics, because another downstream discrete condition is missing.
- If v43 changes no additional action but all gates pass: retain v43 because it fixes checker correctness and deployment-state sufficiency; do not force a new algorithm change solely to increase version number.

---

## v44 — RADS: Regret-Aware Any-Budget Deployment Supervision

### Motivation from the v43 1000-scenario result

The v43 runtime gate passed, but it did not establish planning-quality gain:

- teacher action match remained approximately 0.239;
- full-interface teacher match remained approximately 0.265;
- winner/rival pair-sign accuracy remained approximately 0.642;
- evidence sufficiency remained approximately 0.071;
- only 4 of 53 stage-aware fixed-budget searches recovered the target action;
- the deployment coreset search increased open-loop wall time from roughly
  1.45 s/scenario for the MARS control to roughly 5.27 s/scenario.

The decisive training issue is upstream of v41-v43 search. The frozen v30
training configuration uses:

```yaml
training:
  epochs: 4
  predicted_selector_start_epoch: 6
```

Epoch indexing starts at zero. Therefore the pair-action objective uses oracle
certificate masks for every v30 epoch, while the predicted selector used at
runtime never receives action-level supervision. The v43 result is consistent
with this mismatch: proposal decisive-rival recall is high, but selected recall,
pair-sign accuracy, action match, and evidence sufficiency remain weak.

### Algorithmic change

v44 replaces per-scene combinatorial deployment repair with **RADS**:

1. **Exact deployment-path supervision.** After one oracle warm-up epoch, the
   action objective uses evidence masks generated by the exact runtime MARS
   selector. Gradients therefore train pair margins on the atoms the deployed
   planner actually selects.

2. **Any-budget action supervision.** Each batch evaluates the runtime selector
   at budgets 8, 16, and 24. Teacher-directed action loss is averaged with
   weights 0.75, 1.5, and 0.75. This yields one checkpoint that is trained for a
   budget curve rather than only B=16 and provides a principled basis for the
   paper's budget sweep.

3. **Robust regret-aware weighting.** For each deployment budget, the selected
   action's teacher regret is divided by a per-scene robust teacher-cost scale,
   transformed with `log1p`, clipped, and detached. The resulting weight
   prioritizes high-consequence deployment failures without allowing the
   heavy-tailed raw teacher cost to dominate training.

4. **Deployment schedule validation.** Training now raises a clear error when
   `predicted_selector_start_epoch >= epochs` while pair-action supervision is
   enabled. Oracle-only selector training must be explicitly marked with
   `allow_oracle_only_selector_training: true`.

5. **Fast runtime retained.** v44 uses the v43 MARS control selector at runtime.
   It does not execute DACC/Lex-DACC/PR-DACC/fixed-budget layer search. The
   deployment goal is amortized learned evidence selection, not expensive
   online subset repair.

### Configuration

```yaml
experiment:
  name: v44_rads
training:
  epochs: 12
  action_loss_start_epoch: 0
  predicted_selector_start_epoch: 1
  deployment_budgets: [8, 16, 24]
  deployment_budget_weights: [0.75, 1.5, 0.75]
  deployment_regret_weight: 1.0
  deployment_regret_clip: 3.0
  deployment_regret_min_scale: 100.0
  pair_action_loss_weight: 4.0
  loss_weights:
    action: 10.0
```

The recommended run warm-starts from the v30 best checkpoint with a lower
learning rate (`6e-6`) and validates all 1000 scenarios every epoch using dense
full-interface diagnostics.

### Modified / added files

- `bdse/model/losses.py`
- `bdse/experiments/train.py`
- `bdse/experiments/evaluate_open_loop.py`
- `bdse/configs/v44_bdse_rads_train.yaml`
- `bdse/configs/v44_bdse_rads_fast_cl.yaml`
- `bdse/configs/v44_bdse_rads_b8_fast_cl.yaml`
- `bdse/configs/v44_bdse_rads_b24_fast_cl.yaml`
- `bdse/tests/test_v44_rads.py`
- `run_v44_rads.sh`
- `README_V44_RADS.md`
- `ALGORITHM_UPDATE_LOG.md`

### Validation completed in the delivery environment

- all modified Python files compile;
- `bash -n run_v44_rads.sh` passes;
- complete test suite: **130 passed, 5 warnings**;
- schedule regression tests confirm that the historical v30-style oracle-only
  schedule is rejected unless explicitly marked as an ablation;
- no nuPlan cache or v30 checkpoint is available in the delivery environment,
  so no empirical v44 gain is claimed.

### Required runtime validation

1. Train v44 from the frozen v30 checkpoint.
2. Evaluate 1000 identical validation scenarios at B=16.
3. Also evaluate B=8 and B=24 using the same checkpoint and scenario order.
4. Compare against frozen v30/MARS and report:
   - teacher action match;
   - median, p90, p95, and CVaR teacher regret;
   - winner/rival and near-tie pair-sign accuracy;
   - proposal and selected decisive-rival recall;
   - evidence sufficiency;
   - wall-clock latency p50/p90/p95 and GPU memory.
5. Run CL20 only if open-loop teacher action match improves by at least two
   absolute percentage points and tail regret does not regress.

### Do not repeat

- Do not spend more runtime on v41-v43 beam/layer search before retraining the
  deployed selector.
- Do not select a checkpoint only by aggregate validation loss; retain
  metric-specific best checkpoints for teacher action match, teacher regret,
  and full-interface action match.
- Do not call the current generic external adapters faithful GameFormer, DTPP,
  PlanTF, PLUTO, or PDM-Closed implementations.
- Do not use mean teacher regret alone; its distribution is strongly
  heavy-tailed.
