# v49 DBAP — Deployment-Boundary Action Preservation

## 日期

2026-07-29

## 基于 v48 实验的结论

- v48 gate FAIL，closed-loop 被正确阻止；不是 CL20 调用缺失。
- certificate/fallback 层有效：certified-pair 0.831，fallback 0.162。
- pair interface 仍是主错误：dense near-tie 约 0.683，pair near-tie 0.460；pair-full match 0.272。
- pair-full→budget harmful compression 仅 0.024，说明 compression 已不是主矛盾。
- interaction budget fraction 0.948，跨 family 竞争失败。
- 平均只选 7.403/16 atoms，v48 实际验证的是 anytime early certificate，不是 strict fixed-budget action preservation。
- validation 未计算真实 pair-full/family collapse，best-checkpoint score 使用弱代理。

## 算法修改

### 1. Pure LOO deployment-boundary criticality

文件：`bdse/model/losses.py`

- `positive_support_floor` 默认改为 0；
- 新增 `target_top_k_atoms`；
- 新增 `min_relative_gain`；
- v49 配置关闭旧 `critical_pair` / `critical_proposal` 与 interaction-only critical target；
- 降低 general proposal 与 interaction boost，增强纯 CF pair/proposal listwise objective。

目的：避免所有微小正 support atom 获得 target mass，使 pair head 只拟合真正会增加 boundary deficit 的单位成本 evidence。

### 2. Bounded uncertainty-gated sparse residual

文件：`bdse/model/bdse_model.py`

新增 `_confidence_shrunk_residual_pair_delta_np`。部署 margin 为：

`local + trust * residual`，其中 `trust <= 0.35`，并由以下项共同缩小：

- residual variance；
- local margin 是否接近边界；
- residual 与 local 的原始符号冲突；
- residual/local 幅值异常。

修复 v48 先形成 `local+residual` 后再检测冲突、导致约 80% residual 仍保留的问题。

### 3. Exact-budget nested AOCC

文件：`bdse/planner/selector.py`

新增配置：

- `adverse_certificate_fill_to_budget_after_certified`；
- `adverse_certificate_max_interaction_prefix_fraction`。

行为：

- 首次认证后记录 first-certified prefix；
- 继续生成完整 nested order；
- materialize 严格 fixed-B prefix；
- 即使剩余 atom 的 certificate gain 为 0，也使用确定性成本顺序补齐；
- 每个 prefix 限制 interaction family 比例。

新增诊断：

- `aocc_first_certified_prefix_length`；
- `aocc_first_certified_cost`；
- `aocc_fill_to_budget_after_certified`；
- `aocc_max_interaction_prefix_fraction`。

### 4. Exact validation and checkpoint selection

文件：`bdse/experiments/train.py`

Validation 新增：

- exact pair-full tournament action；
- budget-vs-pair-full；
- dense→pair interface flip/harm；
- pair→budget compression flip/harm；
- selector diagnostics/family composition；
- planner latency；
- fixed-budget fill。

`_validation_fixed_budget_critical_score` 不再在缺失 pair-full 时回退到 sparse-full；缺失 pair/family diagnostics 会被显式惩罚。

### 5. Gate aligned with fixed-budget claim

文件：`bdse/tools/check_v48_dbce_gate.py`

新增：

- `--min-budget-fill-fraction`，默认 0.95；
- 输出 fixed-budget fill。

### 6. Cache provenance

文件：

- `bdse/experiments/preprocess.py`
- `bdse/data/nuplan_dataset.py`

新增：

- `--resume-require-config-match`；
- split 级 `cache_provenance.json`；
- config mismatch 或 legacy cache 无 provenance 时拒绝 strict resume。

### 7. Config and execution

新增：

- `bdse/configs/v49_bdse_dbap_train_2gpu.yaml`；
- `bdse/configs/v49_bdse_dbap_cl.yaml`；
- `run_v49_dbap.sh`；
- `V49_DBAP_NEXT_COMMANDS.sh`；
- `BUILD_MATCHED_TEST_SET.sh`。

两张 A30：训练 DDP；open-loop 与 closed-loop 按场景/token 分片并行。Control freshness 现在绑定 calibration provenance 与 val_tune manifest。

## Test 构建参数修复

相对上传的“快速生成”命令：

- 使用全新 `bdse_test_v49_matched` 输出目录；
- `--include-drivable-polygons`，与 train/val 诊断一致；
- 移除 `--max-samples-per-log 512`，与 train/val 诊断中的 `None` 对齐；
- 保留 stride=10、initial、teacher stride=1、candidate-aware、crosswalk=false；
- 启用 resume validation 与 config provenance。

## 验证

- `python -m compileall -q bdse`：PASS；
- 两个 v49 YAML 加载：PASS；
- 三个 shell 脚本 `bash -n`：PASS；
- `pytest -q`：161 passed，5 warnings。

## 重要限制

代码修复只保证目标、部署接口、验证指标和 fixed-budget claim 更一致。是否改善 teacher match、regret、latency 与 closed-loop，必须由 v49 实验确认；不得预先声称 PASS。

---

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

---

## v45 — PB-RADS: Primary-Budget Exact Deployment Supervision and Mask Distillation

### Empirical diagnosis of the uploaded v44 run

The uploaded `output_v44_rads_fast_2gpu_v2` run does **not** pass the v44
open-loop gate and should not be used as evidence that RADS training improved
the model.

Observed B=16 / 1000-scenario metrics:

| Metric | v44 | frozen v30/MARS control or previous reference | Interpretation |
|---|---:|---:|---|
| teacher action match | 0.246 | 0.238 | +0.008, below the required +0.020 |
| full-interface teacher match | 0.269 | about 0.265 | dense model remains a low ceiling |
| budget-vs-full match | 0.221 | 0.205 | improved, but still low |
| winner/rival pair-sign accuracy | 0.6564 | about 0.6384 | useful improvement |
| evidence sufficiency | 0.0820 | about 0.071 | small absolute gain; median is zero |
| proposal decisive-atom recall | 0.8320 | high | proposal is not the main bottleneck |
| selected decisive-atom recall | 0.3989 | low | fixed-budget allocation remains the bottleneck |
| selected interaction decisive recall | 0.3444 | low | critical interaction evidence is often dropped |
| MARS target-action preservation | 0.944 | 0.951 in prior fixed-budget runs | selector preservation regressed |
| p95 planner latency | 1096.7 ms | deployment target not met | not real-time at the configured replan rate |

The v44 per-sample regret distribution is heavy-tailed: median approximately
26.1, p90 approximately 30.8k, p95 approximately 45.7k, and CVaR90
approximately 52.6k. The frozen control JSONL was not included in the uploaded
archive, so the required median/p90 non-regression condition cannot be verified.

### P0 training-validity failure

Every row of `bdse_v44_rads.train_log.jsonl` has:

```text
loss = NaN
L_cert_frontier = NaN
```

The source is `_certificate_action_gap_loss`. When a scene has hard candidates
but no safe candidate, `best_safe` is a large negative masking sentinel. The
code evaluates `softplus(best_hard - best_safe)` before applying
`frontier_mask=0`; the result is `inf * 0 = NaN`. AMP/GradScaler can then skip
optimizer steps without making the run fail. Therefore the small v44 metric
changes may come from runtime configuration, checkpoint selection noise, or a
partially/fully skipped finetune rather than a clean algorithmic update.

### P0 deployment-supervision mismatch

The v44 README states that epochs 1--11 use exact runtime masks and that each
batch optimizes B=8/16/24. The fast configuration actually uses:

```yaml
deployment_selector_scenes_per_rank: 2  # local batch is 4
deployment_selector_every_n_steps: 2
deployment_budget_strategy: weighted_round_robin
```

Thus exact predicted-selector masks cover about `2/4 * 1/2 = 25%` of local
scene-steps. Weighted round-robin evaluates one budget per rank-step, with B=16
on about half the slots. The primary B=16 exact deployment path therefore
receives only about 12.5% scene-step coverage before the short final exact tail.
The measured epoch metric `selector_exact_fraction=0.25` confirms this mismatch.

### Algorithmic diagnosis

1. **The dense interface is the upstream ceiling.** Full-interface teacher match
   is only 0.269. A perfect budget selector cannot produce theoretical SOTA if
   the complete learned evidence interface still chooses the teacher action in
   only about one quarter of scenes.
2. **Proposal recall is high but selected recall is low.** The 0.832 proposal
   recall versus 0.399 selected recall localizes the current fixed-budget
   failure to within-Top-M allocation and signed-margin preservation, not to
   candidate atom discovery.
3. **Action agreement is not evidence sufficiency.** Budget-vs-full match is
   0.221 while evidence sufficiency is 0.082 with median zero. The selector can
   occasionally reproduce an action without reconstructing or certifying the
   decisive margin. Paper claims must prioritize one-sided decisive-margin
   coverage rather than action agreement alone.
4. **The paper and runtime selector are not the same algorithm.** The paper
   describes uncertainty-aware capped LCB greedy selection with a submodular
   interpretation, while the deployed v44 path is a signed MARS margin coreset
   with swap passes and uncertainty disabled. The theoretical objective,
   implementation, and ablations must be made identical before submission.
5. **Structural safety and decision budget are conflated in reporting.** Hard
   safety is budget-exempt in the implementation. Report structural mandatory
   safety coverage separately from budgeted critical-evidence recall; do not
   call raw selected hard recall a failure when the atom is intentionally in the
   structural channel.
6. **Checkpoint selection was noisy.** Validation used 256 scenes and selected
   by teacher action match, reaching 0.3906 on the small validation subset but
   only 0.246 on the fixed 1000-scene evaluation. This is selection variance,
   not a stable generalization gain.

### v45 changes implemented

1. **Finite certificate losses.** Safety and safe-frontier masks are applied
   before `softplus`, eliminating sentinel overflow and `inf * 0`.
2. **Synchronized fail-fast.** Every scalar loss is checked for finiteness before
   backward. A DDP all-reduce makes every rank abort together on any invalid
   objective.
3. **Primary-plus-aux budget schedule.** B=16 is optimized on every deployment
   step. One auxiliary budget (B=8 or B=24) rotates with the correct aggregate
   any-budget weighting.
4. **Full exact deployment alignment.** The recommended configuration uses all
   local scenes every step after epoch 0 and enforces
   `min_deployment_exact_fraction=1.0` at startup.
5. **Deployment-mask distillation.** Exact stop-gradient MARS masks from the
   active budgets produce a soft selected-frequency target for proposal logits.
   This is the missing direct signal from discrete fixed-budget selection back
   to the learned proposal head; deployment behavior itself is unchanged.
6. **Core-claim checkpointing.** Fixed-budget critical score now includes dense
   full-interface match and effective decisive recall, and treats budget-exempt
   hard safety through effective hard recall. The run uses 1000-scene dense
   validation and saves metric-specific checkpoints.
7. **Strict gate tool.** `check_v45_pb_rads_gate.py` verifies finite training,
   exact selector coverage, +2 point teacher-match gain, median/p90 regret
   non-regression, winner/rival sign gain, material evidence-sufficiency gain,
   and a user-specified p95 latency target.
8. **Optional base normalization support.** Code supports per-scene normalized
   base-cost regression, but it is disabled in the first v45 run to keep the
   validity-controlled comparison isolated. Enable it only as a separate
   ablation after v45.

### Modified / added files

- `bdse/model/losses.py`
- `bdse/experiments/train.py`
- `bdse/configs/v45_bdse_pb_rads_train_exact_2gpu.yaml`
- `bdse/configs/v45_bdse_pb_rads_fast_cl.yaml`
- `bdse/configs/v45_bdse_pb_rads_b8_fast_cl.yaml`
- `bdse/configs/v45_bdse_pb_rads_b24_fast_cl.yaml`
- `bdse/tools/check_v45_pb_rads_gate.py`
- `bdse/tests/test_v45_pb_rads.py`
- `run_v45_pb_rads.sh`
- `README_V45_PB_RADS.md`
- `ALGORITHM_UPDATE_LOG.md`

### Validation completed in the delivery environment

- all Python files compile;
- `bash -n run_v45_pb_rads.sh` passes;
- complete test suite: **142 passed, 5 warnings**;
- a regression test reproduces the all-hard/no-safe v44 NaN case and confirms
  finite certificate losses;
- schedule tests reject the v44 25% exact-supervision configuration when the
  v45 100% floor is requested;
- primary-plus-aux tests confirm B=16 is present on every rank-step and B=8/B=24
  are balanced across the deterministic schedule.

### Required next experiment

Warm-start from the frozen v30 checkpoint, not the v44 checkpoint. The v44
checkpoint cannot be trusted because the optimization objective was non-finite.
Use the exact command in `README_V45_PB_RADS.md`, then run the strict gate against
the same frozen v30/MARS 1000-scenario JSON/JSONL.

### Next decision rule

- **If v45 passes:** run paired CL20 on exactly the same scenario tokens as the
  control. Proceed to CL100 only if safety is non-inferior and the primary
  closed-loop score improves.
- **If full-interface match rises but budgeted match/sufficiency does not:** the
  next algorithm should replace independent signed coreset selection with one
  nested, anytime one-sided certificate ordering; do not tune quotas or add
  another online repair search.
- **If full-interface match remains below 0.30:** stop changing the selector.
  Improve pair-margin representation with hard-rival mining, antisymmetric
  cycle consistency, and teacher-cost calibration first.
- **If latency remains above 500 ms p95:** profile proposal construction,
  pair-score materialization, CPU selector transfer, and utility refinement.
  Do not claim real-time planning from two-GPU shard wall time.

### CCF-A novelty path after a valid v45 result

The strongest theory-aligned extension is an **anytime one-sided adverse-bound
certificate coreset**. For each predicted winner--rival pair, assign every
unqueried atom a calibrated adverse contribution bound. Querying an atom removes
that worst-case penalty and replaces it with its observed lower-confidence
margin contribution. Capped reduction of certificate deficit is monotone
submodular under a knapsack budget, yielding a standard greedy approximation
bound and a single nested ordering valid for every budget. This directly serves
the paper's central idea and avoids another ad-hoc quota/search variant. It
should be implemented only after v45 establishes that the learned dense margin
interface is accurate enough to support such a guarantee.

### Do not repeat

- Do not continue from the v44 checkpoint.
- Do not interpret v44's +0.8 point match change as a trained-model gain.
- Do not use 256-scene validation to select the final checkpoint.
- Do not spend more compute on beam/layer/DACC repair or quota sweeps.
- Do not increase B or Top-M to hide low evidence sufficiency.
- Do not report two-GPU shard throughput as per-replan latency.
- Do not claim a submodular/uncertainty selector theorem while deploying MARS
  with uncertainty disabled.

---

## v45-fast — GPU Margin-Damage Training Surrogate with Sampled Exact Distillation

### Trigger

The full-exact v45 training command required almost one day for five epochs on
2xA30 with a Xeon Gold 5220R. The DDP topology was correct; profiling the code
showed that each rank synchronously executed the CPU/NumPy runtime selector for
all four local scenes and two budgets on every optimizer step.

### Root cause

- `_predicted_pair_certificate_masks()` calls `.detach().cpu().numpy()` on the
  pair-margin field and proposal state, forcing a GPU synchronization.
- The signed MARS coreset then performs backward elimination in Python/NumPy.
- `primary_plus_aux` repeats the full conversion, Top-M construction, and search
  for B=16 plus B=8/B=24.
- At global batch eight and 50k scenarios there are approximately 6250 steps per
  epoch. A synthetic v45-sized batch measured about 0.57 s for one exact budget
  on four scenes; two budgets therefore consume more than one second per rank
  per step before forward/backward.
- DDP amplifies this CPU imbalance because gradient synchronization cannot begin
  until both ranks finish their local selector.
- 1000-scene validation every two epochs with dense diagnostics performs an
  additional dense inference and adds avoidable wall time.

### Changes

1. Added `training.deployment_selector_backend=hybrid_fast`.
2. Added a GPU margin-damage surrogate that computes all configured budgets in
   one nested ranking. Its atom score approximates exact MARS removal damage
   using signed-margin residual, sign mismatch, winner protection, and predicted
   action preservation.
3. Exact runtime masks remain the target for `L_deploy_select`, sampled using
   `deployment_exact_distill_scenes_per_rank` and
   `deployment_exact_distill_every_n_steps`.
4. Added surrogate/exact Jaccard agreement and separate fast/exact selector wall
   times to the train log.
5. Added per-stage epoch timing for data wait, host-to-device transfer, forward,
   loss construction, and backward/optimizer step.
6. Added a fast training configuration and launcher with reduced intermediate
   validation and checkpoint I/O. Final 1000-scene open-loop evaluation remains
   unchanged.
7. Added a synthetic selector benchmark and regression tests for nested masks,
   budget compliance, and fast-backend schedule validation.

### Validation

- Complete suite: **144 passed, 5 warnings**.
- Synthetic CPU benchmark at B=4, K=32, E=128, P=192:
  - exact full local batch, one budget: about 0.57 s;
  - exact one-scene distillation: about 0.137 s;
  - fast all-scene B=8+B=16 masks: about 0.006 s.
- With one exact scene every four steps, the selector-only path is reduced by
  roughly an order of magnitude or more; end-to-end speedup depends on model,
  cache storage, and validation overhead.

### Scientific constraint

Runtime/open-loop/closed-loop evaluation still uses the exact deployed MARS
selector. The GPU surrogate is only a structured training approximation. Final
results must include a smaller exact-all-scenes ablation and report the exact
mask sampling rate plus surrogate/exact agreement.

### Do not repeat

- Do not set all-scene exact CPU selection for every budget on every training
  step merely to claim deployment alignment.
- Do not interpret low GPU utilization as a DDP failure before checking
  `selector_exact_wall_time_s` and `train_loss_ms_per_step`.
- Do not enable dense 1000-scene validation every two epochs during exploratory
  training; retain it for final evaluation and selected checkpoints.

---

## v46 result audit and v47 D3CE — Deployment-Consistent Calibrated Critical Evidence

### Trigger

The exact two-GPU v46 AOCC run completed with finite losses and
`selector_exact_fraction=1.0`, but the open-loop gate still failed. Uploaded
1000-scenario results were audited together with full validation diagnostics,
a partial test diagnostic build, and 1000-scenario per-city training diagnostics.

### v46 measured result

- teacher action match: **0.226**;
- evidence sufficiency: **0.080087**;
- dense full-interface action match: **0.298**;
- pair-full exact-tournament action match: **0.237**;
- budget vs pair-full match: **0.823**;
- dense winner/rival sign accuracy: **0.8076**;
- pair-head winner/rival sign accuracy: **0.5738**;
- dense near-tie sign accuracy: **0.6481**;
- pair-head near-tie sign accuracy: **0.3663**;
- proposal decisive recall: **0.8320**;
- selected decisive recall: **0.5287**;
- AOCC certified-pair fraction: **0.0867**;
- p95 single-scenario planner latency: **867.09 ms**;
- mean prediction stage: **567.26 ms**;
- mean selector stage: **3.04 ms**;
- mean tournament stage: **7.27 ms**.

### Status of the three v45 failure causes

1. **Exact runtime-selector alignment: solved operationally.** Every post-warmup
   epoch reports exact fraction 1.0. This removes the v45 6.25% coverage defect.
2. **Upstream pair/dense interface: not solved.** Dense full match changed only
   from 0.296 to 0.298, while the deployment pair-full path is lower at 0.237.
   The independent pair head is especially weak on near ties.
3. **Budget compression: only partially solved relative to the wrong target.**
   AOCC preserves pair-full actions fairly often, but the pair-full target itself
   is frequently wrong. Dense-to-pair-full loses 154 correct scenes and rescues
   93; pair-full-to-budget loses 18 and rescues 7; dense-to-budget loses 164 and
   rescues 92. End-to-end harmful compression therefore did not improve.

### Additional root causes

- The AOCC target was selected by an internal pairwise preference proxy rather
  than the exact final tournament including utility and safety guards.
- The raw base-cost loss remained around 2,400 with a 0.35 weight, dominating
  normalized action/critical-evidence losses. `normalize_base_loss` was false.
- `action_conditioned_action_loss_weight` was zero, despite the dense/local head
  being much more accurate than the independent pair head.
- The AOCC mean target confidence was only about 0.05 and only 8.7% of target
  pairs were certified at B=16. Fallback was disabled, so uncertified actions
  were used as ordinary decisions.
- Prediction accounts for roughly 90% of end-to-end latency; further selector
  micro-optimization alone cannot meet a 500 ms p95 target.
- v46 reported `aocc_bound_calibrated=1` even though no independent calibration
  run or provenance existed. The flag was implementation-derived, not evidence
  of a valid held-out calibration protocol.

### Dataset audit

Validation oracle capacity is high: full-interface match **0.9657**, B=16 oracle
sufficiency **0.9120**, runtime teacher match **0.7490**. Therefore the main val
bottleneck is learned representation/decision alignment, not evidence-bank
capacity.

The partial test cache is not preprocessing-compatible with validation:

- drivable polygon count: **0.0** test vs **31.74** val;
- safe candidate exists: **0.4937** vs **0.7173**;
- B=16 oracle sufficiency: **0.8065** vs **0.9120**;
- route-distance p95-p90: **20.66 m** vs **3.26 m**.

No final test claim is allowed until the complete test cache passes the new
split-parity gate.

### v47 algorithm changes

1. **Exact-tournament AOCC target.** AOCC receives the full Top-M action produced
   by the exact deployment tournament, rival graph, safety filter, utility
   refinement and all-flagged guard. The proxy Copeland target is no longer used
   when the planner can provide an exact target.
2. **Teacher-targeted exact selector training.** Stop-gradient AOCC masks use the
   teacher action during training, so the selector is taught evidence that
   protects the desired action rather than its current mistaken winner.
3. **Integrable local margin plus sparse residual.** The action-conditioned head
   defines the margin for every pair. The antisymmetric pair head learns only a
   residual and is evaluated at runtime on at most 48 boundary/safety/winner
   pairs; unrefined pairs remain valid local margins.
4. **Counterfactual critical-evidence loss.** Current teacher-vs-hard-rival pairs
   define a cost-normalized distribution over atoms with positive teacher-margin
   support. Pair residual and proposal logits receive listwise supervision on
   that distribution.
5. **Loss-scale correction.** Enable normalized base regression, nonzero
   action-conditioned action loss and stronger normalized full-action,
   hard-rival, deployment-selection and critical-evidence objectives.
6. **Honest calibration provenance.** The calibrated flag requires a
   group-disjoint calibration-only provenance JSON. A hand-set epsilon and
   learned variance no longer imply calibration.
7. **Group-disjoint validation protocol.** Added deterministic log-level
   `val_tune`/`val_calib` manifests. Checkpoints and hyperparameters use
   `val_tune`; one-sided calibration uses `val_calib`; test remains untouched.
8. **Uncertified fallback.** Closed-loop config can trigger a same-budget rival
   expansion/fallback when the AOCC certified-pair fraction is below the
   configured floor.
9. **Error-decomposition metrics.** Evaluator now reports pair-interface flips,
   harmful/beneficial pair compression, harmful/beneficial pair-interface
   changes, fully certified scene rate, and conditional teacher match.
10. **Dataset parity gate.** Added a diagnostic checker for map-feature,
    safe-candidate, oracle-capacity and route-tail drift before final test use.

### Motivation and definition decisions

- The seven existing macro-action labels are retained. Validation oracle capacity
  shows that replacing the maneuver taxonomy is not the current limiting factor.
  A new taxonomy now would confound the fixed-budget evidence contribution.
- Existing evidence families are retained because proposal decisive recall is
  already high and B=16 oracle sufficiency exceeds 0.91 on validation. The model
  must learn *which* atoms change teacher-vs-rival decisions, not merely emit more
  atom types.
- Candidate generation remains a separate error component. Test safe-candidate
  failure and missing map polygons must be fixed in preprocessing/candidate
  coverage before claiming model or selector gains.

### Added / modified files

- `bdse/configs/v47_bdse_d3ce_train_2gpu.yaml`
- `bdse/configs/v47_bdse_d3ce_cl.yaml`
- `bdse/model/bdse_model.py`
- `bdse/model/losses.py`
- `bdse/planner/selector.py`
- `bdse/planner/nuplan_planner.py`
- `bdse/experiments/evaluate_open_loop.py`
- `bdse/tools/build_group_disjoint_calibration_split.py`
- `bdse/tools/calibrate_v47_adverse_bounds.py`
- `bdse/tools/apply_v47_calibration.py`
- `bdse/tools/check_v47_d3ce_gate.py`
- `bdse/tools/check_dataset_diagnostics_parity.py`
- `bdse/tests/test_v47_d3ce.py`
- `run_v47_d3ce.sh`
- `NEXT_COMMANDS_V47_D3CE.sh`
- `README_V47_D3CE.md`

### Validation completed

- complete unit suite: **154 passed, 5 warnings**;
- Python compile check: pass;
- `bash -n run_v47_d3ce.sh`: pass;
- `bash -n NEXT_COMMANDS_V47_D3CE.sh`: pass;
- uploaded partial test diagnostics correctly fail the new parity gate.

### Required experiment order

1. Build group-disjoint `val_tune` and `val_calib` manifests.
2. Retrain from the frozen v30 checkpoint using only `val_tune` for selection.
3. Calibrate adverse residuals only on `val_calib` with provenance.
4. Replay v47 and frozen control on identical `val_tune` rows.
5. Run strict token-paired gate; proceed to CL20 only after PASS.
6. Complete and repair test preprocessing, pass parity gate, then evaluate test
   once as final evidence.

### CCF-A claim boundary

The defensible core contribution is a deployment-identical, calibrated, nested
fixed-budget critical-evidence selector with a three-term error decomposition:
candidate coverage, full-interface decision error, and budget-compression error.
The one-sided certificate preserves the learned full-interface action under its
calibration assumptions; it does not make a wrong full-interface action correct.
The paper must report certificate coverage, conditional action preservation,
budget curves, scenario/city slices, latency stages, and candidate oracle bounds.

---

## v48 DBCE — Deployment-Boundary Critical Evidence (2026-07-27)

### Why v47 did not enter closed loop

The v47 pipeline stopped intentionally at the strict paired open-loop gate.  The
closed-loop directory is empty because the gate command returned nonzero under
`set -e`; this was not a nuPlan simulator startup failure.  The uploaded run had:

- teacher action match `0.249` versus control `0.238`, a `+0.011` gain below the
  required `+0.020`;
- evidence sufficiency `0.073575` versus `0.070400`, a `+0.003175` gain below
  the required `+0.010`;
- pair-full deployment match `0.248 < 0.300`;
- certified target-pair fraction `0.135 < 0.500` and fully certified scene rate
  `0.028`;
- planner p95 latency `1365.15 ms > 500 ms`;
- paired regret regression at both median and p90.

A second experiment-integrity issue was discovered: two launcher logs began
about two minutes apart and the training JSONL contains two rows for every epoch
(24 rows for 12 unique epochs).  These independent jobs wrote to the same output
root.  V47 remains useful for diagnosis, but it is not a clean paper-grade run.

### What v47 learned and what it did not

**Effective components**

1. Exact-selector supervision reached `1.0` after epoch zero.
2. Full-interface match improved from `0.265` to `0.319`.
3. Winner/rival pair sign accuracy improved from `0.638` to `0.692`.
4. Actual sparse query count fell from about `11252` to `2285` per scenario.
5. Budget-to-pair-full preservation reached `0.895`; pair-to-budget harmful
   compression was only `0.018`.

**Ineffective or misleading components**

1. Near-tie pair sign accuracy fell from `0.583` to `0.469`; the residual head
   learned broad corrections but damaged the decision boundary.
2. `L_cf_critical_pair` remained essentially flat (`3.612 -> 3.610`).  The v47
   target rewarded all positive teacher support, not the counterfactual effect
   of removing an atom from the teacher/rival boundary.
3. Proposal loss decreased, but `proposal_top_m=64` covered essentially the
   entire average decision evidence pool (`~29.97` atoms); proposal ranking had
   no deployment consequence.
4. `min_soft_interaction_topm_slots=24` caused about `15.30 / 15.87 = 96.4%` of
   the decision budget to be filled by interaction-family evidence.  High recall
   therefore did not imply causal decision support.
5. AOCC protected the pair-full action well, but the pair-full action itself was
   correct only `0.248` of the time.  Dense-to-pair interface error, not final
   budget compression, remained the dominant action-loss source.
6. The per-atom prior radius `0.10` was roughly five times the observed raw
   atom-pair MAE and accumulated across omitted atoms.  B=16 therefore certified
   only a small fraction even though the full-order target was usually
   certifiable.

### v48 core algorithm changes

1. **Leave-one-out deployment-boundary criticality.**  For teacher action `w`,
   rival `r`, full teacher margin `m_wr`, and atom support `d_i`, the target is

   `relu(gamma - (m_wr - d_i)) - relu(gamma - m_wr)`.

   This is nonzero only when removing the atom increases the boundary deficit.
   It is divided by query cost and used as the shared listwise target for the
   pair residual and proposal heads.
2. **Teacher-nearest + model-confused rival mining.**  Critical evidence is
   learned against the union of teacher-nearest rivals and current model errors,
   rather than predicted-hard rivals alone.
3. **Confidence-shrunk integrable pair interface.**  Runtime margins use the
   integrable local action-conditioned margin plus a sparse antisymmetric
   residual.  The combined margin is shrunk back toward the local interface when
   residual variance is high or signs disagree.  V47 skipped this calibration in
   residual mode.
4. **Tournament-active AOCC frontier.**  AOCC protects near-boundary target
   rivals and safety-crossing rivals instead of requiring every target-incident
   pair.  It reports original frontier size and retained pair-weight fraction.
5. **Real proposal competition.**  `proposal_top_m` is reduced `64 -> 24`, while
   reserved interaction slots are reduced `24 -> 8`.  Interaction evidence must
   now compete with route, progress, regularity, and other decision families.
6. **Less over-conservative bounds.**  Calibration prior radius is reduced
   `0.10 -> 0.02`, protected target rivals `16 -> 6`, and calibration remains
   group-disjoint and provenance-gated.
7. **Boundary-aligned checkpointing.**  Best-checkpoint selection now emphasizes
   teacher match, exact/sparse pair-full match, near-tie sign, budget-to-pair
   preservation, sufficiency, regret, and safety.  Raw interaction recall has
   only a small diagnostic weight.
8. **Latency-oriented caps.**  Pair residual refinement is reduced `48 -> 32`,
   selector pairs `128 -> 96`, runtime pair queries `320 -> 192`, tournament
   rivals `16 -> 12`, and utility refinement `12 -> 8`.
9. **Experiment single-writer enforcement.**  Both the run script and full
   pipeline create PID-bearing lock directories.  A second writer to the same
   `OUT_ROOT` fails immediately.
10. **Stricter gate.**  The gate now rejects duplicate epochs, near-tie
    regression, interaction-budget saturation, excessive fallback, inadequate
    AOCC frontier weight, weak certificate coverage, pair-full error, regret
    regression, and latency failure.
11. **Automatic gated CL20.**  The prior command file contained only a commented
    closed-loop example.  V48 starts CL20 automatically after PASS when
    `RUN_CLOSED_LOOP_AFTER_GATE=1` and `NUPLAN_ROOT` is set.

### Configuration changes

- new train config: `bdse/configs/v48_bdse_dbce_train_2gpu.yaml`;
- new closed-loop config: `bdse/configs/v48_bdse_dbce_cl.yaml`;
- training epochs: `12 -> 16`;
- counterfactual rivals: `3 -> 4`;
- counterfactual pair/proposal weights: `4/3 -> 8/4`;
- old non-causal critical pair/proposal weights: `8/2 -> 2/0.5`;
- online hard-rival weight: `3 -> 5`;
- full-action weight: `4 -> 6`;
- pair calibration enabled for the local+residual interface.

### Files added or modified

- `bdse/model/losses.py`
- `bdse/model/bdse_model.py`
- `bdse/planner/selector.py`
- `bdse/experiments/train.py`
- `bdse/configs/v48_bdse_dbce_train_2gpu.yaml`
- `bdse/configs/v48_bdse_dbce_cl.yaml`
- `bdse/tools/calibrate_v48_adverse_bounds.py`
- `bdse/tools/apply_v48_calibration.py`
- `bdse/tools/check_v48_dbce_gate.py`
- `bdse/tests/test_v47_d3ce.py`
- `bdse/tests/test_v46_aocc.py`
- `run_v48_dbce.sh`
- `V48_DBCE_NEXT_COMMANDS.sh`
- `README_V48_DBCE.md`

### Validation completed

- Python compile check: pass;
- `bash -n run_v48_dbce.sh`: pass;
- `bash -n V48_DBCE_NEXT_COMMANDS.sh`: pass;
- complete unit suite: **158 passed, 5 warnings**;
- v48 strict gate replayed on v47 and correctly detected every original failure,
  the near-tie regression, interaction saturation, and all duplicate epochs.

### Claim boundary

V48 is a code-level and objective-level correction, not an assertion that the
new training run will necessarily pass.  The strongest defensible contribution
is the combination of deployment-boundary leave-one-out criticality, an
integrable local margin with calibrated sparse residuals, and a nested
fixed-budget deployment certificate.  Statistical claims remain conditional on
log-disjoint calibration and exchangeability; teacher-action correctness also
depends on candidate coverage and full-interface representation quality.

---

# v50 DBAP-RI — Deployment-aligned Residual Intervention

## Result trigger

v49 failed the strict open-loop gate. The decisive failure moved to the pair interface:

- teacher action match 0.232 vs control 0.286;
- near-tie sign 0.462 vs control 0.593;
- pair-full action match 0.231;
- dense-to-pair-full flip rate 0.692;
- harmful dense-to-pair intervention 0.176 vs beneficial 0.047;
- pair-full-to-budget harmful compression only 0.009;
- p95 latency 854.84 ms.

AOCC certification, fallback control, nested budget fill, and final compression remained effective. v50 therefore preserves those components and fixes the local-to-pair residual interface.

## Algorithm changes

1. Added a shared deployment residual gate in `bdse/model/residual_gate.py`.
   - NumPy runtime and Torch training use the same trust equation.
   - Trust uses variance, boundary strength, raw local/residual sign disagreement, and magnitude ratio.

2. Added pair-level aggregate residual intervention control.
   - Multiple small atom residuals can no longer accumulate into an unconfident pair-sign flip.
   - Unconfident flips are sign-preserving capped.
   - A flip is allowed only when the residual lower-confidence correction exceeds the local margin plus a configurable flip margin.

3. Changed pair residual learning from full reconstruction to correction-focused learning.
   - Local-sign errors receive the largest residual regression weight.
   - Teacher/local near-boundary pairs receive elevated weight.
   - Already-correct, far-from-boundary pairs receive only a small background weight.
   - All action/rank/certificate losses use the exact deployment-gated interface.

4. Added local-only pair-full decomposition.
   - `local_pair_full_interface_action_match`
   - `local_pair_full_to_residual_flip_rate`
   - `harmful_residual_intervention_rate`
   - `beneficial_residual_intervention_rate`
   - `dense_to_local_pair_full_flip_rate`

5. Reworked checkpoint selection.
   - Rewards local pair-full and residual-combined pair-full.
   - Penalizes residual interface drop and net harmful residual intervention.
   - Missing local/residual diagnostics are treated as a failed diagnostic path.

6. Added validation-aware early stopping.
   - Default patience: 3 validation events.
   - Prevents prolonged residual fine-tuning from overwriting a stronger early local interface.

7. Tightened genuine cross-family budget competition.
   - maximum interaction prefix fraction 0.80 -> 0.75;
   - soft interaction Top-M reserve 4 -> 2;
   - decision family boost 0.10 -> 0.0.

8. Retained effective downstream modules.
   - nested AOCC frontier;
   - independent calibration;
   - exact-budget action preservation;
   - candidate/interface/compression error decomposition;
   - teacher-nearest + model-confused rival union.

## Runtime changes

- max runtime pair queries: 128 -> 96;
- max selector pairs: 64 -> 56;
- tournament `L_infer`: 10 -> 8;
- residual refinement pairs: 24 -> 16;
- utility refinement top-k: 6 -> 4;
- exact decision prefix is force-filled to B=16.

## Test-set protocol changes

1. Added `bdse/tools/check_test_set_readiness.py`.
2. Build integrity is now separate from distribution parity.
3. Hard checks include:
   - config parity with val;
   - missing labels and duplicate identity;
   - test/train and test/val manifest overlap when caches are supplied;
   - failed-preprocess fraction;
   - minimum preliminary sample count.
4. An incomplete cache may return `PRELIMINARY_PASS`, but cannot be reported as the final paper test.
5. Natural val-to-test distribution shift is a warning and must not be tuned away.

## New configs and scripts

- `bdse/configs/v50_bdse_dbap_ri_train_2gpu.yaml`
- `bdse/configs/v50_bdse_dbap_ri_cl.yaml`
- `bdse/tools/check_v50_dbap_ri_gate.py`
- `bdse/tools/check_test_set_readiness.py`
- `run_v50_dbap_ri.sh`
- `V50_DBAP_RI_NEXT_COMMANDS.sh`
- `NEXT_COMMANDS_V50_DBAP_RI.txt`
- `BUILD_MATCHED_TEST_SET.sh`
- `CHECK_PARTIAL_TEST_SET.sh`

## Validation

- compileall: passed;
- YAML loading: passed;
- shell syntax: passed;
- tests: 164 passed, 5 warnings.

No v50 performance claim is made before a fresh two-A30 train/calibrate/paired-gate run. Closed-loop remains strictly gated.

---

# v50-FR — Rebuilt Foundation and Exact-Selector Execution Acceleration

## Trigger

The historical warm-start checkpoint `outputs_v30/train/bdse_v30_pmvrbsr.best.pt` was deleted. The v50 launcher therefore could not start, and direct random initialization would have changed the comparison protocol relative to the previous v49/v50 experiments. The exact CPU selector was also the dominant training wall-time bottleneck.

## Experimental-protocol decision

1. Rebuild a v30-compatible foundation checkpoint from random initialization with the current code and the original v30 objective schedule.
2. Warm-start v50 from the rebuilt best checkpoint.
3. Use the exact same rebuilt checkpoint as the frozen control.
4. Recompute control, calibration, paired replay and gate outputs; never reuse historical control/calibration files.
5. Treat historical v49/v50 metrics as diagnostic only. Absolute cross-run comparison is invalid when the foundation checkpoint differs.
6. Direct v50 scratch training remains an explicit ablation, not the primary experiment.

## Foundation rebuild

Added:

- `bdse/configs/v50_rebuild_v30_from_scratch_2gpu.yaml`;
- `RUN_MODE=foundation` in `run_v50_dbap_ri.sh`;
- automatic rebuild in `V50_DBAP_RI_NEXT_COMMANDS.sh` when `V30_CKPT_IN` is absent;
- mid-epoch resume discovery for the foundation stage;
- `bdse/tools/write_checkpoint_provenance.py`;
- SHA-256/config/cache-manifest provenance at `OUT_ROOT/provenance/foundation_checkpoint.json`.

The rebuilt foundation uses the v30 objective for four epochs, seed 17 by default, two-GPU DDP and `teacher_action_match` checkpoint selection. `allow_oracle_only_selector_training=true` is explicit because the original v30 predicted-selector start is after the foundation epoch count.

## Training hot-path diagnosis

The main bottleneck is exact selector execution within the loss:

- packed CUDA-to-CPU synchronization;
- per-scene Python/NumPy HAB and AOCC logic;
- primary-plus-auxiliary exact budgets;
- DDP waits for both ranks before backward synchronization.

At 50k scenarios and global batch eight there are approximately 6,250 optimizer steps per epoch. Data loading, validation and checkpoint I/O are secondary but measurable through existing per-stage metrics.

## Logic-preserving acceleration

Added an optional exact-selector CPU backend:

```yaml
training:
  deployment_selector_cpu_backend: process
  deployment_selector_cpu_workers: 4
```

Implementation details:

- persistent `ProcessPoolExecutor` per DDP rank;
- `spawn` multiprocessing context to avoid forking an initialized CUDA process;
- one NumPy snapshot per rank/step;
- independent scenes distributed to CPU workers;
- unchanged exact `_predicted_pair_certificate_masks` in every worker;
- deterministic reassembly by scene index;
- one host-to-device mask transfer after all scene results return;
- sequential and thread backends retained for debugging/fallback.

Representative B=4, K=32, E=128, P=56, B16+B8 benchmark:

- sequential steady state: 1.679 s;
- four spawn workers steady state: 0.428 s;
- speedup: 3.92x;
- masks: exactly equal.

The first process invocation includes approximately eight seconds of one-time spawn/import cost. A two-rank nested process-pool smoke test passed.

## Additional execution-only changes

- fused AdamW on supported CUDA builds, with standard fallback;
- foreach gradient clipping, with standard fallback;
- explicit DataLoader workers/prefetch/pinned memory;
- checkpoint interval 2,000 steps;
- foundation and v50 mid-epoch resume;
- included `tools/benchmark_v50_training_hotpath.py`.

The exact selector cadence, exact-scene coverage, budget schedule, losses, validation protocol, calibration and gate were not changed.

## Validation

- compile: passed;
- shell syntax: passed;
- YAML load: passed;
- full unit suite: **165 passed, 5 warnings**;
- exact sequential/process equality: passed;
- two-rank process-pool smoke test: passed.

## Claim boundary

The rebuilt foundation preserves attribution only inside the new matched experiment family. It is not the deleted historical checkpoint and must not be used for direct absolute comparison with old v49/v50 numbers. Component-level v49-versus-v50 claims require variants initialized from the same rebuilt foundation and evaluated on identical rows with identical calibration provenance.

---

# v50.1 — Checkpoint-Independent Foundation Resolution

## Trigger

The server still reported:

```text
Missing checkpoint: outputs_v30/train/bdse_v30_pmvrbsr.best.pt
```

after the historical `outputs_v30` directory had been deleted. The root cause was an engineering dependency retained in both launchers: an unset `V30_CKPT_IN` was immediately repopulated with the deleted hard-coded path. A stale or detached launcher could therefore continue to fail before the intended rebuild protocol became visible.

## Experimental-protocol decision

The paper main run no longer depends on a directory named `outputs_v30`.

The canonical initialization is now a **matched foundation checkpoint**, resolved in this order under `FOUNDATION_POLICY=auto`:

1. an explicitly supplied existing `FOUNDATION_CKPT`;
2. a current-run rebuilt foundation under `OUT_ROOT/foundation_v30_compatible`;
3. a conservatively verified retained copy whose own filename/config/output metadata identifies it as v30-compatible;
4. a fresh rebuild from random initialization using `v50_rebuild_v30_from_scratch_2gpu.yaml`.

Later AOCC/D3CE/DBCE/DBAP checkpoints are rejected by default. They already contain algorithm-specific adaptation and would confound attribution of v50 gains. They may only be enabled with `ALLOW_ALGORITHM_CHECKPOINT_INIT=1`, which is explicitly classified as a transfer-initialization ablation rather than the paper main run.

## Engineering changes

1. Removed the default assignment:

   ```bash
   V30_CKPT_IN=outputs_v30/train/bdse_v30_pmvrbsr.best.pt
   ```

   from the v50 outer and inner launchers.

2. Added canonical variables:

   - `FOUNDATION_CKPT`;
   - `FOUNDATION_POLICY=auto|rebuild|recover|explicit`;
   - `FOUNDATION_SEARCH_ROOT`;
   - `RECOVER_SAFE_FOUNDATION_COPIES`;
   - `ALLOW_ALGORITHM_CHECKPOINT_INIT`.

3. Added `bdse.tools.resolve_foundation_checkpoint`:

   - inventories retained `.pt` files under `outputs_v40*` through `outputs_v50*`;
   - loads each checkpoint safely on CPU;
   - measures current-model key and parameter-shape compatibility;
   - records checkpoint config/output/warm-start metadata;
   - selects only checkpoints with strong v30 identity evidence by default;
   - writes `OUT_ROOT/provenance/foundation_checkpoint_inventory.json`;
   - reports but rejects algorithm-specific checkpoints.

4. The resolved foundation is assigned to both:

   - v50 warm-start;
   - the frozen matched control.

   Control, calibration, replay and gate outputs are therefore recomputed within one provenance family.

5. Added a launcher version banner and changed relative-path resolution to the script directory. This exposes stale server copies immediately in logs.

6. Replaced hard-coded `GPUS=0,1` in downstream stages with the caller-provided `GPUS` value.

7. Added a foundation final-checkpoint fallback if a clean foundation run emits the final checkpoint but no metric-specific `.best.pt` file.

8. Changed the v50 config's `warm_start_recommended` field from a deleted physical path to `resolved_by_V50_DBAP_RI_NEXT_COMMANDS.sh`.

## Recommended interpretation of retained v40-v49 outputs

- A checkpoint whose own metadata identifies it as `bdse_v30_pmvrbsr` may recover the deleted foundation without changing the historical initialization family.
- Directories named `runtime_v30ckpt` often contain evaluation artifacts rather than a copied checkpoint; directory naming alone is not accepted as provenance.
- `outputs_v47_control_val_tune` normally contains control replay outputs, not necessarily model weights.
- v47/v48/v49 trained best checkpoints may be useful as transfer ablations, but not as the main v50 foundation.
- If no verified v30 copy exists, rebuilding a matched foundation is the clean default.

## Validation

- launcher shell syntax: passed;
- resolver unit tests: passed;
- safe v30 identity selection: passed;
- algorithm-specific checkpoint rejection: passed;
- no hard-coded v30 dependency remains in the v50 launch path.

## Claim boundary

A fresh rebuilt foundation does not recreate the deleted v30 weights bit-for-bit. It preserves causal attribution inside the new matched experiment because v50 and control share the same foundation. Historical absolute values remain diagnostic only. For a direct v49-versus-v50 component claim, both variants must be retrained from the same resolved foundation.

---

# v51 FAR-DBAP — Foundation-Anchored Residual Decision-Boundary Action Preservation

Date: 2026-07-30

## Trigger: v50 open-loop gate failed

Fresh v50 output showed:

- teacher action match: candidate `0.068`, matched foundation control `0.134`;
- winner-rival sign accuracy: `0.506` vs `0.678`;
- near-tie sign accuracy: `0.307` vs `0.494`;
- evidence sufficiency: `0.0770` vs `0.0674` (`+0.00965`, not enough to recover decisions);
- teacher regret mean: `13113.5` vs `10486.8`;
- candidate paired regret median/p90/CVaR90 all regressed;
- latency p95: `1161 ms`, with prediction stage, not exact selector, now the dominant cost.

The best v50 epoch was epoch 1. Later validation degraded monotonically enough to identify full-model/interface drift rather than under-training. The rebuilt four-epoch foundation was itself weak and therefore unsuitable as an unquestioned residual anchor.

## What v50 did validate

The v50 best checkpoint still showed a useful residual signal:

- local pair-full match `0.050` -> gated residual pair-full match `0.068`;
- beneficial residual intervention `0.026` > harmful intervention `0.008`.

AOCC execution was also healthy:

- exact tournament target active `1.0`;
- calibrated bound active `1.0`;
- certified pair fraction `0.752`;
- fixed-budget fill `16/16`;
- frontier retained weight `0.804`.

Therefore v51 preserves AOCC/certificate/budget logic and repairs the foundation/residual interface instead of replacing the selector.

## Algorithm diagnosis

1. The v50 foundation was too weak for causal residual attribution.
2. Full-model fine-tuning changed the base/local interface and damaged pair ordering.
3. A direct foundation `pair_head` was silently reused as a residual-over-local head because tensor shapes matched, despite incompatible semantics.
4. Residual authorization used evidence-only local margin and ignored the base margin that participates in the final action decision.
5. Near-boundary pairs received less trust, and full disagreement suppression prevented the residual from correcting anchor-wrong pairs.
6. The v50 control used a different v43 runtime/selector configuration, confounding the comparison.
7. Invalid candidate `inf - inf` subtraction produced runtime warnings and could contaminate margin statistics.

## v51 algorithm changes

### 1. Strong foundation gate

Added `bdse/configs/v51_strong_foundation_anchor_2gpu.yaml` and `bdse.tools.check_v51_foundation_quality`.

The foundation is rebuilt for 12 epochs with normalized base loss and is replayed before v51 training. Residual training is blocked unless the foundation meets minimum teacher-match, pair-interface, sign-accuracy, sufficiency, and latency thresholds.

### 2. Immutable foundation interface

The v51 main run freezes all parameters except:

- `pair_head`;
- `pair_var_head`;
- `proposal_feature_proj`;
- `family_embed`;
- `family_activity_proj`;
- `family_head`;
- `proposal_head`.

This prevents base/local representation drift and makes the residual stage causally interpretable.

### 3. Semantic warm-start reset

Added `training.reinitialize_modules_after_warm_start` and `_reinitialize_modules_after_warm_start`.

V51 resets `pair_head` and `pair_var_head` after loading the foundation, because a direct pair predictor is not a valid residual predictor even if shapes match. Reset parameters are broadcast in DDP.

### 4. Foundation-anchored losses

Added:

- `L_anchor_preserve`: do not reduce teacher-correct far-anchor signed margin below a configured ratio/minimum;
- `L_anchor_correct`: correct foundation-wrong and near-tie pairs toward a capped teacher-directed margin.

Both operate on the deployed gated margin and are configured through `training.foundation_anchor_objective` and loss weights `anchor_preservation` / `anchor_correction`.

### 5. Full-margin certified residual gate

`confidence_shrunk_residual_pair_delta_{numpy,torch}` now accepts the normalized base margin.

The anchor is:

```text
full foundation margin = base pair margin + local evidence pair margin
```

Trust is highest near the full anchor boundary. Non-confident corrections are capped so they cannot flip a non-zero anchor. A flip is allowed only when the uncertainty-shrunk residual magnitude crosses the full anchor plus a positive flip margin.

### 6. Three-way causal protocol

Added matched configs for:

- candidate: v51 FAR residual enabled;
- local control: the same v51 checkpoint and runtime with residual intervention disabled;
- foundation control: the foundation checkpoint with matched v51 runtime/selector.

Each arm is independently calibrated and evaluated on identical deterministic replay keys. `check_v51_far_dbap_gate.py` checks residual-only gain, total gain, paired regret, exact selector coverage, certificate/fill/fallback/latency, and training health. CL20 runs only after PASS.

### 7. Non-finite invalid-candidate fix

Tournament pair-margin construction and BDSE metrics now sanitize invalid-candidate infinite costs using the shared finite-cost policy before subtraction. Scale estimation uses valid-valid margins only.

## New files

- `V51_FAR_DBAP_NEXT_COMMANDS.sh`
- `run_v51_far_dbap.sh`
- `NEXT_COMMANDS_V51_FAR_DBAP.txt`
- `README_V51_FAR_DBAP.md`
- `V51_FAR_DBAP_ANALYSIS_AND_NEXT_STEPS.md`
- `V50_RESULT_DIAGNOSIS.json`
- `bdse/configs/v51_strong_foundation_anchor_2gpu.yaml`
- `bdse/configs/v51_far_dbap_train_2gpu.yaml`
- `bdse/configs/v51_far_dbap_cl.yaml`
- `bdse/configs/v51_far_dbap_local_control_cl.yaml`
- `bdse/configs/v51_far_dbap_anchor_control_cl.yaml`
- `bdse/tools/check_v51_foundation_quality.py`
- `bdse/tools/check_v51_far_dbap_gate.py`
- `bdse/tests/test_v51_far_dbap.py`

## Validation

- Python compile: passed;
- all five v51 YAML files load and training schedule validation passes;
- shell syntax: passed;
- full unit suite: **173 passed, 6 warnings**;
- far full-anchor no-unconfident-flip: passed;
- near-boundary certified flip: passed;
- warm-start reset and frozen-interface test: passed;
- invalid-candidate non-finite margin test: passed.

## Main-run command policy

Use a new `OUT_ROOT`, `FOUNDATION_POLICY=rebuild`, `RECOVER_SAFE_FOUNDATION_COPIES=0`, and `ALLOW_ALGORITHM_CHECKPOINT_INIT=0`. Do not reuse the weak four-epoch v50 foundation. The foundation gate must pass before v51 training.

## Claim boundary

V51 supplies a cleaner novelty and theorem route: budgeted AOCC compression plus uncertainty-certified residual intervention on the complete decision boundary. It has not yet been trained or simulated in the current environment. No empirical SOTA or closed-loop improvement is claimed until fresh three-way open-loop and paired closed-loop results pass the gates.

## v51.1 Engineering/algorithm gate separation

The known prediction-stage latency is now reported separately from the causal algorithm-quality gate by default. `ENFORCE_LATENCY_BEFORE_CL=0` permits paired CL20 when action-quality/AOCC checks pass, while still marking the run as ineligible for a real-time claim. Set `ENFORCE_LATENCY_BEFORE_CL=1` to make the 500 ms target a hard prerequisite. This prevents an engineering bottleneck from hiding whether FAR-DBAP improves closed-loop decisions. The gate marker also depends on the gate source file, so changes to gate logic invalidate stale PASS markers.

---

# v52 BFAR-DBAP — Boundary-Focused Anchor-Residual Decision-Budget Action Preservation

Date: 2026-07-31

## Trigger: v51 stopped at an over-constrained foundation gate

The uploaded v51 run did not train FAR-DBAP or execute closed loop. It stopped after the foundation replay because the old gate required direct pair-head and budget-selector quality from modules that would later be reset/trained.

Observed immutable-interface quality:

- full interface action match `0.359`;
- base winner-rival sign `0.671`;
- dense winner-rival sign `0.803`;
- dense near-tie sign `0.706`;
- dense all-pair sign `0.718`;
- teacher regret `10808`.

Observed downstream-head/selector quality:

- direct pair-full action match `0.060`;
- direct pair near-tie sign `0.342`;
- certified pair fraction `0.210`;
- fallback `0.791`.

The first group is sufficient for an immutable anchor; the second group must be learned in the BFAR stage and is no longer part of the anchor gate.

## Training-speed diagnosis

- non-exact foundation epochs: `2509–2796 s`;
- full-exact epoch: `4725 s`;
- exact selector cost: `0.3146 s/step`;
- data wait and H2D: approximately `1 ms/step` each.

The dominant costs were dense pair/loss/backward and all-scene exact CPU selector work, not data loading.

## V52 algorithm changes

### 1. Immutable anchor-only gate

Added `bdse.tools.check_v52_anchor_quality`. It gates only the base+dense-local interface that remains frozen. Direct pair head, proposal selection, evidence sufficiency and AOCC coverage are explicitly excluded.

The uploaded v51 checkpoint passes this corrected gate. The recommended main run reuses that checkpoint instead of rebuilding another foundation.

### 2. Factorized fast anchor fallback

Added `bdse/configs/v52_factorized_anchor_fast_2gpu.yaml` for cases where no usable anchor exists:

- direct pair-conditioned inference disabled;
- exact deployment selector training disabled;
- only scene/action/evidence/query/base/local modules train;
- normalized base/local decision losses retained;
- batch/chunk settings enlarged.

### 3. Quota-constrained boundary pair curriculum

Added `_boundary_focused_pair_subsample` in `bdse.experiments.train`.

Most optimizer steps keep 64 pairs and reserve overlapping quotas for:

- teacher-winner/rival pairs;
- hard-feasibility crossings;
- near-tie pairs.

Remaining slots use a deterministic joint decision-weight score. This prevents abundant hard pairs from evicting all near-boundary pairs. Full pair graphs are restored on exact AOCC steps and in the final alignment tail.

### 4. Periodic exact full-graph AOCC supervision

The main run uses:

- batch size 8 per GPU;
- exact selector on 1 scene/rank every 4 steps;
- full batch exact selector for the final 128 steps;
- process backend with 4 workers/rank.

Ordinary exact scene coverage is `1 / (8 * 4) = 3.125%`, with a short final full-alignment tail. The gate now requires sparse exact supervision rather than the impossible historical 99% average.

### 5. Primary-budget exactness with sparse auxiliary budgets

B=16 is evaluated at every exact event. B=8/B=24 are sampled only every fourth exact event as robustness regularizers. This preserves the fixed-budget paper objective while reducing expensive CPU selector calls.

### 6. Core-idea evidence gate

`check_v52_bfar_dbap_gate.py` now directly requires:

- proposal decisive recall;
- selected decisive recall;
- effective decisive recall;
- selected interaction-decisive recall;
- AOCC certificate/frontier/fill;
- residual-only and total action gains;
- paired regret non-regression.

This ties the empirical gate to the paper statement that retained evidence must preserve or correctly change action ordering under B=16.

### 7. Training-health gate repair

The gate checks finite critical loss/timing metrics and sparse exact coverage. Optional diagnostics with zero denominators may be NaN without invalidating training. The old requirement that every epoch have near-100% exact coverage was removed.

### 8. Provenance and launcher fixes

- added `FOUNDATION_SOURCE_CONFIG` so an explicitly reused v51 checkpoint is recorded with its actual training config;
- fixed factorized-anchor checkpoint naming across launcher and pipeline;
- exposed batch size, worker count and sparse selector cadence as environment variables;
- retained algorithm/latency gate separation and three-way paired controls.

## Main algorithm statement

V52 BFAR-DBAP learns a fixed-budget evidence coreset over flip-critical action pairs, anchored to the complete base+local decision margin. A residual may change an action ordering only when its uncertainty-shrunk correction is certified against the full anchor margin. Far-correct pairs receive do-no-harm supervision; near-tie and anchor-wrong pairs receive correction supervision.

## New files

- `V52_BFAR_DBAP_NEXT_COMMANDS.sh`
- `run_v52_bfar_dbap.sh`
- `NEXT_COMMANDS_V52_BFAR_DBAP.txt`
- `README_V52_BFAR_DBAP.md`
- `V52_BFAR_DBAP_ANALYSIS_AND_NEXT_STEPS.md`
- `V51_RESULT_DIAGNOSIS.json`
- `bdse/configs/v52_factorized_anchor_fast_2gpu.yaml`
- `bdse/configs/v52_bfar_dbap_train_2gpu.yaml`
- `bdse/configs/v52_bfar_dbap_cl.yaml`
- `bdse/configs/v52_bfar_dbap_local_control_cl.yaml`
- `bdse/configs/v52_bfar_dbap_anchor_control_cl.yaml`
- `bdse/tools/check_v52_anchor_quality.py`
- `bdse/tools/check_v52_bfar_dbap_gate.py`
- `bdse/tests/test_v52_bfar_dbap.py`

## Claim boundary

V51 provides no residual or closed-loop result because the pipeline stopped before those stages. V52 has been code-validated locally but not trained on nuPlan in this environment. No closed-loop or SOTA claim is made before fresh three-way replay and paired CL evaluation.

---

# v53 WC-BFAR-DBAP — Winner-Consistent Boundary-Focused Anchor-Residual Decision-Budget Action Preservation

Date: 2026-07-31

## Trigger: v52 was blocked before the algorithm stage

The uploaded v52 pipeline did not train BFAR-DBAP and did not execute three-way open-loop or closed-loop evaluation. It stopped at the reused-foundation gate. The only hard failure was:

- `teacher_regret=12761.731177 > 12000`.

This was a semantic gate error. `teacher_regret` is the final budgeted/runtime action regret, not the regret of the frozen base+dense-local anchor. Between v51 and v52, the same checkpoint and the same 1000 scenario keys produced identical teacher, dense full-interface and local pair-full actions, while the budgeted action changed in 833/1000 rows because the runtime configuration changed. The immutable anchor metrics remained unchanged:

- full-interface action match `0.359`;
- base winner-rival sign `0.671`;
- dense winner-rival sign `0.803`;
- dense near-tie sign `0.706`;
- dense all-pair sign `0.718`.

Consequently, v52 provides no empirical test of its boundary sampler, periodic exact selector distillation, residual learning, or fixed-budget closed-loop benefit.

## 1. Correct immutable-anchor gate

Added `bdse/tools/check_v53_anchor_quality.py`.

The gate checks only the interface that is actually frozen during residual training:

- base+dense-local action match;
- base winner-rival sign;
- dense winner-rival, near-tie and all-pair sign;
- replay cardinality, unique scenario/timestamp keys and a SHA-256 key fingerprint.

It explicitly excludes budgeted regret, direct pair-head quality, proposal recall, selector quality and AOCC certification. Full-interface regret and latency are diagnostics only. The uploaded v52 replay passes this corrected gate.

## 2. Correct anchor-specific regret metrics

Added explicit metrics so future gates cannot confuse runtime actions with immutable interfaces:

- `base_interface_teacher_regret`;
- `full_interface_teacher_regret`;
- `sparse_full_interface_teacher_regret`;
- `pair_full_teacher_regret`;
- `local_pair_full_teacher_regret`.

## 3. Removed zero-gradient action objectives

V52 assigned high weights to `full_action` and `full_margin`, but both are computed from frozen base+local outputs in the residual stage. They therefore contributed no gradient to the trainable residual or selector heads and only inflated the logged total loss.

V53 sets both weights to zero and skips their forward computation when all frozen full-interface objectives are disabled.

## 4. Winner-consistent residual tournament learning

Added three objectives that act on trainable residual-dependent logits:

- `L_pair_full_action`: teacher-action cross entropy on the all-evidence residual tournament;
- `L_pair_full_winner_margin`: teacher winner versus strongest valid rival margin;
- `L_budget_preserve_pair_full`: B=16 action must preserve the pair-full winner whenever that winner is teacher-correct.

These objectives connect residual margin learning and fixed-budget evidence selection directly to action ordering rather than average pair reconstruction.

## 5. Safe no-op residual initialization

After loading the reused foundation and resetting semantically incompatible heads:

- the last residual pair-head layer is initialized to exactly zero;
- the pair-variance head starts from a configurable constant uncertainty bias;
- base/local parameters remain unchanged.

The candidate therefore starts exactly at the frozen anchor rather than immediately introducing random action flips.

## 6. Retained BFAR main line

The following v52 mechanisms are retained because they were never tested and remain aligned with the paper idea:

- winner-rival, hard-feasibility-crossing and near-tie pair quotas;
- maximum 64 boundary-critical pairs on ordinary steps;
- periodic full-pair-graph exact AOCC supervision;
- B=16 as the primary exact deployment budget;
- sparse B=8/B=24 robustness supervision;
- full-margin uncertainty-certified residual intervention;
- counterfactual decisive-evidence and do-no-harm losses;
- independent calibration and three-way paired controls.

## 7. Additional training-speed changes

- cycle/transitivity topology mining runs every 4 steps instead of every step;
- maximum sampled consistency triangles reduced from 64 to 48;
- final full-graph/full-exact alignment tail reduced from 128 to 64 steps;
- stale frozen full-interface objective computation is skipped;
- exact CPU process backend and v52 boundary-pair sparsification are retained.

No surrogate replaces the deployed exact B=16 selector.

## 8. Two-tier evaluation gate

Added `bdse/tools/check_v53_wc_bfar_dbap_gate.py`.

### Minimum-completeness gate

Controls whether paired CL20 runs. It rejects catastrophic action/regret regression, broken pairing, missing exact supervision, failed calibration, budget violations, severe harmful residual behavior and gross loss of decisive evidence. It does not require a paper-grade improvement before any closed-loop evidence can be collected.

### Competitive gate

Controls CL100/paper-result escalation. It retains strict requirements on:

- candidate gain over foundation and same-checkpoint local control;
- pair-full residual gain;
- beneficial versus harmful interventions;
- decisive and interaction-decisive recall;
- AOCC certification/frontier/fallback;
- paired median and p90 regret non-regression.

Latency remains an independent deployment warning unless `ENFORCE_LATENCY_BEFORE_CL=1`.

## 9. Claim boundary

V53 fixes the gate semantics and the missing winner-gradient path. It has passed local code/unit validation, but has not been trained or simulated on nuPlan in this environment. No gate PASS, closed-loop gain, fixed-budget SOTA or real-time claim is made before a fresh v53 run.

---

# v54 AR-BFAR-DBAP — Anchor-Relative Boundary-Focused Residual Decision-Budget Action Preservation

Date: 2026-07-31

## Trigger: V53 selector succeeded at preserving the wrong pair winner

V53 is the first run in this branch that completed fresh residual/selector training, independent three-way calibration, and three paired 1000-scene open-loop replays. Its immutable anchor gate passed, but both the minimum-completeness and competitive gates failed, and the launcher stopped before CL20.

V53 candidate metrics:

- teacher action match: `0.100`;
- matched foundation teacher action match: `0.116`;
- dense full-interface action match: `0.359`;
- B=16 selected-local sparse action match: `0.141`;
- local pair-full action match: `0.118`;
- residual pair-full action match: `0.100`;
- proposal/selected/effective decisive recall: `0.801/0.588/0.754`;
- interaction-decisive recall: `0.555`;
- AOCC certificate: `0.821`;
- fallback: `0.157`;
- budget-vs-pair-full action match: `0.987`;
- beneficial/harmful residual intervention: `0.015/0.033`.

The selector and certificate therefore reached the previous competitive coverage targets, but they preserved a weak pair-full action target. The primary bottleneck is the pair tournament interface, not exact coreset search.

## 1. Engineering gate repair

V53 computed frozen anchor drift by comparing different interfaces:

- candidate `local_pair_full_interface_action_match`;
- local-control `pair_full_interface_action_match`.

V54 compares row-wise, same-interface actions instead:

- candidate/local `local_pair_full_action`;
- candidate/local `full_action`.

Both corrected drifts are exactly zero on the uploaded V53 replay. The corrected protocol-integrity gate passes.

## 2. Diagnostic closed-loop on minimum failure

V53 stopped after the minimum gate and produced no CL20. V54 introduces three gate tiers:

1. protocol-integrity: failure blocks all closed-loop evaluation;
2. minimum-completeness: determines whether CL20 is a formal minimum-result PASS;
3. competitive: controls CL100 and paper-result escalation.

When protocol integrity passes but minimum performance fails, `RUN_DIAGNOSTIC_CL20_ON_GATE_FAIL=1` runs fully paired candidate/local/foundation CL20 and writes `.diagnostic_cl20`. This prevents repeated open-loop-only iterations while keeping the result clearly separated from a publication PASS.

## 3. Selected-local integrable tournament anchor

V53 reconstructed the pair tournament from `J0` plus queried pair edges. Missing edges fell back to the base margin and discarded the selected evidence's local action costs. The resulting graph could be non-integrable and inconsistent even when dense/local pair signs were strong.

V54 first constructs the selected-local action cost:

`J_B^L(a) = J0(a) + sum_{i in S_B} g_i(a)`.

The learned pair prediction is decomposed into its known local component and a residual:

`r_B(a,b) = pair_delta(a,b) - local_delta(a,b)`.

The tournament margin becomes:

`m_B^AR(a,b) = J_B^L(a) - J_B^L(b) + r_B(a,b)`.

At zero residual, the tournament exactly reproduces the selected-local B=16 planner. This is now enforced in both differentiable training and runtime inference.

## 4. Full-margin action-anchor guard

The runtime records the selected-local anchor action before applying residual pair corrections. A proposed residual action flip is accepted only if:

- the uncertainty-shrunk proposed-vs-anchor pair margin exceeds the configured flip margin;
- the proposed tournament score improves over the anchor winner;
- validity and safety masks allow the action.

Otherwise the final action is restored to the selected-local anchor. New diagnostics report the proposed/anchor actions, robust margin, score gain, and whether the flip was blocked or accepted.

## 5. Anchor-relative winner objectives

Full-evidence training starts from `J0 + sum_all g`; B=16 training starts from `J0 + sum_selected g`. The pair head only supplies a residual correction.

New/updated objectives:

- `pair_full_anchor_preserve`: keep the full-local winner when it is teacher-correct;
- `budget_preserve_pair_full`: keep the selected-local winner when it is teacher-correct;
- teacher-action and strongest-rival correction on anchor-wrong scenes;
- `anchor_wrong_action_weight` upweights correctable anchor errors;
- existing do-no-harm, counterfactual decisive evidence, certificate and proposal losses are retained.

## 6. Further speed reduction

V54 changes the residual/selector schedule:

- epochs: `6 -> 4`;
- ordinary boundary pairs: `64 -> 48`;
- full graph cadence: every `4 -> 8` steps;
- exact B=16 supervision cadence: every `4 -> 8` steps;
- final full-exact tail: `64 -> 32` steps;
- cycle/transitivity cadence: every `4 -> 8` steps;
- maximum consistency triangles: `48 -> 32`.

The deployed B=16 selector remains exact AOCC. No surrogate replaces deployment selection.

## 7. V53 findings by core issue

1. Pair sign to action: not solved. Strong dense signs did not survive the reconstructed pair graph.
2. Decisive evidence selection: coverage/certification is largely solved, but the certified target winner is wrong.
3. Training compute: substantially improved. V53 reduced epoch time to approximately 20–30 minutes from V51's roughly 42–79 minutes.

## 8. New files

- `V54_AR_BFAR_DBAP_NEXT_COMMANDS.sh`;
- `run_v54_ar_bfar_dbap.sh`;
- `NEXT_COMMANDS_V54_AR_BFAR.txt`;
- `README_V54_AR_BFAR.md`;
- `V54_AR_BFAR_ANALYSIS_AND_NEXT_STEPS.md`;
- `V53_RESULT_DIAGNOSIS.json`;
- `V54_GATE_ON_V53.json`;
- `bdse/configs/v54_ar_bfar_dbap_train_2gpu.yaml`;
- `bdse/configs/v54_ar_bfar_dbap_cl.yaml`;
- `bdse/configs/v54_ar_bfar_dbap_local_control_cl.yaml`;
- `bdse/configs/v54_ar_bfar_dbap_anchor_control_cl.yaml`;
- `bdse/tools/check_v54_ar_bfar_dbap_gate.py`;
- `bdse/tests/test_v54_ar_bfar.py`.

## 9. Claim boundary

V54 has been code-validated and the corrected gate has been replayed on V53 outputs. It has not been trained or simulated on nuPlan in this environment. No future gate PASS, closed-loop improvement, fixed-budget SOTA, or real-time claim is made before fresh V54 training and paired CL20.

---

# V55 PC-BFAR-DBAP update — 2026-07-31

Version: **Potential-Consistent Boundary-Focused Anchor-Residual Decision-Budget Action Preservation**.

## 1. V54 experimental diagnosis

The uploaded V54 run completed fresh residual/selector training, three independent calibrations, and three paired 1000-scene open-loop replays. It did not run CL20 or CL100.

V54 gate status:

- immutable anchor gate: PASS;
- protocol gate: FAIL only because `selector_exact_fraction=0.02572` was compared against a hard-coded `0.03`;
- configured train floor: `0.015`;
- minimum failure list: empty;
- competitive gate: FAIL.

Replaying the same results with a config-derived exact-fraction floor gives protocol PASS and minimum PASS, while competitive remains FAIL. Therefore the missing CL20 was caused by an engineering gate mismatch, but the lack of competitive action gain is algorithmic.

Key V54 observations:

- candidate/local/foundation final teacher match: `0.118/0.118/0.118`;
- local pair-full match: `0.118`;
- sparse all-local-evidence match: `0.141`;
- dense full-interface match: `0.359`;
- candidate-local deployed action differences: `0/1000`;
- beneficial/harmful causal residual interventions: `0/0`;
- pair-full match: `0.118`;
- budget-vs-pair-full match: `0.992`;
- proposal/selected/effective decisive recall: approximately `0.800/0.577/0.743`;
- fallback: `0.156`;
- frontier retained: `0.759`;
- epoch time: approximately `17–20` minutes.

The selector and sparse training schedule remain useful. The action aggregation path is the primary bottleneck.

## 2. Engineering gate fix

`check_v55_pc_bfar_dbap_gate.py` now reads `training.min_deployment_exact_fraction` from the actual train config. A command-line value can raise the floor, but the gate no longer silently imposes an unrelated hard-coded threshold.

The V54 replay passes V55 protocol and minimum gates and correctly fails only the competitive gate.

## 3. Pure same-checkpoint local control

V54 disabled residual mean but left residual variance active. Variance still affected AOCC and tournament uncertainty and produced five internal `local_pair_full -> pair_full` action changes in 1000 scenes.

V55 makes `disable_pair_residual_intervention=true` disable both:

- selector/tournament residual mean;
- selector/tournament residual variance.

The local control is therefore causally isolated from the residual head.

## 4. Direct selected-local action anchor

V54 inserted selected-local margins into a restricted-rival tournament, but still selected the action through tournament traversal. Zero residual was not guaranteed to equal the direct B=16 selected-local argmin.

V55 constructs:

`J_B^L(a) = J0(a) + sum_{i in S_B} g_i(a)`

and uses its direct global argmin/scores as the action anchor. At zero residual the deployed action exactly equals the selected-local planner, independently of pair variance and rival graph coverage.

## 5. Potential-consistent residual aggregation

Independent pair residuals can contain cycles and violate global action transitivity. V55 projects the selected residual edge field onto an integrable action potential:

`phi* = argmin_phi sum_(a,b) w_ab ((phi_b-phi_a)-r_ab)^2 + lambda ||phi||^2`.

Boundary pairs receive larger weights. Only the conservative potential component can change action costs:

`J_B^PC(a) = J_B^L(a) + scale * phi(a)`.

The non-conservative component is reported through reconstruction and cycle diagnostics and is penalized during training rather than being allowed to alter the winner through an arbitrary tournament path.

## 6. Global certified residual flip

A potential-corrected action may replace the selected-local winner only when:

- its direct corrected cost is better;
- the uncertainty-shrunk proposed-vs-anchor global margin exceeds the flip threshold;
- validity and safety guards permit the action.

Pair variance no longer changes anchor scores directly; it is used only for certification.

## 7. Action-potential distillation

V55 adds a direct global target for the residual potential based on the teacher-versus-selected-local cost correction. The target is centered per scene and robustly scaled. It upweights:

- the teacher winner;
- the selected-local strongest rival;
- actions near the teacher decision boundary;
- scenes where the selected-local anchor is wrong.

This provides a real gradient from preserved evidence to the final global winner, instead of relying only on average independent pair signs.

## 8. Deployment target alignment

Periodic exact AOCC training can target the full Top-M integrable-potential action. Thus the exact B=16 selector is trained to preserve the same downstream action definition used at deployment.

Deployment still uses exact AOCC. No surrogate selector replaces the B=16 method.

## 9. Gate and diagnostics

V55 records:

- direct selected-local anchor action and regret;
- deployed-vs-selected-local match;
- potential proposed/allowed/deployed flip rates;
- causal paired candidate-vs-local beneficial/harmful rates;
- projection reconstruction RMSE;
- cycle fraction;
- potential correction magnitude.

If protocol integrity passes, paired diagnostic CL20 is run even when the minimum or competitive gate fails. CL100 remains blocked unless the competitive gate passes.

## 10. Preserved designs

The following V52–V54 designs are retained:

- winner/hard/near-tie boundary-pair curriculum;
- ordinary 48-pair training graph;
- periodic full graph and exact B=16 supervision;
- final short full-exact alignment tail;
- exact AOCC and counterfactual decisive-evidence targets;
- B=16 fixed evidence budget;
- same-checkpoint local and matched-foundation controls;
- independent calibration and paired replay;
- algorithm and latency gates remain separated.

## 11. New files

- `V55_PC_BFAR_DBAP_NEXT_COMMANDS.sh`;
- `run_v55_pc_bfar_dbap.sh`;
- `NEXT_COMMANDS_V55_PC_BFAR.txt`;
- `README_V55_PC_BFAR.md`;
- `V55_PC_BFAR_ANALYSIS_AND_NEXT_STEPS.md`;
- `V54_RESULT_DIAGNOSIS.json`;
- `V55_GATE_ON_V54.json`;
- `bdse/model/potential_projection.py`;
- `bdse/configs/v55_pc_bfar_dbap_train_2gpu.yaml`;
- `bdse/configs/v55_pc_bfar_dbap_cl.yaml`;
- `bdse/configs/v55_pc_bfar_dbap_local_control_cl.yaml`;
- `bdse/configs/v55_pc_bfar_dbap_anchor_control_cl.yaml`;
- `bdse/tools/check_v55_pc_bfar_dbap_gate.py`;
- `bdse/tests/test_v55_pc_bfar.py`.

## 12. Validation and claim boundary

Validation completed:

- Python compile: PASS;
- four V55 YAML files: PASS;
- shell syntax: PASS;
- unit tests: `194 passed, 7 warnings`;
- V54 replay with corrected gate: protocol PASS, minimum PASS, competitive FAIL.

No fresh V55 nuPlan training or closed-loop simulation was run in this environment. No future gate PASS, closed-loop gain, fixed-budget CCF-A result, SOTA, or real-time claim is made in advance.

---

# V56 DCIP-BFAR-DBAP update — 2026-08-01

Version: **Dual-Certificate Integrable-Potential Boundary-Focused Anchor-Residual Decision-Budget Action Preservation**.

## 1. V55 experimental diagnosis

The uploaded V55 run completed fresh residual/selector training, independent three-way calibration, paired 1000-scene open-loop replay, and paired diagnostic CL20.

Gate status:

- protocol: PASS;
- minimum: FAIL;
- competitive: FAIL.

Minimum failures were `certificate=0.204 < 0.40` and `fallback=0.801 > 0.60`. The same-checkpoint residual-disabled local control had certificate `0.888`, fully-certified scenes `0.805`, and fallback `0.11`, while candidate and local deployed actions were identical on all 1000 rows. The minimum failure was therefore primarily caused by residual uncertainty contaminating the evidence certificate.

Competitive failure was real:

- candidate/local/foundation teacher match: `0.141/0.141/0.141`;
- pair-full/local-pair-full match: `0.141/0.141`;
- candidate-local deployed flips: `0/1000`;
- beneficial/harmful residual interventions: `0/0`;
- paired regret delta: `0`.

## 2. Closed-loop runtime diagnosis

Three paired diagnostic CL20 branches completed:

- candidate wall time: `22,858 s`;
- local control: `15,924 s`;
- foundation control: `14,914 s`;
- sequential three-way total: `53,696 s` (`14.92 h`).

Each 10-scenario GPU shard constructed and loaded ten independent CUDA planner/model instances. Prediction dominated one open-loop planner call at approximately `685 ms`, compared with selector `89 ms` and tournament `6 ms`. Duplicate CUDA model loading and device contention were the primary avoidable engineering bottlenecks.

The combined closed-loop summary also reported `num_scenarios=10` despite an actual `scenario_count=20` because the combiner weighted the count field like a metric.

## 3. Shared-model closed-loop execution

V56 adds one process-global model cache keyed by checkpoint, model architecture, and device. All planners in one nuPlan worker process reuse the same read-only eval model.

- model construction remains under a cache lock, preventing concurrent duplicate CUDA allocation;
- a per-device reentrant inference lock serializes GPU forward passes while allowing CPU simulation workers to overlap;
- default `CL_WORKERS_PER_GPU` is increased to four;
- OpenMP/MKL/BLAS threads are limited to one per worker;
- summary PDF/histogram rendering is disabled by default;
- per-stage closed-loop timing is aggregated to JSON;
- combined summaries use the true scenario count;
- candidate/local/foundation CL20 scenario-token hashes are checked before accepting the paired result.

No fixed speedup is claimed before a server run.

## 4. Dual evidence/residual certificates

V55 used one certificate for both evidence sufficiency and residual uncertainty. Residual variance could therefore collapse AOCC certification even when the residual did not change the action.

V56 separates:

- **evidence certificate**: exact AOCC over selected-local evidence margins only;
- **residual flip certificate**: a global uncertainty-shrunk guard applied only when the residual action potential proposes a winner change.

The gate consumes `evidence_certificate_fraction` when available and records residual-flip certification separately.

## 5. Direct evidence-attributable integrable potential

V55 learned an arbitrary residual pair field and projected it through Hodge decomposition. The field could be cyclic before projection, and a scene-level potential target did not identify which evidence should correct which action.

V56 predicts one signed residual action potential per evidence/action query:

`h_i(a) = residual_action_head(scene, action, evidence_i, query_i(a))`.

For selected evidence set `S_B`:

`J_B^DCIP(a) = J0(a) + sum_{i in S_B} g_i(a) + scale * sum_{i in S_B} h_i(a)`.

Every pair correction is a difference of one global action cost, so antisymmetry and cycle consistency hold exactly. The legacy pair MLP is skipped in training and deployment.

All V56 train/closed-loop configurations explicitly set `model.evidence_action_residual=true`; this is required to execute the new head rather than returning a zero potential.

## 6. Atomwise causal-potential distillation

The cache provides teacher per-evidence action costs. V56 adds:

`h_i^T(a) = [g_i^teacher(a) - g_i^local(a)] / scene_scale`.

Target and prediction are gauge-centered over valid actions. The loss upweights:

- the teacher winner;
- the selected-local anchor action when it is wrong;
- interaction evidence;
- anchor-wrong scenes;
- atoms/actions with larger teacher-minus-local correction.

This resolves the identifiability problem of scene-level-only potential distillation. The global potential target remains with a lower weight as a consistency loss.

## 7. Exact selected-local no-op and pure controls

With zero residual potential, the deployed action is exactly the direct B=16 selected-local argmin, independent of pair graph coverage and pair variance.

When `disable_pair_residual_intervention=true`, both residual potential and residual variance are removed. Candidate-local differences can therefore be attributed to the residual-potential module.

## 8. Preserved effective designs

V56 retains:

- factorized base+dense-local foundation anchor;
- winner/hard/near-tie boundary-pair curriculum;
- ordinary 48-pair training graph;
- periodic full graph and sparse exact B=16 supervision;
- final short full-exact alignment tail;
- exact AOCC and counterfactual decisive-evidence targets;
- fixed B=16 decision budget;
- same-checkpoint local and matched-foundation controls;
- independent calibration and paired replay;
- diagnostic CL20 whenever protocol integrity passes;
- algorithm and latency gates remain separate.

## 9. Partial test-set policy

The uploaded partial test diagnostics contain `67,042` unique identities with no internal duplicates and the same candidate/evidence/teacher/preprocess configuration as validation. It is harder than validation, but it has no split manifest or train/val overlap audit and represents only about 28% of the intended cache.

V56 adds `RUN_V56_PRELIMINARY_TEST.sh`. It requires `V56_TEST_FROZEN_ACK=YES`, records checkpoint/config hashes, and is intended only for one-shot evaluation after all training and thresholds are frozen. It must not be used for tuning or called a final paper test set.

## 10. New files

- `V56_DCIP_BFAR_DBAP_NEXT_COMMANDS.sh`;
- `run_v56_dcip_bfar_dbap.sh`;
- `NEXT_COMMANDS_V56_DCIP_BFAR.txt`;
- `README_V56_DCIP_BFAR.md`;
- `V56_DCIP_BFAR_ANALYSIS_AND_NEXT_STEPS.md`;
- `V55_RESULT_DIAGNOSIS.json`;
- `V55_PARTIAL_TEST_READINESS.json`;
- `RUN_V56_PRELIMINARY_TEST.sh`;
- `bdse/configs/v56_dcip_bfar_dbap_train_2gpu.yaml`;
- `bdse/configs/v56_dcip_bfar_dbap_cl.yaml`;
- `bdse/configs/v56_dcip_bfar_dbap_local_control_cl.yaml`;
- `bdse/configs/v56_dcip_bfar_dbap_anchor_control_cl.yaml`;
- `bdse/tools/check_v56_dcip_bfar_dbap_gate.py`;
- `bdse/tests/test_v56_dcip_bfar.py`.

## 11. Validation and claim boundary

Validation completed:

- Python compile: PASS;
- four V56 YAML files: PASS;
- shell syntax: PASS;
- unit tests: `201 passed, 8 warnings`;
- direct action-potential integrability/no-op/certified-correction tests: PASS;
- shared model cache-key control test: PASS;
- partial test readiness audit: preliminary PASS with manifest warnings.

No fresh V56 nuPlan training or closed-loop simulation was run in this environment. No future minimum/competitive gate PASS, closed-loop speedup, fixed-budget CCF-A result, SOTA, or real-time claim is made in advance.

### V56 post-validation protocol clarification: explicit NR/R closed-loop mode

The uploaded V55 `CL20` command used `closed_loop_nonreactive_agents`; it was not a reactive closed-loop benchmark.  V56 now makes the challenge explicit through `CL_CHALLENGE` and derives the matching metric aggregator.  The default remains non-reactive for fast diagnostic iteration, while a frozen checkpoint can be evaluated with `CL_CHALLENGE=closed_loop_reactive_agents` under a separate `OUT_ROOT`.  Unsupported challenge names fail before nuPlan starts, preventing NR and R summaries from being silently mixed.

### V56 pre-release exact-selector alignment fix

A final end-to-end code audit found that `skip_pair_head_forward=true` removed `pair_atom_delta`, while the exact-selector training mask still required that legacy tensor.  Without the fix, periodic exact AOCC supervision would silently return an empty mask (or the NumPy cache would raise on the first exact step), so the new direct evidence-potential head could train without the intended deployment-selector supervision.  V56 now derives each evidence atom's certificate delta directly from the selected-local action field, `g_i(b)-g_i(a)`, when the dual-certificate/direct-potential route is active.  The full-TopM exact target is the selected-local anchor action, matching runtime AOCC; the residual action potential is excluded from the evidence certificate and is handled only by the downstream residual-flip certificate.  A regression test verifies exact selection without a legacy pair head.

### V56 frozen reactive CL20 runner

Added `RUN_V56_REACTIVE_CL20.sh`.  The uploaded V55 CL20 was non-reactive; the new runner reuses the already frozen V56 checkpoint and three calibrated configs, runs candidate/local/foundation under `closed_loop_reactive_agents`, writes to a separate output root, and verifies the three scenario-token hashes.  It never retrains and prevents non-reactive and reactive summaries from being mixed.

---

# V57 WC-DCIP-BFAR-DBAP update — 2026-08-02

Version: **Winner-Correction Dual-Certificate Integrable-Potential Boundary-Focused Anchor-Residual Decision-Budget Action Preservation**.

## 1. V56 result diagnosis

The uploaded V56 run completed training, three independent calibrations, and paired 1000-scene open-loop evaluation. The archive contains no closed-loop output; the claimed paired reactive CL20 is therefore not part of the evidence analyzed in this update.

V56 gate state:

- protocol: FAIL;
- formal minimum: FAIL, with `minimum_failures=[]`;
- competitive: FAIL.

Open-loop state:

- candidate/local/foundation teacher match: `0.140/0.141/0.141`;
- pair-full/local-pair-full match: `0.141/0.141`;
- evidence certificate: `0.888033`;
- fallback: `0.111`;
- proposal/selected/effective decisive recall: `0.798152/0.606437/0.772101`;
- interaction decisive recall: `0.575137`;
- deployed residual flip rate: `0.005`;
- beneficial/harmful deployed residual rate: `0.000/0.001`.

Paired row analysis found five candidate-local deployed flips: zero exact teacher-winner corrections, one harmful correction, and four teacher-match-neutral changes. Internal pair-full residual changed 17 winners, with one beneficial and one harmful change. V56 therefore learned non-zero perturbations but not a net useful final-winner correction.

## 2. Root cause of the protocol failure

V56 set the legacy aggregate `training.loss_weights.action` to zero while assigning non-zero weights to deployment selection, pair-full winner, budget preservation, certificate, and action-potential objectives.

The loss implementation incorrectly used `loss_weights.action > 0` as the master switch for the entire action/winner/deployment family. This silently disabled:

- exact deployment selector supervision;
- deployment-mask distillation;
- selected-budget action loss;
- pair-full action and winner-margin losses;
- budget-to-pair-full preservation;
- pair-full anchor preservation;
- global action-potential teacher loss.

All five V56 epochs consequently reported `selector_exact_fraction=0` and zero for every winner-level objective. Only atomwise residual distillation remained active and stayed nearly flat around `0.0117`.

A second audit found that the multi-budget selected-B branch did not pass `residual_action_potential` into the direct-potential logits. Even after enabling the family, the deployed-budget winner objective could therefore remain disconnected from the residual head.

## 3. Minimum-gate interpretation

V56 solved the V55 evidence/residual certificate mixing at the evidence-gate level:

- certificate recovered from `0.204` to `0.888`;
- fallback recovered from `0.801` to `0.111`.

The formal minimum label was false only because the gate defined `minimum_pass = protocol_pass and minimum_metrics_pass`. V57 reports `minimum_metrics_pass` separately.

The dual-certificate implementation was still incomplete:

- residual and dual-certificate aggregate metrics were NaN;
- the tournament did not consume the evidence certificate before authorizing a residual flip;
- a harmful flip was deployed at evidence-certificate fraction `0.2`.

## 4. Winner/deployment action-family activation

V57 replaces the legacy aggregate master switch with a family-level predicate. The winner/deployment branch activates when any relevant sub-objective has non-zero weight, including:

- deployment selection;
- certificate gap/safety/frontier;
- pair-full action and winner margin;
- budget preservation;
- pair-full anchor preservation;
- action-potential teacher distillation;
- residual winner correction.

The V57 main configuration deliberately retains `loss_weights.action=0` as a regression guard: child objectives must execute independently.

Training now reports `action_family_enabled`.

## 5. Direct selected-budget potential gradient repair

All direct-potential action paths now receive `residual_action_potential`:

- full-support action potential;
- early oracle-selected action potential;
- predicted multi-budget/B=16 action potential.

A regression test verifies that a selected evidence potential changes the budgeted winner and backpropagates a non-zero gradient into the residual potential.

## 6. Winner-directed residual correction

V57 adds `L_residual_winner_correction`.

For an anchor-wrong scene, the corrected teacher winner must outrank the exact selected-local anchor winner by a configured correction margin. For an anchor-correct scene, the corrected teacher winner must retain a preservation margin over the strongest valid rival.

This objective directly optimizes the deployed intervention role instead of assuming that atomwise residual reconstruction will automatically improve the final winner.

The term is intentionally named winner correction, not counterfactual or causal intervention: the current cache provides comparative teacher/local labels, not a separate intervention dataset.

## 7. Independent residual uncertainty training

V57 trains `residual_action_var_head` against detached atomwise residual error in log-variance space.

- the variance target cannot absorb or alter the residual mean target;
- residual uncertainty never enters the evidence certificate;
- it is used only by the residual-flip certificate;
- `residual_action_var_head` is explicitly included in the trainable-module set.

## 8. Enforced dual-certificate deployment

The residual winner may change only when both certificates pass:

1. evidence certificate fraction meets the configured threshold;
2. residual robust margin and score gain meet the flip guard.

The main V57 configuration requires a fully certified evidence frontier before a residual flip:

`min_evidence_certificate_fraction_for_residual_flip = 1.0`.

The structural post-processing path is also prevented from reintroducing a residual flip that the certificate rejected.

Full-support pair diagnostics pass evidence-certificate fraction `1.0` and use the same post-structural finalization as deployment.

## 9. Metric and gate observability

Open-loop evaluation now propagates:

- evidence certificate fraction;
- residual flip proposed/deployed;
- residual flip certificate pass;
- dual-certificate deployment status;
- evidence-certificate guard diagnostics.

The V57 protocol gate fails when any configured winner/deployment supervision is silently inactive:

- no exact selector supervision;
- action family never active;
- all winner-level losses zero;
- deployment-selection distillation zero.

The gate separately reports:

- `minimum_metrics_pass`;
- formal `minimum_pass` including the protocol prerequisite;
- `competitive_metrics_pass`;
- formal `competitive_pass`.

## 10. Preserved designs

V57 retains the designs that V56 evidence supports:

- immutable base+dense-local foundation anchor;
- winner/hard/near-tie boundary-pair curriculum;
- sparse periodic exact AOCC and final exact tail;
- fixed B=16 evidence budget;
- direct per-evidence integrable action potential;
- dual evidence/residual certificate definition;
- same-checkpoint local and matched-foundation controls;
- independent calibration and paired replay;
- shared-model closed-loop execution;
- separate non-reactive and reactive closed-loop protocols.

V57 does not restore arbitrary pair fields, Hodge projection, scene-level-only potential supervision, or full-pair/full-scene exact training on every optimizer step.

## 11. Required experiment order

1. Run `RUN_V57_TRAINING_SMOKE.sh` on 1024 train / 256 validation scenes for one epoch.
2. Continue only if action-family, exact selector, deployment-selection, pair-full action, winner-correction, and residual-uncertainty signals are all non-zero.
3. Run the complete six-epoch pipeline, independent calibration, and paired 1000-scene open loop.
4. Require protocol PASS before interpreting any algorithm gate or closed-loop result.
5. Run paired NR-CL20 and frozen paired R-CL20.
6. Run CL100 only after competitive PASS and beneficial residual interventions exceed harmful interventions.
7. Run the partial test once only after checkpoint, calibration, and thresholds are frozen.

## 12. New and changed files

- `V56_RESULT_ANALYSIS_AND_V57_OPTIMIZATION.md`;
- `V56_RESULT_DIAGNOSIS_FOR_V57.json`;
- `NEXT_COMMANDS_V57_WC_DCIP_BFAR.txt`;
- `RUN_V57_TRAINING_SMOKE.sh`;
- `V57_WCD_CIP_BFAR_DBAP_NEXT_COMMANDS.sh`;
- `run_v57_wcdcip_bfar_dbap.sh`;
- `RUN_V57_REACTIVE_CL20.sh`;
- `RUN_V57_PRELIMINARY_TEST.sh`;
- `bdse/configs/v57_wcdcip_bfar_dbap_train_2gpu.yaml`;
- `bdse/configs/v57_wcdcip_bfar_dbap_cl.yaml`;
- `bdse/configs/v57_wcdcip_bfar_dbap_local_control_cl.yaml`;
- `bdse/configs/v57_wcdcip_bfar_dbap_anchor_control_cl.yaml`;
- `bdse/tools/check_v57_wcdcip_bfar_dbap_gate.py`;
- `bdse/tests/test_v57_wcdcip_bfar.py`.

Core modified modules:

- `bdse/model/losses.py`;
- `bdse/planner/tournament.py`;
- `bdse/planner/nuplan_planner.py`;
- `bdse/metrics/bdse_metrics.py`;
- `bdse/experiments/evaluate_open_loop.py`.

Historical V56 gate and unit-test files are left unchanged so the V56 record remains reproducible.

## 13. Validation and claim boundary

Validation completed:

- Python compile: PASS;
- four V57 YAML files: PASS;
- five V57 shell runners: PASS;
- unit tests: `208 passed, 8 warnings`;
- action-family master-switch regression: PASS;
- direct selected-budget potential gradient: PASS;
- evidence-certificate flip blocking/allowing: PASS;
- protocol/minimum-metrics separation: PASS.

No fresh V57 nuPlan training, open-loop evaluation, non-reactive closed loop, reactive closed loop, or CL100 was run in this environment. No future gate PASS, closed-loop gain, real-time performance, SOTA, or CCF-A-level empirical result is claimed in advance.

---

# V58 CSIP-BFAR-DBAP update — 2026-08-02

Version: **Certified Set-Aligned Integrable-Potential Boundary-Focused Anchor-Residual Decision-Budget Action Preservation**.

## 1. V57 result and gate audit

The uploaded V57 gate report recorded:

- protocol: PASS;
- minimum metrics: PASS;
- formal minimum: PASS;
- competitive: FAIL.

The strict engineering interpretation is narrower. V57's training/pairing subprotocol did pass and the V56 winner-family shutdown was repaired, but the complete dual-certificate protocol was not established because:

- the direct residual action-potential uncertainty used at deployment was never calibrated;
- residual proposal/certificate metrics were contaminated by later structural safety changes;
- paired CL20 crashed before any scenario was simulated;
- the protocol gate did not audit those conditions.

Accordingly, V57 minimum *metrics* remain a genuine PASS, but formal minimum is incomplete when full protocol validity is a prerequisite. Competitive remains a true algorithmic FAIL.

V57 open-loop state:

- candidate/local/foundation teacher match: `0.141/0.141/0.141`;
- candidate/local pair-full match: `0.141/0.141`;
- residual gain: `0.000`;
- pair-full residual gain: `0.000`;
- beneficial/harmful deployed residual: `0/0`;
- evidence certificate: `0.888033`;
- fallback: `0.110`;
- proposal/selected/effective decisive recall: `0.799339/0.607881/0.773545`;
- selected interaction decisive recall: `0.577697`.

Training evidence showed real gradients but no discrete winner learning:

- pair-full winner-margin loss: `10.0749 -> 8.1748`;
- residual winner-correction loss: `7.2338 -> 5.3785`;
- residual uncertainty loss: `3.9812 -> 2.3770`;
- atomwise residual loss remained near `0.012`;
- global action-potential teacher loss remained near `0.363`;
- B16 and pair-full action match remained `0.141`.

## 2. Runtime attribution bug

V57 froze neither the raw anchor nor the raw proposed residual action before `_finalize_pair_anchor_after_structural_guard`. The structural all-flagged guard could overwrite `pair_action_anchor_action`, after which the evaluation layer compared the proposed action with the overwritten anchor.

Of 99 reported residual proposals in 1000 scenes:

- 86 were genuine raw residual proposals rejected by the margin/uncertainty certificate;
- 13 were structural-guard action changes misreported as residual proposals;
- deployed residual flips were zero.

V58 stores immutable raw anchor/proposed actions, raw residual margin, residual sigma, and residual conformal epsilon before structural post-processing. Proposal-conditional pass rates, no-proposal abstention, structural changes, and deployed flips are now distinct metrics.

## 3. Closed-loop callback failure

V57 set:

`main_callback.metric_summary_callback=null`

nuPlan dereferenced the callback config and raised:

`AttributeError: 'NoneType' object has no attribute '_target_'`

Both CL20 shards failed before simulation. V58 keeps a valid callback, optionally removes PDFs only after success, and writes `.closed_loop_complete.json` only when all shard and merge steps complete. Three-way CL20 also requires identical scenario-token hashes and counts.

## 4. Calibration redesign

V57 ran candidate, local, and foundation calibration sequentially, consuming about 116 minutes. The three passes calibrated the legacy pair-atom adverse bound, not the direct residual action-potential margin uncertainty used by V57's flip guard. Candidate and local calibration outputs were nearly identical.

V58 replaces them with one dual-certificate collection:

- `val_calib` is sharded across both GPUs;
- shared evidence adverse scores are collected once;
- candidate proposal-conditional residual margin nonconformity is collected once;
- evidence and residual split-conformal epsilons are merged into one artifact;
- controls reuse only the evidence epsilon and have residual mean, variance, certificate, and residual epsilon disabled;
- train/calibration/runtime uncertainty beta is fixed to `1.0`.

Evidence calibration is applied only to AOCC/adverse certificate fields. It no longer mutates the tournament action rule or its independent epsilon.

## 5. Concurrent open-loop suite

V57 ran candidate, local, and foundation formal open-loop sequentially. V58 adds `bdse.tools.run_parallel_open_loop_suite`:

- all systems enter one bounded CPU/GPU worker pool;
- the same modulo shards are used for every system;
- default concurrency is two workers per GPU;
- scenario/timestamp keys must be nonempty and unique;
- all compared systems must have identical count and SHA-256 before the suite is accepted;
- per-task and suite wall times are persisted.

`BENCHMARK_V58_OPEN_LOOP_CONCURRENCY.sh` compares one, two, and three workers per GPU on a short paired run so machine-specific GPU contention can be measured instead of guessed.

Prediction remains the main V57 open-loop cost (~441 ms of ~586 ms mean latency). V58 does not yet reuse candidate/local prediction in one process; that is a future high-value refactor after the corrected protocol is validated.

## 6. Certified set-aligned winner objective

V57 optimized ordinary winner margins but not the robust margin actually required by deployment. It could therefore reduce soft losses while every proposed correction remained below the certificate threshold.

V58 adds `L_certified_residual_winner`, defined on the primary selected evidence set and aligned with deployment:

`robust_margin = corrected_margin - beta * residual_sigma - residual_epsilon`.

For an anchor-wrong scene, a correction is trained only when the frozen teacher establishes a minimum true winner margin. For an anchor-correct scene, the teacher winner must preserve a robust margin over the strongest valid rival. This creates an explicit do-no-harm term.

A configurable `residual_epsilon_reserve` is included during training because the frozen split-conformal residual epsilon is installed only after checkpoint selection.

## 7. Residual learning-rate groups

The selector and anchor-related heads already have useful behavior, while the zero-initialized residual mean remained sub-threshold. V58 introduces named optimizer groups:

- residual action mean head: `5x` base LR;
- residual action variance head: `2x` base LR;
- proposal/family/selector-related heads: `1x` base LR.

Gradient clipping still receives the complete flat parameter list. A smoke test verifies both LR groups are present and all configured winner/certificate losses execute.

## 8. Competitive checkpoint selection

V57 selected epoch 3 using a score dominated by fixed-budget evidence quality, even though residual and pair-full gains were zero for all checkpoints and proposal recall exceeded 0.80 only at epoch 5.

V58 adds `val_competitive_score`, which explicitly rewards:

- candidate teacher match;
- candidate minus selected-local gain;
- pair-full minus local pair-full gain;
- beneficial minus harmful interventions;
- selected and interaction decisive recall;
- low fallback.

The V58 training config uses this score as the primary checkpoint metric. This does not guarantee a positive residual checkpoint; it prevents a selector-only checkpoint from being preferred without accounting for the paper's winner-correction claim.

## 9. Stricter protocol and gate health checks

The V58 gate audits:

- action-family activation;
- nonzero exact-selector fraction;
- every configured winner/deployment loss individually, including certified winner loss;
- residual uncertainty supervision;
- independent evidence calibration;
- shared evidence epsilon across candidate/local/foundation;
- candidate-only residual calibration and residual-enabled deployment;
- residual-disabled controls with zero residual epsilon;
- train/deploy uncertainty beta agreement;
- exact paired scenario hashes and counts.

It reports protocol, minimum metrics, formal minimum, competitive metrics, and formal competitive separately. A protocol failure blocks closed-loop interpretation.

## 10. Preserved algorithmic components

V58 preserves components supported by V57 evidence:

- immutable base+dense-local foundation anchor;
- boundary-focused winner/hard/near-tie pair curriculum;
- sparse periodic exact AOCC and exact tail;
- fixed B=16 evidence budget;
- direct per-evidence integrable action potential;
- separate evidence and residual certificates;
- same-checkpoint local and matched-foundation controls;
- group-disjoint tune/calibration split;
- paired replay and token hashing;
- shared-model closed-loop execution.

V58 does not restore arbitrary pair fields, Hodge projection, scene-level-only potential supervision, or per-step full exact training.

## 11. Required V58 experiment order

1. Run `RUN_V58_TRAINING_SMOKE.sh` on 1024 train / 256 validation scenes.
2. Require nonzero action-family, exact-selector, deployment-selection, pair-full action, winner correction, certified winner, residual uncertainty, and correctable-scene metrics, plus `5x/2x` LR groups.
3. Run eight-epoch training.
4. Run the two-GPU shared dual-certificate calibration.
5. Run simultaneous candidate/local/foundation 1000-scene open-loop with bounded concurrency.
6. Require strict protocol PASS before interpreting minimum, competitive, or closed-loop results.
7. Run paired NR-CL20, then frozen paired R-CL20.
8. Run CL100 only after open-loop residual gain is positive, beneficial exceeds harmful, and CL20 has no safety regression.

If V58 still has zero pair-full residual gain while certified supervision is active, the next algorithmic branch should be a zero-initialized set-conditioned interaction potential head. Threshold relaxation or unconditional residual scale inflation should not be the next action.

## 12. New and changed files

New:

- `V57_RESULT_ANALYSIS_AND_V58_OPTIMIZATION.md`;
- `V57_RESULT_DIAGNOSIS_FOR_V58.json`;
- `NEXT_COMMANDS_V58_CSIP_BFAR.txt`;
- `RUN_V58_TRAINING_SMOKE.sh`;
- `BENCHMARK_V58_OPEN_LOOP_CONCURRENCY.sh`;
- `V58_CSIP_BFAR_DBAP_NEXT_COMMANDS.sh`;
- `run_v58_csip_bfar_dbap.sh`;
- `RUN_V58_REACTIVE_CL20.sh`;
- `bdse/configs/v58_csip_bfar_dbap_train_2gpu.yaml`;
- `bdse/configs/v58_csip_bfar_dbap_cl.yaml`;
- `bdse/configs/v58_csip_bfar_dbap_local_control_cl.yaml`;
- `bdse/configs/v58_csip_bfar_dbap_anchor_control_cl.yaml`;
- `bdse/tools/calibrate_v58_dual_certificates.py`;
- `bdse/tools/apply_v58_dual_calibration.py`;
- `bdse/tools/run_parallel_open_loop_suite.py`;
- `bdse/tools/check_v58_csip_bfar_dbap_gate.py`;
- `bdse/tests/test_v58_csip_bfar.py`.

Core modified:

- `bdse/model/losses.py`;
- `bdse/experiments/train.py`;
- `bdse/planner/tournament.py`;
- `bdse/planner/nuplan_planner.py`;
- `bdse/metrics/bdse_metrics.py`.

## 13. Validation and claim boundary

Completed in the analysis environment:

- Python compile: PASS;
- four V58 YAML files: PASS;
- shell syntax: PASS;
- unit tests: `215 passed, 8 warnings`;
- raw residual attribution regression: PASS;
- certified robust-winner loss regression: PASS;
- dual-calibration application regression: PASS;
- LR parameter-group regression: PASS;
- strict gate audit regression: PASS.

No fresh V58 nuPlan training, calibration, open-loop, non-reactive closed loop, reactive closed loop, or CL100 was executed here. No future gate PASS, closed-loop improvement, real-time claim, SOTA claim, or CCF-A-level empirical result is asserted in advance.

---

# V59 FSCIP-BFAR-DBAP — V58结果诊断、闭环工程修复与集合条件残差（2026-08-03）

## 1. V58冻结结果

V58 原始 open-loop gate 报告：

- Protocol gate：PASS；
- Minimum metrics / formal minimum gate：PASS；
- Competitive metrics / formal competitive gate：FAIL。

V58 在 1000 个配对场景上保持：candidate/local/foundation teacher match 均为 `0.141`，pair-full candidate/local 均为 `0.141`，最终 residual gain、pair-full residual gain、beneficial/harmful 均为零。Competitive FAIL 是真实算法失败，而不是阈值边缘失败。

V58 的正向证据集中在 evidence path：proposal decisive recall `0.80247`、selected decisive recall `0.61065`、effective decisive recall `0.77631`、interaction decisive recall `0.57934`、evidence certificate `0.88803`、fallback `0.110`。因此 V59 不回退 selector/AOCC 主线。

## 2. V58闭环结果不可用

V58 candidate CL20 两个 shard 都完成了长时间仿真，但在 `SimulationLogCallback` 序列化 planner 时触发：

```text
TypeError: cannot pickle '_thread.RLock' object
```

两个 shard 最终均报告 `successful=0, failed=10`。旧脚本仍生成 combined summary 和 `.closed_loop_complete.json`，所以这些文件不能作为闭环结果。Local control 仅开始运行，三路 paired CL20 并未完成。

V59 修复：

- `BDSEPlannerCore.__getstate__/__setstate__` 删除进程本地 RLock；
- 默认删除 `callback.simulation_log_callback`，保留 metric main callbacks；
- 合并前强制逐 shard 检查成功数等于 token 数且失败数为零；
- 只有全部 shard 验证通过后才写 `.closed_loop_complete.json`。

## 3. V58耗时根因

流水线近似耗时：anchor gate 13 分钟、8 epoch 训练 2 小时 40 分钟、双证书 calibration 25 分钟、三路并发 open-loop 9.8 分钟。正式 open-loop 不是主要瓶颈。

Candidate CL20 从 gate 后运行约 3.1–3.6 小时仍以 20/20 失败结束。profile 显示 80% planner call 命中 plan cache，但所有调用平均 core plan 仍为 25.9–29.4 秒，其中 certificate stages 14.5–16.7 秒、final safety 4.76–4.80 秒。非缓存 replanning 的隐含成本约为每次 2 分钟以上。

根因：

1. 同一 replanning 的 route distance、agent envelope、TTC、hard/soft risk 在模型、selector、多级 fallback、rule rerank 和 final diagnostic 中重复计算；
2. 每个 fallback B/M/L stage 重复构建 batch 和执行相同 scene encoder；
3. 4 个 simulation threads 共用一份 GPU 模型与 RLock，CPU 几何工作也形成锁队列；
4. 仿真结束后序列化错误使前面的全部计算失效。

V59 增加 per-planner-call runtime safety memo、scene/context memo，并改为每 GPU 多进程、每进程一个 simulation worker。并发数必须通过 CL4 benchmark 在目标机器上选择，默认 2 process/GPU，不预先声称固定加速倍数。

## 4. V58 residual calibration问题

V58 residual calibration 只在实际 residual proposal 场景上收集分数：5000 个 calibration scenes 仅 90 个 proposal，得到 `epsilon=0.46431`，远大于训练 reserve `0.05`。1000-scene open-loop 中 raw proposal 约 1.2%，conditional residual certificate pass 为 0，所有 residual proposal 被拒绝。

V59 将 residual conformal score 改为 scene-uniform all-rival score：每个场景固定 selected-local anchor，对所有 valid rivals 计算单侧 margin error，并取该场景最坏 rival 作为一个 exchangeable calibration score。actual-proposal calibration 仅保留为 diagnostic。

第二遍工程审计发现并修复一个新实现中的键名错误：runtime 输出为 `residual_action_var`，校准器曾错误读取 `residual_action_variance`。V59 现在使用严格 helper，缺少正确 key、动作维度不一致或 sigma matrix 不完整时立即失败，禁止静默零方差校准。

## 5. V58算法判断

有效并保留：

- immutable foundation anchor；
- fixed B=16 evidence budget；
- boundary pair curriculum；
- sparse exact AOCC；
- direct integrable action potential；
- evidence/residual dual-certificate separation；
- same-checkpoint local 与 matched foundation controls；
- group-disjoint tune/calibration；
- paired open-loop concurrency 与 token hashing。

不足或无效：

- 独立 per-evidence additive residual 没有产生最终 winner correction；
- global action-potential reconstruction loss 从约 `0.372` 恶化至 `0.383`；
- atomwise residual loss 约 `0.012` 基本不变；
- certified winner loss 从约 `3.09` 降到 `2.86` 后停滞，robust winner boundary 未被跨越；
- proposal-conditional calibration 太稀疏；
- V58 checkpoint score 在所有 residual gain 为零时仍可由 selector 指标主导。

## 6. V59算法：Focused Set-Conditioned Integrable Potential

V59 在不增加 queried evidence budget 的前提下加入低秩 selected-set interaction potential：

```text
h_S(a) = sum_{i in S} h_i(a)
       + < psi(a), tanh( sum_{i in S} phi(i) / sqrt(|S|) ) > / sqrt(r)
```

其中 `r=8`。该结构直接输出 action scalar potential，诱导的 pair margins 天然 antisymmetric/cycle-consistent；它表达 additive per-evidence head 无法表达的 evidence-set interaction，同时保持论文的 fixed-budget novelty。

新增 `L_residual_boundary_margin_distill`，只在 selected-local anchor 错误且 teacher margin 足够大的 correctable scenes 上，直接拟合 teacher winner 相对当前 anchor 的 margin。V59 降低全局 reconstruction/atomwise loss 权重，提高 certified winner 与 boundary margin 权重，避免平均重构淹没最终 winner 信号。

Residual mean 与 set heads 使用 5x LR，variance head 使用 2x LR。Warm start 时 additive residual 和 set atom factor 为零，set action factor小随机初始化，保证 step 0 仍为 anchor no-op，同时允许 set head获得梯度。

## 7. V59 checkpoint与gate

Competitive checkpoint score 新增：robust margin、proposal rate，并在 residual gain 与 pair-full gain 同时非正时施加大额 penalty，防止 selector-only checkpoint 被选为 paper main checkpoint。

V59 protocol gate 新增：

- `L_residual_boundary_margin_distill` 必须实际执行；
- scene-uniform residual calibration 必须独立、覆盖至少 1000 scenes、覆盖率至少 80%；
- residual calibration epsilon、uncertainty beta 和 runtime config 必须一致；
- controls 必须关闭 additive/set residual 与 residual epsilon；
- candidate/local/foundation paired hashes、counts 和 frozen anchor rows 必须一致。

## 8. V59推荐实验顺序

1. Smoke test；要求新 boundary loss、set-head LR groups、exact selector、winner/certificate losses 非零。
2. Fresh 训练、scene-uniform calibration、三路并发 1000-scene open-loop；默认不自动运行 closed loop。
3. 先看 pair-full residual gain：若仍为零，问题仍在 residual expressivity/optimization；不得放宽 certificate。
4. 若 pair-full gain > 0 但 B16 gain = 0，强化 selector 与 set-conditioned winner coupling。
5. 若 raw proposal 有益但全部被 calibration 拒绝，检查 uniform epsilon、raw error 和 sigma；不得使用 test 调参。
6. Open-loop competitive gain 转正后运行 CL4 concurrency benchmark，再运行 paired NR-CL20 与 frozen R-CL20。
7. 只有 beneficial > harmful 且安全指标无退化后运行 CL100。

## 9. V59工程文件

新增：

- `bdse/configs/v59_fscip_bfar_dbap_{train_2gpu,cl,local_control_cl,anchor_control_cl}.yaml`；
- `bdse/tools/calibrate_v59_dual_certificates.py`；
- `bdse/tools/apply_v59_dual_calibration.py`；
- `bdse/tools/check_v59_fscip_bfar_dbap_gate.py`；
- `bdse/tests/test_v59_fscip_bfar.py`；
- `run_v59_fscip_bfar_dbap.sh`；
- `V59_FSCIP_BFAR_DBAP_NEXT_COMMANDS.sh`；
- `RUN_V59_TRAINING_SMOKE.sh`；
- `RUN_V59_REACTIVE_CL20.sh`；
- `BENCHMARK_V59_OPEN_LOOP_CONCURRENCY.sh`；
- `BENCHMARK_V59_CLOSED_LOOP_CONCURRENCY.sh`。

核心修改：

- `bdse/model/bdse_model.py`；
- `bdse/model/losses.py`；
- `bdse/planner/tournament.py`；
- `bdse/planner/fallback.py`；
- `bdse/planner/nuplan_planner.py`；
- `bdse/experiments/train.py`。

## 10. 验证边界

本地静态/单元验证通过；未在当前环境执行 fresh V59 训练、calibration、open-loop 或 nuPlan closed-loop。V59 不预先声明三个 gate PASS、闭环提升、实时性、SOTA 或 CCF-A录用级结果。

# V60 — Dense-Winner-Aligned Policy-Calibrated Set-Potential BFAR-DBAP

## 诊断来源

- V59 Protocol PASS；Minimum PASS；Competitive FAIL。
- dense full-interface teacher match 0.359，但 sparse Top-M full match 0.141。
- B=16 对 sparse-full winner 保持率 0.981，说明主要瓶颈位于 dense evidence 到 Top-M proposal。
- V59 pair-full open-loop/validation 漏传 set-conditioned factors，导致 set head 未被正确评估和用于 checkpoint selection。
- V59 all-rival residual calibration epsilon=3.3076，远大于训练 reserve=0.15。

## 保留设计

- fixed B budget、exact AOCC、boundary curriculum、direct integrable potential、dual certificate、same-checkpoint local control、paired foundation control。

## 新增算法

1. `L_proposal_dense_winner`：hard-forward straight-through Top-M 保持 dense-local winner和 strongest-rival margin。
2. teacher-aligned dense scene weighting。
3. proposal-first residual curriculum：2 epoch 低 residual scale，4 epoch ramp。
4. set atom/action factor双侧小随机安全初始化。
5. policy-selected-top-rival split-conformal residual calibration。
6. conformal-only residual certificate，`residual_beta_uncertainty=0`。

## 工程修复

1. open-loop pair-full 传入 `residual_set_atom_factors` 和 `residual_set_action_factors`。
2. validation pair-full 同步修复，checkpoint score可评价 set head。
3. query diagnostics导出 `set_conditioned_residual_*`。
4. checkpoint competitive/fixed-budget score加入 sparse-full、budget-vs-full 和 dense proposal drop。
5. gate要求 proposal dense-winner loss实际执行。
6. 新增 strict budget baseline sweep 工具，验证所有系统 scenario/timestamp hash一致。

## 禁止重复尝试

- 不通过降低 residual certificate 阈值制造 flip。
- 不重新启用 arbitrary pair field/Hodge projection。
- 不继续单独堆高 B=16 selector loss而忽略 Top-M proposal bottleneck。
- 不用不同 loss 权重下的 external adapter val loss直接做模型排名。

## V60-EXT1 — Matched external baseline audit, acceleration and paired evaluation

Scope: external baseline adapters and comparison infrastructure only; no change to the V60 DWAPC-BFAR-DBAP algorithm.

- Reclassified GameFormer, DTPP, PlanTF and PLUTO implementations as `-inspired budget adapters`; reclassified the rule baseline as `PDM-Closed-style budget scorer`.
- Added explicit paper/source/fidelity metadata to code and YAML.
- Replaced per-batch/per-evidence GPU-to-CPU budget selection with vectorized unit-cost top-k and an on-device variable-cost fallback.
- Vectorized selected evidence gathering and added correct Transformer/MultiheadAttention padding masks.
- Fixed runtime fallback budget propagation: stage-specific B now changes the evidence tokens consumed by the external model instead of only changing query accounting.
- Made DTPP `tree_depth` operational and added intermediate-stage deep supervision; added GameFormer level deep supervision.
- Added deterministic matched-data manifests, common seed/protocol metadata and exact train/validation path hashes to checkpoints.
- Added strict external checkpoint loading; wrong variants, missing tensors or shape mismatches now fail instead of silently evaluating random parameters.
- Reduced per-step training synchronization; added fused AdamW, TF32, AMP, prefetching, warmup/cosine scheduling and early stopping.
- Standardized output names to `outputs/external/{gameformer,dtpp,plantf,pluto}_budgeted.best.pt`, matching the V60 SWEEP loader.
- Added two-GPU paired training, paired open-loop comparison, all-metric budget sweep, deterministic CL20/CL50 comparison, CL concurrency benchmark and integrity-checked completion markers.
- Closed-loop now runs two systems concurrently on two GPUs, one worker per process, with optional multiple process copies per model after benchmarking.

# V61 — Deployment-Exact Hierarchical Winner Preservation (DE-HWPP) BFAR-DBAP

## 1. V60 结果复核

V60 官方 gate report 为：Protocol PASS、Minimum FAIL、Competitive FAIL。Minimum 的唯一失败项是 proposal decisive recall `0.70011 < 0.72`。Competitive 同时失败于 candidate-local / candidate-foundation teacher-match gain、pair-full residual gain、净有益 residual intervention、proposal recall 与 selected recall。

严格的论文级可归因复核不再把 V60 Protocol 视为充分：candidate 配置启用了 rank-8 set-conditioned residual 与 `evidence_action_potential`，但 1000 条 candidate JSONL 中 `set_conditioned_residual_active/rank/abs_mean/scale` 覆盖率均为 0。因此，最终 winner 没有改善这一事实可信，但不能从现有结果归因 set head 是否真正进入了实际评测二进制/路径。

## 2. V60 dense-winner proposal 目标没有解决旧瓶颈

V60 训练日志中的 `proposal_dense_topm_match≈0.965` 不是运行时 HAB Top-M 指标，而是一个 global Top-M proxy。真实验证始终为：

```text
dense full-interface teacher match = 0.359
sparse-full teacher match          = 0.141
B16 teacher match                  = 0.141
budget-vs-dense winner match       = 0.172
budget-vs-sparse winner match      = 0.981
```

V60 实现存在两个根本问题：

1. straight-through threshold 在未约束 proposal logit 空间中 detach，允许所有 proposal logits 同时上移的无效优化方向；
2. loss hard-forward 使用 global Top-M，但部署使用 HAB family slots、family score、soft interaction reserve、group diversity 与 structural safety bypass。

训练的 `L_prop` 从 `15.09` 增长到 `1452.33`（96.2x），`L_deploy_select` 从 `6.45` 增长到 `833.63`，而真实 proposal recall 从 epoch 1 的 `0.7260` 降到 epoch 7 的 `0.6439`。因此 V60 不仅没有保留 dense winner，还在后期持续破坏 proposal 分支。

## 3. V61 核心算法

### 3.1 Deployment-exact hierarchical winner preservation

新增部署一致的 dense-winner proposal 路径：

- 所有训练场景使用 GPU HAB forward：family slot allocation、family-conditioned atom acquisition、soft interaction reservation、structural evidence exclusion/refill；
- 旋转抽样场景使用与运行时完全相同的 NumPy HAB Top-M hard mask；
- exact hard-forward 与 fast GPU hard-forward 共用 translation-invariant、family-conditioned straight-through surrogate；
- global Top-M 只保留为诊断，不再作为算法成功证据；
- 部署 evidence budget、proposal M 与 query budget均不增加。

新增训练诊断：

```text
proposal_fast_hab_topm_match
proposal_global_topm_match
proposal_exact_hab_topm_match
proposal_exact_hab_fraction
proposal_fast_exact_mask_jaccard
proposal_logit_abs_mean
proposal_logit_rms_mean
L_proposal_logit_stability
```

### 3.2 Stable proposal surrogate

straight-through Top-M 在 active atoms 上先中心化，去除 uniform-logit null direction；soft mask归一化到 M 个 atoms 的总质量；forward 始终为 hard mask。新增 logit center/RMS regularization，gate 对 RMS runaway 与 proposal loss runaway 直接判 Protocol FAIL。

### 3.3 Stage-decoupled residual routing

对 selected sparse anchor 选错的场景分两类：

- dense local 已选对 teacher：proposal bridge failure；residual winner/certificate/boundary loss权重降为 `0.1`；
- dense local 仍选错 teacher：intrinsic residual correction；保留 `1.0` 权重。

这避免 residual 被迫补偿 proposal 丢失的 evidence，并新增：

```text
residual_proposal_failure_scene_fraction
residual_intrinsic_correction_scene_fraction
```

### 3.4 Gate-feasible checkpoint selection

V60 epoch 1 的 proposal recall `0.7260` 已满足 Minimum，但 competitive score 选择了 recall `0.7001` 的 epoch 3。V61 checkpoint score 对正式 Minimum gate 的 shortfall 加入高权重惩罚，并记录 `val_minimum_gate_feasible`。任何 gate-feasible checkpoint 都优先于 gate-infeasible checkpoint，competitive score只在同一可行层内排序。

### 3.5 Strict result provenance

V61 gate 新增：

- candidate JSONL 中 set-conditioned residual rank/activation/amplitude/scale 必须与 candidate config 一致，覆盖率至少 99%；
- exact runtime HAB 必须在训练中实际抽样；
- fast/exact/global HAB 指标和 proposal logit稳定性必须完整；
- proposal loss runaway、proposal logit RMS runaway直接判 Protocol FAIL；
- 启用 residual stage routing 时必须导出两类场景比例。

## 4. 保留、升级与暂缓

继续保留：immutable foundation anchor、fixed planner-interface budget、HAB、exact AOCC、boundary curriculum、direct integrable action potential、dual certificates、same-checkpoint local control、paired foundation control、group-disjoint calibration、paired scenario/timestamp evaluation。

升级：proposal supervision 从 atom recall/global Top-M 升级为 deployment-HAB winner preservation；checkpoint selection 升级为 gate-feasible lexicographic selection；set potential 升级为强制 end-to-end observability；residual training 升级为 proposal/intrinsic error routing。

暂不放大：set-conditioned residual、learned residual uncertainty 与 policy calibration可保留，但 V60 没有净有益 winner 证据。必须先修复 proposal bridge，再判断 residual expressivity。不得通过降低 certificate 阈值制造 flip。

## 5. 新的禁止重复尝试

- 不再用 global Top-M train metric 代替 runtime HAB Top-M。
- 不再对 uncentered proposal logits 使用 detached threshold straight-through。
- 不再允许 checkpoint score选择 formal Minimum gate 已失败的 epoch。
- 不再把“代码单元测试能导出字段”当作“实际实验 JSONL 已使用该路径”的证据。
- 不让 residual 主要学习 dense-correct/sparse-wrong 的 proposal failure 场景。
- test set 未构建完成且存在负 full-interface teacher regret 等诊断异常，在修复前禁止用 test 调参、报主结果或做模型选择。
- open-loop Minimum 与 winner-level信号未转正前不运行 CL100；默认不在 gate fail 后自动跑 diagnostic CL20。

## 6. V61 运行判断顺序

1. smoke：exact HAB fraction > 0，fast/exact Jaccard 可观测，proposal RMS < 20，proposal loss无爆炸；
2. epoch 1/3：`val_proposal_decisive_atom_recall >= 0.72`，且不随训练单调下降；
3. `sparse_full_interface_action_match > 0.141`、`budget_vs_full_match > 0.172`；
4. `budget_vs_sparse_full_match` 保持高位，确认 B16 selector未退化；
5. raw residual proposal 中 teacher-directed beneficial > harmful；
6. set diagnostics覆盖率 100%，pair-full gain转正；
7. candidate-local teacher match gain转正后，才运行 paired CL20；Minimum + Competitive 通过后再运行 CL100。

## 7. 验证边界

V61 本地验证完成：Python compile、4 个 V61 YAML、3 个 shell 脚本语法、231 个单元测试、补丁 dry-run 与 ZIP 完整性。当前环境未执行 fresh V61 训练、calibration、open-loop 或 nuPlan closed-loop，因此不预先声明 V61 gate PASS、闭环提升、实时性或 CCF-A 级性能。

# V62 — Deployment-Complete Action Bridge + Exact Winner-Flip Criticality BFAR-DBAP (DCAB-EWFC)

## 1. V61 结果复核后的状态修正

V61 gate 不能继续写成“三个算法 gate 全部失败”：

- Protocol：FAIL 来自 `set_conditioned_residual_*` 在 metrics 末端被过滤，属于结果可观测性/导出错误；现有旧 JSONL 不能追认 PASS，必须 fresh rerun。
- Minimum：`minimum_metrics_pass=true` 且 failures 为空；official `minimum_pass=false` 只是被 Protocol 串联阻断。proposal decisive recall=0.803054，selected=0.611521。
- Competitive：真实 FAIL。candidate/local/foundation teacher match=0.141，total/residual/pair-full gain 都为 0。

V61 已经消除 V60 proposal-logit runaway：L_prop 稳定下降、proposal RMS<2、fast/exact HAB mask Jaccard=1、每次 validation minimum feasible。禁止再次退回 global Top-M hard forward、未中心化 threshold，或仅通过放大 proposal loss重做 V60 路线。

## 2. 新发现的主要上游缺陷：action-query bridge

V61 训练的 deployment-HAB winner preservation 只对 evidence atom 维度做 HAB mask，但对全部 actions 使用 dense `g(i,a)`；真实 runtime 只查询 rival graph 中 actions，其他 action contribution 置零。训练指标和部署路径不等价：

```text
training: HAB Top-M atoms × all actions
runtime:  HAB Top-M atoms × rival-graph actions
```

V61 的 dense full match=0.359、runtime sparse-full=0.141、B16 vs sparse-full=0.981、B16 vs dense=0.172。当前 selector 对给定 sparse interface 基本忠实。禁止在 action-query bridge 未验证前把 0.359→0.141 全部归因于 potential/residual。

## 3. V62 算法变更

### 3.1 Deployment-complete action query

主配置新增 `runtime.action_query_mode: all_valid`。对固定 B 个 queried evidence atoms，向固定 candidate bank 的全部 valid actions 计算 contribution，查询上界为 `B*K`（B=16，K<=32）。evidence atom budget 不变，不增加 atom、不使用 dense oracle、不绕过 selector。保留 `rival_graph` 同 checkpoint ablation。

新增诊断：`action_query_mode_all_valid`、`valid_action_count`、`queried_valid_action_fraction`、`hab_topm_dense_value_action_match`、`hab_topm_dense_value_vs_runtime_sparse_full_match`、`runtime_sparse_value_bridge_flip_rate`、`selected_budget_dense_value_action_match`、`selected_budget_dense_value_vs_deployed_match`。

### 3.2 Exact winner-flip critical evidence

新增 literal leave-one-atom-out label：atom i 仅在移除后 dense winner action 改变时为 critical。severity 只用于 critical atoms 内排序，不能把“margin 改变但 winner 不变”标成 critical。

新增 `L_exact_winner_flip_critical_proposal`、Top-M/selected critical recall、critical atom/scene fraction、teacher-aligned scene fraction。主权重为 8.0；dense-winner proposal 权重从 24 调整为 20，避免总目标无控制增长。

### 3.3 Residual routing 不变，归因顺序改变

继续保留 proposal-failure residual weight=0.1、intrinsic correction=1.0。先验证 all-valid action bridge；只有 sparse-full 恢复而 pair-full/residual 仍失败时，才升级 potential projection 或 residual target。禁止直接关闭 conformal epsilon、扩大 residual scale 或降低 flip margin制造未经 teacher direction 验证的 flips。

## 4. 工程与效率修复

- metrics 导出 `set_conditioned_residual_*`、`pair_potential_*`、`direct_evidence_action_potential_*`。
- gate 同时报告 metrics pass 与 official protocol-blocked pass。
- runner 接受 `BDSE_VAL_CACHE -> BDSE_SPLIT_CACHE -> BDSE_VAL_CACHE_ORIGINAL` fallback；主 pipeline 仍显式 export。
- signed scalar delta 与 nonnegative regret 分离。
- 顶层 action loss=0 时跳过 CPU deployment certificate mask。
- local uncertainty 关闭时跳过 local variance head。
- 修复极端 cost range 下 invalid-action sentinel。
- literal criticality 使用 `[B,E,K]` 张量化 LOO，无 Python per-atom loop。

## 5. 论文表述同步

核心 novelty 保留：固定 planner-interface evidence budget、可审计 evidence atoms、预算内确定性 selector、literal winner-flip criticality、双证书。正文新增 fixed `B*K` action expansion。

“exact selector”仅指对论文定义的 deterministic fixed-budget AOCC operator 精确执行/审计；当前 acquisition order 是 greedy/anytime，不宣称求解全局 combinatorial optimum。

## 6. V62 决策门

1. Protocol：set 字段 coverage>=0.99；all-valid mode=1；queried valid fraction>=0.99。
2. Minimum metrics：proposal recall>=0.72；与 official protocol blocking 分开报告。
3. Bridge：HAB dense-value vs runtime sparse-full match>=0.95，bridge flip<=0.05。
4. Criticality：Top-M recall>=0.80，selected>=0.50，同时报告 critical scene fraction。
5. Winner：sparse-full/pair-full/candidate match 必须超过 V61 0.141，budget-vs-sparse 保持高位。
6. Residual：raw teacher-directed proposal出现且 beneficial>harmful，再分析 calibration。
7. 先 paired CL20；Competitive PASS + CL20无安全退化后再 CL100。

## 7. 不重复尝试清单

- 不再使用 global Top-M 替代部署 HAB 作为主要训练成功指标。
- 不再使用未中心化 logits + detached threshold 的可平移 proposal surrogate。
- 不再按 Competitive score 选择 Minimum-infeasible checkpoint。
- 不再用 incomplete test 调参、选 checkpoint 或选择版本。
- 不在字段 coverage=0 时宣称某 head 生效或无效。
- 不在 action-query bridge 未对齐时让 residual 补偿 upstream missing values。
- 不通过增加 evidence atom budget 换结果。

## 8. 验证边界

V62 已完成 Python compile、YAML parse、shell syntax、targeted/full unit tests、TeX build 和合成 microbenchmark。当前环境未执行 fresh V62 train/calibration/open-loop/closed-loop，因此不声明 gate PASS、闭环提升或 SOTA。


# V62.1 Engineering Hotfix — Dense Mask Alignment, Query Accounting, and Evaluation Efficiency

## 1. Nature of this update

This is an engineering/measurement hotfix, not a new algorithm version.  It does not
change the BDSE objective, architecture, evidence budget, HAB selector, winner-flip
labels, tournament, calibration thresholds, or fallback policy.

## 2. Fixed issues

- Dense diagnostics use configured padded `[E_max,K]`, while cached evidence masks may
  store only actual scene atoms.  Explicit zero-padding/truncation now preserves padded
  semantics and fixes the `(128,32) * (48,1)` crash.
- Literal LOO criticality uses `np.where` masking so malformed inactive/padded NaNs
  cannot leak through `0 * NaN`.
- V62 all-valid bridge query accounting now reports selected-stage `B*K_valid` rather
  than incorrectly preferring `B*|rival_pairs|`.
- Dense evaluation reuses the certificate-stage encoded context and skips unused full
  forward heads.
- Open-loop metric/JSONL aggregation is streaming rather than retaining all scene
  objects in memory.
- Redundant evaluator-wide CUDA synchronizations were removed; NumPy-returning model
  adapters already block on relevant CUDA-to-CPU transfers.
- `deterministic_order` now handles one-shot iterables correctly.

## 3. New non-regression constraints

- Never repair an actual-vs-padded atom mismatch by cropping dense predictions; align
  masks and preserve the configured interface dimensions.
- For action-conditioned all-valid deployment, paper-facing selected query count is
  `B*K_valid`; rival-pair count is a separate tournament diagnostic.
- Dense diagnostic optimization is acceptable only when `J0`, `g`, and `g_var` are
  numerically equivalent to the full-forward reference under active/valid masks.
- Do not change Transformer `norm_first` merely to suppress the nested-tensor warning;
  that would alter model/checkpoint semantics.

## 4. Validation boundary

Python compile, 315 YAML files, 41 shell scripts, V62-specific regression tests, and
the full 240-test suite pass.  A reduced CPU combined certificate+dense benchmark
improved by 12.70% on average.  Fresh GPU training/calibration/open-loop/closed-loop
has not been executed in this environment, so no model-performance improvement is
claimed from this hotfix alone.


# V63 — Deployment-Consistent Query Contract + Teacher-Flip Criticality Ranking BFAR-DBAP (DCQC-TFCR)

## 1. V62 uploaded-result status correction

The uploaded V62 package does not contain completed dual calibration, candidate/local/foundation open-loop outputs, or `v62_dcab_ewfc_gate_report.json`. The pipeline logs stop after checkpoint reuse and the two calibration workers terminate around 46--47% without raw shard outputs. Therefore:

- Protocol: **not evaluated**, not PASS/FAIL;
- Minimum: **not officially evaluated**; train-time feasibility is only a positive proxy;
- Competitive: **not officially evaluated**, with strong failure warning from the 0.141 teacher-match plateau and zero residual gain.

The V53 factorized-anchor replay in the V62 output directory is an immutable foundation control only. It must never be reported as a V62 candidate gate result.

## 2. Engineering confound discovered in the V62 bridge metric

The old dense-to-sparse comparison mixed different interfaces:

- dense/training query features could come from a shape-compatible cached tensor;
- sparse runtime query features were recomputed by the current canonical implementation;
- dense used learned `J0_model`;
- deployment used `J0_model` plus runtime base prior and structural safety residual prior.

Thus `budget_vs_full_match=0.172` combined query-cache drift, base/prior drift, actual sparse-value drift, and selection loss. Since `budget_vs_sparse_full_match=0.981`, the B=16 selector is not the first component to modify.

## 3. V63 algorithm update

### 3.1 Deployment-consistent query/base contract

- Add `runtime.dense_query_feature_source` with safe `runtime_recompute`, debug `cache_verified`, audited `cache`, and legacy fallback modes.
- Main V63 uses `runtime_recompute`.
- `predict_dense_numpy` now returns both `J0_model` and deployment-consistent `J0_deployment`, applying the exact same base and structural priors as sparse runtime.
- Open-loop exports direct MAE/max/allclose/pass checks for deployment base values and queried atom-action values.
- Protocol gate requires these numerical contracts before any bridge/selector conclusion is valid.

### 3.2 Teacher-interface literal winner-flip criticality

The primary target is now:

```text
critical_T(i) = 1[removing teacher atom i changes the teacher scalar winner]
```

Training excludes scalar/lexicographic teacher-winner mismatch scenes, preserves literal leave-one-out semantics, and adds hardest-negative pairwise proposal-logit ranking. Model-self criticality remains available only as an ablation. Severity ranks already-critical atoms and cannot create a critical label.

### 3.3 Fixed-budget semantics and honest compute accounting

The retained planner-interface evidence budget remains B=16. V63 reports separately:

- acquisition pool M=24;
- action scores for acquisition M*K;
- retained certificate payload B*K;
- pair-conditioned query scores.

Never describe B*K as total internal acquisition compute. Never increase B to hide proposal or bridge failure.

### 3.4 Layered attribution metrics

Report transitions separately:

1. learned model base -> deployment base;
2. deployment dense full -> HAB Top-M dense value;
3. HAB Top-M dense value -> runtime sparse value;
4. deployment dense full -> B16 selected dense value;
5. selected dense value -> deployed action;
6. local same-checkpoint -> residual candidate.

Teacher exact critical Top-M/selected recall, scene rate, and scalar alignment are exported. If the frozen suite contains fewer than approximately 20 critical scenes, gate output warns that recall is high variance.

## 4. Pipeline and efficiency update

- Full-pipeline detach and OUT_ROOT lock.
- Reuse fresh calibration shards; launch only missing shards.
- Preserve worker failure markers and tail failed logs.
- Atomic calibration merge.
- Same-checkpoint V62 contract attribution script with nominal/no-base/no-structural/no-runtime-priors controls and freshness reuse.
- Optional cached-query speed path requires a PASS audit plus matching code/config fingerprints.
- Keep pair-head scoring disabled, local variance disabled when unused, vectorized LOO criticality, and B/K unchanged.

## 5. V63 gate order

1. Protocol numerical contract, provenance, all-valid coverage, fixed retained B;
2. Minimum completeness metrics;
3. Competitive teacher-match gains, paired regret, net-beneficial residual flips, bridge match, and teacher-critical recall;
4. paired CL20 after Protocol PASS;
5. CL100 only after Minimum + Competitive PASS and no CL20 safety regression;
6. completed frozen test exactly once after readiness audit.

## 6. New do-not-repeat list

- Do not interpret a missing gate report from an interrupted pipeline as gate FAIL.
- Do not use a V53 anchor replay as a V62/V63 candidate result.
- Do not diagnose selector quality until base/query numerical contracts pass.
- Do not use global Top-M as the deployment-success metric.
- Do not use model-self criticality as the primary target.
- Do not relax residual calibration to create flips before raw teacher-directed proposals are net beneficial.
- Do not increase B or restore the expensive pair head to compensate for upstream contract failure.
- Do not use the incomplete test set for tuning or model/version selection.

## 7. Validation boundary

V63 local validation: Python compile PASS, 9 V63 YAML files PASS, 3 shell entrypoints PASS, full test suite **245 passed / 0 failed**. Fresh GPU training, calibration, open-loop, and closed-loop have not been executed, so no gate or performance improvement is claimed yet.


# V64 — Support-Aware Query Adapter + Budgeted Critical Coverage BFAR-DBAP (SAQA-BCC)

## 1. V63 gate 状态纠正

上传的 V63 runtime-recompute pipeline 在 immutable anchor replay 后停止；cached-query pipeline 在 cache audit 后停止。两条路径都没有完成 V63 training、dual calibration、candidate/local/foundation paired open-loop 或 gate checker。因此 Protocol=NOT EVALUATED，Minimum/Competitive=NOT OFFICIALLY EVALUATED。禁止把缺失 report 当作算法 FAIL，禁止用 V53/V62 anchor replay 代替 V63 candidate result。

## 2. V63 工程混杂

- 历史 query cache 为 12-D；V63 直接生成 18-D，并把新 6 个非零通道送入旧 checkpoint 的 `query_proj`。V62 虽声明 18-D，但缓存尾 6 维始终为零，因此旧 projection 的对应 columns 没有得到数据支持。相同场景、相同 checkpoint 的 foundation action match 从 V62 0.359 降到 V63 0.170。
- V63 cache audit 抽样 512、实际比较 0、shape failure 512；这是 audit contract 错误，不是缓存数值已经失败。
- nominal base contract 与 base allclose 均为 1.0。旧 query gate 比较不同 CUDA batch shape 的 neural `g`，MAE=1.59e-4、max=7.93e-4，却使用 1e-5 max-error；同时 Top-M dense 与 sparse-full winner match=1.0。禁止再把该单项指标解释为 raw query semantic failure。
- 部分 checkpoint loader 静默丢弃 shape-mismatched tensors。核心 foundation/query tensor 不允许部分初始化。
- V64 配置元数据必须使用 V64 experiment/algorithm 名称，防止 provenance 混淆。

## 3. Support-Aware Query Adapter

- 固定 `query_legacy_support_dim=12`。
- 旧 `query_proj` 仅接收 12-D prefix，尾部补零，以保持旧 checkpoint step-zero 行为。
- 新 6-D extension 通过独立低秩 residual adapter：LN(6)->Linear(6,32)->SiLU->Linear(32,256)，末层零初始化。
- adapter 单独训练、LR multiplier=3；可做 adapter-on/off audit。
- 历史 cache 只负责 12-D supported prefix；新 6-D online recompute。prefix audit PASS 后走速度路径，失败自动回退 runtime recompute。

## 4. 三层 query contract 与 strict load

Protocol 分开要求：

1. raw effective query feature contract；
2. tolerance-aware neural score contract；
3. winner-decision contract。

score contract 使用混合 `atol/rtol`，不能替代 raw feature contract。新增 strict core checkpoint loader：只允许声明的新 adapter/residual heads 缺失；任何 core missing/shape mismatch 直接失败。

## 5. Budgeted Critical Coverage

有效信号：proposal decisive recall=0.8036、selected decisive recall=0.6117；V62 HAB Top-M dense match=0.358，几乎等于 dense-full 0.359。HAB broad proposal 值得保留。

但 teacher exact winner-flip critical scene rate=0.4657、critical atom fraction=0.01519，Top-M/selected critical recall 仅 0.3548/0.3376。selected 基本保留 Top-M critical atoms，主要瓶颈在 acquisition。

V64 在 literal teacher winner-flip label 不变的前提下增加 BCC：hard forward 仍是 deterministic HAB Top-M；backward 使用 ST mask，直接最小化未被固定 Top-M interface 覆盖的 teacher-critical utility。部署不使用 teacher，不改变 M/B/selector，不增加证书预算。

## 6. Runtime prior 与 residual

同 checkpoint V63 ablation 中 nominal teacher match=0.141，no-base=0.214，no-runtime-priors=0.212，base prior 替换 learned winner rate=0.991。绝对质量受 support drift 污染，但方向强烈反对 opaque base prior。V64 nominal 关闭 continuous base prior 和 structural residual prior；结构安全仍由 hard feasibility、candidate validity 和 auditable atoms 表达，structural prior 仅作 ablation。

Residual raw proposed flip、beneficial、harmful 均为 0。保持 curriculum/calibration；在 raw beneficial direction 出现前禁止放宽 epsilon、flip margin 或 scale。

## 7. V64 决策门

1. 先做同 checkpoint support-contract audit；旧 anchor应恢复约0.359，adapter step-zero必须与legacy一致。
2. Protocol：support=12、extension=6、adapter enabled、base/raw/score/decision contract、B=16 provenance。
3. Minimum：broad decisive recall不能明显低于0.804/0.612。
4. Competitive：critical Top-M/selected recall先超过0.355/0.338；teacher match脱离0.141；beneficial residual>harmful。
5. Top-M critical升而selected不升时再升级selector tie-break，不增加B。
6. selected critical升而teacher action不升时转向atom-action value/candidate bank，不继续堆proposal loss。
7. Protocol PASS后paired CL20；Minimum+Competitive+CL20 safety通过后CL100。

## 8. 新增不重复尝试清单

- 不激活 checkpoint 未训练支持的新 query channels；
- 不把缓存完整维度等同于 checkpoint support dimension；
- 不使用 neural-score bitwise tolerance代替raw feature/decision contract；
- 不允许核心 checkpoint tensor静默随机初始化；
- 不使用 opaque runtime prior 绕过 fixed evidence interface；
- 不在 raw residual direction 无效时放松 calibration；
- 不增加 B 掩盖 proposal/bridge 缺陷；
- 不用 interrupted V63 pipeline 宣称 gate 状态；
- 不用 incomplete test 调参。

## 9. 验证边界

V64 本地验证：Python compile PASS，8/8 V64 YAML PASS，4/4 shell syntax PASS，full pytest **250 passed / 0 failed**。当前环境未执行 fresh V64 GPU training、calibration、open-loop、closed-loop，因此不声明 gate PASS、闭环提升或 SOTA。

---

# V64.2 — GateFix + HAB-Consistent Critical Boundary Exchange (HCBE) (2026-08-06)

## Result trigger

The uploaded V64 run completed training but never entered calibration/open-loop. The primary pipeline log stopped at `V64_SAQA_BCC_NEXT_COMMANDS.sh: line 390: sid: unbound variable`. A later direct rerun was independently blocked by a stale failed prefix-cache audit/config combination. Therefore Protocol, Minimum, and Competitive were not officially evaluated.

The corrected support-contract replay passes every deployment-relevant hard check: legacy anchor recovery, deployed step-zero action identity, base/raw-query/score/decision contracts, and runtime/prefix deployed-action identity. The only mismatch is the internal `full_action` diagnostic, whose semantics differ under the support-aware interface and must remain warning-only.

## Engineering fixes

1. Split Bash `local` declarations before interpolating `sid`, for both V64 and the inherited V63 launcher, so `set -u` cannot terminate calibration startup.
2. Fix cached-query audit `max_abs_error`: `arr.max(initial=nan)` always propagated NaN; use `arr.max()` for non-empty arrays.
3. Make support-aware audit hard criteria deployment/interface based. Internal `full_action` mismatch is reported but no longer invalidates an otherwise exact deployed contract.
4. Add a strict V64 pipeline-config contract and persisted query-path/checkpoint provenance to prevent stale inherited configs and silent checkpoint substitution.
5. Add explicit checkpoint-evaluation mode (`SKIP_V64_TRAINING=1`, `V64_CANDIDATE_CHECKPOINT=...`) so primary, teacher-best, and fixed-budget-critical-best checkpoints can be evaluated without retraining or overwriting their identity.
6. Add a pipeline-status inspector that distinguishes missing official gates from metric failures and extracts terminal log errors.

## Algorithm diagnosis from the uploaded training proxies

At the primary best checkpoint (epoch 1): broad proposal/selection remained healthy (`0.795/0.600` decisive recall), the dense-HAB/runtime bridge was `0.999`, and all query/budget contracts passed. However teacher exact winner-flip critical recall was only `0.345/0.329`, evidence certificate fraction was `0.046`, and no residual proposal reached the deployed winner. Epoch 5 improved teacher match and critical recall (`0.275`, `0.352/0.334`) but certificate fraction fell to `0.029`; the primary competitive-score checkpoint therefore selected epoch 1 even though both checkpoints were far below the certificate gate.

The AOCC diagnostic explains the certificate bottleneck: initial weighted deficit was about `0.485`, B=16 reduced it by about `0.320`, mean selected marginal certificate gain was exactly the per-atom prior radius `0.02`, and the final deficit remained about `0.175`. This is a conservative omitted-atom bound bottleneck, not a reason to lower B, redefine criticality, or relax calibration.

## V64.2 HCBE

V64 BCC used a global straight-through Top-M surrogate and an additional hardest-negative rank term. With rare critical atoms, requiring each critical atom to outrank the strongest non-critical atom is stronger than fixed-M inclusion and can conflict with broad decisive recall.

HCBE keeps all deployment semantics unchanged:

- hard forward remains deterministic HAB Top-M;
- B=16, M, evidence atoms, exact selector, and literal teacher winner-flip labels are unchanged;
- only missed critical atoms receive exchange supervision;
- the target boundary is the weakest retained non-critical atom in the same HAB family when available, otherwise the family-conditioned global retained boundary;
- no teacher is used at deployment.

The original hardest-negative term remains as a weak regularizer (`1.0 -> 0.25`) and HCBE receives weight `1.0`. This is an ablation-worthy acquisition improvement, not a claimed empirical gain until fresh training/open-loop/closed-loop results exist.

## Efficiency fixes

Teacher-interface criticality previously computed both model-interface and teacher-interface leave-one-out tensors, then discarded the model labels. V64.2 computes only the configured target source, reducing loss-stage work without changing targets or deployment. The wrapper also exposes data-loader prefetch/validation-worker controls instead of hard-coding them. Uploaded profiling shows data wait and loss construction, not model forward/backward, dominate training; speed tuning should therefore benchmark workers/prefetch and avoid reducing exactness or model capacity.

## No-repeat constraints retained

- do not increase B to hide acquisition/certificate failures;
- do not replace literal winner-flip criticality with margin-deficit proxies;
- do not make global Top-M the deployment selector;
- do not relax residual calibration before deployed beneficial flips exist;
- do not use opaque priors outside the evidence budget;
- do not tune on the incomplete test set;
- do not treat missing gate artifacts as gate failures.

## Validation boundary

Static/CPU validation after the changes: all YAML files parse, modified shell scripts pass `bash -n`, and the full unit suite passes **254/254**. No fresh GPU training, calibration, official open-loop gate, or closed-loop simulation was run in this environment, so V64.2 performance and SOTA claims remain unverified.

---

# V64.3 — Calibration-Consistent AOCC + Anchor-Preserving Winner-Conditioned Critical Acquisition (CC-AOCC / AP-WCCA) (2026-08-07)

## Result trigger

V64.2 completed calibration and the official 1000-scene paired candidate/local/foundation open-loop suite. The historical V64.2 checker reported Protocol PASS, Minimum FAIL, Competitive FAIL. Minimum failed only on evidence certificate coverage (`0.048 < 0.40`) and fallback (`0.952 > 0.60`). Competitive additionally failed teacher-match gain (`+0.007 < +0.015`), residual gain (`0`), proposal decisive recall (`0.7558 < 0.80`), literal teacher winner-flip critical Top-M/selected recall (`0.3482/0.3321`), certificate/fallback, and paired-regret regression.

## Newly identified engineering root cause: evidence calibration/deployment contract mismatch

The V64.2 calibration collector was invoked with `beta=0.0`, while the calibrated deployment configs retained `selector.adverse_certificate_beta=1.0` and `adverse_certificate_prior_radius=0.02`. `apply_v61_dual_calibration.py` copied only epsilon and never copied the evidence beta/prior-radius that define the calibrated nonconformity score. The old protocol checker audited residual beta but did not audit evidence AOCC beta/prior-radius.

With `proposal_top_m=24`, the uncalibrated deployment-only prior term can contribute about `24 * 0.02 = 0.48` adverse deficit. This matches the observed AOCC initial deficit (`~0.47–0.48`). B=16 recovered only about `0.32`, leaving most scenes uncertified and producing `fallback_rate=0.952`. Therefore the V64.2 Minimum failure is primarily an engineering/protocol inconsistency, not evidence that fixed B=16 is intrinsically insufficient.

V64.3 adds a calibration-consistent evidence contract:

- raw calibration shards persist `beta` and `prior_radius`;
- merged calibration verifies every shard used identical values;
- calibrated deployment configs copy evidence `beta`, `prior_radius`, and epsilon exactly;
- the formal gate now rejects calibration/deployment evidence-beta or prior-radius mismatch;
- V64.2's reported Protocol PASS is retained as a historical checker result, but under the corrected paper-grade contract it is flagged with `calibration beta=0, deployment beta=1`.

No gate threshold is relaxed.

## V64.2 model-state diagnosis

What the model learned:

- fixed-budget and query-interface contracts are stable;
- HAB Top-M preserves its own dense learned winner very well (`~0.981`);
- selected/effective decisive recall remains useful (`~0.582/~0.747`), with interaction decisive recall `~0.597`;
- candidate/local teacher match is `0.238`, a small `+0.007` over foundation (`0.231`);
- pair-full teacher match is `0.249` versus foundation `0.235`, so the learned sparse pair interface contains some beneficial information before final certificate/residual gating.

What it did not learn:

- literal teacher winner-flip critical acquisition did not improve: candidate Top-M/selected `0.348/0.332` versus foundation Top-M/selected about `0.355/0.331`;
- broad proposal decisive recall regressed from foundation `~0.804` to `~0.756`;
- the query-extension path caused severe dense-interface drift: candidate full-interface teacher action match `0.182` versus foundation `0.359`;
- the residual proposes changes in `~51.6%` of scenes but produces zero deployed flips; calibrated residual epsilon is `0.989`, residual raw-error MAE is `1.691`, and the winner-margin signal is orders of magnitude smaller, so this is not a threshold-tuning problem;
- the primary checkpoint was selected at epoch 1 while residual-curriculum scale was only `0.05`; therefore the formal candidate checkpoint was selected before the residual path had received meaningful full-strength training.

The critical Top-M→B16 drop is small (`0.348 -> 0.332`), so the main critical-evidence bottleneck remains acquisition, not the exact B=16 selector.

## V64.3 algorithm change: AP-WCCA

V64.2 HCBE changed supervision but still fine-tuned the whole legacy proposal/family stack. The resulting run lost broad proposal recall without gaining literal teacher-critical recall. The legacy proposal is therefore restored as an immutable acquisition anchor.

V64.3 adds a small **zero-initialized winner-conditioned critical proposal residual**:

```text
legacy HAB atom score (frozen)
  + r_crit(evidence, proposal features, scene, candidate-set summary,
           evidence family, frozen base-winner action embedding)
```

Properties:

- final residual layer is zero initialized, so step-zero proposal ranking is exactly the legacy anchor;
- legacy `proposal_head`, `family_head`, family embedding/activity encoder, proposal feature encoder, and query-extension adapter are frozen;
- nominal query-extension scale is `0`, because V64.2 showed a large harmful dense-interface drift; the 12-D checkpoint-supported path remains the nominal anchor;
- teacher labels are used only in training losses; deployment conditioning uses the planner-available frozen base winner, never teacher information;
- hard forward remains deterministic HAB Top-M;
- fixed evidence budget remains B=16 and proposal M remains 24;
- auditable evidence atoms and literal leave-one-atom-out winner-flip criticality are unchanged.

The objective is rebalanced toward the missing quantity rather than adding more global proposal pressure: exact critical loss `8 -> 12`, BCC coverage `2 -> 4`, HCBE exchange `1 -> 2`; dense-winner proposal pressure `20 -> 6`; the old hardest-negative rank remains weak (`0.10`).

## Residual training/checkpoint correction

V64.2 used `proposal_only_epochs=3`, `ramp_epochs=4`, `initial_scale=0.05`, yet primary checkpoint selection was allowed from epoch 0. The selected epoch-1 checkpoint therefore had residual losses scaled to only 5%.

V64.3:

- curriculum: proposal-only `1` epoch, ramp `2` epochs, initial scale `0.10`;
- adds `--best-min-epoch`; paper pipeline default is zero-based epoch `3`, so no primary/metric-specific best checkpoint can be promoted before residual training reaches full strength;
- early stopping minimum is moved to epoch 5.

This fixes checkpoint-selection semantics rather than relaxing the residual certificate. Residual epsilon must still come from independent calibration; do not lower it manually.

## Runtime/experiment speed changes

Observed V64.2 wall time after pipeline start was approximately:

- anchor audit: 15.1 min (4.3%);
- training: 299.2 min (84.8%);
- calibration: 24.4 min (6.9%);
- three-system open-loop: 14.3 min (4.0%).

Training profiling showed data wait and loss construction dominate forward/backward. V64.3 therefore changes the default paper launcher to:

- 2 GPUs;
- per-GPU batch `16` (global batch 32);
- 8 DataLoader workers/GPU and prefetch factor 3; persistent workers were already correctly enabled;
- base LR `1.7e-5` (conservative sqrt(2)-style increase from 1.2e-5 for the doubled per-GPU batch);
- only the small critical adapter + residual heads are trainable, reducing backward/optimizer work;
- query extension and legacy proposal/family modules are frozen;
- formal open-loop remains two workers/GPU because it is already only ~4% of the run.

These changes reduce compute without approximating the exact deployed selector or changing B.

## Closed-loop policy after gate failure

A paired diagnostic CL20 is recommended whenever the **corrected Protocol gate passes**, even if Minimum or Competitive fails. It is diagnostic only: it must not waive the formal gate, select hyperparameters on test, or support a SOTA claim. Run candidate/local/foundation on identical val-tune tokens and inspect paired safety/progress/regret/action-change cases. This can distinguish open-loop certificate conservatism from candidate-dynamics/reactive/replan failures.

Do not run CL20 when the corrected Protocol contract fails. For V64.2, first run the calibration-consistent open-loop replay; then run diagnostic CL20 if protocol is valid.

## No-repeat constraints

- do not increase B to hide certificate/acquisition failures;
- do not redefine criticality using margin deficit or severity proxies;
- do not unfreeze the legacy proposal stack unless AP-WCCA first demonstrates that a frozen-anchor residual lacks capacity;
- do not re-enable the query extension in nominal until a controlled ablation shows dense/interface benefit;
- do not lower residual conformal epsilon or evidence certificate thresholds to manufacture flips;
- do not select a checkpoint before residual curriculum reaches full strength;
- do not use the incomplete test set for tuning;
- do not treat diagnostic CL20 as publication evidence.

## Validation

Static/CPU validation in the delivery environment:

- Python compile: PASS;
- V64.3 config contract: PASS;
- shell syntax: PASS;
- targeted V64.3 tests: 4/4 PASS;
- full pytest: **258 passed, 0 failed**.

Fresh GPU training/calibration/open-loop/closed-loop were not run in this environment; all V64.3 performance claims remain to be measured.

---

# V64.3.1 — Provenance-Correct AP-WCCA + Decision-Aligned Exact Preservation Certificate (DA-EPC) (2026-08-07)

## Uploaded V64.3 result is not a valid AP-WCCA training run

The uploaded output directory is named `outputs_v64_3_cc_aocc_apwcca_fast_2gpu_v1`, but the immutable run evidence shows that the training process actually selected:

```text
bdse/configs/v64_2_saqa_bcc_hcbe_train_2gpu.yaml
```

The training log contains 10 epochs and reports the V64.2 trainable module set (`proposal_head`, family modules, `query_extension_proj`, residual heads). The V64.3 `critical_proposal_adapter` activation diagnostics are exactly zero for every epoch. Therefore this run evaluates **V64.2-trained weights through later V64.3 calibration/evaluation code**; it cannot be used to conclude that AP-WCCA succeeded or failed.

A second engineering defect allowed this contamination to survive reruns: `training_complete()` accepted an existing checkpoint by file timestamp and did not bind reuse to the training-config SHA-256 or foundation-checkpoint SHA-256. A later launcher invocation could overwrite the config-contract JSON with a valid V64.3 contract while silently reusing the older V64.2-trained checkpoint.

## Corrected gate interpretation

The historical checker reports Protocol PASS, Minimum FAIL, Competitive FAIL. Under the corrected V64.3.1 training-health contract the uploaded run is **Protocol-invalid for AP-WCCA evaluation**, because a configured zero-init trainable `critical_proposal_adapter` never leaves exact zero.

The historical Minimum failures are:

```text
evidence certificate fraction = 0.066 < 0.40
fallback rate                 = 0.934 > 0.60
```

Competitive additionally fails teacher-match gain, residual gain, proposal decisive recall, teacher literal critical Top-M/selected recall, certificate/fallback, and paired regret.

Unlike V64.2, the V64.3 calibration/deployment beta contract is now consistent (`beta=0`), so the low certificate rate can no longer be blamed on the old calibration beta mismatch.

## New certificate diagnosis

The uploaded 1000-scene paired candidate output shows:

```text
exact B16 vs pair-full winner preservation = 0.901
AOCC evidence certificate                  = 0.066
AOCC initial weighted deficit mean         = 0.019389
AOCC B16 deficit reduction mean            = 0.000744
AOCC final deficit mean                    = 0.018695
full-TopM target pair-certified fraction   = 0.0
```

The certificate/preservation quadrants are:

```text
certified + preserved       = 39
certified + not preserved   = 27
uncertified + preserved     = 862
uncertified + not preserved = 72
```

Thus the current pairwise AOCC certificate is not merely conservative relative to the exact downstream decision. It is also not a sound proxy for the actual winner under the current action-potential + safety + utility-refinement deployment operator: only 59.1% of pairwise-certified scenes preserve the exact full-TopM winner, while 92.3% of uncertified scenes do preserve it. On teacher-critical scenes, exact B16 winner preservation is about 90.5%.

This is a certificate-definition / deployment-semantics mismatch. It is not evidence that B=16 is intrinsically too small.

## V64.3.1 algorithm change: DA-EPC

V40--V43 already tried expensive deployment-aligned combinatorial search/repair. That line eventually reached high target preservation but increased open-loop latency substantially and did not solve upstream planning quality. **Do not repeat that search.**

V64.3.1 keeps the existing AOCC/HAB B=16 selection exactly unchanged and adds a no-search **Decision-Aligned Exact Preservation Certificate (DA-EPC)**:

1. Use the already queried Top-M evidence and the exact residual-disabled downstream deployment evaluator to obtain the full-TopM target action.
2. Run the same deterministic evaluator once on the selected B=16 evidence atoms.
3. Certify the planner-interface evidence set iff the two winner actions are literally identical.
4. Export only B=16 evidence atoms as before; the full-TopM evaluation remains selector-internal and adds no neural evidence query.
5. Retain the historical one-sided AOCC pair bound as a separate robustness diagnostic (`aocc_pairwise_certified_pair_fraction_raw`); it no longer masquerades as exact winner sufficiency.

DA-EPC changes neither B, M, evidence atoms, HAB hard forward, selected atom identities, nor literal winner-flip criticality. It also does not lower a certificate threshold. The certificate semantics are changed from a surrogate pair-margin condition to the same literal winner-preservation semantics already central to the paper's critical-evidence definition.

This is intentionally distinct from V40--V43 DACC: there is no deletion beam, swap repair, preservation search, or candidate-set expansion. It is one exact post-selection audit over already available model outputs.

## AP-WCCA remains the acquisition experiment to test

Because the uploaded run never trained AP-WCCA, V64.3.1 does **not** replace AP-WCCA with another proposal algorithm. The frozen legacy HAB proposal + zero-init winner-conditioned critical residual remains the next acquisition hypothesis to test.

The uploaded stale run still reinforces why this test matters:

- foundation proposal decisive recall is about `0.804`, uploaded candidate is `0.751`;
- teacher literal critical Top-M recall remains `0.3548`;
- teacher literal critical selected recall is `0.2726`;
- candidate teacher action match `0.224` is essentially foundation `0.225`;
- residual calibrated epsilon is `1.2259`, residual deployed flips remain zero.

Because AP-WCCA was inactive, none of these numbers can be attributed to the intended frozen-anchor winner-conditioned adapter.

## Training provenance hardening

V64.3.1:

- ignores inherited generic `MAIN_CONFIG` / `SPEED_CONFIG` variables in the V64.3 wrapper;
- requires version-scoped `V64_3_MAIN_CONFIG`, `V64_3_SPEED_CONFIG`, `V64_3_EVAL_CONFIG` overrides;
- strictly rejects stale V64.2 train configs even when they satisfy generic V64 budget/query checks;
- requires metadata/provenance algorithm versions to match;
- writes the selected train-config SHA-256 into query-path provenance;
- binds checkpoint reuse to both train-config SHA-256 and foundation-checkpoint SHA-256;
- refuses to reuse a checkpoint if the training provenance marker does not match;
- gate training health fails if a configured trainable zero-init AP-WCCA adapter never becomes non-zero.

## Runtime and training speed diagnosis

The uploaded 10-epoch run spent about `19,702.8 s = 5.47 h` in training. Profile attribution:

```text
DataLoader/data wait   15,195.8 s  = 77.1%
loss construction       1,775.3 s  =  9.0%
forward                    435.5 s =  2.2%
backward/step              457.4 s =  2.3%
pair sampling              525.9 s =  2.7%
H2D                         10.7 s =  0.05%
```

The mean data wait is about `972 ms/step`, while forward is only tens of milliseconds. The GPU is primarily starved by NPZ decode/tensorization/storage supply; more GPU compute alone will not solve the bottleneck.

The current main pipeline does **not** use `bdse_test_2` for training, calibration, or the 1000-scene paired gate. The large test-set size therefore did not cause this run's slowdown. Test should remain untouched until one final frozen evaluation.

V64.3.1 speed changes:

- intended AP-WCCA full training is 8 epochs rather than the accidentally executed V64.2 10 epochs;
- 2-GPU DDP remains mandatory for the paper pipeline, per-GPU batch is 16;
- default training workers are 12/GPU with prefetch 2, while a new input-pipeline microbenchmark can compare 8/12/16 on the actual storage server before the long run;
- foundation quality replay is sharded across both GPUs with two workers/GPU instead of one serial 1000-scene replay;
- calibration is split into four deterministic shards (two workers/GPU) and merged exactly;
- three-system paired open-loop stays at two workers/GPU because it is not the dominant wall-time stage;
- a 12k-scene, 4-epoch AP-WCCA activation screen is added so an inactive or harmful adapter can be rejected before paying for full training/calibration/open-loop.

## Most valuable next experiment

Do **not** go directly to another full version change. First run the V64.3.1 AP-WCCA activation screen on train/val only:

- 12k training scenes;
- 4 epochs;
- 2 GPUs, batch 16/GPU;
- val_tune 500 every epoch;
- require AP-WCCA residual RMS > 0;
- require teacher-critical Top-M recall to move above the ~0.355 foundation plateau while proposal decisive recall stays at least ~0.78.

If this screen fails, AP-WCCA capacity/conditioning should be changed before any calibration or closed loop. If it passes, run the full provenance-correct 50k/8-epoch V64.3.1 pipeline and inspect the causal sequence:

1. AP-WCCA activation and critical Top-M recall;
2. selected critical recall;
3. DA-EPC exact winner preservation / fallback;
4. teacher action match and paired regret;
5. residual raw error / calibrated epsilon / beneficial vs harmful flips;
6. diagnostic paired CL20 after corrected Protocol PASS, even if Minimum/Competitive still fail.

If true AP-WCCA improves Top-M critical recall but selected recall does not follow, revisit selector tie-breaking. If both improve but teacher action/regret do not, the next algorithmic bottleneck is atom-to-action value learning, not proposal acquisition. If open-loop improves but diagnostic CL20 does not, move to candidate dynamics/reactive interaction/replanning rather than adding more evidence-selection losses.

## Additional no-repeat constraints

- Do not interpret the uploaded V64.3 directory as an AP-WCCA result; its adapter was never trained.
- Do not reuse an OUT_ROOT across algorithm configs unless the config/foundation SHA contract passes.
- Do not repeat V40--V43 deployment coreset beam/swap search to manufacture winner preservation.
- Do not replace DA-EPC with a relaxed pair-margin threshold merely to raise certificate coverage.
- Do not increase B or M to solve the current certificate mismatch.
- Do not change AP-WCCA again before a provenance-correct activation screen establishes whether it has capacity.
- Do not tune on the incomplete/large test set; use test once after model/protocol freeze.

## V64.3.1 final engineering validation and provenance hardening

After the algorithm/code changes above, the long-stage reuse contracts were audited again.

Additional hardening:

- calibration raw shards now persist `source_checkpoint_sha256` and `source_config_sha256`;
- calibration merge rejects mixed checkpoint/config shards and exports the common source SHA values;
- the main pipeline treats legacy calibration shards/merged files without source SHA provenance as stale;
- paired open-loop suite reports now persist config/checkpoint SHA-256 for candidate/local/foundation;
- open-loop reuse also depends on the current `val_tune/manifest.jsonl`, not only checkpoint/config mtimes;
- the uploaded V64.3 open-loop artifacts were verified to have been freshly recomputed at 22:32--22:41; the causal contamination was the reused V64.2-trained checkpoint, not stale metrics files.

Final static/CPU regression in the delivery environment:

- Python compile: PASS;
- strict V64.3.1 train/eval config contract: PASS;
- shell syntax for main wrapper, main pipeline, and activation-screen launcher: PASS;
- targeted V64.3/V64 tests: **11 passed, 0 failed**;
- full test suite executed in two bounded batches: **102 + 158 = 260 passed, 0 failed**;
- synthetic calibration SHA-provenance test: PASS (mixed-checkpoint shards are rejected).

No fresh nuPlan GPU training/calibration/open-loop/closed-loop was run in this delivery environment. Performance claims for AP-WCCA and DA-EPC must therefore come from the next provenance-correct server run.

# V64.3.2 — Auditable AP-WCCA ScreenFix + Anchor-Centered Residual Alignment (ACRA) / optional AP-WRCCA (2026-08-07)

## Uploaded V64.3.1 activation-screen audit

The uploaded screen cannot be interpreted as a clean AP-WCCA algorithm failure.
Three engineering defects contaminate the decision:

1. `RUN_V64_3_1_APWCCA_ACTIVATION_SCREEN_2GPU.sh` requested
   `val_teacher_exact_winner_flip_critical_recall_topm/selected`, but the
   training-time open-loop validation path never emitted these formal-evaluator
   metrics.  The reported NaNs were therefore instrumentation NaNs, not measured
   zero/failed critical recall.  Training-side literal teacher-critical Top-M
   recall was finite (~0.365 -> ~0.368 over four epochs).
2. The screen declared AP-WCCA inactive from forward residual diagnostics alone.
   The uploaded current source would report a nonzero RMS floor even for an exact
   zero residual, while the run logs contain exact `0.0`.  This source/result
   semantic mismatch makes that scalar unsuitable as an activation contract.
   V64.3.2 measures adapter parameter delta directly and stores source SHA-256.
3. The generic launcher unconditionally overrode the screen YAML's intended
   `deployment_selector_scenes_per_rank=1` and `deployment_selector_every_n_steps=4`
   with CLI defaults `0/1`, forcing exact CPU selector supervision on the full
   local batch every step.  In the uploaded screen, loss construction consumed
   ~280--302 s of ~335--366 s per epoch (~83--85%), so the override materially
   changed both runtime and optimization schedule.

Other trustworthy uploaded signals: validation proposal decisive recall drifted
`0.7718 -> 0.7670`; proposal interaction recall `0.7663 -> 0.7618`; selected
recall stayed ~0.567--0.568; teacher action match stayed ~0.260--0.262.  DA-EPC
validation certification was already high (~0.962--0.970) with zero fallback and
budget-vs-pair-full winner preservation ~0.95--0.956.  Therefore the current
screen does not justify changing B=16 or the hard selector.

## Engineering fixes

- Training open-loop validation now emits the same literal teacher winner-flip
  Top-M/selected recall metrics as formal open-loop evaluation.
- Activation is based on `critical_proposal_adapter` parameter delta RMS/max,
  not a forward scalar.  A step-zero validation row (`epoch=-1`) is run on the
  exact same validation subset before optimizer updates, so screen decisions use
  within-subset deltas instead of an absolute historical 0.78 threshold.
- Frozen top-level foundation modules remain in eval mode during head-only
  finetuning, preventing dropout from making the claimed immutable anchor
  stochastic.
- Generic launcher selector CLI overrides are now opt-in.  When environment
  variables are unset, the YAML schedule is preserved exactly.
- Checkpoints and screen provenance carry source SHA-256 for the model/loss/train
  implementation.
- A present-but-shape-incompatible optional adapter is now fatal at evaluation;
  only a genuinely missing newly introduced module may be tolerated when loading
  an older foundation.

## Minimal algorithm stabilization: ACRA

V64.3.2 keeps AP-WCCA and literal teacher winner-flip criticality.  It adds a
small Anchor-Centered Residual Alignment objective directly on the AP-WCCA
residual logits.  The target is the same literal critical mask, centered per
scene so the residual cannot solve the task with a global logit shift.  This
provides an explicit gradient to a zero-initialized residual while preserving the
legacy HAB anchor, fixed M, fixed B=16, and deterministic hard selector.

This is deliberately not another global BCE/ranking rewrite and does not unfreeze
legacy proposal/family modules, avoiding the V64.2 failure mode where broad
proposal recall fell without improving literal critical coverage.

## Conditional next algorithm: AP-WRCCA

Do not run AP-WRCCA first.  Only if the corrected AP-WCCA+ACRA screen shows a
nonzero adapter parameter delta but no positive validation critical-TopM delta,
run the second screen with deployment-available base winner + strongest base
rival action embeddings.  This tests the previously identified hypothesis that
criticality is winner-rival boundary-relative rather than winner-only.  Teacher
information remains training-label-only; deployment conditioning uses only the
foundation action set.

## Screen decision policy

Primary screen A (AP-WCCA+ACRA):
- parameter-delta activation > 0;
- validation literal teacher-critical Top-M recall improves relative to the exact
  step-zero anchor on the same 500 rows;
- proposal decisive recall delta >= -0.02.

If A passes, run the full AP-WCCA V64.3.2 pipeline.  If A is instrument-valid but
fails specifically because critical Top-M does not improve, run screen B
(AP-WRCCA).  Do not alter selector/B/atom-action value until acquisition has been
measured cleanly.  If critical Top-M and selected recall improve but teacher
match/regret do not, the next bottleneck is atom-to-action value learning.
