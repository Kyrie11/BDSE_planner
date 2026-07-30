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
