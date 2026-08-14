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

# V64.3.3 — Full-Support Criticality Audit + Wired ACRA + Conditional Literal-Critical Value Probe (2026-08-08)

## Trigger: V64.3.2 Phase-1A/1B screens were not valid acquisition tests

Uploaded V64.3.2 AP-WCCA and AP-WRCCA screens both reported an activated adapter but `delta_val_critical_topm_recall=0`. Source SHA-256 provenance matches the uploaded source for both runs, so this is not another code/result version mismatch. Re-audit found three engineering defects:

1. **Training open-loop critical Top-M recall was tautological.** `_run_certificate_stage()` returns `stage_atom_active`, which is already the HAB Top-M mask. V64.3.2 passed this mask as the `active_atoms` universe to `_criticality_metrics()`, then asked what fraction of those critical atoms are in Top-M. Therefore anchor and last `val_teacher_exact_winner_flip_critical_recall_topm` were exactly `1.0` in both Phase-1A and Phase-1B by construction. The resulting `delta=0` contains no information about acquisition quality.
2. **ACRA was disconnected from the actual training forward contract.** `encode_context()` computed `critical_proposal_residual_logits`, but `BDSEModel.forward()` omitted the tensor from its output dict. `compute_bdse_losses()` therefore received `None`; the ACRA alignment branch and residual diagnostics were disabled. The adapter still moved because BCE/ranking/coverage losses backpropagated through the combined `proposal_logits`, explaining the non-zero parameter delta together with exactly-zero forward diagnostics.
3. **AP-WRCCA screen had a train/eval conditioning provenance mismatch.** Its train config used `frozen_base_winner_rival_actions`, but the launcher still named the AP-WCCA eval config. `RUN_MODE=train` prevented this from changing the screen weights, but the old config contract failed to reject the mismatch and the same mistake would be unsafe in a full pipeline.

## What the uploaded screens do and do not prove

Phase-1A AP-WCCA:
- adapter parameter delta RMS max: `0.005239` (real parameter movement);
- proposal decisive recall: `0.79151 -> 0.77514` (`-0.01637`);
- teacher action match: `0.264 -> 0.258`;
- reported validation critical Top-M: `1.0 -> 1.0`, invalid/tautological;
- training-side literal critical Top-M diagnostic: approximately `0.3653 -> 0.3685 -> 0.3682`, but there is no matching step-zero train diagnostic, so this is not a clean causal gain estimate.

Phase-1B AP-WRCCA:
- adapter parameter delta RMS max: `0.005010`;
- proposal decisive recall: `0.79151 -> 0.76280` (`-0.02871`);
- teacher action match: `0.264 -> 0.258`;
- reported validation critical Top-M: `1.0 -> 1.0`, invalid/tautological;
- training-side literal critical Top-M diagnostic peaks near `0.3689`, essentially indistinguishable from AP-WCCA without a step-zero train anchor.

Therefore **neither AP-WCCA nor AP-WRCCA is proven ineffective by these screens**. AP-WRCCA has a provisional negative broad-recall signal, but it must be re-tested only after the full-support metric and ACRA routing are fixed.

## V64.3.3 engineering corrections

1. Training validation now computes literal teacher criticality over `sample.evidence_bank.active_mask`, identical to formal open-loop support semantics. Top-M and selected sets are only evaluated *against* that full support.
2. `_criticality_metrics()` exports critical counts and Top-M/selected hit counts. Validation additionally reports micro recall (`total hits / total literal critical atoms`) for stable same-subset screening while retaining formal macro recall.
3. `BDSEModel.forward()` exports `critical_proposal_residual_logits`; ACRA now receives its intended direct residual tensor. The training log exports `L_critical_adapter_residual_alignment`.
4. Screen activation requires all three: parameter delta, non-zero forward residual, and non-zero ACRA alignment loss. Parameter movement alone is no longer considered evidence that the intended local objective is wired.
5. V64.3.3 train/eval config contract requires acquisition conditioning to match across train and eval configs.
6. New regression tests construct a literal critical atom outside Top-M and require measured Top-M recall `0`, preventing the Top-M-as-support bug from returning.

## Algorithm decision hierarchy after the repair

The historical full-support formal results still support **acquisition as the likely upstream bottleneck**: literal critical Top-M recall has remained near ~0.35 while B=16 generally preserves most of what Top-M already acquired. However V64.3.2 Phase-1A/1B do not add evidence for or against a particular acquisition representation.

Run controlled screens in this order:

1. **AP-WCCA + actually wired ACRA**, same frozen 500-row validation anchor.
2. If valid but non-improving, **AP-WRCCA + actually wired ACRA** on the same protocol.
3. Only if both binary-target screens are valid and non-improving, run a **Literal-Critical Value (LCV) probe**. LCV does not redefine criticality: all non-critical atoms retain exactly zero target. Among literal winner-flip critical atoms only, the adapter target is scaled by the exact post-removal winner-gap severity. This tests whether acquisition needs a richer value target rather than a binary critical label.

LCV is deliberately distinct from v48 DBCE. v48 could reward a boundary-deficit increase without an actual winner flip; V64.3.3 LCV is gated first by the literal winner-flip mask and is zero on every non-flipping atom.

## Interpretation after clean screens

- Corrected critical Top-M improves and selected recall follows, but teacher action/regret does not: **move to atom-to-action value / pair-margin representation**. Do not add more proposal losses.
- Corrected critical Top-M improves but selected recall does not: revisit fixed-B allocation/tie-breaking while keeping B=16.
- AP-WCCA/AP-WRCCA/LCV are all cleanly activated but critical Top-M does not improve: **acquisition representation/conditioning is the bottleneck**. The next representation experiment should expose a richer deployment-available multi-rival boundary state, not alter B/M/certificate.
- Open-loop teacher match improves but diagnostic closed loop does not: move to candidate dynamics, reactive interaction, and replanning.

## Positive components retained

- fixed planner-interface budget `B=16`;
- auditable evidence atoms and literal removal-induced winner-flip criticality;
- strong legacy HAB family-aware proposal as immutable anchor;
- DA-EPC exact downstream winner-preservation certificate (high screen coverage / zero fallback in the recent branch);
- strict checkpoint/config/source provenance;
- sparse exact-selector supervision cadence, which reduced screen loss-stage cost versus the accidental full-batch/every-step V64.3.1 run.

## Do not repeat

- Do not interpret V64.3.2 `critical Top-M = 1.0` or `delta = 0` as an acquisition result.
- Do not declare ACRA ineffective before a run where `L_critical_adapter_residual_alignment` and residual RMS are non-zero.
- Do not increase B or M to solve the current acquisition question.
- Do not relax DA-EPC/certificate thresholds to manufacture a gate pass.
- Do not unfreeze the complete legacy proposal/family stack; V64.2 lost broad recall without critical-recall gain.
- Do not simply increase BCC/HCBE/global hardest-negative weights; those objectives have not broken the historical ~0.35 critical-acquisition plateau and can conflict with broad recall.
- Do not repeat v40-v43 beam/swap/repair coreset search; it raised CPU latency without solving teacher-action quality.
- Do not re-enable the harmful 6-D query extension or opaque base prior in nominal runs.
- Do not lower residual conformal epsilon merely to create deployed flips.

# V64.3.4 — Frontier-Pair Conditioned Critical Acquisition (FPCCA) + Literal Boundary Attribution (LBA) (2026-08-10)

## Trigger: V64.3.3 corrected experiments make the acquisition negative result real

The uploaded V64.3.3 source SHA-256 values exactly match the source used by both
AP-WCCA and AP-WRCCA activation screens. Re-audit found one remaining
instrumentation defect, but unlike V64.3.2 it does **not** invalidate the
optimization itself:

- `_exact_winner_flip_critical_proposal_loss()` computes ACRA and includes
  `adapter_residual_alignment_weight * L_adapter_residual` in
  `L_exact_winner_flip_critical_proposal`;
- `compute_bdse_losses()` omitted only the standalone
  `L_critical_adapter_residual_alignment` key from the returned logging dict.

Therefore `acra_wired=false` in the uploaded V64.3.3 screen JSON is a false
negative caused by missing logging. The AP-WCCA/AP-WRCCA residuals were actually
optimized with ACRA, and their flat literal-critical acquisition is valid
algorithmic evidence. V64.3.4 restores the standalone diagnostic key.

Corrected same-subset screens:

- AP-WCCA: critical Top-M micro `0.360153 -> 0.360153` (delta `0.000000`),
  selected critical micro `0.260536 -> 0.252874` (delta `-0.007663`), proposal
  decisive `0.791509 -> 0.767043` (delta `-0.024466`), adapter parameter delta
  RMS `0.005158`, residual RMS max `0.96157`.
- AP-WRCCA: critical Top-M micro `0.360153 -> 0.360153` (delta `0.000000`),
  selected critical micro `0.260536 -> 0.256705` (delta `-0.003831`), proposal
  decisive `0.791509 -> 0.773034` (delta `-0.018475`), adapter parameter delta
  RMS `0.004834`, residual RMS max `1.02724`.

Formal 1000-scene full-support open-loop confirms the plateau:

- teacher literal-critical Top-M recall: AP-WCCA `0.354762`, AP-WRCCA `0.354762`;
- teacher literal-critical selected recall: `0.280476` / `0.282015`;
- candidate teacher action match: both `0.224`, below matched foundation `0.227`;
- pair-full teacher action match: both `0.236`;
- proposal decisive recall: `0.784096` / `0.780520`;
- dense->HAB Top-M dense-value action preservation: `0.970` / `0.968`;
- budget-vs-pair-full winner preservation: `0.899` / `0.902`;
- deployed residual flips: `0` / `0`.

Thus neither AP-WCCA nor AP-WRCCA should be promoted to the main algorithm.
DA-EPC, fixed B=16, literal criticality, auditable atoms, immutable legacy HAB,
and sparse exact-selector supervision remain retained components.

## New mechanism diagnosis: the conditioning anchor is semantically wrong in most scenes

AP-WCCA conditions every atom on the foundation/base top-1 action. AP-WRCCA adds
only the second-lowest foundation action as one rival. But the matched foundation
teacher-action match is only `0.227`. Consequently, in most scenes the
winner-conditioned acquisition residual is centered on an action that is not the
teacher-relevant winner used by the literal winner-flip label. A large residual
can therefore move proposal logits substantially while never exposing the actual
decision boundary associated with a literal critical atom.

This explains the otherwise contradictory V64.3.3 pattern:

- adapter/residual activation is large;
- broad proposal ranking moves/degrades;
- literal critical Top-M recall is exactly flat;
- strongest-rival conditioning does not recover the lost semantic boundary.

This is a representation/conditioning failure, not evidence for increasing M/B,
relaxing the certificate, or stacking larger global ranking weights.

## FPCCA: deployment-available frontier-pair representation

V64.3.4 adds **Frontier-Pair Conditioned Critical Acquisition (FPCCA)** as an
optional residual over the same immutable HAB proposal anchor.

Instead of privileging foundation top-1, FPCCA:

1. obtains the top-F valid actions under deployment-available frozen `J0`;
2. constructs every unordered action pair in this frontier (`F=6` -> 15 pair
   tokens in the default screen);
3. represents a pair with action embeddings, relative/product interaction, and a
   normalized foundation-cost gap;
4. lets each evidence atom attend to this boundary set;
5. maps the atom-specific boundary context to a zero-initialized proposal
   residual.

The forward deployment path remains teacher-free. B=16, evidence M, auditable
atoms, HAB hard selection, candidate bank, and DA-EPC are unchanged.

The representation is aligned with the paper's decisive-rival framing: the
proposal no longer has to guess a single universal winner/rival pair before it
can decide which evidence is decision-sufficient.

## LBA: literal boundary identity supervision without redefining criticality

FPCCA additionally supports **Literal Boundary Attribution (LBA)**. For a literal
winner-flip critical atom only, leave-one-atom-out evaluation provides the exact
flip target action. If `(teacher winner, leave-one-out flip target)` lies inside
the deployment base frontier, LBA supervises the atom's boundary attention to
that pair.

Important invariants:

- non-critical atoms receive **no** LBA target;
- the definition of critical remains exactly “removing this atom changes the
  winner”;
- LBA is not margin-deficit, soft-criticality, or severity-as-criticality;
- teacher boundary identity is training-only and is not required at deployment;
- main LBA weight is conservative (`0.25` inside the exact-critical objective).

This creates a clean CCF-A-style ablation:

1. AP-WRCCA + binary ACRA (uploaded negative control);
2. AP-WRCCA + LCV (target/value granularity probe);
3. FPCCA without LBA (representation-only);
4. FPCCA + LBA (representation + literal boundary attribution).

## Frontier representability diagnostic

Training/open-loop criticality metrics now report whether the literal critical
boundary is representable inside foundation top-2/top-3/top-5/top-6/top-9 action
frontiers. The default FPCCA uses F=6. If the same-subset top-6 critical-boundary
representability is low, a prepared F=8 screen is allowed as a *representation
support* experiment. This does not increase evidence M or B.

Do not expand F when top-6 representability is already high but critical Top-M
remains flat; that outcome means the representation/learning rule failed rather
than the frontier support being too small.

## Efficiency and engineering corrections

1. The missing standalone ACRA diagnostic is restored. New FPCCA training also
   exports `L_critical_boundary_attribution` and
   `critical_boundary_representable_fraction`.
2. Train/eval provenance now binds the complete critical-proposal adapter
   signature (conditioning, rank, scale, frontier size/count, gap-bias setting),
   not only the conditioning string. A silent F6/F8 train/eval mismatch is a
   hard configuration error.
3. The generic V64.3 pipeline writes the current YAML `metadata.algorithm_version`
   into the training contract instead of a stale hard-coded V64.3.1 string.
4. MR/FPCCA frontier extraction uses `torch.topk` rather than sorting all K
   actions. FPCCA pair construction is vectorized with `torch.triu_indices`.
5. The boundary-focused training pair sampler was a major avoidable execution
   cost: V64.3.3 averaged about `251 s/epoch` in pair sampling on AP-WCCA and
   `250 s/epoch` on AP-WRCCA. Its row-wise CUDA `.item()/nonzero/topk` loop is
   replaced by batched quota-preserving top-k selection with no host-device
   scalar synchronization. A regression test requires exactly the same selected
   pair tensors as the V64.3.3 row-wise rule on a deterministic fixture. CPU
   microbenchmark for B=16/P=112 is ~`1.2 ms` per call; the server-side CUDA gain
   must be measured in the next run rather than assumed.
6. Two-GPU launchers keep 12 training workers/GPU, 4 validation workers/GPU,
   sparse exact-selector cadence, and screen-before-full execution. No loss,
   supervision source, pair quota, evidence budget, or selector objective is
   removed for speed.

## Remaining protocol/closed-loop engineering blocker

Both uploaded full runs have `dense_runtime_query_decision_match=0.997`, below the
existing Protocol requirement `0.999`, so their formal pipeline correctly blocks
closed loop. Re-audit of the three AP-WCCA mismatched scenes shows:

- dense/runtime raw query-feature max absolute error = `0.0`;
- neural-score max absolute difference < `0.001`;
- the action difference is therefore a near-tie dense-vs-sparse CUDA batch-shape
  numerical sensitivity, not query-feature provenance drift.

Do **not** lower the 0.999 threshold. The next full run should retain the strict
raw-feature/score/action audit. If 0.997 repeats after the algorithm winner is
chosen, isolate strict-matmul / near-tie numerical reproducibility as an
engineering protocol experiment; do not mix that change into the acquisition
ablation.

## V64.3.4 promotion policy

Run short 2-GPU screens before any new full pipeline. A screen is considered a
meaningful acquisition improvement only if, on the exact same validation anchor:

- instrumentation/adapter/ACRA (and LBA when applicable) are wired;
- literal critical Top-M micro gain >= `+0.01` absolute;
- literal critical selected gain >= `-0.005`;
- broad proposal decisive delta >= `-0.02`;
- teacher action-match delta >= `-0.005`.

Among eligible screens, prioritize literal-critical Top-M gain, then selected
critical gain and teacher action stability. Only the winning configuration gets a
50k/8-epoch full pipeline.

If FPCCA raises critical Top-M and selected recall but pair-full/candidate teacher
match remains flat, acquisition has finally been fixed and the next main problem
becomes **atom-to-action / pair-boundary value representation**. Do not add more
acquisition losses at that point.

If full open-loop teacher match improves but diagnostic CL20 still does not,
move the research direction to candidate dynamics, interactive prediction, and
replanning rather than evidence-budget relaxation.

## V64.3.4 no-repeat constraints

- Do not retry AP-WCCA or AP-WRCCA binary as candidate main algorithms; V64.3.3
  now provides valid negative evidence.
- Do not declare LCV positive before its dedicated screen; it is still an
  untested value-target probe.
- Do not unfreeze the complete legacy proposal/family stack.
- Do not increase B or evidence M to hide acquisition failure.
- Do not relax DA-EPC or conformal residual epsilon to manufacture winner flips.
- Do not re-run V40--V43 beam/swap/repair coreset search.
- Do not stack larger BCC/HCBE/hardest-negative weights without a new causal
  diagnosis.
- Do not expand FPCCA from F=6 to F=8 unless the new frontier-representability
  diagnostic shows that F=6 is actually missing literal boundaries.
- Do not mix numerical protocol fixes with algorithm ablations; isolate them.

### V64.3.4 delivery hardening amendment

- Added V64.3.4-specific AP-WRCCA+LCV candidate, anchor-control, and local-control closed-loop configs instead of reusing V64.3.2 metadata. The model semantics are unchanged, but train/eval provenance now reports the same `V64.3.4-CC-AOCC-AP-WRCCA-LCV-DA-EPC` version end-to-end.
- Final repository regression status after FPCCA/LBA, pair-sampler vectorization, diagnostics, and provenance hardening: `280 passed, 0 failed`.

# V64.3.5 — Complete-Candidate Boundary Routing (CCBR) + Literal Endpoint Attribution (LEA) (2026-08-11)

## Trigger: V64.3.4 FPCCA/LBA screen is a clean negative acquisition result

The uploaded V64.3.4 same-subset acquisition matrix gives the same literal teacher winner-flip critical Top-M micro recall `0.3601532567` for AP-WRCCA+LCV, FPCCA-noLBA, FPCCA+LBA F6, and FPCCA+LBA F8. Adapter parameters/residuals and ACRA/LBA losses are non-zero, so another instrumentation/activation explanation is no longer credible.

The four causal deltas relative to the common anchor are:

- AP-WRCCA+LCV: Top-M `+0.0000`, selected `-0.00383`, proposal decisive `-0.01852`, teacher match `-0.006`;
- FPCCA-noLBA: Top-M `+0.0000`, selected `-0.00766`, proposal decisive `-0.02829`, teacher match `-0.006`;
- FPCCA+LBA F6: Top-M `+0.0000`, selected `-0.00766`, proposal decisive `-0.01456`, teacher match `-0.004`;
- FPCCA+LBA F8: Top-M `+0.0000`, selected `-0.00766`, proposal decisive `-0.01573`, teacher match `-0.004`.

FPCCA's literal-boundary support is the decisive failure signal: learned LBA representability is only `0.12408` for F6 and `0.20250` for F8. The independent teacher diagnostic reports literal boundary in deployment base top-6 `0.20479` and teacher winner in base top-6 only `0.36626`. Expanding F from 6 to 8 raises support but produces zero critical Top-M gain, so further F expansion is not the next algorithm.

**Decision:** none of AP-WRCCA+LCV / FPCCA-noLBA / FPCCA+LBA F6 / FPCCA+LBA F8 is promoted into the main algorithm. They become negative/mechanism ablations.

## Invalid full-style runs are excluded from algorithm evidence

The uploaded FPCCA-noLBA/F6/F8 DA-EPC fast archives each found zero safe retained foundation checkpoints (`candidate_count=0`, `safe_candidate_count=0`), rebuilt a fresh fast foundation, and failed the immutable anchor gate with base winner-rival sign accuracy about `0.346--0.354 < 0.62`. These runs are not evidence about FPCCA. V64.3.5 full launchers therefore set `FOUNDATION_POLICY=explicit`, disable recovery/rebuild, and require both an existing `FOUNDATION_CKPT` and a screen promotion report before any full pipeline starts.

## Algorithm change: Complete-Candidate Boundary Routing (CCBR)

V64.3.5 removes the top-F pair-support assumption while preserving the fixed evidence interface.

For each evidence atom, CCBR reuses the already-computed candidate action embeddings plus normalized deployment `J0` cost/rank. Two atom-conditioned O(EK) attention routers cover the **complete valid candidate bank**:

1. a winner-endpoint router;
2. a flip-endpoint router.

Their pooled endpoint contexts are composed as `[winner, flip, flip-winner, winner*flip]` and mapped to a proposal residual. The final residual layer is zero initialized, so the legacy HAB anchor is exactly unchanged at step zero. CCBR does not construct K^2 pair tokens and does not issue any additional atom-action evidence query. Fixed `B=16`, proposal M, literal criticality, DA-EPC and exact selector semantics remain unchanged.

This is intentionally not a repeat of V55 action-potential projection or V59 generic set-conditioned interaction potential. Those versions modified downstream action/value representation. CCBR modifies **acquisition boundary conditioning** only and uses exact literal boundary endpoints as its semantic support.

## LEA: exact endpoint identity supervision

V64.3.5 adds Literal Endpoint Attribution (LEA) only for exact teacher-interface winner-flip critical atoms. If the full teacher winner is `w` and removing atom `i` changes the scalar-aligned teacher winner to `f_i`, LEA supervises:

- winner router -> `w`;
- flip router -> `f_i`.

Non-critical atoms receive no endpoint target. Therefore LEA does not turn margin proximity, severity, uncertainty, or any proxy into the definition of criticality. Teacher labels remain training-only.

The clean mechanism screen is now only:

1. CCBR-noLEA (representation-only);
2. CCBR+LEA (representation + exact endpoint identity).

No AP-WCCA/AP-WRCCA/FPCCA rerun is part of this phase.

## New diagnostic: frozen-family-slot oracle ceiling

A second possible acquisition ceiling was not directly measured in V64.3.4: the critical residual can reorder atoms but the legacy HAB family allocation remains frozen. V64.3.5 open-loop diagnostics therefore add `teacher_exact_winner_flip_frozen_family_slot_oracle_topm_recall`.

For this diagnostic only, exact critical atoms receive oracle-dominant proposal logits while runtime family logits, family ids, B and M remain fixed. It never changes training or deployment.

Stop rule:

- oracle >= `0.90`, learned critical Top-M flat -> family allocation is not the main blocker; do not unfreeze/tune family stack;
- oracle < `0.90` -> only then allow one small zero-init boundary-aware family-residual screen. Do not repeat full proposal/family unfreezing.

## Promotion and next bottleneck rules

CCBR promotion keeps the V64.3.4 causal gate:

- critical Top-M micro gain >= `+0.01` absolute;
- selected critical micro delta >= `-0.005`;
- proposal decisive recall delta >= `-0.02`;
- teacher action match delta >= `-0.005`;
- CCBR+LEA additionally requires endpoint representability > `0.95` and non-zero LEA loss.

If CCBR improves Top-M/selected but pair-full/candidate teacher match remains flat, acquisition work stops and the next algorithm must be explicit literal/decisive **pair-boundary value representation**. Do not repeat V55 global action potential or V59 generic set-conditioned potential. If open-loop improves but CL20 remains flat, switch to candidate dynamics/interactive prediction/replanning.

## Engineering fixes and efficiency

1. Fixed acquisition screen checker substring bug: `FPCCA-noLBA` no longer incorrectly requires LBA merely because `NOLBA` contains `LBA`.
2. Added analogous explicit `noLEA` handling and LEA instrumentation checks.
3. Corrected the old V64.3.4 FPCCA-LBA screen metadata text that incorrectly called the algorithm MR-BCCA.
4. Added V64.3.5 config-contract support and complete adapter-signature matching including endpoint cost bias.
5. Full wrappers hard-require explicit foundation + promotion report and use a V64.3.5 validation split cache derived from `bdse_val_v2`; automatic foundation rebuild is disabled.
6. Retained the V64.3.4 vectorized pair sampler. Uploaded screen matrix pair-sampling time is only about `1.35--1.76 s/epoch`, versus ~`250 s/epoch` in historical V64.3.3. Current timing bottlenecks are loss construction (`~40--113 s/epoch`) and variable data wait (`~10--136 s/epoch`), not pair sampling.
7. CCBR complexity is O(EK) and reuses existing action embeddings/J0, avoiding complete O(K^2) pair materialization.

Final repository regression after V64.3.5: **285 passed, 0 failed, 30 warnings**. No nuPlan GPU training was performed in the delivery environment; CCBR/LEA performance claims require the next two-GPU screen.

# V64.3.6 — Boundary-Coupled Hierarchical Admission (BCHA) + Literal Boundary Pair Residual (LBPR) (2026-08-11)

## Trigger: V64.3.5 proves complete boundary support is necessary but not sufficient

The uploaded V64.3.5 causal screen used the same V62 DCAB-EWFC warm-start anchor and correctly blocked full training because neither CCBR arm met the acquisition promotion gate.  The result is informative rather than an instrumentation failure:

- common anchor literal critical Top-M micro recall: `0.3601532567`;
- CCBR-noLEA final/best Top-M gain: `+0.0000`;
- CCBR+LEA final/best Top-M gain: `+0.0000`;
- CCBR+LEA endpoint representability: `1.000` at activation and `0.9973` at the last epoch;
- LEA loss is active (`max 3.2463`) and CCBR residual RMS reaches `0.4676`;
- CCBR adapter parameter delta RMS reaches `0.00418`;
- selected-critical recall drops from `0.26054` to `0.25287` (`-0.00766`);
- proposal decisive recall drops from `0.79151` to `0.77632` (`-0.01519`);
- teacher action match drops from `0.264` to `0.260` (`-0.004`).

Therefore CCBR **did solve FPCCA's representation-support ceiling** (F6/F8 had only about 12%/20% literal boundary support), and LEA demonstrably learns exact winner/flip endpoint identity, but that semantic support does not by itself change HAB Top-M admission.  CCBR+LEA is retained as a useful representation primitive, not promoted as a completed main algorithm.

## The previously potential downstream value bottleneck is now confirmed

V64.3.5 also makes the downstream ceiling explicit on the same validation anchor.  Across CCBR-noLEA and CCBR+LEA epochs:

- `pair_full_interface_action_match` remains only about `0.256--0.262`;
- `local_pair_full_interface_action_match` remains about `0.258--0.264`;
- `budget_vs_pair_full_match` is much higher, about `0.928--0.948`;
- `pair_full_to_budget_flip_rate` is only about `0.052--0.072`.

Thus the B=16 compression is **not** the dominant source of the roughly 74% teacher mismatch. Even giving the pair pathway full active evidence leaves teacher match near 26%. The atom/action-to-pair-boundary value representation is a real performance bottleneck, not merely a hypothetical second stage.

This changes the optimization regime from "acquisition first, value later" to a **verified dual bottleneck**. Both must eventually improve, but they must remain separately switchable so the paper can attribute gains to evidence admission versus evidence value.

## V64.3.5 family-oracle metric was not evidence either way

`teacher_exact_winner_flip_frozen_family_slot_oracle_topm_recall` was `null` in the uploaded V64.3.5 screens. Audit shows the function existed only in the optional dense open-loop diagnostic, while short training screens used the teacher-only validation path. This was an instrumentation-placement bug. It does **not** prove that frozen HAB family slots are or are not a bottleneck.

V64.3.6 computes two oracle diagnostics on every screen validation:

1. `teacher_exact_winner_flip_frozen_family_slot_oracle_topm_recall`: critical atoms get oracle-dominant atom logits while the frozen family gate/slots remain intact;
2. `teacher_exact_winner_flip_global_oracle_topm_recall`: the same oracle atom scores with HAB family allocation disabled.

Their gap is the direct family-admission ceiling. BCHA is justified only if frozen-family oracle `<0.90` and the global-vs-frozen gap is at least `0.05`. Otherwise BCHA must be dropped rather than tuned repeatedly.

## Algorithm A: Boundary-Coupled Hierarchical Admission (BCHA)

CCBR currently changes atom proposal logits **after** the immutable family gate has allocated family support. A correct literal-boundary atom signal can therefore be invisible if its family receives too little Top-M capacity.

BCHA reuses the same CCBR atom residual, centers it over active atoms, max-pools it within each semantic family, centers the active-family vector, clips it, and adds only a bounded residual to the frozen family logits before HAB computes family probabilities. It is parameter-free beyond CCBR itself.

Important invariants:

- legacy family/proposal heads remain frozen;
- CCBR final layer is zero initialized, so BCHA is an exact step-zero no-op;
- B=16 and proposal M are unchanged;
- auditable evidence atoms and exact winner-flip criticality are unchanged;
- no extra atom-action evidence query is introduced;
- BCHA is O(E+F), not a larger selector or beam/swap search.

This module directly serves the paper's hierarchical budget interface: exact decision-boundary evidence can influence **where the fixed proposal capacity is allocated**, not only ranking inside an already-allocated family.

## Algorithm B: Literal Boundary Pair Residual (LBPR)

LBPR addresses the now-proven downstream pair-value ceiling without repeating V46/V49 broad arbitrary pair fields or V55/V56/V59 global/integrable action potentials.

For evidence atom `i` and ordered candidate pair `(a,b)`, LBPR computes a low-rank atom factor and multiplies it by an action/query **difference** factor. The residual therefore satisfies exact antisymmetry by construction:

`r_i(a,b) = -r_i(b,a)`.

The output linear layer has **no bias**; a regression test explicitly swaps the pair after non-zero output weights are installed and requires exact sign reversal. The final output weights are zero initialized, so step zero is exactly the frozen local pair interface.

LBPR is additionally gated by CCBR/LEA endpoint compatibility. The gate uses only the deployment-available CCBR endpoint distributions and is detached from value gradients, so downstream value learning cannot corrupt the literal endpoint-attribution mechanism.

Training reserves a bounded pair-batch quota for exact teacher winner -> leave-one-atom-out flip edges and upweights only the matching literal atom/pair labels. Runtime still evaluates only the existing rival graph and selected evidence. It does not build an O(K^2) evidence lattice and does not alter the evidence budget.

## Clean causal design

V64.3.6 exposes four independently interpretable arms:

1. `LOCAL`: CCBR+LEA, no BCHA, no LBPR — causal control with frozen local pair value;
2. `BCHA`: admission-only intervention;
3. `LBPR`: literal pair-value intervention only;
4. `BCHA_LBPR`: joint intervention, run only when family oracle and/or individual screens justify it.

The adaptive matrix always runs LOCAL and LBPR. BCHA runs only when the frozen-family oracle indicates an admission ceiling (or the user explicitly forces the 2x2). The joint arm runs only when BCHA is justified and an individual mechanism has a meaningful signal, unless `RUN_FULL_2X2=1` is requested for the paper ablation.

## Warm-start policy

Keep the user's existing immutable warm-start for the next causal screens:

`outputs_v62_dcab_ewfc_fast_2gpu_v1/train/bdse_v62_dcab_ewfc.best.pt`.

Do **not** scratch-retrain the full foundation yet. The current question is whether the new admission/value residuals can correct a fixed representation, and historical broad unfreezing/retraining has not provided a reliable positive result. A scratch/full encoder retrain would confound this diagnosis.

Only reconsider selective foundation retraining if LBPR has verified non-zero gradients/parameter movement and literal-pair coverage but cannot raise pair-full match. At that point the next hypothesis would be frozen action/evidence embedding capacity, and the preferred experiment is a **small selective encoder unfreeze**, not an indiscriminate full-model restart. A final paper can later include scratch-vs-warm-start robustness after the winning mechanism is fixed.

## Runtime/training efficiency changes

- V64.3.4 vectorized pair sampling remains; no quota reduction is used for speed.
- In V64.3.6 causal configs, the historically negative residual-action and set-conditioned potential branches are disabled (`evidence_action_residual=false`, `set_residual_rank=0`) and their loss weights are zero. They are not consumed by the legacy direct pair tournament in this phase, so this removes unused forward/loss work without changing the tested mechanism.
- LBPR rank is 32; training keeps max 48 cached pairs/scene with 12 literal-boundary reserved slots. Runtime refines at most 32 existing rival pairs.
- BCHA adds only O(E+F) pooling.
- Two-GPU defaults remain batch 16/GPU, 12 train workers/GPU, 4 validation workers/GPU, prefetch 2.

## V64.3.6 promotion rules

Acquisition signal remains strict: literal critical Top-M gain >= `+0.01`, selected-critical delta >= `-0.005`, proposal-decisive delta >= `-0.02`, teacher-match delta >= `-0.005`.

Value signal requires pair-full teacher-match gain >= `+0.01`, pair-full advantage over the same-epoch local pair-full control >= `+0.005`, budget-vs-pair-full delta >= `-0.02`, and teacher-match stability >= `-0.005`.

Automatic full training is even stricter: the selected row must improve final teacher action match by at least `+0.005` and have either a meaningful acquisition or value gain. This prevents a diagnostic-only pair-full gain from being promoted as a planner result.

## V64.3.6 no-repeat constraints

- Do not retry AP-WCCA/AP-WRCCA/LCV/FPCCA F6/F8 as candidate main algorithms.
- Do not increase B or M to hide admission failure.
- Do not unfreeze the complete legacy family/proposal stack; BCHA is the only permitted family intervention unless its oracle-conditioned experiment fails for a new reason.
- Do not repeat V46/V49 broad arbitrary pair residuals.
- Do not repeat V55 Hodge/global action potential, V56 generic per-evidence action potential, or V59 generic set-conditioned potential.
- Do not relax DA-EPC, residual confidence gates, or protocol thresholds to manufacture flips.
- Do not scratch-retrain the full foundation before the targeted V64.3.6 causal screens resolve whether frozen representation capacity is actually limiting.

# V64.3.7 — Decisive Anchor-Relative Margin Refinement (DARM) + Decisive Boundary Residual (DBR) (2026-08-11)

## Trigger: V64.3.6 rules out family admission and exposes a value/aggregation confound

The uploaded V64.3.6 dual-bottleneck screen correctly returned `winner=null`. This is not an activation failure.

**BCHA is now a definitive negative.** The teacher exact winner-flip oracle diagnostics are

- frozen-family-slot oracle Top-M recall = `1.000`;
- global oracle Top-M recall = `1.000`;
- oracle gap = `0.000`.

Therefore the frozen HAB family allocation is not limiting literal-critical Top-M admission on this validation subset. Do not tune BCHA, do not unfreeze the family stack, and do not reinterpret the unchanged learned Top-M as a family-quota problem.

**LBPR has a real but weak positive value signal.** In the LBPR arm the adapter trains (`parameter delta RMS=0.005838`, residual RMS=`0.030978`). At epoch 3, pair-full teacher match is `0.176` versus the same-epoch local pair-full control `0.174`; pair-full teacher regret improves from the local control `11286.84` to `11030.27`; beneficial residual intervention is `0.002` and harmful residual intervention is `0.000`. However the +0.2 percentage-point action-match advantage is far below the +1pp value promotion threshold and final deployed teacher match remains `0.178`. LBPR is therefore not promoted as a completed main module.

The weak LBPR gain is mechanistically informative. CCBR endpoint support is complete, but the LEA endpoint CE is still diffuse (`3.1849` at activation, `2.9401` at the final epoch). With about 27 valid candidates, a uniform endpoint CE is roughly `log(27)=3.30`; representability=1 means the correct endpoint exists in support, not that the endpoint posterior is sufficiently sharp. Multiplying LBPR by this immature endpoint posterior suppresses an already sparse residual.

A second confound is more important: V64.3.6 used `legacy_tournament` to carry LBPR. The same V62 warm-start had a much stronger direct selected-local/evidence-action-potential anchor in V64.3.5 (`teacher action match≈0.264`, local pair-full≈`0.264` at epoch -1), whereas the V64.3.6 zero-residual legacy-tournament anchor is only `0.180/0.172`. Thus V64.3.6 tested a new residual on top of a historically weaker aggregation operator. This does not invalidate the tiny LBPR positive signal, but it prevents a clean conclusion about whether pair residuals can correct the strongest fixed evidence-value anchor.

## Updated bottleneck diagnosis

The optimization priority is now:

1. **First: decisive pair-value / final aggregation.** Pair-full remains extremely low even with full evidence, and V64.3.6 introduced an avoidable weak-aggregator confound. Establish whether a pair residual can improve a strong selected-local anchor before changing the foundation representation.
2. **Second: learned proposal-score generalization.** Literal critical Top-M remains `0.360153`, but the family/global oracle both equal 1.0 and CCBR already removed the representation-support ceiling. The remaining acquisition problem is learned atom ranking/generalization under rare literal-critical supervision, not family capacity and not top-F boundary support.

Do not optimize these two bottlenecks jointly in V64.3.7. Freeze acquisition while isolating value. If the value mechanism succeeds, its realized decisive-margin correction becomes a better future acquisition target than another critical-atom classification loss.

## Algorithm: Decisive Anchor-Relative Margin Refinement (DARM)

DARM replaces the global soft-min pair tournament used in V64.3.6 with a theorem-aligned star refinement around the already strong budgeted selected-local action.

For selected evidence `S_B`, define the immutable local anchor cost

`J_B^L(a) = J0(a) + sum_{i in S_B} g_i(a)`

and anchor

`a0 = argmin_a J_B^L(a)`.

For each challenger `b`, DARM uses only the anchor-relative margin

`M_DARM(a0,b) = J_B^L(b)-J_B^L(a0) + sum_{i in S_B} r_i(a0,b)`.

The challenger score is `score(a0)-M_DARM(a0,b)`. Missing learned pair edges fall back exactly to the local margin. Residuals on non-anchor pairs cannot change the final action. With zero residual, DARM is exactly the direct selected-local planner (before the same existing safety/utility post-processing), rather than a different global tournament.

This is closer to the paper's one-sided preservation theorem: final decision preservation requires protecting the teacher-relevant winner-versus-decisive-rival margins, not constructing a globally consistent all-pairs utility field.

## Algorithm: Decisive Boundary Residual (DBR)

DBR retains only the useful core of LBPR: a low-rank evidence-attributable signed pair correction with exact antisymmetry by construction. It removes the LEA/CCBR endpoint posterior gate.

For evidence atom `i` and runtime ordered pair `(a,b)`, DBR uses an atom factor multiplied by an action/query **difference** factor and a bias-free output. Consequently

`r_i(a,b) = -r_i(b,a)`

holds after training, not merely at zero initialization. The output layer is zero initialized, so the V62 selected-local anchor is an exact step-zero no-op.

DBR is trained directly against teacher atom-pair residual labels on the existing sampled decision pairs. It does not require endpoint identity to be confident before value learning can act. Two causal pair-sampling arms are provided:

- `BROAD`: existing decision-weighted winner / hard / near-margin pairs;
- `LITERAL`: the same broad support plus a bounded exact teacher-winner -> leave-one-atom-out-flip pair quota and matching literal atom weight.

The comparison tests whether literal-critical emphasis is useful **after** endpoint gating is removed. It does not redefine criticality and does not make teacher futures runtime inputs.

## Fixed-interface and no-repeat invariants

- Keep `B=16`, proposal M, auditable evidence atoms, DA-EPC, exact selector and runtime rival construction unchanged.
- Freeze the V62 foundation, proposal/family stack, CCBR/BCHA and legacy residual-action/set-potential branches.
- Do not retry BCHA: the family oracle is 1.0/1.0 with zero gap.
- Do not retry AP-WCCA/AP-WRCCA/LCV/FPCCA-F6/F8 or proposal-only CCBR/LEA as main acquisition algorithms.
- Do not increase B/M, relax certificate thresholds, or rerun beam/swap/brute-force selector search.
- Do not repeat V46/V49 broad arbitrary global pair tournaments, V55 Hodge/global potential, V56 generic evidence action potential, or V59 generic set-conditioned potential. DARM only consumes anchor-challenger margins.
- Do not scratch-retrain the complete foundation in this screen. Keep `outputs_v62_dcab_ewfc_fast_2gpu_v1/train/bdse_v62_dcab_ewfc.best.pt` to preserve causal attribution.

## V64.3.7 screen / promotion rules

A V64.3.7 run is invalid unless the zero-residual epoch restores a strong selected-local anchor: `val_teacher_action_match >= 0.24`. This explicitly catches accidental regression back to the weak V64.3.6 legacy tournament.

DBR activation additionally requires non-zero adapter parameter movement and residual RMS. When emitted, DARM runtime activation must be >0.99 and training anchor-pair coverage must be at least 0.20.

A meaningful value signal requires:

- pair-full teacher-match gain >= `+0.01` absolute versus the same run's epoch -1 anchor;
- pair-full advantage over same-epoch local pair-full >= `+0.005`;
- teacher action match delta >= `-0.005`;
- budget-vs-pair-full delta >= `-0.02`;
- beneficial residual intervention minus harmful residual intervention >= 0.

Automatic full promotion is stricter: the same row must also improve deployed teacher action match by >= `+0.005`. A pair-full-only gain is a mechanism result, not a full-planner winner.

## Warm-start / next decision rule

Continue the immutable V62 warm start for V64.3.7. If DBR has adequate pair coverage, non-zero parameter/residual activation and a restored strong selected-local anchor but cannot improve pair-full, then the next hypothesis is frozen action/evidence pair-feature capacity. The next experiment should be a **small selective pair-feature/action-evidence adapter or selective encoder unfreeze**, not full scratch training.

If DARM+DBR materially improves pair-full and final teacher action, freeze the value mechanism and return to acquisition. The preferred future acquisition target is then the learned/realized **decisive-margin utility** of an atom under DARM, not another generic literal-critical classifier. This preserves the paper chain: fixed budget -> auditable atoms -> decisive margins -> budgeted evidence -> decision preservation.

## Engineering hardening

1. DBR output is bias-free; a non-zero-weight regression test enforces exact pair antisymmetry.
2. Training DARM averages duplicated directed observations of the same anchor edge before action scoring, avoiding accidental double counting when both directions are sampled.
3. Non-anchor residual edges are tested to have zero effect on the DARM action; anchor margin crossing is tested to change it.
4. Safety atoms are included consistently in DBR pair regression and pair-action loss (`exclude_safety_atoms_from_pair_regression=false`, `exclude_safety_atoms_from_pair_action_loss=false`) so train and deployment do not silently use different evidence supports.
5. V62 checkpoint loading explicitly permits the newly introduced `decisive_boundary_pair_adapter.*` to be absent while still rejecting a missing/shape-mismatched foundation tensor.
6. Fixed a copied provenance error where full-train YAMLs still said `screening_only=true`; the validator now binds metadata/provenance screening flags.
7. Fixed the V64.3.7 config validator's cross-config strict-family set to include `v64.3.7` rather than silently skipping V64.3.7 signature matching.
8. CCBR/LEA/BCHA are disabled in V64.3.7 value-isolation configs; only DBR is trainable. This removes unused loss/forward work without weakening the tested value supervision.

Final static/regression status: all **299 collected tests passed, 0 failed** when executed in bounded chunks (the monolithic command exceeded the execution harness time limit without reporting a test failure); 33 warnings are pre-existing Transformer nested-tensor warnings. Six V64.3.7 YAMLs parse, four launchers pass `bash -n`, broad/literal screen and full train/eval config contracts pass, and `compileall` passes. No nuPlan GPU training is claimed in the delivery environment.

# V64.3.7.1 — DARM+DBR screen protocol hotfix (2026-08-11)

## Uploaded V64.3.7 result correction

The uploaded `outputs_v64_3_7_darm_dbr_screen_matrix_2gpu_v1` contains only the BROAD arm because the old checker returned exit code 3 under `set -e` before LITERAL could run. The BROAD algorithm itself did **not** fail to activate.

Immutable epoch -1 anchor:

- teacher match = `0.264`;
- local pair-full = `0.264`;
- pair-full = `0.264`;
- teacher regret = `14484.4613`;
- pair-full regret = `14079.9774`;
- critical Top-M micro recall = `0.3601533`;
- selected critical micro recall = `0.2605364`;
- proposal decisive recall = `0.7915091`.

The robust positive row is epoch 3:

- teacher match = `0.282`, **+1.8pp**;
- pair-full = `0.274`, **+1.0pp**;
- local pair-full stays frozen at `0.264`;
- pair-full-over-local advantage = **+1.0pp**;
- teacher regret = `13367.1395`, improvement `-1117.32`;
- pair-full regret = `13673.7794`, improvement `-406.20`;
- beneficial/harmful residual intervention = `0.022/0.012`, net **+1.0pp**;
- beneficial/harmful pair-full->budget compression = `0.016/0.008`, net **+0.8pp**;
- DBR parameter delta RMS max = `0.005218`;
- DBR residual RMS max = `0.003149`;
- critical Top-M, selected-critical, and proposal-decisive metrics remain exactly unchanged.

This is the first clean evidence in this branch that downstream decisive pair-value/final aggregation can improve the fixed-budget final decision while acquisition is held constant. DARM+DBR-BROAD is therefore a **provisional positive main-algorithm candidate**, not a negative result. Statistical/generalization confirmation still requires the missing LITERAL arm and a larger full pipeline.

## Engineering/protocol errors fixed

1. The old screen checker used `decisive_anchor_full_pair_coverage >= 0.20` as a validity gate. The uploaded run reached `0.1990928`. This metric is all-challenger anchor-star coverage under a discrete sampled pair graph, not exact teacher-correction-edge coverage. The 0.20 boundary was arbitrary and caused a false invalid result. V64.3.7.1 keeps it diagnostic-only.
2. The old value gate required `budget_vs_pair_full_delta >= -0.02`. That is semantically misaligned with the BDSE target: the paper seeks teacher decision preservation under fixed B=16, not imitation of a learned pair-full surrogate. In the uploaded epoch 3, B16 divergence is net teacher-beneficial. V64.3.7.1 gates on teacher/pair-full action gain, regret, residual intervention direction, and beneficial-vs-harmful compression instead.
3. A non-promoted scientific screen returned process exit code 3, so `RUN_V64_3_7_DARM_DBR_SCREEN_MATRIX_2GPU.sh` aborted under `set -e` before LITERAL. The checker now exits zero for an interpretable negative result; malformed inputs still fail normally.
4. The matrix now re-audits any existing train log with the current checker instead of trusting stale provenance. This allows the completed BROAD training to be reused without GPU retraining.
5. Validation export now includes `decisive_anchor_margin_*` tournament diagnostics. Earlier `runtime_darm_active_min` was null because the runtime DARM diagnostic existed in the tournament result but was filtered from validation metrics.

## Revised promotion semantics

`instrumentation_valid` requires a restored strong selected-local anchor, non-zero DBR parameter movement/residual, and observed pair-graph supervision. All-challenger coverage is not a hard algorithm gate.

`meaningful_value_gain` requires:

- pair-full gain >= `+1.0pp` vs epoch -1;
- pair-full advantage over same-epoch local pair-full >= `+0.5pp`;
- pair-full teacher regret non-worse;
- beneficial residual intervention strictly exceeds harmful intervention.

`deployment_gain` requires:

- final fixed-B teacher match >= `+1.0pp`;
- final teacher regret non-worse;
- pair-full->budget compression net is non-harmful when available.

`full_promotion = instrumentation_valid AND meaningful_value_gain AND deployment_gain`.

Re-auditing the uploaded BROAD log with this definition selects epoch 3 and returns `instrumentation_valid=true`, `meaningful_value_gain=true`, `deployment_gain=true`, `full_promotion=true`.

## Algorithm priority after the uploaded result

The priority ordering remains, with stronger evidence:

1. **Decisive pair value + final aggregation stays first priority and is partially validated.** DARM+DBR improves final teacher decisions while all acquisition metrics remain frozen.
2. **Acquisition proposal-score generalization remains second priority.** Critical Top-M is still `0.3601533`, but it is no longer justified to modify acquisition before completing DARM+DBR causal validation.

Do not introduce another pair architecture, global tournament/potential, BCHA, CCBR/LEA gate, larger B/M, scratch training, or acquisition loss before completing the missing LITERAL screen and the guarded full pipeline.

If the DARM+DBR gain survives the full pipeline, freeze the value/aggregation mechanism. The next acquisition hypothesis should be **DARM-consistent decisive-margin marginal utility**: train the cheap proposal path to rank auditable atoms by how much they reduce the one-sided decisive-margin deficit under fixed B=16. This connects acquisition directly to the theorem-aligned DARM decision objective instead of repeating sparse binary critical classification.

# V64.3.8 — BDMU: Budgeted Decisive-Margin Marginal Utility Acquisition (2026-08-12)

## New uploaded V64.3.7 matrix result

The latest uploaded `outputs_v64_3_7_darm_dbr_screen_matrix_2gpu_v1` now contains **both BROAD and LITERAL**. This supersedes the previous V64.3.7.1 note that only BROAD was available.

Common immutable epoch -1 anchor: teacher/pair-full/local-pair-full action match `0.264/0.264/0.264`, teacher regret `14484.4613`, pair-full regret `14079.9774`, exact critical Top-M `0.3601533`, selected exact-critical `0.2605364`, proposal decisive recall `0.7915091`.

BROAD selected epoch 3: teacher `0.282` (+1.8pp), pair-full `0.274` (+1.0pp), local pair-full unchanged `0.264`, teacher regret `13367.1395` (-1117.32), pair-full regret `13673.7794` (-406.20), residual intervention net +1.0pp, compression net +0.8pp.

LITERAL selected epoch 2: teacher `0.286` (+2.2pp), pair-full `0.280` (+1.6pp), local pair-full unchanged `0.264`, teacher regret `14027.5786` (-456.88), pair-full regret `14031.3813` (-48.60), residual intervention net +1.6pp, compression net +0.6pp.

Acquisition metrics are exactly unchanged in both arms. Therefore LITERAL is the provisional main **value-supervision** candidate because it best preserves the teacher decision, while BROAD's stronger regret reduction exposes an action-identity versus severity tradeoff. LITERAL already contains BROAD support plus the bounded exact-boundary quota; do not construct a redundant BROAD+LITERAL mixture.

The value-side causal question is now sufficiently positive to justify moving to acquisition **only after LITERAL reproduces on the full 50k/1k pipeline**.

## Bottleneck update

The first bottleneck (decisive pair value + final aggregation) is partially resolved by DARM+DBR. The unresolved bottleneck is now acquisition generalization: exact critical Top-M remains `0.3601533` and selected exact-critical remains `0.2605364` despite downstream value gains.

Do not retry AP-WCCA/AP-WRCCA, LCV, FPCCA-F6/F8, CCBR/LEA objectives, BCHA, larger B/M, full proposal unfreeze, binary literal-critical BCE, global pair tournament/potential, evidence/set potential, beam/swap/bruteforce selection, or scratch foundation training.

## Algorithm: Budgeted Decisive-Margin Marginal Utility (BDMU)

BDMU trains acquisition on the same one-sided decision-margin object used by DARM instead of predicting whether an atom is a sparse literal-critical event.

For full teacher winner `w`, nearest teacher rivals `b`, and immutable frozen-foundation B-set `S_B`, define a normalized full teacher margin `m_T(w,b)` and preservation threshold

`gamma_b = min(m_T, max(rho*m_T, margin_floor), margin_cap)`.

The reference deficit is

`delta_b(S_B) = [gamma_b - m_{S_B}(w,b)]_+`.

Selected-atom utility is the deficit increase under removal. Missed-atom utility is the best deficit reduction under a **budget-feasible single exchange** `S_B-{j}+{i}`; direct addition is allowed only when the reference has true budget slack. This revision is important: the supervision never relies on a B+1 counterfactual and therefore respects the fixed planner-interface budget even in the teacher target.

Utilities are softly aggregated over the nearest `R=4` teacher rivals and divided by query cost (`cost_power=1`). Exact winner-flip atoms are a high-value limiting case, not the only positive label. The R=1 and no-cost settings are reserved as theory ablations.

## Strict causal isolation

- Warm-start only from a **full-pipeline promoted V64.3.7 DARM+DBR-LITERAL checkpoint**.
- Freeze DARM, DBR and all foundation modules.
- Train only the existing zero-init `critical_proposal_adapter` with complete-candidate boundary-routing representation.
- CCBR is representation only; all old CCBR/LEA/HCBE/ACRA/literal-critical objectives are disabled.
- Reconstruct immutable foundation proposal logits by subtracting the trainable residual before computing the reference B-set, preventing target chasing.
- The training reference B-set uses the existing all-GPU one-shot MARS/HAB budget surrogate for throughput; it is explicitly an immutable budget-feasible reference approximation, while exact deployed selector behavior remains an evaluation target.
- Keep B=16, proposal Top-M, evidence atoms, selector, DA-EPC and runtime final aggregation unchanged.
- V64.3.8 checkpoint contract does **not** allow `decisive_boundary_pair_adapter.*` to be missing. Pointing at V62 directly must fail.

Tightened paper chain:

`fixed planner-interface budget -> auditable evidence atoms -> budget-feasible BDMU decisive-margin utility -> budgeted acquisition -> DARM one-sided margin preservation -> final decision preservation`.

The theorem bridge is conditional and reuses the DARM certificate: for a complete decisive-rival set, `D(S)=sum_b pi_b[gamma_b-m_S(w,b)]_+` with positive weights satisfies `D(S)=0 =>` every decisive margin is positive; BDMU is the budget-feasible local decrease of this same deficit. Practical R=4 is a training approximation, while the exact downstream certificate remains the verifier.

## Promotion protocol

1. Reproduce DARM+DBR-LITERAL on full train/val and require the existing teacher-oriented full audit to pass. Otherwise BDMU launchers block.
2. BDMU 12k/500 screen must activate the adapter, expose non-zero teacher utility support, improve BDMU Top-M utility capture by >=2pp plus selected-capture or exact-critical corroboration, and produce a teacher-action >=+0.5pp or regret >=2% gain without teacher/regret harm.
3. BDMU 50k/1k full run must reproduce the same mechanism+deployment gate before test/closed-loop or ablations.
4. R1 and no-cost ablations are blocked until **both** main BDMU screen and main BDMU full audit pass.
5. Held-out test is run once after checkpoint/hyperparameters are frozen. CL20 is integration debug only; CL100 non-reactive and reactive use the same frozen checkpoint.

## Engineering / efficiency changes

- Added a strict BDMU-only loss fast path. It is active only when BDMU is enabled and is the sole positive `loss_weights` entry; legacy configs keep the original loss graph.
- Fixed a config-default trap: `load_config` injects a default family loss if omitted, so V64.3.8 explicitly sets `family: 0.0`. Without this, the intended fast path would silently be disabled.
- BDMU diagnostics are opt-in; V64.3.7-and-earlier evaluation pays no new utility-diagnostic overhead.
- Full V64.3.7 and V64.3.8 launchers save every epoch and train before evaluation. Checkpoint selection/audit occurs on validation only, then open/closed-loop runs use the frozen selected epoch.
- Uploaded LITERAL e3 profiling shows loss construction can be a material cost (`73.63 s` of `217.81 s` epoch; pair sampling `1.92 s`). The fast path removes zero-weight legacy loss construction, but **no V64.3.8 speedup is claimed until the next GPU run measures it**.

## New files / tests

- `bdse/model/decisive_margin_utility.py`
- BDMU objective + strict fast path in `bdse/model/losses.py`
- BDMU validation metrics in `bdse/experiments/evaluate_open_loop.py` and training validation export
- `bdse/tools/check_v64_3_8_bdmu_screen.py`
- `bdse/tools/select_decision_pareto_checkpoint.py`
- V64.3.8 main train/screen/closed-loop configs plus R1/no-cost configs
- 2-GPU screen/full/theory-ablation launchers
- regression tests covering Torch/NumPy utility equivalence, cost normalization, scalar-teacher mismatch exclusion, fixed-budget no-B+1 behavior, strict acquisition isolation, zero-init preservation, checkpoint contract, fast-path opt-in and legacy diagnostic opt-in.

# V64.3.9 — AF-BDMU: Adaptive-Frontier Budgeted Decisive-Margin Utility Acquisition (2026-08-13)

## Re-audit of the uploaded V64.3.7 full pipeline

The previous V64.3.8 note treated the full-pipeline stop as a missing-checkpoint / calibration-launch failure. That diagnosis is **not the failure represented by the newly uploaded artifacts**. The current full DARM+DBR-LITERAL training log contains epochs `-1,0,...,5`, early-stops after epoch 5, and the launcher log records a final model save. The uploaded archive omits `.pt` weights, but the training process itself completed.

The sequence was blocked by the old V64.3.7 promotion checker. It required the epoch -1 validation anchor to satisfy `teacher_action_match >= 0.24`. On the full 1000-scene validation set the anchor is `0.180`, although the zero-residual pair-full and selected-local interfaces agree (`0.181/0.181`). This is a protocol bug: an absolute score floor tied to the old first-500 screen cannot be used as an instrumentation contract on a different validation composition.

The distribution shift is large. On the selected epoch-1 checkpoint, the first 500 validation samples have teacher/pair-full/local match about `0.284/0.286/0.264`, while the second 500 have about `0.112/0.110/0.098`. Exact-critical Top-M recall is actually higher on the difficult second half, so the lower action-match ceiling cannot be attributed to acquisition alone.

V64.3.7.2 therefore keeps the historical absolute anchor floor only for the original screen protocol. For `*FULL*` variants it instead checks the zero-residual **interface consistency** contract (`pair-full ~= selected-local`, tolerance 0.005) and the existing DARM activation/value/deployment causal gates. Re-auditing the uploaded full log now selects epoch 1 and gives:

- teacher match `0.180 -> 0.198` (`+1.8pp`);
- pair-full match `0.181 -> 0.198` (`+1.7pp`);
- selected-local match unchanged at `0.181`;
- teacher regret `19759.44 -> 16496.54` (`-3262.91`);
- pair-full regret `19545.80 -> 16461.69` (`-3084.11`);
- residual intervention beneficial/harmful = `0.019/0.002`, net `+1.7pp`;
- acquisition metrics are exactly unchanged.

Under the repaired protocol `meaningful_value_gain=true`, `deployment_gain=true`, and `full_promotion=true`. This is stronger full-pipeline causal evidence that DARM+DBR improves the downstream teacher decision under frozen acquisition.

## Corrected bottleneck interpretation

Two certificate diagnostics must not be conflated:

- `selector_aocc_pairwise_certified_pair_fraction_raw = 0.031` is the conservative AOCC pairwise lower-bound surrogate;
- `evidence_certificate_fraction = 0.946` is the exact runtime B=16 -> Top-M downstream winner-preservation certificate used by the deployed guard.

Therefore the current primary controllable bottleneck is **not** “certificate utilization inside the selected B=16 set”. The selected B-set preserves the current Top-M downstream action in 94.6% of scenes.

The clearest acquisition failure is before that stage:

- exact winner-flip critical Top-M recall = `0.4279` micro / `0.3838` macro;
- exact winner-flip selected recall = `0.3076` micro / `0.2777` macro;
- proposal decisive recall = `0.8005`;
- frozen-family-slot oracle Top-M critical recall = `1.000`;
- global oracle Top-M critical recall = `1.000`.

Thus the existing semantic family capacity can contain the decisive atoms, but learned proposal ranking does not put enough of them into Top-M. This is the acquisition bottleneck that can be causally tested while value is frozen.

There is also a **global value/model ceiling**: even the selected full-interface pair pathway reaches only `0.198` teacher match. Acquisition cannot explain the remaining roughly 80% teacher mismatch. The correct next experiment is therefore not “acquisition is the only bottleneck”; it is: **under a frozen, causally validated DARM+DBR value interface, test whether fixing Top-M decisive-utility ranking produces further teacher decision gain.** If the acquisition mechanism moves without an endpoint gain, stop selector iterations and pivot to value/frontier representation.

## Keep / modify / reject from V64.3.8 BDMU

Keep:

- fixed planner-interface evidence budget `B=16`;
- auditable evidence atoms and existing HAB/selector interface;
- frozen DARM+DBR/foundation for causal attribution;
- immutable frozen-foundation reference B-set;
- budget-feasible direct-add/single-exchange utility (never B+1);
- cost-normalized continuous one-sided margin deficit;
- exact winner-flip labels as a limiting-case diagnostic, not the primary loss.

Upgrade:

1. **Adaptive decisive rival frontier.** Fixed `R=4` can omit a teacher rival that lies on the same local decision frontier. AF-BDMU always includes at least four nearest positive-margin rivals, additionally includes rivals with normalized teacher margin below `max(0.05, 2 * nearest_margin)`, and caps the set at eight.
2. **One-sided weakest-rival preservation.** The old soft weighted mean can hide one decisive rival behind several easy rivals. The reference deficit becomes a convex mixture of weighted-mean and worst-frontier deficit, with `worst_rival_weight=0.35`.
3. **Top-M swap ranking.** Generic absolute/listwise utility regression is not sufficient when the diagnosed failure is the proposal boundary. The new pairwise term is formed only from a positive-utility atom currently missed by Top-M and a lower-utility atom currently occupying Top-M. It directly trains the feasible one-for-one proposal-boundary correction without increasing M or B. Main weights are listwise `0.65` + Top-M swap rank `0.35`.

Do **not** add a large binary `certificate contribution` term to the utility in this version. The exact downstream evidence certificate is already `0.946`; optimizing the conservative 3.1% AOCC surrogate as if it were the deployed certificate would target the wrong bottleneck. Likewise, do not reintroduce LITERAL winner-flip BCE as the main acquisition objective; the uploaded experiments already show that binary boundary identity is useful but too sparse to encode margin severity.

## CCF-A mainline and novelty position

Keep the tightened paper chain:

`fixed planner-interface budget -> auditable evidence atoms -> budget-feasible decisive-margin marginal utility -> budgeted acquisition -> DARM one-sided decisive-margin preservation -> final decision preservation`.

The novelty claim should be framed as **Auditable Budgeted Decision Preservation**, not generic decision-aware selection. AF-BDMU makes the acquisition stage match the same one-sided margin object used by DARM, while the exact interface budget and frozen-value experiment make the causal contribution auditable. The paper should not claim a distribution-free safety certificate; the certificate is an empirical/calibrated decision-preservation interface certificate.

## New protocol exit rule

V64.3.9 promotion requires:

- AF-BDMU adapter moves and the adaptive frontier / Top-M swap-rank objective is active;
- Top-M continuous utility capture improves by at least `+1.5pp`;
- either selected utility capture improves by `+0.5pp` or exact critical Top-M recall improves by `+0.5pp`;
- fixed-B teacher match improves by `+0.5pp` or teacher regret improves by at least 2%;
- teacher match/regret are non-harmful within the stated tolerances.

**Exit rule:** if Top-M utility/critical recall improves but teacher match/regret does not, classify acquisition as no longer binding under the frozen value interface and pivot the next algorithm cycle to a decisive value/frontier model. Do not keep tuning acquisition losses.

## Engineering and efficiency changes

1. **Full-pipeline gate fix.** V64.3.7 full promotion uses interface consistency instead of the stale first-500 absolute anchor floor.
2. **Training artifact contract.** `validate_training_artifacts.py` requires a parseable trained-epoch log plus final checkpoint, and can require an epoch checkpoint. This separates genuine training completion from launcher/promotion failures.
3. **Representative capped validation.** `PreprocessedBDSEDataset` supports `max_scenarios_strategy={first,uniform,uniform_blocks}`. V64.3.9 screens use `uniform_blocks` so a 500-scene screen covers the whole ordered validation cache while retaining short contiguous blocks for I/O locality.
4. **Input pipeline benchmark.** The worker benchmark now sweeps both worker count and prefetch factor and reports the measured best combination instead of assuming prefetch=2.
5. **BDMU fast path remains.** Only the acquisition utility graph is built when BDMU is the sole positive loss.
6. **Do not change open-loop model semantics for speed in this causal round.** The selected-checkpoint 1000-scene open loop averages `592.55 ms` internal planner latency: prediction is `458.57 ms` (~77%), selector `66.98 ms`, tournament `8.95 ms`. The next safe speed work is batched/cached prediction after AF-BDMU semantics are frozen, not an approximate selector/value rewrite in the same experiment.
7. DARM full training profiling shows data wait is large (`452--702 s/epoch` out of `963--1482 s/epoch`), so worker/prefetch measurement is justified before the next long run.

## New files / tests

- adaptive-frontier + mean/worst decisive-deficit support in `bdse/model/decisive_margin_utility.py`;
- Top-M swap-ranking acquisition loss in `bdse/model/losses.py`;
- AF-BDMU diagnostics in `bdse/experiments/evaluate_open_loop.py`;
- representative capped-cache selection in `bdse/data/nuplan_dataset.py` and CLI/wrapper plumbing;
- `bdse/tools/check_v64_3_9_af_bdmu_screen.py` with an explicit acquisition->value pivot diagnosis;
- `bdse/tools/validate_training_artifacts.py`;
- V64.3.9 train/screen/closed-loop configs and 2-GPU launchers;
- regression tests for Torch/NumPy adaptive-frontier equivalence, strict frozen-value acquisition isolation, representative uniform-block validation sampling, and full-gate interface consistency.

## V64.3.9 no-repeat constraints

Continue all V64.3.8 no-repeat constraints. Additionally:

- do not tune an AOCC binary certificate bonus unless a future experiment first shows that the exact B->TopM evidence certificate has become binding;
- do not use prefix-500 validation for promotion on the current cache ordering;
- do not weaken the full-pipeline gate to an unconditional bypass; the replacement is a causal interface contract, not `skip_audit`;
- do not change DARM/DBR while testing AF-BDMU;
- if AF-BDMU improves Top-M utility without improving teacher decision/regret, do not run another acquisition-loss family next.

# V64.3.9-R1 — Engineering correction: exact runtime Top-M parity (2026-08-13)

## Status of the uploaded V64.3.9 screen

The uploaded `outputs_v64_3_9_af_bdmu_screen_2gpu_v1` is **not a clean AF-BDMU algorithm-negative result** and must not be used to trigger a new acquisition/value redesign.

The launcher/config/checkpoint contracts all passed and optimization was active: train loss decreased from `2.2914` (epoch 0) to `2.1394` (epoch 3), listwise loss from `3.2052` to `3.0523`, Top-M swap-rank loss from `0.5941` to `0.4409`, and the proposal adapter residual RMS increased from `0.2316` to `1.0341`. However, validation exact-critical Top-M recall stayed exactly `0.23734`, while validation BDMU Top-M utility capture did not improve over the epoch -1 anchor (`0.47680`; selected epoch 1=`0.47511`).

Code audit found a material training/deployment interface mismatch exactly on the mechanism AF-BDMU is intended to test:

- real runtime Top-M post-processing in `BDSEModel.predict_certificate_numpy` uses **structural-safety exclusion/refill first, then group-aware soft-interaction reservation**;
- the BDMU-only training fast mask used **soft-interaction reservation first, then structural-safety exclusion/refill**;
- the training fast reservation ignored `evidence_agent_group_ids`, while runtime deliberately reserves distinct interaction-agent groups;
- the frozen-foundation reference B-set was also conditioned on this fast/surrogate Top-M pool.

The active V64.3.9 config has both `decision_budget_excludes_structural_safety=true` and `min_soft_interaction_topm_slots=2`, so this is not a dormant code path. It changes which atoms are labeled `missed`/`occupied` by the AF-BDMU Top-M swap loss and can also change the utility reference set. Therefore the V1 screen cannot cleanly answer whether AF-BDMU itself works.

**Causal decision:** do not run V64.3.9 full/test/closed-loop from the V1 screen, but also do not pivot to a new algorithm yet. Re-run Phase 1 after restoring exact training/deployment Top-M semantic parity.

## Engineering-only fix

No AF-BDMU algorithm object, utility formula, loss weight, budget, DARM/DBR value model, or promotion threshold was changed.

1. Added `finalize_runtime_topm_policy(...)` in `bdse/planner/selector.py` as the canonical deployment Top-M post-processing helper. Its order is structural exclusion/refill (or mandatory-hard handling) -> group-aware soft-interaction reservation, always at fixed M.
2. `BDSEModel.predict_certificate_numpy` and the planner rule fallback now use the same canonical helper.
3. `_runtime_hab_topm_hard_mask(...)` now calls the same canonical helper and is the source of **every scene's** hard membership mask in BDMU-only training. The previous GPU fast mask remains diagnostic only.
4. The frozen V64.3.7 reference B-set now first obtains the exact frozen-foundation runtime Top-M pool and only then applies the pre-existing fast budget selector. Thus the BDMU target is no longer conditioned on a different proposal interface.
5. Added explicit V64.3.9 config contracts:
   - `topm_membership_source: exact_runtime_hab`
   - `reference_topm_pool_source: exact_runtime_hab`
6. Added `bdse/tools/check_v64_3_9_runtime_topm_contract.py`. The launcher runs it before spending GPU compute. A synthetic group-diversity fixture deliberately exposes the historical mismatch: canonical/exact Top-M is `{1,3,5}`, while the legacy fast surrogate is `{1,3,4}`.
7. Added training diagnostics `bdmu_runtime_topm_surrogate_jaccard` and `bdmu_runtime_topm_exact_fraction`; the latter must be `1.0` in the repaired BDMU-only path.
8. Screen/full launcher defaults use new `*_v2_runtime_parity` output roots to prevent contamination by the invalid V1 artifacts.

## Validation

- targeted V64.3.9 tests: `5 passed`;
- complete repository suite after the fix: `315 passed`, `34 warnings` (existing PyTorch Transformer nested-tensor warnings only);
- V64.3.9 train/eval config contract: PASS;
- runtime Top-M semantic contract: PASS;
- shell syntax for repaired screen/full launchers: PASS.

## Next scientific decision rule

The no-repeat/algorithm stop rules from V64.3.9 remain unchanged, but they may be applied **only after the repaired Phase-1 screen**. In particular, do not interpret the V1 failure as evidence to pivot to value/frontier modeling. A value pivot is justified only if an exact-interface rerun shows that acquisition mechanism metrics improve while teacher match/regret fail to respond.

# V64.3.10 — HAP-BDMU: HAB-Projected Budgeted Decisive-Margin Utility (2026-08-13)

## Evidence status of the repaired V64.3.9 runtime-parity screen

The uploaded `outputs_v64_3_9_af_bdmu_screen_2gpu_v2_runtime_parity` is the first V64.3.9 screen that is valid for **partial algorithm attribution**.  The R1 engineering contracts are satisfied: the exact runtime Top-M semantic contract passes, `bdmu_runtime_topm_exact_fraction=1.0`, the intended acquisition-only adapter is optimized, and trained checkpoints/artifacts were produced.  Therefore the old V1 training/deployment Top-M mismatch is no longer a sufficient explanation for the stop.

The repaired screen still fails both mechanism and deployment promotion:

- anchor teacher match/regret: `0.178 / 20133.34`;
- selected epoch 1 teacher match/regret: `0.182 / 20038.65` (`+0.4pp`, only `0.47%` regret improvement);
- moving-reference BDMU Top-M utility capture: `0.47680 -> 0.47511`;
- exact winner-flip Top-M recall: `0.23734 -> 0.23734` (no movement);
- proposal decisive recall: `0.75366 -> 0.73941` (decreases);
- exact B->TopM evidence certificate: `0.928 -> 0.922`;
- optimization is active: train loss `2.2914 -> 2.1394`, AF-BDMU listwise `3.2052 -> 3.0523`, legacy swap-rank `0.5941 -> 0.4409`, and adapter residual RMS reaches `1.0341`.

A stronger negative mechanism signal is visible even before considering validation generalization: the train-side hard Top-M utility capture changes only from about `0.56435` at epoch 0 to `0.56684` at epoch 3 (`+0.25pp`) while the swap-ranking objective falls by about 26%.  Thus the learned score surrogate is being optimized, but it rarely crosses the actual hierarchical hard-admission boundary.

**Do not loosen the V64.3.9 screen threshold to promote this run.**  Full/test/closed-loop would spend compute before the acquisition mechanism itself has demonstrated a meaningful change, so downstream endpoint variation would not identify the cause.

## Protocol correction discovered from the V64.3.9 V2 audit

V64.3.9 training defines BDMU utility against an immutable **frozen-foundation reference B-set**.  However, the open-loop metric `val_teacher_bdmu_topm_utility_capture` reconstructs utility relative to the **current checkpoint's selected B-set**.  It is therefore an endogenous/moving target and cannot be the primary causal mechanism gate for an acquisition-only experiment.

This protocol issue does **not** erase the independent negative evidence above (exact-critical Top-M is unchanged and proposal decisive recall decreases), but it means V64.3.9 cannot cleanly quantify how much of the fixed BDMU acquisition gap the adapter closed.  V64.3.10 therefore sets `VAL_MODE=both` and promotes only from `val_bdmu_*` validation-loss diagnostics computed from the exact same frozen reference target used by training.  Moving-reference open-loop BDMU metrics remain descriptive only.

## Bottleneck update

The next bottleneck is not generic acquisition score regression.  It is the mapping

`continuous decisive-margin utility -> realizable HAB hard Top-M admission`.

The V64.3.9 AF-BDMU swap loss compared any positive-utility missed atom with low-utility atoms occupying Top-M.  Even after exact runtime Top-M membership parity, that pair need not describe a membership state the **frozen hierarchical interface** can realize after family-slot allocation, structural bypass/exclusion, and group-aware interaction reservation.  A large differentiable score update can therefore reduce ranking loss without producing the intended discrete admission.

This is distinct from the historical V64.3.4--V64.3.6 selector beam/swap/bruteforce failures: those repaired the **B-layer selector inside a fixed proposal pool**.  V64.3.10 operates at the proposal admission boundary and keeps the downstream B selector frozen.

## Algorithm: HAP-BDMU

Keep the V64.3.9 continuous budget-feasible BDMU target, adaptive decisive frontier, weakest-rival deficit, B=16, proposal size M, DARM, DBR, evidence atoms, family gate, and foundation checkpoint unchanged.

Add an **exact HAB-feasible utility projection**:

1. Compute continuous one-sided BDMU utility `u_i` using the immutable frozen-foundation reference B-set.
2. Treat detached `u_i` as oracle proposal scores and pass them through the **same deployed HAB Top-M operator**: frozen family slots -> structural policy -> group-aware interaction reserve.
3. The resulting `S_HAB^U` is the best utility-induced target that is directly realizable by this fixed hierarchical interface.  It is not an unconstrained global Top-M oracle.
4. Form structured admission positives only from `S_HAB^U \ S_current` and negatives only from `S_current \ S_HAB^U`.
5. Rank each missed oracle atom primarily against displaced atoms from the **same frozen HAB family stratum**; use cross-family fallback only when finalization/global refill produces no same-family displacement. Rank by the original continuous utility gap.  Main weights are `0.15` listwise support + `0.85` HAB-feasible admission ranking; the V64.3.9 unconstrained swap rank is disabled.

This keeps the optimization object aligned with the same planner-interface feasibility constraints used at deployment instead of changing M/B or bypassing HAB.

## Causal attribution protocol implemented in code

V64.3.10 makes the following decomposition executable and auditable in every validation epoch:

- **C0 fixed target:** `val_bdmu_reference_selected_utility_capture` and the frozen-reference BDMU utility use exactly the training target.
- **C1 interface ceiling:** `val_bdmu_hab_oracle_topm_utility_capture` projects that same utility through the exact frozen-family deployed HAB policy.  `val_bdmu_hab_oracle_gap` measures available proposal-admission headroom.
- **C2 learned admission:** `val_bdmu_current_topm_utility_capture` measures the learned proposal on the same fixed target and same interface.  Promotion requires meaningful oracle-gap closure rather than a moving target.
- **C3 downstream transmission:** teacher match/regret is evaluated with budget, DARM, DBR and the foundation value interface frozen.

Decision rules:

1. If the C1-C2 oracle gap is too small, acquisition capacity is not binding; **do not tune another proposal loss**.
2. If C1 leaves headroom but C2 does not move, structured admission learning is the bottleneck.
3. If C2 closes a meaningful fraction of the C1 gap but C3 does not improve, trigger `pivot_to_value_frontier=true`; the next cycle must target decisive value/frontier modeling, not acquisition.
4. Only if C2 and C3 both improve is HAP-BDMU promoted to full/test/closed-loop.

This protocol is intended to be part of the paper contribution: under one fixed planner interface it separates *support/interface capacity*, *learned acquisition*, and *downstream value transmission* instead of attributing all endpoint changes to a monolithic planner update.

## Promotion thresholds

The screen uses fixed-reference validation metrics and requires:

- instrumentation valid and exact runtime Top-M semantics;
- anchor HAB oracle gap >= `1pp` utility capture (otherwise acquisition is classified non-binding);
- learned fixed-reference Top-M capture improves >= `1pp`;
- at least 10% of the anchor HAB oracle gap is closed;
- teacher match improves >= `0.5pp` **or** teacher regret improves >= 2%;
- teacher match/regret remain within non-harm tolerances.

These thresholds are not weakened to rescue V64.3.9.  If HAP-BDMU does not move the fixed-reference mechanism, full/test/CL remain blocked.

## Engineering changes

- `run_v64_saqa_bcc.sh` supports `VAL_MODE` from the environment; default remains `open_loop` for legacy runs.  V64.3.10 launchers set `VAL_MODE=both` so fixed-reference loss diagnostics are logged without changing older configs.
- Added generic exact `_runtime_hab_topm_mask_from_scores(...)`; current proposal and utility-oracle projections now use the same canonical deployed Top-M operator.
- Added `check_v64_3_10_hap_bdmu_contract.py` with a synthetic structural/group-diversity fixture and strict config checks.
- Added `check_v64_3_10_hap_bdmu_screen.py`; it refuses a log missing fixed-reference validation metrics and emits explicit `acquisition_capacity_not_binding`, `mechanism_gain`, `deployment_gain`, and `pivot_to_value_frontier` diagnoses.
- V64.3.10 continues the strict training-artifact and frozen-value config contracts.

## V64.3.10 no-repeat constraints

All V64.3.9 no-repeat constraints remain active.  In particular, do not respond to another screen failure by lowering M/B, re-enabling global proposal unfreezing, retrying AP-WCCA/AP-WRCCA/LCV/FPCCA/CCBR/LEA/BCHA, adding a binary AOCC/certificate bonus, running selector beam/swap/bruteforce, or changing DARM/DBR in the same causal experiment.

If HAP-BDMU closes the fixed-reference HAB oracle gap but teacher endpoint does not respond, the next algorithm cycle is **value/frontier**, not another acquisition objective.

# V64.3.11 — BTP-BDMU: Budget-Transmission-Projected Decisive-Margin Utility (2026-08-14)

## V64.3.10 HAP-BDMU screen: valid algorithm evidence

The uploaded `outputs_v64_3_10_hap_bdmu_screen_2gpu_v1` is valid for algorithm attribution.  All V64.3.10 engineering contracts pass: exact runtime Top-M semantics are active (`bdmu_runtime_topm_exact_fraction=1.0`), the intended zero-init proposal adapter is the only trainable module, DARM/DBR/foundation remain frozen, and epoch/final checkpoints were produced.  Therefore the STOP is not explained by the V64.3.9 training/deployment Top-M bug.

The executable C1/C2 result is:

- anchor C2 learned exact-HAB Top-M utility capture: `0.478576`;
- C1 exact-HAB utility-oracle capture: `0.504656`;
- exact-HAB acquisition headroom: `+0.026081` (`+2.61pp`);
- selected epoch 3 C2 capture: `0.482552`, a `+0.003977` (`+0.40pp`) gain;
- C1-C2 gap closure: `15.25%`;
- feasible-admission rank loss: `1.6695 -> 1.0195` on validation;
- feasible-admission pair count: `156.97 -> 72.47`;
- same-family pair fraction: `0.2268 -> 0.4109`.

Thus HAP-BDMU **partially solved** the V64.3.9 bottleneck.  The learned adapter can now cross the exact hierarchical hard-admission boundary and close a measurable fraction of the realizable HAB utility gap.  It is no longer correct to describe the primary failure as simply “continuous utility cannot move hard Top-M membership.”

However, the moved admissions are not decision-useful enough:

- proposal decisive recall: `0.75366 -> 0.67370` (`-8.00pp`);
- exact winner-flip Top-M recall: `0.23734 -> 0.23418`;
- exact winner-flip B-selected recall: unchanged at `0.14873`;
- teacher match: `0.178 -> 0.172` (`-0.6pp`);
- teacher regret: `20133.34 -> 20378.10` (`1.22%` worse);
- pair-full teacher match stays `0.174`;
- exact B->TopM evidence certificate improves `0.928 -> 0.936` while teacher decision degrades.

The last point is a useful negative result: preserving the *current learned Top-M action* more often does not imply preserving the teacher decision.  Do not resurrect a binary certificate objective.

Train-side evidence leads to the same conclusion.  HAP rank loss falls strongly (`1.394 -> 1.052` over epochs 0->3), while train hard Top-M capture remains roughly `0.5641 -> 0.5613`; the optimizer learns the surrogate/oracle ordering but does not produce a robust decision-useful support shift.  Loss construction is also the dominant training hot path (`~99--116 s` of `~189--203 s/epoch`).

## Bottleneck transition

V64.3.10 moves the diagnosed break one mediator downstream:

`continuous BDMU utility -> exact HAB hard admission` **partially repaired**

but

`exact HAB admission -> fixed B=16 decision evidence -> frozen DARM/DBR endpoint` **not repaired**.

The exact-HAB C1 headroom is only `2.61pp`, so another stronger HAP/listwise ranking loss has limited upside and is not justified.  More importantly, HAP moves C2 while decisive recall and C3 degrade.  The next question is whether the utility-supported proposal changes can survive the existing B=16 selector without displacing already useful evidence.

The global value ceiling remains important: pair-full teacher match is only about `0.174` on this screen.  This strongly suggests that decisive value/frontier modeling will become the next bottleneck once acquisition transmission is either saturated or shown not to affect C3.  V64.3.11 is therefore a **final mediation-aligned acquisition test**, not permission for indefinite proposal-loss iteration.

## Algorithm: BTP-BDMU

Keep unchanged:

- fixed planner-interface budget `B=16` and proposal size `M`;
- auditable evidence atoms and exact runtime HAB policy;
- V64.3.9 adaptive decisive frontier and weakest-rival one-sided deficit;
- immutable frozen-foundation BDMU target/reference;
- CCBR proposal representation as a representation primitive only;
- promoted V64.3.7 DARM+DBR value/aggregation path, completely frozen;
- exact DA-EPC downstream winner-preservation diagnostic.

Replace broad HAP supervision with **budget-transmission-projected admission**:

1. Compute the same fixed-reference continuous BDMU utility.
2. Project utility through the exact frozen-family HAB policy to obtain the realizable utility Top-M oracle.
3. During training, pass current and utility-oracle Top-M pools through the frozen vectorized pair-margin B=16 selector surrogate.
4. A positive proposal atom must be (a) in the exact-HAB utility oracle, (b) selected by the B=16 projection under that oracle pool, and (c) absent from the current deployed Top-M.
5. Current evidence that already survives the B=16 selector is **protected** and cannot be used as a negative.  This is a minimum-intervention / one-sided preservation prior, not a binary critical label.
6. Rank only same-family positive/negative replacements.  V64.3.10 allowed cross-family fallback and only `41.1%` of selected-epoch pairs were same-family; the broad fallback coincided with a severe decisive-recall drop.  V64.3.11 disables cross-family fallback entirely because frozen HAB family slots normally make such comparisons non-actionable.
7. Disable V64.3.9 unconstrained swap, V64.3.10 broad HAP feasible-admission rank, and broad listwise utility loss.  The only acquisition objective is the B-transmitted admission rank plus a small residual L2 preservation prior.

Main configuration:

- `budget_transmission_rank_weight=1.0`;
- `budget_transmission_margin=0.35`;
- `budget_transmission_positive_k=6`, `negative_k=6`;
- `budget_transmission_same_family=true`;
- `budget_transmission_cross_family_fallback=false`;
- `budget_transmission_protect_current_budget=true`;
- `listwise_weight=0`;
- `feasible_admission_rank_weight=0`;
- `topm_swap_rank_weight=0`;
- `residual_l2_weight=0.005`.

The new pair loss is fully vectorized over the small Top-M set, removing the HAP per-scene/per-positive Python ranking loop.  No runtime planner query or selector search is added.

## Exact C0/C1-M/C2-M/C1-B/C2-B/C3 causal protocol

V64.3.11 strengthens the protocol so the paper does not mistake a training selector surrogate for the deployed B-layer mediator:

- **C0:** immutable frozen-foundation decisive-margin utility target.
- **C1-M:** exact-HAB utility-oracle Top-M capture.
- **C2-M:** learned exact-HAB Top-M capture.
- **C1-B:** C1-M passed through the **exact runtime pair-conditioned B=16 selector** during validation.
- **C2-B:** C2-M passed through the **same exact runtime B=16 selector** during validation.
- **C3:** teacher match/regret with B, DARM, DBR and foundation value frozen.

Training uses the established vectorized pair-margin selector surrogate for tractability, but validation runs under `torch.no_grad()` and recomputes C1-B/C2-B with the exact runtime selector.  It additionally reports current/oracle surrogate-vs-exact B-mask Jaccard.  Promotion requires `bdmu_budget_projection_exact_fraction=1.0`; therefore a surrogate mismatch can be diagnosed rather than silently entering the paper claim.

This deterministic same-interface mediation protocol is a stronger paper contribution than another selector heuristic.  It should be described as **causal attribution under controlled planner-interface interventions**, not as a causal intervention on the physical world.

## V64.3.11 decision rules

1. If anchor exact `C1-B - C2-B < 0.5pp`, classify budget-transmitted acquisition capacity as non-binding and **pivot to decisive value/frontier**.  Do not invent another acquisition loss.
2. If C1-B headroom exists but C2-B does not improve by at least `0.5pp` and close at least `15%` of the gap, B-transmitted admission learning remains the bottleneck.
3. C2-B mechanism gain is accepted only if proposal decisive recall, exact-critical Top-M/selected recall, and C2-M remain within explicit non-harm tolerances.  This prevents another HAP-like proxy improvement that destroys decisive support.
4. If C2-B improves under non-harm constraints but C3 teacher match/regret does not, emit `pivot_to_value_frontier=true`.  Acquisition is no longer the next bottleneck.
5. Only C2-B + C3 improvement may promote to full.  Full must reproduce before held-out test and closed loop.

## Engineering changes

- `_predicted_pair_certificate_masks` now accepts an exact `topm_mask_override` and optional proposal-score override, allowing validation to run the real pair-conditioned B selector on controlled C1-M/C2-M pools without rebuilding a different HAB policy.
- BTP training uses the vectorized B selector surrogate; validation automatically switches to exact B projection because `_run_validation_loss` is `@torch.no_grad()`.
- New diagnostics: current/oracle B utility capture, B transmission gap, BTP rank loss/pair/scene statistics, protected-negative fraction, exact B projection fraction, and surrogate/exact Jaccard for current/oracle pools.
- New V64.3.11 config/semantic contract rejects broad listwise/HAP/AF losses, cross-family fallback, missing budget protection, or missing exact validation projection.
- New screen checker reports separate best-mechanism and best-endpoint epochs and their concordance instead of hiding a mediator/endpoint mismatch behind one score.

## V64.3.11 no-repeat constraints

Continue all V64.3.10 constraints.  In addition:

- do not strengthen HAP feasible-admission ranking or re-enable its cross-family fallback;
- do not use C1-M/C2-M alone to promote acquisition; C1-B/C2-B exact B transmission is now mandatory;
- do not optimize DA-EPC/certificate fraction as a teacher objective;
- do not replace the exact validation B selector with the training fast surrogate in the causal report;
- if exact C1-B has little headroom, or exact C2-B moves without C3, the next version must be decisive value/frontier rather than another proposal objective.


## V64.3.11 post-implementation engineering hardening: exact B must remain inside injected Top-M

A targeted regression test of the new exact C1-B/C2-B validation path exposed an evaluation-adapter bug before any V64.3.11 experiment was run.  `_predicted_pair_certificate_masks(...)` correctly accepted an already-finalized `topm_mask_override`, but the helper subsequently re-applied the soft-interaction Top-M reservation step.  With an injected Top-M that did not already satisfy that reserve, atoms outside the controlled pool could be pulled back into the candidate domain.  The underlying deployed pair-conditioned selector was not the source of the violation; the bug was specific to the controlled exact-TopM injection path used for the new causal mediation metric.

This is material for the paper protocol because C1-B/C2-B are only interpretable if the nested interface is exact: `B subset injected final Top-M`.  The fix is intentionally semantic-only and does not change BTP utility, rank loss, DARM/DBR, M, B, or runtime planner behavior:

- when `topm_mask_override` is supplied, it is treated as the fully finalized canonical Top-M and no HAB/structural/soft-interaction Top-M post-processing is allowed to run again;
- validation logs `bdmu_budget_projection_topm_violation_fraction`; promotion requires it to be exactly zero (within `1e-9`);
- the V64.3.11 preflight contract now runs a synthetic adversarial fixture with high-scoring soft-interaction atoms outside the injected pool and requires the returned B set to contain no outside atom;
- unit coverage explicitly requires the exact budget mask to be nested in the injected Top-M.

This correction prevents a new attribution error at the B mediator and is a reason to keep exact interface contracts as executable experimental artifacts rather than prose-only assumptions.

# V64.3.12 — RET/CET-BDMU: Runtime-Exact Transmission + Controlled Exact Transmission (2026-08-14)

## Trigger: V64.3.11 BTP-BDMU screen exposes a train/promotion B-interface mismatch

The uploaded V64.3.11 BTP-BDMU screen is a valid STOP, but it does **not** yet establish that acquisition capacity is exhausted.  The exact runtime budget oracle still has material headroom:

- anchor C1-M exact-HAB oracle capture: `0.5046561`;
- anchor C2-M learned Top-M capture: `0.4785755`;
- anchor C1-B exact-runtime oracle capture: `0.4151051`;
- anchor C2-B exact-runtime learned capture: `0.3805248`;
- exact C1-B/C2-B headroom: `0.0345803` = **3.458pp**.

The selected epoch 1 moves C2-M in the expected direction (`+0.0030805`, +0.308pp), but exact C2-B moves in the wrong direction (`-0.0019379`, -0.194pp).  Every trained epoch has negative exact C2-B delta (`-0.250pp`, `-0.194pp`, `-0.478pp`, `-0.288pp`).  Teacher match rises only `17.8% -> 18.2%` while regret worsens `0.691%`; pair-full teacher match remains `17.4%`.

Code audit identifies the first-order mechanism mismatch. V64.3.11 training constructs BTP positives/negatives with `_fast_pair_margin_surrogate_masks`, while exact `_predicted_pair_certificate_masks` is used only under no-grad validation.  The uploaded validation reports current/oracle surrogate-to-exact B-set Jaccard only about `0.775/0.769`.  Thus the gradient target is not the same discrete B=16 mediator used by C1-B/C2-B promotion.

A second restriction is blanket current-B protection.  Validation protects about `74.7%--76.7%` of raw displaced negatives; train epochs protect about `84.6%--89.2%`.  This strongly limits direct B-set exchange pairs even when the exact oracle leaves transmission headroom.

According to the V64.3.11 stop protocol itself, `C1-B - C2-B >= 0.5pp` with no C2-B movement means **budget-transmitted admission remains the unresolved acquisition bottleneck**.  Therefore one final semantics-correct transmission experiment is warranted before a value/frontier pivot.  This is not permission for another family of acquisition proxies: CET failure below is terminal for this proposal-only branch.

## Arm A: RET-BDMU — Runtime-Exact Transmission control

RET changes only the training B mediator:

1. Keep the immutable frozen-foundation BDMU decisive-margin target unchanged.
2. Keep exact frozen-family HAB projection unchanged.
3. On training scenes with a missed positive utility-oracle Top-M atom, select a deterministic rotating subset of at most four actionable scenes per rank/step.
4. Run the **exact runtime pair-conditioned B=16 selector** under stop-gradient for current Top-M and oracle Top-M.
5. Only those exact-sampled rows may form B-transmission ranking pairs.  Non-sampled rows never fall back to the fast surrogate as a training target.
6. Keep the V64.3.11 blanket current-B protection, same-family competition, no cross-family fallback, old AF/HAP/listwise objectives disabled, and `residual_l2_weight=0.005`.
7. Retain the fast B projection only as a diagnostic and report its Jaccard against the exact training target.

This isolates whether V64.3.11 failed because `surrogate B target -> exact promotion B target` was not deployment-consistent.

Main RET config:

- `budget_transmission_selector_source=exact_runtime_sampled`;
- `budget_transmission_exact_scenes_per_rank=4`;
- `budget_transmission_exact_every_n_steps=1`;
- `budget_transmission_exact_candidate_only=true`;
- `budget_transmission_exact_eval=true`;
- `budget_transmission_allow_controlled_budget_exchange=false`.

## Arm B: CET-BDMU — Controlled Exact Transmission (main novelty candidate)

CET keeps RET's exact-runtime training target and replaces **blanket protection** with a controlled budget-exchange criterion.

A current transmitted atom may become a proposal-ranking negative only when all of the following are true under the same exact runtime selector:

- it lies in `current Top-M \ oracle Top-M`;
- it is in `exact current B`;
- it is not in `exact oracle B` after the oracle-TopM intervention;
- it competes with an exact-oracle transmitted positive from the same frozen HAB family;
- the positive has larger fixed decisive-margin utility.

Everything else in the current B-set remains protected.  Controlled exchange pairs are down-weighted by `0.5` to retain one-sided minimum-intervention bias.

This is not a binary critical/certificate reward and does not optimize the certificate metric.  It is an **interface intervention condition**: the exact frozen B selector itself must show that the current transmitted atom is displaced under the oracle proposal intervention before acquisition learning may push it down.

Main CET config is identical to RET except:

- `budget_transmission_allow_controlled_budget_exchange=true`;
- `budget_transmission_controlled_exchange_weight=0.5`.

## Causal screen and terminal stop rules

Run RET and CET on the same 12k-train / representative-500-val screen, DARM/DBR/foundation/B/M frozen.

The checker now requires both training and validation interface evidence:

- training exact projection fraction must be non-zero;
- training actionable exact-candidate fraction must be non-zero;
- validation exact projection fraction must be `1.0`;
- exact `B subset injected Top-M` violation fraction must be zero;
- surrogate/exact B Jaccards are logged as diagnostics only;
- C1-M/C2-M/C1-B/C2-B/C3 use the same fixed reference as V64.3.11.

Decision rules:

1. If exact C1-B/C2-B anchor headroom `<0.5pp`, pivot to decisive value/frontier.
2. If RET passes C2-B + C3 under non-harm, it is a full candidate; CET is still a controlled ablation and is not forced into the main method if worse.
3. If RET fails but CET passes, blanket current-B protection is identified as the remaining B-transmission constraint; promote CET.
4. If exact C2-B improves but C3 does not, acquisition is causally cleared and the next version must be decisive value/frontier.
5. If **CET** has valid exact training, exact C1-B headroom remains, and C2-B still does not improve, set `exact_acquisition_exhausted=true` / `pivot_to_value_frontier=true`.  **Do not design V64.3.13 as another proposal/acquisition loss.**

This gives V64.3.12 a hard endpoint: after correcting train/eval selector semantics and testing exact-controlled B-set exchange, another acquisition surrogate would no longer have a new falsifiable interface hypothesis.

## Paper/mainline interpretation

The main paper chain remains:

`fixed planner-interface budget -> auditable evidence atoms -> budget-feasible decisive-margin marginal utility -> budgeted acquisition -> one-sided margin preservation -> final decision preservation`.

If CET is supported, the mechanism-level novelty can be stated as **Controlled Exact Budget Transmission for Auditable Decision Preservation**, nested inside the broader controlled interface-level causal attribution protocol.  This is consistent with the paper theorem/regret decomposition: proposal miss and budget-selection loss are separate error terms, and selected evidence must protect one-sided decisive margins rather than merely improve an upstream ranking proxy.

Pair-full teacher match remains only `0.174`, so decisive value/frontier remains the likely global ceiling after transmission is either solved or falsified.  The future value pivot should target decisive rival/pair-boundary frontier representation, not repeat generic global action/set potentials.

## No-repeat constraints

All V64.3.11 constraints remain.  In particular, V64.3.12 does not retry:

- AP-WCCA/AP-WRCCA/LCV/FPCCA;
- CCBR/LEA objectives (CCBR remains representation only);
- BCHA/family-capacity tuning;
- broader HAP/cross-family ranking;
- binary literal/AOCC/certificate bonuses;
- selector beam/swap/bruteforce;
- larger B/M;
- global proposal/family unfreeze;
- DARM/DBR changes inside this acquisition mediation cycle;
- V55/V59 generic action/set-potential branches.

If CET reaches its terminal failure rule, the no-repeat list additionally includes **RET/CET proposal variants and current-B protection tuning**; the next algorithm must move to decisive value/frontier.

## Engineering changes and validation

- Added sampled exact-runtime B targets to `bdse/model/losses.py`; exact target selection is stop-gradient and restricted to actionable scenes.
- Added CET controlled-exchange pair construction and diagnostics.
- Added `bdmu_budget_exact_candidate_scene_fraction`, `bdmu_budget_current_oracle_jaccard`, `bdmu_budget_controlled_exchange_negative_fraction`, and `bdmu_budget_controlled_exchange_pair_fraction`.
- Added RET/CET screen/train/closed-loop configs and launchers.
- Added strict V64.3.12 config contract, exact interface contract, RET/CET screen checker, and two-arm comparison tool.
- Added gradient-level unit tests proving RET blocks an exact-current-B displaced atom while CET unlocks **only** the exact-oracle-controlled exchange.
- Removed stale `.bak` copies of source files; generated bytecode is excluded from final delivery archive.
- `python -m compileall -q bdse`: PASS.
- V64.3.9--V64.3.12 targeted regression: **17 passed**.
- Full repository regression: **328 passed, 34 warnings**; all warnings are existing PyTorch Transformer nested-tensor warnings.

# V64.3.13 — EAF-DMVR: Evidence-Attributed Frontier Decisive-Margin Value Residual (2026-08-14)

## Trigger: V64.3.12 RET/CET terminally falsifies the proposal-only acquisition branch

V64.3.12 resolves the remaining BTP acquisition hypotheses rather than merely producing another negative screen.

**RET result.** The V64.3.11 diagnosis that BTP trained against a fast B-selector surrogate while C1-B/C2-B promotion used the exact runtime B=16 selector was correct. RET replaces the training mediator with sampled stop-gradient exact runtime B projection and obtains valid exact instrumentation, but selected epoch 1 changes C2-B only `0.3805248 -> 0.3805309`, i.e. **+0.000606pp**, while C2-M improves about `+0.2396pp`. Teacher match changes `17.8% -> 18.2%`, but teacher regret worsens by `181.08`. Therefore train/runtime selector mismatch was a real semantic defect and needed repair, but it was **not the root performance bottleneck**. Do not make RET-v2 by increasing exact-sampled scenes, rank weight, margin, or epochs.

**CET result.** CET activates exact-oracle-controlled current-B exchange, so its hypothesis is genuinely tested rather than blocked by implementation. The selected epoch has a large non-zero controlled-exchange pair fraction, but C2-B changes `0.3805248 -> 0.3763801` (**-0.4145pp**), proposal decisive recall falls about `-2.18pp`, exact-critical Top-M recall about `-0.63pp`, exact-critical selected recall about `-0.32pp`, pair-full match falls `-0.2pp`, and teacher regret worsens despite a noisy `+0.8pp` teacher-match change. Controlled exchange is therefore an **active but harmful mechanism**, not a direction to tune further.

The two-arm causal conclusion is terminal:

- RET = semantics-correct exact target, no C2-B transmission gain;
- CET = controlled B-set exchange active, negative C2-B/critical-support gain;
- `exact_acquisition_exhausted=true` and `pivot_to_value_frontier=true` are accepted;
- **no V64.3.13 acquisition/proposal loss is permitted.**

The old 3.458pp C1-B/C2-B oracle gap should no longer be interpreted as permission to keep inventing selector losses. It is an oracle intervention headroom under a value interface that has now failed to turn semantics-correct acquisition supervision into endpoint value.

## New bottleneck: selected evidence -> decisive value/frontier

The shared V64.3.12 anchor exposes a much larger downstream failure:

- base teacher-winner/rival sign accuracy: `0.62789`;
- dense teacher-winner/rival sign accuracy: `0.62612`;
- selected-B pair/tournament teacher-winner/rival sign accuracy: **`0.06082`**;
- selected pair margin MAE: `1.3320`, signed error `-1.3315`;
- pair-full action match: `0.174`;
- local pair-full action match: `0.174`;
- final teacher match: `0.178`;
- evidence certificate fraction remains high at about `0.928`.

Runtime frontier coverage is also weak. The teacher winner is inside base Top-2/3/5/6/9 only about `16.9%/21.8%/31.7%/34.6%/48.6%`, and only about `21.9%` of exact-critical boundaries lie inside base Top-9. Thus the current decisive-value problem has two coupled pieces:

1. **pair-value error:** selected evidence is not converted into correct teacher decisive margins;
2. **frontier-coverage error:** the sparse DARM/DBR rival graph often never exposes the true decisive challenger to downstream correction.

This is consistent with the historical V64.3.5/V64.3.7 evidence: budget-vs-pair-full could already be high while pair-full stayed low, and DARM+DBR improved endpoint with acquisition frozen. The next branch therefore extends the historically positive literal/decisive-boundary value direction rather than repeating acquisition.

## Algorithm: EAF-DMVR

**Evidence-Attributed Frontier Decisive-Margin Value Residual** keeps the paper chain:

`fixed planner-interface budget -> auditable evidence atoms -> budget-feasible decisive-margin marginal utility -> frozen budgeted acquisition -> exact selected B=16 evidence -> evidence-attributed complete decisive frontier -> one-sided margin preservation -> final decision preservation`.

V64.3.13 freezes:

- foundation;
- proposal/critical-proposal adapter;
- HAB/family slots;
- Top-M `M=24`;
- exact runtime B selector and `B=16`;
- DARM;
- DBR;
- calibrated evidence certificate and one-sided flip guard.

Only `decisive_anchor_frontier_value_adapter` is trainable.

### Complete selected-local anchor frontier

The frozen selected-local value first defines the DARM anchor `a`. EAF-DMVR then exposes the value head to **every valid challenger** `b != a`, not only the sparse runtime pair graph. This removes the historical base-Top-L/rival-edge coverage hole without adding any evidence query or changing the candidate bank.

### Selected-evidence-attributed pair residual

For each already-selected atom `e_i`, the new head produces a bounded atom factor `z_i`; each candidate produces a signed factor `u_a` and context factor `c_a`. With the symmetric pair gate

`c(a,b)=tanh(c_a+c_b+c_a*c_b)`,

the residual is

`r_S(a,b)=sum_{i in S}<tanh(z_i)*c(a,b),u_b-u_a>/sqrt(|S|*d)`.

Properties required by the implementation contract:

- exact antisymmetry: `r_S(a,b)=-r_S(b,a)`;
- explicit selected-atom additive attribution (for the fixed selected set);
- only exact runtime-selected B evidence participates;
- no selector/proposal score is changed;
- no additional planner evidence query is created;
- zero-initialized final atom layer makes V64.3.13 an exact step-zero no-op from the promoted V64.3.7 value checkpoint.

This is deliberately **not** V55 Hodge/global action potential, V56 generic evidence-action potential, or V59 generic selected-set potential. The correction is pair-specific and selected-evidence-attributed, and it only fills the complete selected-local anchor frontier.

### Training and one-sided preservation

Training uses the exact B=16 selected mask as stop-gradient input and reconstructs the frozen V64.3.7 DARM+DBR star as the baseline. The EAF residual is additive and is supervised against teacher complete anchor-star margins with:

- boundary-weighted robust margin regression;
- teacher-winner weighting;
- pair-sign loss;
- wrong-anchor teacher-winner correction;
- correct-anchor strongest-rival preservation.

At runtime the residual is added only to the DARM anchor star. The existing `pair_action_anchor_guard` remains authoritative; a changed winner still needs the configured robust margin/score evidence and the existing certificate policy. V64.3.13 creates no new formal certificate concept.

## V64.3.13 causal screen rules

Instrumentation must show:

- new value adapter parameter delta > `1e-7`;
- value residual RMS > `1e-6`;
- exact selected-B training scene fraction and complete anchor-star coverage valid;
- runtime EAF-DMVR active;
- critical proposal adapter and DBR parameter deltas remain zero;
- proposal decisive / exact-critical Top-M / exact-critical selected metrics do not drift beyond `1e-4` from the frozen anchor.

Mechanism gain requires:

- complete-frontier pair-sign accuracy at least `+2pp` over anchor;
- complete-frontier action match at least `+1pp`;
- already-correct anchor preservation at least `97%`.

Endpoint gain requires either teacher match `+1pp` with regret no worse than `1%`, or regret improvement at least `2%` with teacher match non-harm (`-0.4pp` tolerance).

Only instrumentation + frozen acquisition + mechanism + endpoint may promote full.

If the head is active and complete frontier is covered but mechanism does not improve, **do not make EAF-DMVR-v2 and do not reopen acquisition**. The next hypothesis is frozen action/evidence representation capacity and the permitted next experiment is a small selective action/evidence representation adapter/unfreeze test. If mechanism improves but endpoint does not, audit frontier-to-final one-sided guard/calibration instead.

## Engineering hardening

Two implementation errors were caught and fixed before delivery:

1. An automated patch initially wrote literal `\\n` text into the new `losses.py` block, producing a syntax-valid but effectively non-executing section. The block was rewritten as real code and a gradient-level test now requires the zero-init output layer to receive non-zero gradient.
2. Pair-full diagnostics initially received the new B=16-trained EAF factors. This would contaminate the full-evidence ceiling with an out-of-distribution selected-B correction. Pair-full/local-pair-full now preserve frozen V64.3.7 semantics; EAF-DMVR is applied only to the real selected-B deployment tournament. New `decisive_frontier_value_*` diagnostics are exported explicitly.

Final validation:

- `python -m compileall -q bdse`: PASS;
- V64.3.13 tests: **8/8 PASS**;
- V64.3.7--V64.3.12 targeted regression: **37/37 PASS**;
- full repository: **336/336 PASS, 36 warnings**;
- warnings are existing PyTorch Transformer nested-tensor warnings;
- train/eval config contract: PASS;
- V64.3.13 exact-interface contract and adversarial `B subset injected Top-M` fixture: PASS;
- screen/full launcher shell syntax: PASS.

## V64.3.13 no-repeat constraints

All earlier no-repeat constraints remain. In addition:

- RET/CET/current-B-protection tuning is terminally closed;
- do not interpret remaining C1-B oracle gap as permission for another proposal loss;
- do not change B=16 or M=24;
- do not unfreeze proposal/HAB/family gate during EAF-DMVR screen;
- do not re-enable V46/V49 arbitrary pair fields;
- do not repeat V55 Hodge/global action potential;
- do not repeat V56 generic evidence-action potential;
- do not repeat V59 generic set-conditioned potential;
- do not bypass the existing one-sided anchor guard or relax certificate gates to manufacture endpoint gain;
- if EAF-DMVR mechanism fails with valid instrumentation, the next branch is selective action/evidence representation capacity, not another value-loss variant with the same frozen embeddings.

## 2026-08-14 engineering hotfix after first V64.3.13 screen launch

This patch is **engineering-only** and does not change EAF-DMVR, DARM/DBR, HAB, proposal, B=16/M=24 budgets, losses, promotion gates, or the execution protocol.

Observed failure during the pre-training validation (`val-loss -1`):

`UnboundLocalError: local variable 'topm_mask_override' referenced before assignment`

Root cause: `_predicted_pair_certificate_masks_multi_budget(...)` accepts only `outputs`, `batch`, `budget_cfgs`, and optional `scene_indices`.  A stale block copied from the single-budget selector helper attempted to slice non-existent local variables `topm_mask_override` and `proposal_scores_override` whenever `scene_indices` was provided.  V64.3.13 uses sampled exact-selector scenes, so this previously untested branch was exercised immediately.

Fix:

- remove only the stale undefined-variable slicing block from the multi-budget helper;
- preserve the existing `outputs`/`batch` scene slicing and exact runtime selector implementation unchanged;
- add a regression comparing multi-budget subset selection with repeated exact single-budget calls;
- add a second regression for the actual V64.3.13 `spawn` process backend with subset/reordered scenes, requiring bitwise-equal masks to the sequential exact backend.

Causal/algorithm contract after the fix is unchanged: acquisition remains terminally frozen, only `decisive_anchor_frontier_value_adapter` is trainable, and exact selected B=16 remains the value-head input.
