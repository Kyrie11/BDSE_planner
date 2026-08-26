# External-baseline fidelity and fixed-budget fairness audit

## Scope

This audit targets the matched-interface comparison used by BDSE-PTMC. It distinguishes:

1. **matched fixed-budget adapters** that use the BDSE candidate/evidence interface and therefore permit a controlled comparison under the same retained-evidence budget; and
2. **official/native baselines** whose original inputs, proposal generators, prediction heads, rollout/scoring stacks, and computational interfaces are different.

The matched adapters are **not verbatim reproductions** of the official methods. Paper tables must use the labels in the table below. If the row is named simply `GameFormer`, `DTPP`, `PlanTF`, `PLUTO`, or `PDM-Closed`, the corresponding official/native code path should be run instead.

## Reference mapping and fidelity

| Local key | Paper / public implementation | Defining original algorithmic components | Current matched implementation after patch | Fidelity verdict | Required label |
|---|---|---|---|---|---|
| `gameformer` | GameFormer, ICCV 2023; official `MCZhi/GameFormer` and nuPlan `MCZhi/GameFormer-Planner` | Transformer scene encoder, hierarchical/level-k decoder, joint ego/agent multimodal futures; nuPlan planner additionally performs feature processing, path planning, model query, trajectory refinement | Transformer scene/candidate/evidence reasoning with iterative levels, trained by logged-expert imitation on the common candidate bank | Low–medium: preserves iterative Transformer/game-inspired structure but omits the defining joint multimodal agent prediction and native nuPlan path/refinement stack | **GameFormer-inspired budget adapter** |
| `dtpp` | DTPP, ICRA 2024; official `MCZhi/DTPP` | Ego trajectory tree, ego-conditioned scenario-tree prediction, query-centric Transformer, context-aware learned cost, differentiable joint prediction/cost training, tree search / dynamic programming | Maneuver-conditioned candidate branch reasoning and refinement over the common candidate bank, trained by logged-expert imitation | Low: the main DTPP tree-policy and joint conditional prediction/cost machinery is not reproduced | **DTPP-inspired budget adapter** |
| `plantf` | PlanTF, ICRA 2024; official `jchengai/planTF` | Pure imitation learning; state6 + attention State Dropout Encoder (SDE), token dropout 0.75; state perturbation/re-normalization; Transformer feature pipeline; paper recipe batch 128, lr 1e-3, wd 1e-4, 25 epochs, cosine decay | Logged-expert imitation target; attention-token SDE with 0.75 dropout; lr/wd/epochs aligned; gradient accumulation can realize effective batch 128 | Medium: core imitation/SDE/training recipe is materially closer, but official object-token features and state-perturbation/re-normalization remain absent | **PlanTF-inspired budget adapter** |
| `pluto` | PLUTO, 2024; official `jchengai/pluto` | Longitudinal–lateral aware query architecture, differentiable interpolation auxiliary loss, Contrastive Imitation Learning (CIL), data augmentations | Logged-expert imitation; longitudinal/lateral decomposition retained; SDE-style state encoding; optimizer/effective-batch recipe aligned with public repo | Low–medium: still omits the defining query architecture, auxiliary-loss framework, CIL and official augmentations | **PLUTO-inspired budget adapter** |
| `pdm_closed_style` | PDM-Closed, CoRL 2023; official `autonomousvision/tuplan_garage` | Route/centerline selection, batches of IDM longitudinal policies + lateral offsets, dynamics rollout, proposal scoring approximating nuPlan closed-loop metrics, safety/emergency logic | Deterministic static geometric/progress/comfort/safety cost on the **existing BDSE candidate bank** | Low: this is not PDM-Closed and cannot be named as such. Its `B` is BDSE interface accounting, not PDM's native number of proposals | **PDM-Closed-style budget scorer (NOT official PDM-Closed)** |

Public references:
- GameFormer paper: https://openaccess.thecvf.com/content/ICCV2023/html/Huang_GameFormer_Game-theoretic_Modeling_and_Learning_of_Transformer-based_Interactive_Prediction_and_ICCV_2023_paper.html
- GameFormer nuPlan planner: https://github.com/MCZhi/GameFormer-Planner
- DTPP: https://github.com/MCZhi/DTPP
- PlanTF: https://github.com/jchengai/planTF
- PLUTO: https://github.com/jchengai/pluto
- PDM-Closed / tuPlan Garage: https://github.com/autonomousvision/tuplan_garage

## Fairness problems found in the uploaded version

### 1. Test leakage / wrong final split

The supplied commands selected checkpoints on validation **and also reported open-loop/closed-loop results on validation**. The corrected protocol is:

- train: `bdse_train_v2/{train_boston,train_pittsburgh,train_singapore,train_vegas_2}`
- model selection: `bdse_val_v2/val`
- final open-loop and closed-loop token manifest: `bdse_test_2/public_set_test`

No final metric should be selected, tuned, thresholded, or calibrated on `public_set_test`.

### 2. External-planner supervision was teacher-distillation-like

The uploaded adapter loss used BDSE teacher candidate cost/action labels for the external planner. That turns the rows into architecture ablations of a shared teacher rather than faithful external imitation/planning baselines.

Patched behavior:

- read `label_logged_ego` only as an offline training label;
- project the logged expert future to the common candidate bank via finite-horizon trajectory error;
- train candidate action/cost heads toward that expert projection;
- keep the BDSE-specific oracle evidence mask only for training the **common imposed evidence selector**, not the external planner preference/cost target.

This preserves the paper rule that future logged states are label-only and are not runtime features.

### 3. Budget was entangled with proposal-pool compute

The fixed-budget comparison must hold the upstream proposal pool at `M=24`. An older sweep changed `proposal_top_m` with `B`, mixing retained-interface budget with extra upstream computation.

Patched behavior for B in `{8,16,24}`:

- `evidence.budget = B`
- `external_baseline.budget = B` (trainable adapters)
- `selector.min_selected_atoms = B`
- `selector.force_fill_budget = true`
- `selector.proposal_top_m = 24` for every B
- fallback disabled
- no additional fallback evidence stage

### 4. One B=16 checkpoint reused at B=8/24 would create an input-budget distribution shift

For the **primary matched-adapter comparison**, each trainable adapter is now trained separately for B=8, B=16, and B=24, and validation selects the best checkpoint separately within each B. A shared checkpoint across B is supported only behind an explicit cross-budget-ablation flag and should not be presented as the strict primary comparison.

### 5. Closed-loop NPZ cache and raw nuPlan DB were conflated

`evaluate_closed_loop.py` launches the nuPlan simulator, which needs raw `.db` logs. The custom NPZ cache is used to determine the **exact test scenario tokens**; it is not a replacement for raw DB files.

Correct pairing:

- `--split-cache /data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2`
- `--token-split public_set_test`
- `--nuplan-db-root <RAW TEST DB ROOT CONTAINING .db FILES>`

The runner extracts the ordered unique `scenario_token` list from the test NPZs, hashes the list, pins the exact same tokens across every system/B, and checks simulation success/failure counts.

## Fixed-budget closed-loop protocol

For every reportable arm:

- same ordered test `scenario_token` manifest;
- same raw nuPlan logs and map data;
- same challenge and metric aggregator;
- same candidate-bank mechanism for matched adapters;
- fixed `M=24`;
- exact deployed B in `{8,16,24}`;
- no fallback budget expansion;
- all requested scenarios must complete (`successful == manifest_count`, `failed == 0`);
- retain the token SHA256 and resolved config next to each result;
- aggregate each system once using the normal nuPlan metric aggregation (do not average per-shard aggregate scores).

The default challenge is `closed_loop_nonreactive_agents`, matching the supplied command. A reactive suite can be run separately, but do not mix NR-CLS and R-CLS in a single column.

## Important manuscript caveat for B=24

The current manuscript states a current configuration with `B=16`, `M=24`, and later frames the experiment as operating under an unchanged `B <= 16` evidence interface. Therefore:

- `B=8`: valid lower-budget scaling / cross-budget ablation relative to the fitted B=16 setting.
- `B=16`: current primary manuscript setting.
- `B=24`: **outside the current stated `B <= 16` experimental scope**. Treat it as an extended-budget appendix ablation unless the manuscript is revised.
- At `B=24`, keep `M=24` fixed, but do **not** interpret the numerical equality `B=M=24` as equal resource types: `B` counts retained decision-evidence atoms while `M` counts upstream proposals. They must still be reported separately, exactly as the manuscript specifies.

For the user's V64.3.48 model, the final V48 fit/promotion flow is B=16. The new own-model runner therefore records B=8/B=24 as **frozen-policy cross-budget ablations** unless all B-dependent calibration/fitting artifacts are independently refit for each B.

## Recommended paper organization

Use two conceptually separate comparison tables.

### Table A — matched fixed-budget interface (primary causal comparison)

Rows:
- BDSE-PTMC
- GameFormer-inspired budget adapter
- DTPP-inspired budget adapter
- PlanTF-inspired budget adapter
- PLUTO-inspired budget adapter
- PDM-Closed-style budget scorer

This table answers the paper's constrained-interface question because every row is forced through the same candidate/evidence accounting protocol.

### Table B — native/official contextual benchmark

Run the official repositories with their native preprocessing/model/planner stack, but pin the same raw test scenario tokens whenever the repository supports it. These rows can be named simply GameFormer, DTPP, PlanTF, PLUTO and PDM-Closed. Do **not** claim that their native compute is matched to BDSE evidence budget B; report native runtime/latency/compute separately.

GameFormer and DTPP especially require their own richer raw-log preprocessing and cannot be made into high-fidelity official reproductions merely by consuming the compressed BDSE NPZ tensor schema.

## Commands

### 0. Dataset split/token integrity audit

```bash
python -m bdse.tools.export_custom_dataset_token_manifests \
  --train-root /data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2 \
  --val-root /data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2 \
  --test-root /data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_2 \
  --output-dir outputs/fairness_manifests
```

The command exits nonzero if scenario tokens overlap between train/val/test and records counts + SHA256 hashes.

### 1. Train 4 learnable matched adapters separately for B=8/16/24

```bash
bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

Default output layout:

```text
outputs/external_fixed_budget/
  B8/{gameformer,dtpp,plantf,pluto}_budgeted.best.pt
  B16/{gameformer,dtpp,plantf,pluto}_budgeted.best.pt
  B24/{gameformer,dtpp,plantf,pluto}_budgeted.best.pt
```

The wrapper uses all training scenarios found under the four train splits, validates on all `val` scenarios, and keeps validation-only checkpoint selection. PlanTF/PLUTO use gradient accumulation to obtain effective batch 128 on the two-GPU execution layout.

### 2. Final open-loop test on `public_set_test`

```bash
bash RUN_FAIR_EXTERNAL_OPEN_LOOP_TEST_B8_B16_B24_2GPU.sh
```

### 3. Point closed-loop to raw nuPlan test DBs

```bash
export NUPLAN_ROOT=/data0/senzeyu2/dataset/nuplan
export NUPLAN_TEST_DB_ROOT=/path/to/raw/nuplan/test/db/root
find "$NUPLAN_TEST_DB_ROOT" -type f -name '*.db' | head
```

`NUPLAN_TEST_DB_ROOT` must contain raw `.db` files. Do not set it to `bdse_test_2`.

### 4. External matched-adapter closed-loop, all budgets

```bash
bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
```

Individual budget examples:

```bash
BUDGETS=8  bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
BUDGETS=16 bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
BUDGETS=24 bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
```

### 5. Own V64.3.48 closed-loop, B=8/16/24

```bash
bash RUN_V64_3_48_OWN_FIXED_BUDGET_CLOSED_LOOP_TEST_2GPU.sh
```

Individual budgets:

```bash
BUDGETS=8  bash RUN_V64_3_48_OWN_FIXED_BUDGET_CLOSED_LOOP_TEST_2GPU.sh
BUDGETS=16 bash RUN_V64_3_48_OWN_FIXED_BUDGET_CLOSED_LOOP_TEST_2GPU.sh
BUDGETS=24 bash RUN_V64_3_48_OWN_FIXED_BUDGET_CLOSED_LOOP_TEST_2GPU.sh
```

The wrapper refuses to present V48 as the promoted model if the recorded double-fresh screen did not pass, resolves the preferred promotion arm and selected EAF checkpoint, and then runs the paired fixed-budget suite.

### 6. One combined suite when own config/checkpoint are already resolved

```bash
export OWN_CONFIG=/path/to/resolved_v48_config.yaml
export OWN_CKPT=/path/to/selected_eaf_checkpoint.pt
bash RUN_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
```

## Verification performed in the supplied code package

The following targeted tests passed after the changes:

```text
python -m pytest -q bdse/tests/test_external_baselines.py bdse/tests/test_v2_budget_pipeline.py
10 passed
```

All modified Python files compile, strict B=8/16/24 config generation was smoke-tested, and the five new shell wrappers pass `bash -n`.

What could not be executed in the sandbox: the user's `/data0/...` custom dataset and the raw nuPlan `.db` simulator data were not mounted. Therefore actual split counts/hashes, GPU training, and nuPlan closed-loop metric values must be produced on the user's workstation/server using the commands above.
