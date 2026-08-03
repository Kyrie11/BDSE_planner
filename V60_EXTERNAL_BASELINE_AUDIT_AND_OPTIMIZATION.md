# V60 external baseline audit and optimized comparison protocol

## Scope

This update changes only the external-baseline training/evaluation path. It does not modify the V60 DWAPC-BFAR-DBAP algorithm.

## 1. What the five evaluated baselines actually are

The repository implementations share the BDSE candidate bank, runtime features, evidence atoms and fixed evidence-budget interface. They are controlled structural adapters, not verbatim copies of the full official systems.

### GameFormer-inspired budget adapter

Reference: *GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving*, ICCV 2023. Public code: `MCZhi/GameFormer` and `MCZhi/GameFormer-Planner`.

Retained:

- Transformer scene encoding.
- Hierarchical level-k iterative refinement.
- Current-level candidate state responds to selected interaction evidence and the previous level.
- Deep supervision of intermediate reasoning levels.

Omitted:

- Joint multi-agent future prediction.
- The official hierarchical multi-agent decoder and level-specific prediction targets.
- The full GameFormer-Planner path planning/model-query/trajectory-refinement stack.

Conclusion: the core iterative reasoning motif is retained, but the implementation must be reported as a **GameFormer-inspired budget adapter**.

### DTPP-inspired budget adapter

Reference: *DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving*, 2023.

Retained:

- Maneuver-level branch costs.
- Multi-stage/tree-depth candidate refinement.
- Context-dependent candidate cost.
- Deep supervision of tree stages.

Omitted:

- Query-centric ego-conditioned prediction of other agents.
- Explicit ego trajectory tree and multimodal scenario tree.
- Joint differentiable training of prediction and planning cost.

The previous code exposed `tree_depth` in YAML but did not use it. This update makes tree depth operational. The implementation remains a **DTPP-inspired budget adapter**.

### PlanTF-inspired budget adapter

Reference: *Rethinking Imitation-based Planner for Autonomous Driving*, ICRA 2024. Official code: `jchengai/planTF`.

Retained:

- Pure Transformer imitation-based candidate scorer.
- State dropout on ego-state features.
- Scene/candidate/object-style token interaction under a fixed budget.

Omitted:

- The exact official feature builder and object-token representation.
- The full official augmentation and training pipeline.

This is the closest structural adapter among the four learning models, but it must still be labeled **PlanTF-inspired budget adapter**.

### PLUTO-inspired budget adapter

Reference: *PLUTO: Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving*, 2024. Official code: `jchengai/pluto`.

Retained:

- PlanTF-like Transformer backbone.
- Longitudinal/lateral cost decomposition.
- State dropout.

Omitted:

- PLUTO auxiliary-loss formulation.
- Contrastive imitation learning.
- Official behavior-regulating data augmentations and hidden projection path.

Therefore it is a **PLUTO-inspired budget adapter**, not a full PLUTO reproduction.

### PDM-Closed-style budget scorer

Reference: *Parting with Misconceptions about Learning-based Vehicle Motion Planning*, CoRL 2023. Official code: `autonomousvision/tuplan_garage`.

Retained:

- Centerline/route proximity prior.
- Progress, comfort and safety penalties.
- Deterministic non-learning scoring.

Omitted:

- Official centerline proposal generation and lane-graph search.
- IDM-based rollout.
- PDM proposal sampling, observation construction and complete scoring stack.

This implementation has low reproduction fidelity and must be named **PDM-Closed-style budget scorer**. It should not be presented as official PDM-Closed.

## 2. Engineering audit and fixes

### Removed GPU-to-CPU synchronization in budget selection

The previous `_top_budget_mask` executed a nested batch/evidence Python loop and called `.cpu()` for each evidence cost. With batch 32 and up to 128 evidence atoms, this could introduce thousands of CUDA synchronizations per forward.

The new implementation:

- uses exact vectorized `topk` when `evidence.unit_cost=true`;
- keeps the variable-cost greedy fallback entirely on device;
- returns ranked selected indices and valid-slot masks without data-dependent CPU synchronization.

### Vectorized evidence gather and correct padding masks

The previous code built selected evidence tensors with a Python loop over batch rows and computed a data-dependent maximum using GPU-to-CPU synchronization. It also passed zero padding tokens to attention without a key padding mask.

The new implementation performs batched gather and supplies attention padding masks to Transformer/MultiheadAttention modules.

### Runtime fallback budget now reaches the model

The previous planner changed the stage configuration from B=16 to B=32 during fallback, but the external model kept using the budget captured at construction time. Query accounting could therefore report B=32 while candidate scoring still used B=16 evidence. The runtime stage budget is now passed explicitly into the model forward path, and a regression test verifies monotonic selected-token counts under budget overrides.

### Reduced training synchronization

The old trainer copied every loss to CPU every batch and updated the progress bar every step. The new trainer:

- accumulates loss tensors on GPU and synchronizes once per epoch;
- updates the progress display only every configurable number of steps;
- supports fused AdamW, AMP, TF32, deterministic workers and prefetching;
- supports warmup plus cosine decay and early stopping.

### Strict checkpoint loading

The old evaluation loader silently loaded only shape-compatible tensors and left missing tensors randomly initialized. This can create apparently valid but meaningless comparisons.

External checkpoints now fail immediately if:

- the checkpoint variant does not match the configuration;
- model keys are missing or unexpected;
- any parameter shape differs.

Partial loading is disabled unless the explicit emergency variable `BDSE_ALLOW_PARTIAL_EXTERNAL_CHECKPOINT=1` is set. It must never be enabled for reported comparisons.

### Matched dataset/protocol manifest

Each checkpoint stores:

- ordered train/validation path-size hashes;
- train and validation counts;
- split names;
- seed;
- config hash;
- common optimization protocol.

Before open-loop, budget sweep or closed-loop, the validator checks that all four checkpoints use the exact same train and validation manifests and have the expected names.

### Exact checkpoint naming

Training output must be:

```text
outputs/external/gameformer_budgeted.pt
outputs/external/dtpp_budgeted.pt
outputs/external/plantf_budgeted.pt
outputs/external/pluto_budgeted.pt
```

The trainer automatically writes:

```text
outputs/external/gameformer_budgeted.best.pt
outputs/external/dtpp_budgeted.best.pt
outputs/external/plantf_budgeted.best.pt
outputs/external/pluto_budgeted.best.pt
```

These names exactly match the V60 SWEEP loader. Passing an output name that already ends in `.best.pt` would produce an incorrect `.best.best.pt` name and is prohibited by the wrapper.

## 3. Fairness interpretation

The matched protocol guarantees the same:

- four training city splits;
- 50,000 selected training samples, at most 12,500 per city;
- frozen `val_tune` cache and 500 checkpoint-selection samples;
- random seed;
- common batch size, optimizer family, learning-rate schedule, maximum epochs and early-stopping rule across the four adapters;
- candidate bank, evidence atoms and B=16 interface;
- ordered evaluation scenarios/timestamps/tokens.

It does not make the objectives identical to V60 because V60 has algorithm-specific selector, certificate and residual objectives. Calling that “identical training” would be scientifically inaccurate. The correct claim is **matched data, compute opportunity and evaluation protocol**.

## 4. Closed-loop optimization

The generic comparison runner:

- creates one deterministic scenario-token manifest shared by every system;
- schedules two models concurrently on two GPUs;
- uses one nuPlan worker per process;
- optionally runs multiple process copies per model/GPU;
- disables `simulation_log_callback` to avoid planner/RLock serialization;
- constrains BLAS/OpenMP threads to prevent CPU oversubscription;
- resumes only from an integrity-checked completion marker;
- requires `successful == expected` and `failed == 0` for every shard;
- requires an aggregator parquet with a `final_score` row;
- combines every finite closed-loop metric by scenario-count weighting;
- writes all metrics to CSV/JSON and a compact Markdown summary.

Run the CL4 concurrency benchmark before CL20/CL50. Use `CL_PROCESSES_PER_MODEL=2` only if it reduces wall time without GPU contention or failures.

## 5. Delivered commands

- `RUN_V60_EXTERNAL_BASELINES_MATCHED_TRAIN_2GPU.sh`
- `RUN_V60_EXTERNAL_OPEN_LOOP_2GPU.sh`
- `RUN_V60_BUDGET_BASELINE_SWEEP.sh`
- `BENCHMARK_V60_EXTERNAL_CLOSED_LOOP_CONCURRENCY.sh`
- `RUN_V60_EXTERNAL_CL20_2GPU.sh`
- `RUN_V60_EXTERNAL_CL50_2GPU.sh`
- `RUN_V60_EXTERNAL_CLOSED_LOOP_SUITE_2GPU.sh`

The complete environment setup and recommended execution order are in `NEXT_COMMANDS_V60_EXTERNAL_BASELINES.txt`.
