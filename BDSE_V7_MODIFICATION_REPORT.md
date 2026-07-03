# BDSE v7 normalized safety-guard + interaction-selector modifications

This package modifies BDSE around the current empirical failure mode: the oracle budget sweep shows strong decision sufficiency, while learned open-loop margins/calibration are weak.  The code now treats safety evidence as structural guard evidence and trains the learnable pair head primarily on normalized interaction margins.

## Modified modules

- `bdse/model/bdse_model.py`
  - Added `model.pair_margin_normalized`.
  - Expanded the pair-conditioned head from 6 feature blocks to 10 relation-aware blocks:
    `[a, b, e, q_a, q_b, scene, b-a, a*b, q_b-q_a, q_a*q_b]`.
  - Runtime `predict_certificate_numpy` now exports `pair_margin_scale`, `rival_pair_margin_scale`, and `pair_margin_normalized`.
  - Structural safety mask is now family-aware: hard atom flag OR feasibility family.

- `bdse/model/losses.py`
  - Pair regression/residual/rank/calibration losses now operate in normalized margin space.
  - Safety/feasibility atoms are excluded from learned pair-margin regression by default.
  - Hard-action pairs are excluded from pair regression by default.
  - Pair loss is decision-weighted: teacher-involving pairs, near ties, hard/safe crossings, and interaction atoms get higher weight.
  - Proposal target is focused on interaction evidence; hard/safety atoms are structurally retained instead of learned as proposal positives.
  - Added oracle-to-predicted selector curriculum: early action loss uses oracle selected interaction atoms, then switches to predicted selector after `predicted_selector_start_epoch`.
  - Early epochs skip CPU greedy-selector action loss before `action_loss_start_epoch`, reducing noisy gradients and training overhead.

- `bdse/planner/selector.py`
  - Added `structural_safety_mask()` and `margin_normalization_scale()`.
  - Pair-conditioned greedy selector can now run on normalized margins.
  - Added selector modes for internal baselines: `proposal_top`, `hard_safety_only`, `interaction_only`, `rule_map_only`, `risk_only`.

- `bdse/planner/tournament.py`
  - Pair-conditioned tournament supports normalized margin matrices and reports margin scale diagnostics.

- `bdse/planner/nuplan_planner.py`
  - Closed-loop runtime now passes normalized margin scales consistently into selector/tournament.
  - Added internal `planner.baseline_mode` dispatch:
    `base_only`, `no_evidence`, `dense_full`, `random_budget`, `hard_safety_only`, `proposal_top`, `interaction_only`, `rule_map_only`, `risk_only`, and default `bdse`.
  - `oracle_budget` is explicitly guarded because it requires teacher labels and is therefore only meaningful in offline diagnostics.

- `bdse/experiments/calibrate.py`
  - Calibration residuals now compare predicted normalized margins against normalized teacher margins when `model.pair_margin_normalized=true`.

- `bdse/metrics/bdse_metrics.py`
  - Open-loop diagnostics now compare normalized certificate margins against normalized teacher margins, while dense/base diagnostic margins remain in raw cost units.

- Configs added:
  - `bdse/configs/v7_normalized_fast.yaml`
  - `bdse/configs/v7_bdse_pair_conditioned.yaml`
  - `bdse/configs/v7_baseline_base_only.yaml`
  - `bdse/configs/v7_baseline_dense_full.yaml`
  - `bdse/configs/v7_baseline_random_budget.yaml`
  - `bdse/configs/v7_baseline_hard_safety_only.yaml`
  - `bdse/configs/v7_baseline_proposal_top.yaml`
  - `bdse/configs/v7_baseline_interaction_only.yaml`
  - `bdse/configs/v7_baseline_rule_map_only.yaml`
  - `bdse/configs/v7_baseline_risk_only.yaml`

## Validation performed

- `python -m compileall -q bdse`
- `pytest -q`
- Result: 60 tests passed.
