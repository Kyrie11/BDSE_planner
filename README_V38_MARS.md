# BDSE v38 — MARS-BDSE

**MARS**: Margin-Aligned Residual Sparsification.

v38 addresses the v37 failure mode in which evidence-family recall and safety coverage were already high, but the fixed `B=16` certificate did not preserve the signed action margins of the full decision interface.

## Core change

The runtime already evaluates every decision Top-M atom on the logical pair graph. v38 treats that Top-M result as a predicted full signed margin field and compresses it into a `B=16` margin coreset.

1. **Adaptive local-head calibration** blends the pair-conditioned head with the existing action-conditioned local head when pair uncertainty or sign disagreement is high.
2. **Signed margin coreset** starts from the Top-M predicted margin and removes atoms while minimizing signed-margin residual, sign flips, target winner/rival certificate loss, and target-action disagreement.
3. Safety remains complete and budget-exempt; soft feasibility, interaction, route and comfort evidence remain selectable.
4. No teacher labels, future labels, new learned parameters, larger evidence budget, larger proposal pool, or larger pair cap are used.

## Fixed budgets

- Decision evidence budget: `B=16`
- Proposal Top-M: `64`
- Runtime logical pair cap: `480`
- B=16 vs full-interface action match gate: `>= 0.17`
- Margin coreset target-action/sign preservation: `>= 0.90 / >= 0.90`
- Effective query gate: `<= 8500`
- Total sparse query gate: `<= 33000`

## Open-loop causal configurations

- `v38_bdse_mars_balanced_fast_cl.yaml`: local-head calibration + signed margin coreset.
- `v38_bdse_mars_winner_strong_fast_cl.yaml`: stronger winner/sign preservation.
- `v38_bdse_mars_pair_only_fast_cl.yaml`: coreset without local-head calibration.
- `v38_bdse_mars_actionrank_control_fast_cl.yaml`: calibrated margins with the old action-rank selector.

The two controls distinguish whether gains come from margin-preserving sparsification, local calibration, or both.

## Run

```bash
SKIP_TRAIN=1 \
V38_CKPT=outputs_v30/train/bdse_v30_pmvrbsr.best.pt \
V30_CKPT_IN=outputs_v30/train/bdse_v30_pmvrbsr.best.pt \
OUT_ROOT=outputs_v38_runtime_v30ckpt \
V35_BASELINE_JSON=outputs_v35_runtime_v30ckpt/open_loop/open_loop_v35_dice_hard7.json \
V35_BASELINE_JSONL=outputs_v35_runtime_v30ckpt/open_loop/open_loop_v35_dice_hard7.jsonl \
RUN_MODE=open_loop \
OPEN_LOOP_MAX_SCENARIOS=1000 \
ENFORCE_RUNTIME_GATE=1 \
bash run_v38_mars.sh
```

Do not run CL20 or training unless at least one main v38 configuration passes the runtime gate.
