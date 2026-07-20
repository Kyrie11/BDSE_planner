# BDSE v37 SAGE

**SAGE-BDSE: Safety-Always-on Graded Evidence Planning**

v37 fixes the two regressions exposed by the v36 runtime-only gate while preserving the fixed decision-evidence budget (`B=16`) and proposal size (`M=64`).

## Why v36 failed

1. `selected_action_safety_flag_rate=0.035` was dominated by samples where every valid candidate was flagged. The final hard guard already prevents an avoidable flagged choice whenever any safe candidate exists, so the old metric mixed an unavoidable candidate-bank condition with a selector error.
2. v36 removed every hard atom and the entire feasibility family from the decision certificate. This also removed soft feasibility evidence such as route connectors and speed-limit evidence.
3. Binary hard flags preserved constraint feasibility but discarded the continuous clearance/TTC/route-boundary information that the v30 teacher margin had used to rank feasible actions.
4. Hard restriction of the pair graph changed deployment support relative to the unchanged v30 checkpoint.

## v37 design

- Complete hard constraints remain budget-exempt and lexicographic.
- Only truly hard atoms bypass the selector; soft feasibility remains eligible for the `B=16` decision certificate.
- Continuous structural risk is compressed into a budget-free graded residual added to the base action cost.
- Pair comparisons are softly reweighted by viability instead of being deleted.
- All-flagged candidate banks use a continuous minimum-hard-risk guard with the certificate as a tie-breaker.
- Safety diagnostics separate avoidable unsafe selections from all-flagged candidate banks.

No neural parameter is added. A v30 checkpoint remains structurally compatible.

## Strict runtime-only gate

```bash
SKIP_TRAIN=1 \
V37_CKPT=outputs_v30/train/bdse_v30_pmvrbsr.best.pt \
V30_CKPT_IN=outputs_v30/train/bdse_v30_pmvrbsr.best.pt \
OUT_ROOT=outputs_v37_runtime_v30ckpt \
V35_BASELINE_JSON=outputs_v35_runtime_v30ckpt/open_loop/open_loop_v35_dice_hard7.json \
RUN_MODE=open_loop \
OPEN_LOOP_MAX_SCENARIOS=1000 \
ENFORCE_RUNTIME_GATE=1 \
bash run_v37_sage.sh
```

The gate evaluates four causal configurations:

- `v37_sage_balanced`: structural residual weight 0.22, soft viability weighting.
- `v37_sage_risk35`: stronger structural residual weight 0.35.
- `v37_sage_full_graph`: full pair graph control.
- `v37_sage_interaction7`: stronger interaction allocation control.

Do not run CL20 or training unless at least one configuration passes.
