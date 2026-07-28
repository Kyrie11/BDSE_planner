# BDSE v48 DBCE

**DBCE** stands for **Deployment-Boundary Critical Evidence**.  This branch is a
correction of v47 D3CE after the uploaded paired val-tune experiment failed the
open-loop gate and never reached closed-loop simulation.

## Result that motivated v48

The v47 candidate improved several intermediate quantities but failed the strict
gate:

| Metric | v47 | control / gate |
|---|---:|---:|
| Teacher action match | 0.249 | 0.238; required gain +0.020 |
| Evidence sufficiency | 0.073575 | 0.070400; required gain +0.010 |
| Pair-full deployment match | 0.248 | required 0.300 |
| AOCC certified-pair fraction | 0.135 | required 0.500 |
| Pair near-tie sign accuracy | 0.469 | control 0.583 |
| Planner latency p95 | 1365.15 ms | required <=500 ms |

The closed-loop directory is empty because the strict gate returned nonzero and
the pipeline runs under `set -e`.  In addition, two independent v47 launchers
wrote to one output root, producing duplicate epoch rows.  Do not use that run as
a paper result.

## Core change

V47 trained evidence ranking with positive teacher support.  DBCE trains the
counterfactual contribution of atom `i` to a teacher-action/rival boundary:

```text
criticality_i(w,r) =
    relu(gamma - (m_wr - d_i(w,r))) - relu(gamma - m_wr)
```

The target is cost normalized and shared by the pair-residual and proposal
heads.  It is learned over the union of teacher-nearest and model-confused
rivals.

Runtime uses:

1. an integrable local action-conditioned margin;
2. a sparse antisymmetric pair residual only on boundary/safety/winner pairs;
3. uncertainty/disagreement shrinkage toward the local margin;
4. an exact-tournament AOCC frontier containing near-boundary and
   safety-crossing target rivals;
5. a nested fixed-budget evidence prefix.

## Important configuration corrections

- `proposal_top_m: 64 -> 24`;
- reserved interaction slots: `24 -> 8`;
- pair residual refinement cap: `48 -> 32`;
- selector pair cap: `128 -> 96`;
- runtime pair-query cap: `320 -> 192`;
- tournament rivals: `16 -> 12`;
- AOCC target rivals: `16 -> 6`;
- calibration prior radius: `0.10 -> 0.02`;
- pair calibration is enabled for local-plus-residual margins;
- best-checkpoint scoring prioritizes teacher match, pair-full/near-tie
  correctness, regret and fixed-budget preservation instead of raw recall.

## Clean experiment protocol

Use a new `OUT_ROOT`.  The scripts enforce one writer per output root.

```bash
export NUPLAN_ROOT=/path/to/nuplan
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2
export BDSE_VAL_CACHE_ORIGINAL=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2
export V30_CKPT_IN=outputs_v30/train/bdse_v30_pmvrbsr.best.pt
export OUT_ROOT=outputs_v48_dbce_exact_2gpu_v1

PIPELINE_DETACH=1 \
RUN_CLOSED_LOOP_AFTER_GATE=1 \
bash V48_DBCE_NEXT_COMMANDS.sh
```

The pipeline performs:

1. log-disjoint `val_tune` / `val_calib` construction;
2. clean two-GPU training from v30 using `val_tune` only;
3. independent one-sided calibration on `val_calib`;
4. candidate and frozen-control replay on identical val-tune rows;
5. strict paired gate;
6. automatic CL20 only after PASS.

Monitor:

```bash
tail -f "$OUT_ROOT"/logs/pipeline_*.log
```

## Gate policy

The v48 gate does not relax the original requirements.  It additionally fails
on:

- duplicate epoch rows;
- near-tie sign regression;
- interaction evidence consuming more than 85% of decision budget;
- AOCC frontier retaining less than 45% of target-pair weight;
- fallback rate above 50%;
- missing independent calibration provenance.

## Validation performed in this package

```text
158 passed, 5 warnings
```

Python compilation and shell syntax checks for both run scripts pass.

## Research claim boundary

The proposed contribution is not “critical evidence recall.”  It is
**deployment-boundary action preservation under a fixed evidence budget**, using
counterfactual criticality, an integrable-plus-residual margin interface and a
nested calibrated certificate.  A certificate preserves the learned
full-interface action under its stated calibration assumptions; it does not
correct candidate-bank or full-interface representation errors.
