# BDSE v34 — ABIQ

**ABIQ: Antisymmetric Budgeted Influence Queries for Decision-Sufficient Planning**

v34 is a runtime-selector correction built for the unchanged evidence budget `B=16`.
It does not add a learned head and is therefore compatible with the v30 checkpoint for
causal runtime-only evaluation.

## Core changes

1. Canonicalize reciprocal action pairs before neural pair scoring.
2. Apply the reciprocal collapse to the actual pair-conditioned selector.
3. Reserve interaction/precedence evidence separately from broad decision families.
4. Enforce a logical pair-query cap before model execution.
5. Report the actual unique selector/tournament union query count.
6. Add a feasibility-first runtime gate and four causal selector configurations.

## First command

```bash
SKIP_TRAIN=1 \
V34_CKPT=outputs_v30/train/bdse_v30_pmvrbsr.best.pt \
OUT_ROOT=outputs_v34_runtime_v30ckpt \
RUN_MODE=open_loop \
bash run_v34_abiq.sh
```

Do not run closed loop or training unless at least one configuration passes the printed
runtime gate.
