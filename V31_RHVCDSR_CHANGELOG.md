# v31 RH-VCDSR

RH-VCDSR is a runtime recovery update for BDSE:

- receding-horizon hard (4 s) and soft (6 s) runtime safety checks;
- scene-adaptive normalization for agent, route, soft, hard, and certificate risks;
- component viability guards with explicit relaxation diagnostics;
- true epsilon-Pareto frontier over risk, BDSE certificate loss, and progress loss;
- evidence-conditioned recovery that reuses existing tournament scores without extra evidence queries;
- separate runtime-only output roots in `run_v31_rhvcdsr.sh` to preserve causal ablations.

Validation:

```text
50 passed, 2 warnings
bash -n run_v31_rhvcdsr.sh: passed
```

Recommended first run:

```bash
export SKIP_TRAIN=1
export V31_CKPT=outputs_v30/train/bdse_v30_pmvrbsr.best.pt
export OUT_ROOT=outputs_v31_runtime_v30ckpt
export RUN_MODE=open_loop
bash run_v31_rhvcdsr.sh
```
