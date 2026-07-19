# BDSE v32 CAVR execution

## 1. Replace current code

```bash
unzip -q BDSE_v32_cavr_optimized.zip
cp -a BDSE_v32_cavr_optimized/. .
```

## 2. Runtime-only open-loop with the v30 checkpoint

```bash
export SKIP_TRAIN=1
export V32_CKPT=outputs_v30/train/bdse_v30_pmvrbsr.best.pt
export OUT_ROOT=outputs_v32_runtime_v30ckpt
export RUN_MODE=open_loop
export OPEN_PARALLEL4=1
bash run_v32_cavr.sh
```

## 3. Runtime-only CL20

```bash
export SKIP_TRAIN=1
export V32_CKPT=outputs_v30/train/bdse_v30_pmvrbsr.best.pt
export OUT_ROOT=outputs_v32_runtime_v30ckpt
export RUN_MODE=cl20
export CL_PARALLEL4=1
export CL_WORKERS_PER_RUN=2
bash run_v32_cavr.sh
```

## 4. Runtime-only CL50

```bash
export SKIP_TRAIN=1
export V32_CKPT=outputs_v30/train/bdse_v30_pmvrbsr.best.pt
export OUT_ROOT=outputs_v32_runtime_v30ckpt
export RUN_MODE=cl50
export RUN_CL50_ALL4=1
export CL_WORKERS_PER_RUN=2
bash run_v32_cavr.sh
```

Do not finetune unless runtime-only CL50 passes the gate in `V32_CAVR_ANALYSIS_AND_CHANGELOG.md`.

## 5. Critical-head finetune

```bash
unset SKIP_TRAIN
export V30_CKPT_IN=outputs_v30/train/bdse_v30_pmvrbsr.best.pt
export OUT_ROOT=outputs_v32
export TRAIN_MAX_SCENARIOS=12000
export VAL_MAX_SCENARIOS=1000
export TRAIN_EPOCHS=3
export NPROC_PER_NODE=2
export RUN_MODE=all
bash run_v32_cavr.sh
```

## 6. CL50 with the v32 checkpoint

```bash
export SKIP_TRAIN=1
export V32_CKPT=outputs_v32/train/bdse_v32_cavr.best.pt
export OUT_ROOT=outputs_v32
export RUN_MODE=cl50
export RUN_CL50_ALL4=1
export CL_WORKERS_PER_RUN=2
bash run_v32_cavr.sh
```
