# BDSE current runnable package

This package intentionally keeps only the current V64.3.50.1 PIOR entrypoint, the two external-baseline entrypoints used by the current workflow, the BDSE source/config/test tree, the algorithm changelog, and the small legacy launcher/science-lock fixtures required by the V50 startup regression suite.

## Repository/output layout

The repository root is resolved as `BDSE_ROOT` (default: the directory containing the launcher). All experiment outputs are rooted at:

```text
$BDSE_ROOT/outputs/
```

For the user's deployment this is:

```text
/home/senzeyu2/code/BDSE_planner/outputs/
```

## Current V50.1 PIOR TRAIN

```bash
cd /home/senzeyu2/code/BDSE_planner
bash RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh
```

Default output:

```text
$BDSE_ROOT/outputs/outputs_v64_3_50_1_eaf_icer_pior_train_2gpu_v1/
```

Historical prerequisites are read from:

```text
$BDSE_ROOT/outputs/outputs_v64_3_49_eaf_icer_siir_screen_2gpu_v1/
$BDSE_ROOT/outputs/outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1/
```

## External baselines

Training:

```bash
GPUS=0,1 bash RUN_FAIR_EXTERNAL_BASELINES_TRAIN_B8_B16_B24_2GPU.sh
```

Checkpoints/configs:

```text
$BDSE_ROOT/outputs/external_fixed_budget/
```

Closed-loop test:

```bash
GPUS=0,1 \
CL_WORKERS_PER_JOB=4 \
bash RUN_EXTERNAL_FIXED_BUDGET_CLOSED_LOOP_TEST_B8_B16_B24_2GPU.sh
```

Closed-loop outputs:

```text
$BDSE_ROOT/outputs/closed_loop/external_fixed_budget_test/
```

## Intentionally retained legacy root files

A small number of old launchers and `V64_3_48_OCRR_SCIENCE_LOCK.sha256` remain only because the current V13→V50 regression tests read them as frozen historical fixtures. They are not current experiment entrypoints and should not be used for new runs.

## V50.1 target-spec hash hotfix

The current package fixes a pre-simulation target-spec checksum bug. The target JSON now has two distinct integrity concepts: a canonical semantic SHA256 used for scientific target identity and a file-byte SHA256 used for resume/file integrity. Do not re-enable V50.0 legacy full-arm resume; only V50.1 manifest-bound batch certificates are reusable. The user-facing command remains unchanged:

```bash
cd /home/senzeyu2/code/BDSE_planner
bash RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh
```
