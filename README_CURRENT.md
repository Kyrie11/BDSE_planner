# Current branch: V64.3.52 EAF-ICER-HODR

V51 POCR is scientifically attributable: the minimal operator-relative QPE+D state is identified, but both sign-only retention arms fail the unchanged paired deployment gate. V52 follows the preregistered `state identified but deployment fail` branch and changes only the paired selected-outcome functional.

HODR factorizes structural null/effect support from the conditional outcome order. The first arm keeps conditional sign ranking; the second replaces sign compression with an unweighted Pareto order over paired official-score and hard-safety deltas. Runtime state remains `[Q,P-Q,E-P,D]`, RSMR remains frozen, and the operator is still same-winner-or-incumbent with no rerank/fallback.

Run `RUN_V64_3_52_EAF_ICER_HODR_TRAIN.sh`. It reuses V50.5 paired evidence and V51 fit result; no 502x2 closed-loop rerun is required.

# BDSE current runnable package

This package intentionally keeps only the current V64.3.50 PIOR entrypoint (V50.2 engineering revision), the two external-baseline entrypoints used by the current workflow, the BDSE source/config/test tree, the algorithm changelog, and the small legacy launcher/science-lock fixtures required by the V50 startup regression suite.

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


## V50.2 frozen-anchor time-alignment repair

The current package keeps the V64.3.50 PIOR algorithm unchanged but repairs nuPlan scenario pre-roll handling. A cached `*_it000000` V49 proposal is an anchor event, while some nuPlan tagged scenarios begin roughly 3 seconds before that anchor. The paired runner now binds the scenario at iteration 0, preserves the incumbent in both arms during pre-roll, and executes the treatment/control split only at the exact frozen anchor timestamp. Cached-plan reuse cannot skip the anchor. Obsolete iteration-0 manifest probe configs fail closed.

The user-facing command and output root are unchanged:

```bash
cd /home/senzeyu2/code/BDSE_planner
bash RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh
```

The existing failed preflight batch has no valid completion certificate and is automatically rerun. Do not copy or force-resume V50.0/V50.1 legacy arms.
