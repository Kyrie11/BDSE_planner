# v50 checkpoint dependency diagnosis and corrected experiment protocol

## 1. Why the deleted v30 path still appeared

The uploaded launchers still contained two hard-coded fallbacks:

```bash
V30_CKPT_IN="${V30_CKPT_IN:-outputs_v30/train/bdse_v30_pmvrbsr.best.pt}"
```

and, in the outer launcher, the same value was also used to initialize `CONTROL_CKPT`.

Therefore `unset V30_CKPT_IN` did not mean “no v30 checkpoint”. It immediately restored the deleted path. The previous full-retrain launcher was intended to intercept that missing path and rebuild a foundation, but an error that still prints the exact historical path indicates one of the following engineering states:

1. the server is executing an older/stale `V50_DBAP_RI_NEXT_COMMANDS.sh` or `run_v50_dbap_ri.sh`;
2. a detached pipeline was launched before the updated file was copied;
3. the outer launcher resolved a new foundation, but the inner launcher again replaced an empty value with its historical default;
4. the command was read from an earlier output log rather than the current versioned launcher.

The corrected launcher prints:

```text
[v50] pipeline_version=v50.1-checkpoint-independent
```

at startup and resolves all relative paths from its own directory. If that banner is absent, the server is not running this revision.

The command in the user message also contains an unrelated shell typo:

```bash
export NUPLAN_ROOT= /data0/senzeyu2/dataset/nuplan/
```

There must be no space after `=`:

```bash
export NUPLAN_ROOT=/data0/senzeyu2/dataset/nuplan
```

This typo does not explain the v30 checkpoint error, but it would break the gated closed-loop stage later.

## 2. Should a v40-v49 best checkpoint replace v30?

### Paper main run: no, unless it is a true retained v30 copy

A trained v40-v49 checkpoint already contains method-specific adaptation:

- v46 contains AOCC adaptation;
- v47 contains D3CE adaptation;
- v48 contains DBCE adaptation;
- v49 contains DBAP adaptation.

Warm-starting v50 from one of these checkpoints changes the experiment from “evaluate v50 algorithm design from a shared foundation” to “continue fine-tuning an earlier algorithm into v50”. Any improvement can then be caused by extra optimization history, inherited selector/residual heads, or favorable proximity to the v50 objective. This contaminates the causal interpretation of v50 effectiveness.

### Safe candidates

The following are safe when their checkpoint file and internal metadata confirm the identity:

1. a retained file named `bdse_v30_pmvrbsr.best.pt`;
2. a prior rebuilt file named `bdse_v30_pmvrbsr_rebuilt.best.pt`;
3. a checkpoint whose stored `args.config` or `args.output` explicitly identifies the v30 PMVRBSR training run.

A likely location worth checking first is:

```text
outputs_v50_dbap_ri_fulltrain_fast_2gpu_v1/foundation_v30/train/
```

because the previous full-retrain launcher may have completed the foundation stage before failing later.

### Directory names are not enough

Directories such as:

```text
outputs_v40_lexdacc_runtime_v30ckpt
outputs_v41_prdacc_runtime_v30ckpt
outputs_v42_cbldacc_runtime_v30ckpt
outputs_v43_sapdacc_runtime_v30ckpt
```

may only contain runtime evaluation artifacts produced *using* v30. Their names do not prove that a copy of the v30 checkpoint exists inside them. Similarly, `outputs_v47_control_val_tune` usually stores control replay JSON/JSONL rather than model weights.

### Transfer ablation

A v49 best checkpoint can be used for a separate transfer-initialization experiment. It must be labeled as such and must not replace the main matched-foundation experiment. For a meaningful transfer comparison, v49-objective continuation and v50-objective continuation should start from the exact same v49 checkpoint, reset optimizer/scaler/epoch, and use identical data and seeds.

## 3. Corrected foundation resolution

The canonical variable is now `FOUNDATION_CKPT`; `V30_CKPT_IN` is only a compatibility alias for the inner launcher.

Under `FOUNDATION_POLICY=auto`, the pipeline uses:

1. an explicit existing `FOUNDATION_CKPT`;
2. an already rebuilt foundation under the current output root;
3. a conservatively verified retained v30 copy found under outputs v40-v50;
4. otherwise, a fresh current-code v30-compatible foundation rebuild.

No physical `outputs_v30` directory is required.

The resolver writes:

```text
OUT_ROOT/provenance/foundation_checkpoint_inventory.json
OUT_ROOT/provenance/foundation_checkpoint.json
```

The first file records all visible checkpoints, tensor-shape compatibility, stored config/output metadata, algorithm markers, rejection reasons, and the selected safe candidate. The second records the checkpoint actually used by the experiment.

## 4. Effect on v50 algorithm conclusions

### Rebuilt matched foundation

A newly rebuilt foundation changes the absolute starting weights, so old and new v49/v50 numerical results are not directly comparable. However, v50 effectiveness can still be evaluated cleanly when:

- v50 and frozen control use the same rebuilt foundation;
- validation rows are identical;
- calibration provenance is identical;
- the open-loop gate is paired;
- closed-loop tokens are identical.

The new result supports claims within this matched experiment family.

### Recovered exact v30 copy

If an exact retained v30 copy is recovered, historical comparability is stronger, although control/calibration/replay should still be regenerated because code and validation protocol changed.

### Later algorithm checkpoint initialization

Using v47-v49 as the main v50 initialization weakens the novelty argument and makes gate improvements ambiguous. It should not be used for the paper main result.

## 5. Recommended server procedure

First verify the launcher revision:

```bash
grep -n "PIPELINE_VERSION" V50_DBAP_RI_NEXT_COMMANDS.sh
```

Expected output includes:

```text
v50.1-checkpoint-independent
```

Inventory retained checkpoints:

```bash
bash INSPECT_RETAINED_CHECKPOINTS.sh retained_checkpoint_inventory.json
```

Then run the clean automatic policy:

```bash
unset V30_CKPT_IN
unset FOUNDATION_CKPT
unset CONTROL_CKPT

export NUPLAN_ROOT=/data0/senzeyu2/dataset/nuplan
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2
export BDSE_VAL_CACHE_ORIGINAL=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2
export OUT_ROOT=outputs_v50_dbap_ri_checkpoint_independent_2gpu_v1

PIPELINE_DETACH=1 \
RUN_CLOSED_LOOP_AFTER_GATE=1 \
RUN_CL100_AFTER_CL20=0 \
FOUNDATION_POLICY=auto \
REBUILD_FOUNDATION_IF_MISSING=1 \
RECOVER_SAFE_FOUNDATION_COPIES=1 \
ALLOW_ALGORITHM_CHECKPOINT_INIT=0 \
EXACT_SELECTOR_CPU_BACKEND=process \
EXACT_SELECTOR_WORKERS_PER_RANK=4 \
GPUS=0,1 \
bash V50_DBAP_RI_NEXT_COMMANDS.sh
```

To force a clean rebuild and ignore all retained checkpoints:

```bash
FOUNDATION_POLICY=rebuild \
REBUILD_FOUNDATION_IF_MISSING=1 \
... \
bash V50_DBAP_RI_NEXT_COMMANDS.sh
```

To use a manually verified retained v30 copy:

```bash
FOUNDATION_POLICY=explicit \
FOUNDATION_CKPT=/absolute/path/to/bdse_v30_pmvrbsr.best.pt \
... \
bash V50_DBAP_RI_NEXT_COMMANDS.sh
```

A fresh output root is recommended because the old root may contain logs, locks, calibration, or control files produced before foundation resolution was corrected.
