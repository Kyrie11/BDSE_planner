# V64.3.50 V49 prerequisite-lock schema repair

## Symptom

Launching V50 stopped at:

`STOP V50: V49 SIIR AUC signature changed`

If that line was manually removed, the launcher then reported one source-manifest checksum mismatch.

## Root cause

The V49 fit report written by `fit_v64_3_49_eaf_icer_siir.py` stores risk identification as flat fields:

- `aggregate_ego_ref_auc`
- `aggregate_obs_sign_auc`
- `aggregate_siir_auc`
- `siir_better_ego_fold_count`
- `siir_better_obs_fold_count`
- `identified`

The initial V50 prerequisite checker instead attempted to read the nonexistent nested field:

`risk_identification.siir.aggregate_nonpositive_risk_auc`

and defaulted to `-1`. This made the AUC signature check fail even when the preceding exact SHA256 locks had already proved that the V49 result artifacts were the intended ones.

## Repair

The five byte-level V49 prerequisite SHA locks and the exact preregistered V49 failure diagnosis are retained. Only the redundant malformed AUC check is changed. V50 now validates the actual persisted V49 schema and recomputes the preregistered identification condition. The expected V49 prerequisite remains `identified=False`.

The same fix is applied inside the V50 fit tool, so the later nested fit cannot fail on the same malformed schema assumption.

## Why manually deleting the check caused a checksum warning

`RUN_V64_3_50_EAF_ICER_SIOR_SCREEN_2GPU.sh` is included in the V50 source manifest. Editing the launcher changes its SHA256, so `sha256sum -c V64_3_50_SOURCE_MANIFEST.sha256` correctly reports a mismatch. Use an untouched extraction of the repaired package instead.

## Why old V48.2/V49 launchers/manifests show one mismatch in the V50 tree

V50 intentionally changes `bdse/planner/nuplan_planner.py` for its one-shot selected-outcome probe. The historical V48.2/V49 manifests contain the old planner hash, while the V50 manifest contains the new one. Therefore the old manifests each show one planner mismatch when applied to the V50 tree. This is expected provenance behavior, not source corruption. To rerun V49, use the exact V49 package/tree; to run V50, use the V50 manifest.

## Scientific status

Engineering-only repair. V64.3.50 SIOR selector, paired intervention, state, loss, calibration, gates, and runtime containment are unchanged.
