# V64.3.50 PIOR raw-DB resolution hotfix

## Failure signature

The first speed/resume package may stop in `build_v64_3_50_pior_train_manifest.py` with:

```text
V64.3.50 PIOR STOP: cannot map token=... log_name=2021.08.23.18.41.38_veh-28 ... exact DB restriction cannot be proven
```

This is an engineering bug in the speed patch, **not** a scientific V50 failure.

## Root cause

Two independent assumptions were too strong:

1. `Path(log_name).stem` was used on a nuPlan log identity containing dots. Python interprets the final dot-delimited part as a suffix, so `2021.08.23.18.41.38_veh-28` becomes `2021.08.23.18.41`.
2. Raw nuPlan DB filenames may append a temporal crop range such as `_00718_00912.db`, while cached NPZ `log_name` stores the stable log identity without that suffix. The repository's existing feature/dataset code already normalizes this suffix.

## Correctness-preserving fix

The frozen scientific identity remains the exact 502 V49 full-set RSMR `scenario_token`s. Raw DB paths are only the I/O search space.

For each token the manifest now uses:

1. exact raw DB stem if available;
2. stable nuPlan log-family match after removing only a literal `.db` suffix and the known numeric crop suffix;
3. for multi-chunk log families, optional read-only SQLite membership lookup against `lidar_pc.token` / `scenario_tag.lidar_pc_token`;
4. if a unique file cannot be proven, the whole stable-log family; if naming cannot prove a family, only the token's known city split.

The paired runner still passes the exact `scenario_filter.scenario_tokens`, so widening the raw DB candidate set cannot introduce a different scientific sample. It can only cost additional scenario-builder I/O. Any batch resume certificate hashes the actual DB candidate-file union, so a changed DB search set invalidates reuse.

## Server command

Use the original V50 command with this hotfixed package:

```bash
cd bdse_v64_3_50_eaf_icer_pior
bash RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh
```

The manifest stage now prints `raw_db_resolution_counts`. Typical fast paths are `exact_stem`, `stable_log_single`, and `stable_log_sqlite_token_exact`. A small number of `stable_log_family_*` fallbacks is safe. `city_split_fallback` is also scientifically safe but may make affected batches slower; if many tokens use it, inspect the raw filenames before doing further speed optimization rather than changing the experiment.

Do not modify the 502 token file, scenario challenge, RSMR, candidate bank, treatment/control action, PIOR label, or nested gate to work around DB naming.
