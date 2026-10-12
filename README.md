# BDSE research code

This is a cleaned research snapshot with stable, role-based entry points. It
contains no historical patch chains, result postmortems, `NEXT_COMMANDS` files,
version-named experiment scripts, version-named test files, or hundreds of
generated YAML variants.

## Stable entry points

- `bdse/configs/training.yaml`: current training configuration.
- `bdse/configs/deployment.yaml`: frozen deployment/profiling configuration.
- `RUN_TRAIN.sh`: stable two-GPU training launcher (override `NPROC_PER_NODE`).
- `RUN_BUDGET_PROFILE.sh`: cached-runtime core cost profile.
- `RUN_RUNTIME_BUDGET_PROFILE.sh`: actual nuPlan planner-call profile.
- `bdse/experiments/train.py`: training entry point.
- `bdse/experiments/evaluate_open_loop.py` and `evaluate_closed_loop.py`:
  evaluation entry points.

Create new experiments through an output directory, manifest, and immutable
configuration snapshot. Do not clone source modules or add numeric revisions to
filenames. A new algorithmic component belongs behind a named configuration key
and a focused unit test. Git commits/tags and artifact manifests carry revision
history; source paths describe roles.

## Compatibility boundary

Some internal diagnostic keys and comments remain stable for compatibility with
existing checkpoints and result readers. They are not new experiment entry
points and must not be copied to create another revision. Removing or renaming
checkpoint-facing tensor/schema keys would invalidate the frozen model and is
therefore outside a safe cleanup.

See `BUDGET_PROFILE_README.md` for profiling instructions and the corrected
semantic-attribution behavior.
