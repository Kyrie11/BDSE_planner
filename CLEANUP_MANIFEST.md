# Cleanup manifest

The uploaded source was treated as immutable. This package was rebuilt by an
allowlist rather than deleting files in place.

| Area | Uploaded | Cleaned | Policy |
|---|---:|---:|---|
| Regular files | 1103 | 166 | Keep deployable source, stable tools, focused tests, and role configs |
| YAML configs | 456 | 20 | Replace experiment revisions with `training.yaml` and `deployment.yaml`; retain generic baselines |
| Python tools | 275 | 22 | Remove revision-specific fit/check/calibration scripts |
| Tests | 138 | 31 | Remove revision snapshots; retain invariant and current behavior tests |
| Repository-root files | 147 | 8 | Remove patches, manifests, postmortems, next-command notes, and revision launchers |

The cleaned tree contains no references to version-named configs, launchers,
fitters, or contract-check scripts. Checkpoint-facing tensor names and diagnostic
schema keys in the production runtime were preserved because renaming them would
break the supplied frozen checkpoint and downstream result readers.

Functional changes in this package are limited to:

1. Correct optional semantic attribution when selected-atom metadata exists but
   decisive-frontier contribution rows were not produced.
2. Add an explicit availability diagnostic while preserving strict alignment for
   nonempty attribution.
3. Add a pure NumPy regression-testable semantic attribution module.
4. Give the profiling launchers stable defaults and explicit path checks.
5. Replace historical YAML entry points with role-based training/deployment names.

Future experiments should create a resolved config and manifest in the output
directory. They should not create another source/config/script family named after
an experiment revision.
