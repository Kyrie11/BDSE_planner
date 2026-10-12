# External baselines

The adapters in this directory expose DTPP-, GameFormer-, PLUTO-, PPAD-, and
PDM-Closed-style baselines through the same cached inputs, candidate budget, and
evaluation entry points as BDSE.

Generate role-based configs with
`python -m bdse.tools.prepare_external_fixed_budget_configs`, train with
`python -m bdse.external_baselines.train`, and evaluate with
`bdse.tools.run_external_open_loop_comparison` or
`bdse.tools.run_external_closed_loop_comparison`.

Store every run's resolved config, checkpoint hash, dataset manifest, and hardware
metadata inside its output directory. Do not create revision-named source files or
launcher scripts for a new run.
