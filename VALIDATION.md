# Validation performed in the delivery environment

- `test_semantic_attribution_contract`: 2/2 passed.
- `test_planning_budget_unittest`: 12 passed, 1 skipped because PyTorch is not
  installed in the delivery container.
- `python -m compileall -q bdse`: passed.
- `bash -n` for all three root launchers: passed.
- Both generic YAML files load through `bdse.config.load_config` and validate with
  `evidence.budget=16`, `selector.proposal_top_m=24`.
- Revised TeX has balanced environments and braces. A PDF build was not possible
  in this container because its TeX installation has no `pdflatex.fmt`; the user
  bibliography file was also not included in the upload.

Still required on the target machine:

1. Load the real checkpoint with `bdse/configs/deployment.yaml`.
2. Run the 10-scene native smoke profile and confirm no semantic-attribution
   exception.
3. Compare action indices and trajectory hashes between the original frozen
   deployment and the fixed package on the same workload.
4. Run native and attribution profiles separately on the target GPU.
5. Run the actual nuPlan planner-call profile with raw DB/maps and an audited
   scenario-token manifest.
