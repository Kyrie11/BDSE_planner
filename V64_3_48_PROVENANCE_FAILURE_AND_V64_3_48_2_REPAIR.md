# V64.3.48 provenance failure and V64.3.48.2 repair

## Decision

**ENGINEERING / REPRODUCIBILITY STOP.** The uploaded V64.3.48 result is not used for algorithm attribution in this round.

The server run logged a successful 896-file source-manifest verification. But the uploaded current source package fails the exact same preregistered manifest at 17 paths. Two mismatches (`bdse/experiments/evaluate_open_loop.py` and `bdse/planner/nuplan_planner.py`) are on the open-loop evaluation path used by every V48 fresh arm. Therefore the uploaded source package cannot independently reproduce the exact source semantics of the result-producing server tree.

This is a source-identity failure, not evidence that OCRR itself succeeded or failed.

## Mismatched paths in the uploaded source package

- `bdse/configs/external_dtpp_budgeted.yaml`
- `bdse/configs/external_dtpp_budgeted_fast_cl.yaml`
- `bdse/configs/external_gameformer_budgeted.yaml`
- `bdse/configs/external_gameformer_budgeted_fast_cl.yaml`
- `bdse/configs/external_plantf_budgeted.yaml`
- `bdse/configs/external_plantf_budgeted_fast_cl.yaml`
- `bdse/configs/external_pluto_budgeted.yaml`
- `bdse/configs/external_pluto_budgeted_fast_cl.yaml`
- `bdse/data/tensorizer.py`
- `bdse/experiments/evaluate_open_loop.py`
- `bdse/external_baselines/README.md`
- `bdse/external_baselines/losses.py`
- `bdse/external_baselines/models.py`
- `bdse/external_baselines/train.py`
- `bdse/planner/nuplan_planner.py`
- `bdse/tests/test_external_baselines.py`
- `bdse/tools/run_budget_baseline_sweep.py`

## Why attribution is intentionally withheld

The V47 preregistration made source identity and untouched evidence part of the scientific protocol. The user also explicitly required reliability to pass before any algorithm attribution. Interpreting the already-produced A/B metrics while the source package is not byte-identical would create an avoidable ambiguity: a later change could be wrongly attributed to OCRR when it arose from evaluation-stack drift.

Accordingly, this repair does not report a V48 mechanism verdict and does not use A/B outcomes to choose a V49 branch.

## V64.3.48.2 repair

V64.3.48.2 keeps the OCRR scientific core byte-identical to the preregistered V48 files and makes the currently uploaded execution stack the new fully byte-locked rerun baseline. It adds:

- `V64_3_48_2_SOURCE_MANIFEST.sha256` — whole rerun-source lock;
- `V64_3_48_OCRR_SCIENCE_LOCK.sha256` — original V48 OCRR mechanism lock;
- `bdse/configs/v64_3_48_consumed_fresh1000_tokens.txt` — immutable ledger of the 1000 old A/B tokens;
- `RUN_V64_3_48_2_EAF_ICER_OCRR_SCREEN_2GPU.sh` — clean rerun launcher with a new label-free seed and permanent exclusion of those 1000 tokens.

The scientific GO/STOP conditions are unchanged. No V49 mechanism is added.

## Additional packaging incompleteness found by full-repository testing

The uploaded archive also omits the historical root `V64_SAQA_BCC_NEXT_COMMANDS.sh`, which causes `bdse/tests/test_v64_2_gatefix.py` to fail before any V48-specific logic is exercised. The changelog already records the same historical packaging issue and the prior policy: when exact legacy bytes are unavailable, restore a **fail-closed compatibility entrypoint** rather than inventing old experiment behavior. V64.3.48.2 restores such a refusing compatibility entrypoint; it is not used by OCRR.
