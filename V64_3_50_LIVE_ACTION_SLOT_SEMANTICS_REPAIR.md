# V64.3.50 live action-slot semantics repair

## Observed stop

The repaired first-live-event collector reached token `03dac455f9ec5792` and found a synchronized first live proposal at iteration 7 in both arms, but stopped because the live proposal slot was `6` while the V49 offline audit stored `full_selected_action=23`.

## Why this is not evidence that V49 or RSMR changed

The current V50 tree was checked against `V64_3_49_SOURCE_MANIFEST.sha256`.  `bdse/planner/candidate_generator.py` and `bdse/planner/tournament.py` exactly match the V49 hashes.  The only expected historical-manifest mismatch is `bdse/planner/nuplan_planner.py`, which V50 intentionally extends with evidence-collection instrumentation.

More importantly, the generator does not define `action` as a persistent trajectory ID.  Every planner tick creates a new candidate pool from the current route, ego state, agent conflicts, dynamic feasibility, history priors, and other runtime-only state.  When the pool is larger than K, `_prune_candidate_pool()` ranks and preserves candidates using state-dependent scores and returns the selected entries in a new order.  The final action integer is therefore only a slot in that tick's post-pruning candidate bank.  The code itself stores `pool_original_index`, proving that final slot numbers can change after pruning.

Consequently, comparing `offline V49 action=23` from the cached initial-state bank to `live action=6` from a later state is not a valid identity test.  Rerunning V49 cannot make cross-state slot numbers globally meaningful.

## Correct V50 causal object

V50 is intended to test whether selected-outcome interventional evidence improves retention for the frozen RSMR **selection policy**.  The scientifically meaningful treatment is therefore:

1. freeze the V49 scene cohort/folds and RSMR algorithm/config;
2. run CONTROL and TREATMENT from the same native nuPlan scenario state;
3. require the two arms to remain identical until the first live RSMR proposal;
4. at that same live pre-intervention state, require both arms to produce the same live RSMR winner and the same live Q/P/E;
5. CONTROL vetoes to incumbent;
6. TREATMENT executes that live winner exactly once and then returns to incumbent behavior.

This is stronger on-policy evidence than forcing an integer slot copied from a different offline state.

## New live proposal identity certificate

The planner now records a candidate identity certificate at proposal time.  It includes a SHA256 fingerprint of the proposal trajectory after 1e-4 quantization plus maneuver semantics (`maneuver_id`, `pool_original_index`, maneuver label and theta).  The fingerprint is instrumentation-only.  CONTROL and TREATMENT must have identical certificates.  This certifies same-state proposal identity without assuming action-slot persistence across states.

The offline V49 action remains in the paired row as `offline_v49_action_slot` and a diagnostic `live_vs_offline_action_slot_equal`; it is not a gate.

## What remains frozen

No change is made to RSMR, the candidate generator, tournament ordering, Q/P/E definitions, SIOR pairwise model, regularization, calibration budget, no-rerank/no-second-best/no-fallback containment, or scientific TRAIN/fresh promotion gates.

This is an outcome-blind protocol correction made before a valid paired row is admitted under the corrected action-identity contract.

## V49 rerun / server checkout recommendation

- **Do not rerun V49 to fix this mismatch.** With the same V49 source and data, rerunning only regenerates the same offline action slots; it cannot make a state-local slot globally stable at a later live state.
- A **clean checkout/worktree is recommended** to prevent stale tracked/untracked files or editable-install confusion, but it will not by itself change `23` versus `6`.
- Prefer a new clone or `git worktree` for V50 and keep historical output roots outside that code tree. Point `V49_ROOT` to the existing absolute V49 output path.
- Before running, check `python -c 'import bdse; print(bdse.__file__)'`; V50 launcher now performs this check automatically.

## Packaging restoration

`pyproject.toml` is restored as the canonical metadata/build configuration and `setup.py` is restored as a minimal compatibility shim. In a server without package-index access use:

```bash
python -m pip install -e . --no-build-isolation
```

This installs the current checkout in editable mode without attempting to download an isolated build environment.
