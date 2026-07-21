# v43 SAB-DACC

SAB-DACC is a stage-aware extension of v42 CBL-DACC. It keeps the checkpoint, Top-M, B=16, pair queries and all search limits fixed.

## Why v42 appeared to fail

The v42 runtime result reached 0.951 target-action preservation and passed every absolute and paired quality gate. The checker incorrectly averaged a conditionally executed recovery-iteration diagnostic over all 1000 scenes, including 947 scenes where recovery was correctly not invoked.

## Remaining algorithm issue

In 12 of the 49 residual failures, the target already ranked first under post-safety tournament scores, but utility refinement selected another action. v42 discarded utility-stage diagnostics, leaving a zero recovery deficit.

## v43 change

The deployment callback returns exact downstream diagnostics. Fixed-budget search distinguishes raw-score failures from utility overrides and minimizes the exact distance needed to exclude the utility-selected rival from either the score-slack band or pair-certificate band.

## Fast gate checker

`check_v43_sabdacc_gate.py` streams aligned JSONL files, computes online paired moments, validates conditional recovery diagnostics on attempted scenes, and writes the analysis report in the same process. `run_v43_sabdacc.sh` invokes it with `python -S`.

## Re-checking the uploaded v42 result

The backward-compatible v42 checker is also corrected and can be run directly with:

```bash
python -S -m bdse.tools.check_v42_cbldacc_gate \
  open_loop_v42_cbldacc.json open_loop_v42_mars_control.json
```

It validates iterations conditionally on the 53 attempted scenes and reports PASS for the uploaded v42 result.
