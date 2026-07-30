# V51 FAR-DBAP

V51 implements **Foundation-Anchored Residual Decision-Boundary Action Preservation** for the BDSE planner.

Main changes:

- blocks residual training unless a newly rebuilt foundation passes a decision-quality gate;
- freezes the foundation base/local interface in the main run;
- resets direct-pair heads before reinterpreting them as residual heads;
- trains do-no-harm preservation on teacher-correct far margins and correction on anchor-wrong/near-tie margins;
- authorizes residual flips against the full normalized foundation margin (`base + local`), with uncertainty-aware confidence;
- evaluates candidate, same-checkpoint local control, and matched foundation control with independent calibration and identical replay rows;
- sanitizes invalid-candidate infinite costs before pair-margin construction and metrics.

Canonical entry point:

```bash
bash V51_FAR_DBAP_NEXT_COMMANDS.sh
```

See `NEXT_COMMANDS_V51_FAR_DBAP.txt` for the full environment and `V51_FAR_DBAP_ANALYSIS_AND_NEXT_STEPS.md` for the algorithm/result analysis.

Latency is reported separately from the algorithmic gate by default. Set `ENFORCE_LATENCY_BEFORE_CL=1` only when the 500 ms deployment target should hard-block CL simulation.
