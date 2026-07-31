# V54 AR-BFAR-DBAP

V54 fixes the action-aggregation bottleneck exposed by V53.

## Core change

The fixed-budget tournament is anchored to the integrable selected-local action cost:

```text
J_B(a) = J0(a) + sum_{i in selected B=16} g_i(a)
```

The pair head contributes only a residual correction. At zero residual, the pair tournament exactly reproduces the selected-local planner. A residual action flip is accepted only by the full-margin action-anchor guard.

## What remains unchanged

- fixed primary budget B=16;
- exact AOCC deployment selector;
- winner/hard-cross/near-tie boundary curriculum;
- decisive-evidence targets;
- independent candidate/local/foundation calibration;
- three-way paired open-loop and closed-loop evaluation.

## Gate policy

- protocol-integrity failure blocks all closed loop;
- minimum failure still permits a labelled diagnostic paired CL20;
- competitive PASS is required for automatic CL100 escalation.

## Main run

See `NEXT_COMMANDS_V54_AR_BFAR.txt` or `V54_AR_BFAR_DBAP_NEXT_COMMANDS.sh`.
