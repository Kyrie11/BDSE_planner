# V52 BFAR-DBAP

V52 converts V51 into a boundary-focused, faster fixed-budget training protocol.

## Main changes

- reuse the existing V51 checkpoint as an immutable base+local anchor when available;
- gate only modules that remain frozen;
- reset residual pair/variance heads after warm start;
- reserve training-pair quotas for teacher-winner, hard-crossing, and near-tie pairs;
- keep 64 pairs on most steps and restore the full graph on exact AOCC steps;
- run exact B=16 supervision on one scene/rank every four steps plus a short final full-alignment tail;
- sample B=8/B=24 only as sparse robustness regularizers;
- retain full-margin certified residual intervention and three-way paired controls;
- directly gate decisive-evidence recall as part of the paper claim.

See `V52_BFAR_DBAP_ANALYSIS_AND_NEXT_STEPS.md` and `NEXT_COMMANDS_V52_BFAR_DBAP.txt`.
