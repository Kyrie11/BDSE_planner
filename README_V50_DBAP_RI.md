# v50 DBAP-RI

This package is the optimized successor to the uploaded v49 code.

The primary change is not another global residual-weight reduction. Training and deployment now share the same residual intervention gate, and aggregate pair-level corrections cannot flip the integrable local boundary without sufficient confidence.

## Main entry points

- Full two-A30 pipeline: `V50_DBAP_RI_NEXT_COMMANDS.sh`
- Training/evaluation launcher: `run_v50_dbap_ri.sh`
- Concise commands: `NEXT_COMMANDS_V50_DBAP_RI.txt`
- Partial test readiness: `CHECK_PARTIAL_TEST_SET.sh`
- Continue/fresh test preprocessing: `BUILD_MATCHED_TEST_SET.sh`
- Detailed analysis: `V49_RESULT_V50_DBAP_RI_REPORT.md`
- Algorithm changes: `ALGORITHM_UPDATE_LOG.md`

Use a fresh `OUT_ROOT`. Do not reuse v49 checkpoints as resume state. The intended warm start remains the frozen v30 checkpoint.

The official test split must not be used for checkpoint selection, calibration, threshold changes, or gate tuning. The current partial test cache is suitable only for a frozen-model preliminary stress test until manifest integrity and preprocessing completion are verified.
