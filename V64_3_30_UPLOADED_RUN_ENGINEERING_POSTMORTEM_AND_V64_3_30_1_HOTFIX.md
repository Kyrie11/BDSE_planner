# V64.3.30 uploaded-run engineering postmortem and V64.3.30.1 hotfix

## Decision

The uploaded V30 result is **not scientifically analyzable**. It is an incomplete engineering-failure run, not a negative or mixed capacity result.

The B16 TRAIN control completed, but the B24 capacity arm crashed before emitting a single per-scene or frontier-edge row. Consequently no B24 DRC fit, untouched token selection, split A/B evaluation, or double-fresh screen exists. The question "did B=16 discard useful candidate-specific signal, or does the downstream recovery consumer fail to use available signal?" therefore remains unresolved by this upload.

## Concrete evidence from the uploaded output

- `train_b16_v20_rows.jsonl`: 3000 rows.
- `train_b16_v20_edges.jsonl`: 75,133 rows.
- `train_b24_v20_rows.jsonl`: 0 bytes / 0 rows.
- `train_b24_v20_edges.jsonl`: 0 bytes / 0 rows.
- `train_b24_v20.out` terminates with:
  `ValueError: selected evidence count 24 exceeds attribution spectrum budget 16`.
- output `configs/baseline_v25_b16` and `configs/fbic_b24` contain no generated fitted configs because the script stopped in paired TRAIN replay.
- stage timing records only `prerequisites`; fresh selection was never reached.

## Root cause

The FBIC design intentionally keeps upstream AOCC at B16 and then exposes all already-queried M=24 atoms downstream. ICER always constructed a historical V24 attribution-resolved diagnostic matrix even when the current recovery path did not use that representation. That diagnostic had been hard-coded to a 16-atom spectrum. A legal B24 FBIC scene therefore raised before any V30 scientific measurement was possible.

The current V30 V20 control and V30 DRC both do **not** use the attribution-resolved representation as their recovery-risk feature view; the DRC remains the unchanged 18-D `evidence_only` aggregate view. Thus the correct fix is to repair diagnostic capacity compatibility, not to alter the scientific recovery rule.

## Hotfix

`tournament.py` now:

- retains the exact historical B16 32-D default schema;
- derives a V30 instrumentation ceiling of 24 only when FBIC is enabled;
- emits 48 attribution diagnostic coordinates for B24 (24 candidate + 24 delta);
- validates dynamic attribution diagnostic schemas by their actual width;
- leaves learned-memory stored-schema checks strict and unchanged.

No selector score, learned head, DRC feature, threshold, candidate rank, evidence query, or structural guard was modified.

## Regression protection

Added tests cover:

1. B24 attribution instrumentation = 48-D and includes all 24 entries;
2. B16 default attribution instrumentation remains exactly 32-D;
3. a full synthetic ICER tournament under the exact V30 config with 24 selected atoms completes without the historical B16 guard failure;
4. dynamic 48-D attribution diagnostic view has a self-consistent schema;
5. generated V30 B24 DRC still uses `evidence_only` recovery risk and retains the FBIC B16->B24 capacity contract.

Validation: targeted 119/119 PASS; full repository 449/449 PASS; 36 pre-existing Transformer warnings only.

## Rerun discipline

The invalid upload never reached fresh-token selection. Therefore the scientifically correct action is to rerun the same V30 experiment using the same exclusion manifest, same frozen 3000 TRAIN set, and same `FRESH_HASH_SEED=v64.3.30-eaf-icer-fbic-double-fresh-v1`. A new fresh seed would unnecessarily spend another untouched population and weaken causal continuity.

Use the hotfix wrapper so repaired outputs are written to a new directory and cannot mix with the invalid provenance.
