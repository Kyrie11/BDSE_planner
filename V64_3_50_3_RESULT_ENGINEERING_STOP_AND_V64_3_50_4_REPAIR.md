# V64.3.50.3 result audit and V64.3.50.4 engineering repair

## Verdict

**ENGINEERING/DATA STOP. Do not perform V50 mechanism attribution. Do not design V51 from this run.**

The V50 preregistration requires complete paired selected-outcome evidence before algorithmic attribution. The uploaded run stops during the first 64-scene production batch after the 4-scene preflight and therefore does not contain a valid 502-pair scientific dataset.

## Uploaded evidence

- frozen TRAIN proposal manifest: 502 unique full-set RSMR proposal tokens;
- source manifest verified on server;
- targeted regression: 269 passed;
- paired `batch_0000`: 4/4 valid in both control and treatment;
- paired `batch_0001`: 63 successful + 1 failed in both arms, 63/64 probe events;
- identical failed token in both arms: `188773ba51c15df2`;
- failed frozen target: action slot 3, frozen CandidateBank.K=32, maneuver id 6, frozen proposal SHA256 `a900cf53cf97f1c4b74612c30514bc826fe07ffc77ded52e0b483241ea6fb472`;
- no complete paired outcome JSONL/report and no nested PIOR fit/gate result.

The direct child error is:

```text
RuntimeError: V64.3.50.3 PIOR frozen proposal equals incumbent at exact anchor:
token=188773ba51c15df2 action=3
```

The top-level `invalid return=0 success=63 failed=1 probe_fired=63 expected=64` is the paired runner correctly rejecting the incomplete child batch even though nuPlan itself exits with return code 0 after recording a per-scenario failure.

## Why this is an engineering identity bug

V50.3 intentionally stopped using regenerated action slots as treatment identity. It persists and directly executes the exact cached V49 local proposal trajectory. However, one stale guard still equated the historical frozen proposal's integer slot with the regenerated runtime incumbent's integer slot.

That comparison is not valid across replay. In the uploaded successful events:

- preflight: online proposal matches frozen target 0/4; current-slot geometry median error ≈53.49;
- first production batch: online proposal matches frozen target only 3/63; current-slot geometry median error ≈47.32; only 5/63 have geometry error ≤1e-6.

Thus `slot_id == slot_id` is not a physical-action identity certificate. It is exactly the identity concept V50.3 had already demoted to diagnostic-only for the treatment path.

## V50.4 repair

V50.4 removes only the invalid slot-equality STOP. It does not change the PIOR algorithm.

At the exact frozen anchor:

1. load the exact cached V49 proposal trajectory and verify its SHA256;
2. compute the runtime incumbent trajectory SHA256 from the regenerated current candidate bank;
3. record whether the integer slots collide;
4. record frozen-proposal vs runtime-incumbent physical max-absolute trajectory error and exact hash equality;
5. treatment executes the cached frozen proposal trajectory once;
6. control executes the runtime incumbent;
7. both arms are incumbent-only afterwards;
8. all existing token/time/K/hash/one-shot/no-fallback/metric fail-closed checks remain active.

If the two physical trajectories happen to be exactly equal, that event is audited as a null physical contrast; it is not silently converted to a different proposal and it is not rejected merely because integer slots happen to collide.

## Heartbeat diagnosis

`[PIOR-TICK] ... last='... Simulation failed!'` was a status monitor showing the last relevant child-log line. It did not mean a fresh failure every 30 seconds. The child summary contains exactly one failed simulation per arm. V50.4 keeps the heartbeat but reports an unchanged tail as `last_unchanged=true`, preventing this ambiguity.

## Validation

- focused V50.4: 28/28 PASS;
- V13→V50 targeted: 270/270 PASS;
- full repository: 607/607 PASS across four exhaustive shards (152+152+152+151);
- compileall: PASS;
- launcher syntax: PASS.

## What is deliberately not done

No V50 GO/FAIL algorithm judgment, no PIOR AUC interpretation, no causal retention attribution and no V51 mechanism design are made from this incomplete run. Those are unlocked only after V50.4 yields a complete 502/502 paired dataset and executes the already frozen V50 identification and causal-retention gates.
