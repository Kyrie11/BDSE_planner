# V64.3.50 uploaded result: engineering STOP and V64.3.50.1 probe-identity repair

## Executive verdict

The uploaded run is **not sufficient for V50 algorithm attribution**.  It is useful only for diagnosing the collection implementation.

The server executed the exact uploaded V50 source (913/913 source-manifest match) and passed the server targeted regression (254/254), but paired TRAIN collection stopped in the first 64-scene batch because each nuPlan arm completed 64/64 simulations while only 50/64 required PIOR interventions fired.  Consequently the exact 502-scene paired table, the nested PIOR fit, and the preregistered identification/causal-retention gates do not exist.

According to the V49→V50 preregistration, this is an **ENGINEERING/DATA STOP**.  No PIOR promotion/falsification, convergence claim, dominant-bottleneck update, or V51 algorithm design is valid from this run.

## Why "closed loop started" is not a GO signal

V50 differs from the earlier offline versions: paired closed-loop outcomes are the **TRAIN supervision source**.  The order is:

1. exact V49 failure/science lock;
2. exact 502 frozen-proposal TRAIN manifest;
3. paired control/treatment TRAIN closed-loop outcome collection;
4. only then: nested PIOR outcome identification and retention gate;
5. only if TRAIN passes: freeze the artifact and later run untouched validation.

The uploaded run stopped inside step 3.  Therefore it passed neither step 4 nor any untouched/final closed-loop gate.

## Evidence in the uploaded result

- code ZIP SHA256: `62f9937884b70e9aeece881604c8a31421a319a328a4926c0d418aa213713ca6`;
- result ZIP SHA256: `1a509941fee856a52bdc569a736e5e5058ae28921769909c25534c8ed88fa7f7`;
- source manifest against uploaded code: **913/913 PASS**;
- server targeted tests: **254/254 PASS**;
- TRAIN manifest: **502 unique proposal tokens**, all from `*_it000000.npz`;
- DB resolution: all 502 were resolved by exact SQLite token lookup in the provided run;
- batch 0: 64 requested scenarios per arm;
- nuPlan: control **64 success / 0 fail**, treatment **64 success / 0 fail**;
- PIOR probe events: control **50/64**, treatment **50/64**;
- fired iterations in each arm: iteration 0 only 16 events; the remainder occur at later iterations 5..140;
- old probe events omit `scenario_token`;
- no complete paired-outcome report/table;
- no `v64_3_50_pior_fit.json` and no emitted PIOR config.

The 50/64 subset must not be used for mechanism metrics: inclusion is conditioned on the online planner happening to regenerate a proposal, creating an implementation-induced selected subset.

## Root cause

The manifest already contains the exact frozen V49 `full_selected_action` for each of the 502 scenes.  However, the original closed-loop probe ignored that field.  At every simulator replan it read the *current* tournament diagnostics and waited for `proposal_exists`.  If a new online proposal appeared, treatment executed that newly observed proposal and control kept the incumbent; otherwise both arms waited.

That implementation does not match the preregistered causal object: **execute the already-frozen full-set V49 proposal once versus the same incumbent**.

Because every frozen V49 proposal cache sample in the uploaded manifest is `it000000`, the correct intervention event for this preregistered population is scenario iteration 0.

## Repair

V64.3.50.1 is an engineering-only repair; it is not a new algorithm version.

### Frozen target map

The manifest builder stores `cache_iteration`, NPZ `timestamp_us`, and `full_selected_action`.  A batch-local `pior_probe_targets.json` is passed to the nuPlan planner.  The planner's first call must be iteration 0 and is bound to exactly one frozen target using the scenario-start timestamp.  Duplicate start timestamps are deterministically separated into different subprocess batches.

### Exact action contract

At iteration 0:

- treatment final action = manifest frozen `full_selected_action`;
- control final action = current incumbent/baseline;
- manifest proposal must be a valid current candidate and must differ from incumbent;
- online tournament proposal is diagnostic only and cannot choose/replace the target.

After the one-shot event, both arms preserve incumbent-only behavior as before.

### Token-level proof

Every probe certificate now contains exact token, target timestamp, current timestamp, frozen action, incumbent, final action and structural containment flags.  Batch acceptance requires one and only one valid event for every requested token.

### Fail-fast paired preflight

Before the expensive 502-scene collection proceeds, each arm runs a 4-scene first batch.  A paired barrier requires both arms to validate successfully before either can continue.  This protects server time without altering scientific data or folds.

### Resume

The invalid uploaded batch has no V50.1 target-bound completion certificate and is not scientifically reusable.  Use the new default output root.  Future completed batches are resumable only when token/config/checkpoint/DB/target/event/metric hashes all match.

## What is intentionally unchanged

- exact 502 V49 full-set RSMR proposal population;
- candidate generator and action indexing;
- RSMR ordinal selector;
- V44 occupancy / V45 response / V47 EGO-REF inputs already embedded in Q/P/E;
- `[Q, P-Q, E-P]` PIOR state;
- zero-bias pairwise loss and `lambda=1`;
- `closed_loop_nonreactive_agents` TRAIN challenge;
- official score and hard-safety metric definitions;
- PIOR positive/negative outcome label;
- nested folds and preregistered AUC/retention gates;
- same winner or incumbent, no rerank, no second best, no fallback.

## Next scientific decision

Rerun V50.1.  If the 502 paired collection completes, then and only then inspect the preregistered TRAIN gate.  A TRAIN failure is a scientific V50 result and must stop before untouched validation.  A TRAIN pass freezes PIOR and permits a separately preregistered untouched validation.  No V51 design is justified before that result.
