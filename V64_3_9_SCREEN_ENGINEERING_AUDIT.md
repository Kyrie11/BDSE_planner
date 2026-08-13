# V64.3.9 AF-BDMU Screen Engineering / Causal Audit

## Decision

The uploaded `outputs_v64_3_9_af_bdmu_screen_2gpu_v1` should **not** be promoted to full/test/closed-loop, but it also should **not** be treated as a clean negative result for the AF-BDMU algorithm. A material training/deployment Top-M semantic mismatch is active in the exact mechanism being tested. Per the experimental branch rule, this revision contains engineering fixes only; no new algorithm is introduced.

## What did run correctly

The experiment is not a generic launcher/checkpoint failure:

- config contract: PASS;
- final training checkpoint: present;
- epoch checkpoints 1--4: present;
- trained epochs 0--3: present;
- AF-BDMU adaptive frontier / worst-rival / swap-ranking flags: active;
- optimizer was active: train loss `2.2914 -> 2.1394`, listwise `3.2052 -> 3.0523`, swap-rank `0.5941 -> 0.4409`;
- proposal adapter moved substantially: residual RMS `0.2316 -> 1.0341`.

Yet the measured deployment mechanism did not move in a useful direction:

- validation BDMU Top-M utility capture anchor: `0.47680`;
- selected epoch 1: `0.47511` (`-0.17pp`);
- validation exact winner-flip Top-M recall: `0.23734 -> 0.23734`;
- proposal decisive-atom recall: `0.75366 -> 0.73941` (`-1.42pp`);
- teacher match: `0.178 -> 0.182` (`+0.4pp`, below gate);
- teacher regret: `20133.34 -> 20038.65` (`-0.47%`, below gate);
- exact B->Top-M evidence certificate: `0.928 -> 0.922`.

Therefore the original checker correctly refused to spend compute on full/test/CL. However, these values alone are not sufficient to say AF-BDMU is ineffective because the training target did not use the same hard Top-M interface as validation/deployment.

## Material engineering mismatch

AF-BDMU's V64.3.9 swap-ranking term is defined only for:

- a positive-utility atom **missed by the deployed Top-M**, and
- a lower-utility atom **occupying the deployed Top-M**.

The uploaded code violated that definition in the BDMU-only training path.

### Real runtime path

`BDSEModel.predict_certificate_numpy` used:

1. HAB Top-M;
2. structural-safety exclusion/refill when `decision_budget_excludes_structural_safety=true`;
3. group-aware soft-interaction reservation using agent group IDs.

### Old BDMU training path

`_fast_topm_mask_torch` used:

1. HAB Top-M;
2. soft-interaction reservation;
3. structural-safety exclusion/refill;
4. no agent-group-aware reservation.

The active V64.3.9 config has both:

- `decision_budget_excludes_structural_safety: true`;
- `min_soft_interaction_topm_slots: 2`.

Thus the mismatch is active. It changes the binary `deployment_topm` mask used to form AF-BDMU positive/negative swap pairs. The frozen-foundation reference B-set was also conditioned on the same fast Top-M surrogate, so the continuous utility target could be built on the wrong feasible proposal pool.

A deterministic regression fixture exposes the mismatch: with two high-scoring interaction atoms attached to the same agent and a third slightly lower-scoring atom attached to another agent, canonical runtime Top-M is `{1,3,5}`, whereas the historical fast training surrogate is `{1,3,4}`. The difference is exactly the group-diversity behavior intended by runtime.

## Why full/test/closed-loop are not worth running from V1

The causal chain intended by V64.3.9 is:

`teacher decisive-margin utility -> proposal score update -> deployed Top-M membership -> B=16 selector -> frozen DARM/DBR -> teacher decision -> closed-loop`.

The V1 engineering mismatch breaks the chain at the first intervention-to-interface edge:

`proposal score update -> [training surrogate Top-M != deployed Top-M] -> deployed Top-M`.

Consequently:

- a falling training loss does not establish that the model learned the actual deployed Top-M boundary;
- a flat validation Top-M metric cannot be cleanly attributed to the AF-BDMU utility formulation;
- full training would only scale an already semantically confounded intervention;
- test/closed-loop would measure the consequence of that confound rather than provide a clean algorithm comparison.

The correct next step is therefore a **clean Phase-1 rerun after exact runtime Top-M parity is restored**, not an algorithm redesign and not a downstream promotion.

## Engineering fix

This package makes no algorithmic change to AF-BDMU. It:

1. introduces one canonical `finalize_runtime_topm_policy(...)` helper;
2. uses that helper in learned runtime and rule fallback;
3. uses exact runtime HAB Top-M on every BDMU training scene;
4. conditions the frozen-foundation reference B-set on exact frozen runtime Top-M;
5. adds explicit config contracts for exact Top-M membership/reference pools;
6. adds a runtime Top-M parity preflight that is executed before GPU training;
7. writes the repaired screen/full runs to new `*_v2_runtime_parity` roots.

Only after the repaired Phase-1 screen can the existing mechanism/deployment gates be interpreted algorithmically.
