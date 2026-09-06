# V64.3.55 DMOR Result Postmortem and V64.3.56 RCPR Design

## Executive verdict

V55 is engineering/provenance valid and can be scientifically attributed. The exact uploaded code ZIP SHA256 is
`f99b678f1674add6f260af4bd68d8019e71430be67a026345aba5ec87aebdee7`, matching the preregistered V55 delivery. The uploaded result ZIP SHA256 is
`c211f6de0661688f72eee255678a6bef037c995db8c8a4e370359581177c65f6`.

The result-defining science manifest is 19/19 PASS, the server targeted suite is 77/77 PASS, an independent replay of the exact uploaded code is 77/77 PASS, all parent/evidence hashes match, and the sequential branch protocol was obeyed: PREDICTED-DOMINANCE was not fit or reported after REALIZED-DOMINANCE failed its deployment gate.

The preregistered scientific verdict is therefore:

- V52 QPE+D effect support: **GO / retain**.
- V54 realized one-replan endpoint mediator: **GO / retain as an identified diagnostic mediator**.
- V55 REALIZED-DOMINANCE: **Identification GO + Deployment STOP**.
- V55 PREDICTED-DOMINANCE: **NOT EVALUATED by preregistered branch order**; no scientific conclusion about mediator predictability is allowed.
- No V55 runtime retention policy is promoted.

The core result is not a near-miss threshold problem. V55's unweighted Pareto functional is genuinely more aligned with paired deployment outcomes than the earlier binary sign/static Pareto control, but the combination of **one-replan realized ego motion + static Pareto order** remains fold-unstable. The next and final eligible internal state family is therefore the realized interaction/safety constraint process.

---

## 1. Reliability audit

### Exact code and evidence provenance

- code ZIP SHA256: `f99b678f1674add6f260af4bd68d8019e71430be67a026345aba5ec87aebdee7`;
- result ZIP SHA256: `c211f6de0661688f72eee255678a6bef037c995db8c8a4e370359581177c65f6`;
- V55 fit SHA256: `cf7d91b9cf20d62978e766e6b8c739eee75e00011f5a208c2892af419e56dc88`;
- V50.5 paired outcome SHA256: `d592baa3508ae9dc084eebcbb3505d9accadc724601ab35a1253d3536af39d43`;
- V52 fit SHA256: `7a21fead5383ebd6aafaeb5da586346a77e5707050cb359511a06971b742a16b`;
- V53 planned operator profile SHA256: `9a69c196a1d76e9c5d424068df223ec26f0e481252f25f67d5bb17fd355aaef6`;
- V54 fit SHA256: `10f3e60c82bb8b82f1f688e866a27008e1498b67d2e194b0c7aadec5368536d8`;
- V54 dynamic response SHA256: `dd2bdd809a757ce74973d7ce2c3189fad60dc0d3e0125d7fcec9ca7ad1bda373`.

V55 did not recollect outcomes and did not consume untouched validation. Its fit used the exact metric-safe V50.5 502-scene paired outcomes plus frozen V53/V54 state artifacts.

### Sequential preregistration discipline

The V55 fit report contains:

`predicted_dominance = NOT_EVALUATED_BY_PREREGISTERED_BRANCH_ORDER`.

The source also returns from the oracle failure path before the predicted branch is fitted. Therefore the V55 result does not leak information from the t0-distillation branch into the next design decision.

Reliability verdict:

**PASS — valid for TRAIN scientific attribution.**

---

## 2. V55 preregistered branch verdicts

### 2.1 REALIZED-DOMINANCE — Identification GO

The V55 change relative to V54 is only the effectful conditional outcome functional. The state remains the exact V54 realized endpoint mediator and the V52 effect-support hurdle remains frozen. The binary sign ranker is replaced by an unweighted Pareto pairwise order over the existing paired closed-loop utility and hard-safety delta vector.

Identification:

- aggregate Pareto concordance: **0.5677562327**;
- exact V52 QPE+D static Pareto control: **0.4977285319**;
- folds above random: **4/5**;
- folds better than V52 static Pareto: **5/5**.

This satisfies the preregistered identification gate. The structured outcome order is real; V52's earlier Pareto failure was not evidence that all Pareto/partial-order formulations were useless. It failed when the state was QPE+D. Once V54 supplies a true realized mediator, the same conceptual outcome order becomes identifiable.

### 2.2 REALIZED-DOMINANCE — Deployment STOP

Aggregate paired results:

| metric | frozen RSMR | V55 realized dominance |
|---|---:|---:|
| selected | 502 | **447** |
| beneficial | 121 | **111** |
| beneficial recall | 100% | **91.7355%** |
| nonbeneficial | 381 | **336** |
| hard harm | 25 | **17** |
| score-delta sum | -4.2409 | **-2.2999** |
| NegRMS | 0.15486 | **0.14347** |

The mechanism improves hard harm, nonbeneficial count, aggregate utility, and negative RMS. Nevertheless:

- beneficial-retention gate: **FAIL**;
- hard-tail: PASS;
- nonbeneficial reduction: PASS;
- utility nonharm: PASS;
- all-fold nonharm: **FAIL**;
- population: PASS.

The frozen retention requirement is `1-alpha = 0.9220814479`. With 121 beneficial events, at least 112 must be retained. V55 retains 111. Numerically this is only one beneficial short, but it must **not** be interpreted as a threshold near-miss because the fold structure exposes a deeper failure.

### 2.3 Fold 2 is the decisive falsification

Fold 2:

- Pareto concordance: **0.464879**, below random;
- V52 static Pareto: 0.389527, so V55 still improves the old control;
- selected: 79 -> 78;
- beneficial: 15 -> 14;
- nonbeneficial: 64 -> 64;
- hard harm: 3 -> 3;
- score sum: **-0.01319 -> -0.80556**;
- NegRMS: 0.15714 -> 0.15815.

In this fold the policy essentially removes one beneficial event and no nonbeneficial events, causing a large utility reversal. No global threshold adjustment can explain away a fold in which the local outcome ordering itself points in the wrong direction.

Fold 3 also has structural instability: beneficial retention is only 22/28 = 78.57%, despite improvements in aggregate hard harm and negative RMS.

Therefore the proper scientific conclusion is:

**structured functional identifiability does not imply fold-stable deployment sufficiency when the state contains only realized ego motion.**

### 2.4 PREDICTED-DOMINANCE — not evaluated

Because REALIZED-DOMINANCE did not fully pass, V55 correctly did not fit or score the predicted-mediator branch. It is scientifically invalid to claim either that the mediator is predictable or that it is not predictable from V55.

---

## 3. Why the V55 phenomenon occurs

V54 proved that actual treatment-control ego displacement is a genuine mediator. V55 proves that adding a better outcome functional to that mediator still leaves a missing variable.

The key distinction is:

- ego mediator: **how did the ego vehicle actually move differently?**
- deployment consequence: **what did that motion do to agent interaction and physical constraint margins?**

Two scenes can have similar realized ego longitudinal displacement/speed change but very different consequences because:

- an agent may be close versus far;
- the relative closing velocity may differ;
- the ego motion may move toward versus away from an interaction envelope;
- a lateral displacement may remain in the route corridor in one scene but enter an off-route/drivable-risk regime in another.

Thus the same realized ego transition is not itself a sufficient statistic for safety-utility ordering. This explains the combination of positive aggregate concordance and regime-dependent fold reversals.

The important negative result is that the remaining failure cannot reasonably be assigned to another ego DCT mode, endpoint transform, wider MLP, or threshold. V53/V54 already isolated planned versus realized ego geometry, and V55 changed only the functional. The missing object is now directly pointed to by the outcome semantics: **realized operator-relative constraint consequence**.

---

## 4. Promotion / retention / closure after V55

### Retain / promote as supporting mechanisms

1. Frozen full-set RSMR ordinal selection.
2. V44 prospective interaction support.
3. V45 agent-local response as supporting layer.
4. V47 EGO-REF as supporting consequence coordinate.
5. V50.5 metric-safe paired selected-outcome supervision.
6. V52 QPE+D effect-support hurdle.
7. V54 realized one-replan endpoint as an identified diagnostic mediator.
8. **V55 unweighted Pareto outcome order as an identified structured functional** — but only as a supporting functional, not a deployable policy.

No V55 runtime arm is promoted.

### Newly closed family

Close, as deployment-sufficient solutions:

**static one-replan realized ego mediator + tested static sign/Pareto functional family.**

This closure includes attempts to rescue the same combination by:

- threshold/alpha changes;
- lambda or pairwise-loss tuning;
- class/focal/catastrophe weighting;
- safety scalarization/weights;
- bigger MLP/attention;
- more ego endpoint/DCT/horizon/basis/peak/early statistics;
- post-hoc union of V54 sign and V55 Pareto policies.

Do not close V54's mediator as a diagnostic finding, V55's Pareto order as a functional finding, V50.5 paired evidence, or V52 effect support.

All historical no-repeat constraints remain active.

---

## 5. Evidence chain through V55

The cumulative mechanism chain is now:

1. bounded evidence can be sufficient for ordinal extremal proposal selection;
2. selected-tail errors are real and cannot be fixed by generic value capacity;
3. prospective interaction support and agent-local response contain decision-relevant information;
4. better prediction does not imply better extremal deployment decisions;
5. identifiable future representation does not imply deployment sufficiency;
6. observational selected-risk identification does not imply fresh transportability;
7. intervention on the offline selection measure does not identify the deployed selected-outcome law;
8. deployment-aligned paired outcome evidence does not make QPE state sufficient;
9. effect-support sufficiency is distinct from conditional effect-order sufficiency;
10. planned operator geometry is distinct from realized treatment mediation;
11. realized treatment mediation is distinct from deployment outcome-order sufficiency;
12. **V55: structured outcome-functional identification is still not sufficient when the state omits realized interaction/constraint consequence.**

The paper headline remains stable:

**Selection–Valuation–Outcome Sufficiency under a Bounded Auditable Planner Interface.**

V55 deepens the Outcome Sufficiency claim rather than requiring a new paper title.

---

## 6. Dominant bottleneck after V55

Because V55 genuinely improved functional identification, the bottleneck can be narrowed from generic “structured outcome functional” to:

**operator-relative realized constraint-consequence process sufficiency for stable safety–utility outcome order.**

The model currently knows:

- which extremal proposal to consider;
- whether the proposal is likely to have any real effect;
- how the ego actually moves differently after one intervention;
- a meaningful partial order over paired utility/safety outcomes.

It still needs to learn:

**how that realized ego intervention changes the interaction and hard-constraint state relative to the incumbent/control.**

This is the last scientifically justified internal state family before convergence by falsification.

---

# V64.3.56 EAF-ICER-RCPR — Realized Constraint-Process Retention

## 7. Scientific role

V56 freezes V55's successful structured Pareto functional and changes only the mediator/state family.

The hypothesis is not “more safety features are better.” It is:

> For an effectful frozen extremal intervention, stable selected-outcome order requires an **intervention-relative realized constraint process** rather than ego motion alone.

This is tested using the same short one-shot treatment/control exposure window introduced by V54. Final outcome labels remain the exact V50.5 paired outcomes; no final metric is recollected.

## 8. Fixed realized constraint process

At every current simulated state from t=0 through the first scheduled replan, record three lower-is-safer continuous constraint coordinates using already frozen runtime geometry:

1. **agent occupancy risk** — ungated soft interaction potential using the existing box/radius safety geometry and closing-speed buffer;
2. **agent TTC risk** — radial constant-velocity TTC risk gated by the frozen soft interaction envelope and `agent_ttc_safe_s`;
3. **hard off-route excess** — continuous excess beyond the frozen route-corridor width plus hard off-route margin.

No label future, teacher outcome, final nuPlan metric, new evidence query, or tuned safety coefficient enters these channels.

For post-intervention ticks 1..5, form:

`constraint_delta(t) = risk_control(t) - risk_treatment(t)`.

Thus every coordinate has the same semantics: larger is better for treatment. The 15-dimensional process is retained raw; there is no attention, pooling, DCT basis, peak statistic, or safety scalarization.

The collector hard-checks the V53 structural identity signature: all **38** planned-physical-equality interventions must have a near-zero realized constraint process (`L_inf <= 1e-6`).

## 9. Arm A — REALIZED-CONSTRAINT-PROCESS

Outcome state:

`QPE+D + exact V54 realized endpoint + 15-D paired realized constraint process`.

The outcome functional is **exactly V55**:

- frozen V52 effect-support hurdle;
- same unweighted Pareto pairwise outcome ranker;
- same lambda=1;
- same conformal alpha;
- same paired deployment gate;
- same full-set frozen RSMR winner and no-fallback operator.

Only the state changes.

Identification must satisfy:

- aggregate Pareto concordance > 0.5;
- >=4/5 folds > random;
- aggregate concordance > exact V55 REALIZED-DOMINANCE (`0.5677562327`);
- >=4/5 folds > exact V55 control;
- then the unchanged deployment gate must PASS.

This arm is diagnostic/oracle because post-intervention process state is not available at t0.

## 10. Arm B — PREDICTED-CONSTRAINT-PROCESS

This branch is eligible **only if Arm A fully passes both identification and deployment**. The source is genuinely sequential fail-closed: if the oracle fails, this branch is not fitted or scored.

Runtime state uses:

- cross-fitted predicted V54 realized endpoint;
- cross-fitted predicted V56 constraint process.

The endpoint predictor remains the exact V55 zero-bias, zero-preserving, lambda=1 ridge.

The new process predictor is also zero-bias, zero-preserving, lambda=1 multi-output ridge. Its fixed t0 input is:

- V53 four signed planned endpoint channels;
- V53 eight fixed DCT-II k=1,2 planned channels;
- `D * t0_constraint_risk` for the three current constraint coordinates.

Dose-gating enforces the identity contract:

`planned D = 0 => predictor input = 0 => predicted realized process = 0`.

No hidden post-intervention state enters runtime.

Nested nuisance isolation:

- outer test and calibration folds are excluded from predictor fitting;
- the outer-fit outcome ranker sees inner-OOF predicted mediators/processes;
- calibration/test see predictors trained only on the outer fit folds.

Both endpoint and process predictors must beat their zero-response baselines in normalized MSE aggregate and >=4/5 folds, after which the same outcome identification/deployment gates apply.

## 11. Strict internal convergence protocol

V56 is the **final internal state-family experiment**, not the beginning of another sweep.

### If PREDICTED-CONSTRAINT-PROCESS passes

Freeze immediately. No more TRAIN tuning. Perform only:

1. engineering-only runtime integration;
2. untouched paired validation;
3. if untouched passes, external baselines and official benchmarking.

### If REALIZED-CONSTRAINT-PROCESS fails

Internal algorithm search converges **by falsification**. Do not create V57/V58 feature/state variants. The evidence then says that even the final physically aligned short-horizon realized constraint process does not close the static deployment operator. Proceed to external baselines/official benchmarking with the strongest frozen deployable backbone and present the limitation as the outcome-sufficiency boundary.

### If oracle passes but predicted branch fails

Internal state-family search also stops. The physical mechanism exists, but the one predeclared t0 bridge is insufficient. Do not increase nuisance capacity, add horizons/bases, or tune outcome gates. Treat deployability of the mediator as a limitation/future-work boundary unless a separate paper revision is justified.

### If predicted state is identified but deployment still fails

Stop internal search. The problem is no longer state representation; further TRAIN tuning would violate the convergence discipline.

---

## 12. Compute policy

V56 needs a new paired short-horizon state replay because V54 did not log agent/TTC/route constraint channels. It does **not** need new final outcomes or final metrics.

Use the same 502 proposal population; do not reduce sample size because V55's dominant failure is fold stability.

Acceleration remains:

- only the exact 5-tick one-shot exposure window;
- `run_metric=false`;
- reuse V50.5 final outcomes;
- treatment/control parallel on two GPUs when available;
- one-GPU sequential fallback;
- 8-scene first batch sentinel;
- collision-safe resumable batches.

---

## 13. External-baseline readiness

The remaining internal gap is now explicit: **one t0-deployable outcome-retention arm must pass the unchanged nested identification + paired deployment gate.** The gap is not generic planning performance or model capacity.

V56 is the final attempt to close that gap internally. External baseline scripts can be prepared now, but baseline results must not be used to tune V56. After V56's convergence decision, benchmarking should proceed rather than extending the internal version series indefinitely.
