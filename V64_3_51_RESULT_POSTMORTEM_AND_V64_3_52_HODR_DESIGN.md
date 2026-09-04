# V64.3.51 POCR result postmortem and V64.3.52 HODR design

## 1. Reliability verdict

**PASS for TRAIN-level scientific attribution.**

The first V51 run failed the full-package SHA check only on two archival, non-runtime files: `V64_3_50_7_TO_V64_3_51_POCR.patch` and `V64_3_50_PAIRED_OPERATOR_CONTRAST_DIAGNOSTIC.csv`. The first-attempt manifest log reports the V51 fit source, tests, launcher and all other executable science sources as `OK`. The subsequently uploaded launcher differs from the preregistered launcher only by disabling the full-tree manifest fail-fast and the paired-evidence hash assert (plus one comment-only duplicate line). The actual reused paired evidence SHA reported by the successful run is still the exact preregistered value `d592baa3...39d43`, and the V50.7 parent fit hash is exact.

Therefore the SHA incident is a **provenance-hygiene defect, not a result-defining science-source ambiguity**. V52 repairs it by replacing the fragile full-tree manifest with a curated result-defining science manifest. Documentation, patches, diagnostics and changelog are excluded from the hard source gate by design.

Uploaded V51 fit report SHA256: `54d3664e378c85ba482485c49ddcb3e10e83a59d5d04cd049ccaf05fdbb23049`.

## 2. Preregistered branch verdicts

| Branch | Identification | Deployment | Overall |
|---|---:|---:|---:|
| V50 QPE control | FAIL (0.508709; 3/5 vs EGO, 3/5 vs OBS) | FAIL | STOP |
| **QPE+DOSE** | **GO** (AUC 0.576712; 5/5 > QPE, 4/5 > EGO, 4/5 > OBS) | **STOP** | **STOP / no promotion** |
| **QPE+DOSE-X** | **GO** (AUC 0.548491; 4/5 > QPE, 5/5 > EGO, 4/5 > OBS) | **STOP** | **STOP / no promotion** |
| scalar D effect support | **GO** (AUC 0.697547, 5/5 > random) | diagnostic only | retain as identified state evidence |
| preregistered `state identified but deployment fail` branch | **GO** | n/a | **V52 structured paired outcome functional** |

The V51 termination string is exactly `operator_contrast_state_identified_but_low_capacity_sign_retention_functional_insufficient`.

## 3. Aggregate paired-outcome metrics

| Policy | selected | beneficial | nonbeneficial | hard harm | score-sum | NegRMS |
|---|---:|---:|---:|---:|---:|---:|
| Frozen RSMR | 502 | 121 | 381 | 25 | -4.240928 | 0.154855 |
| V50 QPE | 460 | 115 | 345 | 24 | -5.426207 | 0.156268 |
| **QPE+DOSE** | 465 | 115 | 350 | 24 | -4.729640 | 0.155829 |
| **QPE+DOSE-X** | 436 | 113 | 323 | 23 | -4.143927 | 0.157927 |

`QPE+DOSE` passes beneficial retention, nonbeneficial reduction and population, but fails hard-tail, utility-nonharm and all-fold nonharm. `QPE+DOSE-X` passes beneficial retention, nonbeneficial reduction, population and aggregate utility-nonharm, but still fails hard-tail and all-fold nonharm.

This is the critical causal pattern: **the state is now identifiable, yet the binary sign functional is not aligned with the deployment tail/utility constraints.**

## 4. Why V51 is not a state failure

`QPE+DOSE` improves selected-outcome AUC from the V50 QPE control `0.508709` to `0.576712` and beats the QPE control in all five outer folds. This is strong evidence that explicit proposal-vs-incumbent execution contrast carries real selected-outcome information.

The interaction arm is not required as the next state: it is more complex, has lower aggregate identification AUC than the additive state, and still fails deployment. V52 therefore freezes the **minimal identified state** `[Q, P-Q, E-P, D]` and does not carry `D*QPE` interactions forward.

## 5. Why V51 is a functional failure

The sign-only target compresses all paired outcomes into `beneficial` versus `nonbeneficial`. That discards at least two structures already visible in the paired evidence:

1. **Structural null / effect support.** The scalar execution contrast D has effect-support AUC `0.697547` and all 38 physically equal proposal/incumbent pairs are null-effect.
2. **Deployment partial order.** The deployment gate separately constrains official score, hard safety and negative tail. V51's binary sign loss cannot distinguish a harmless zero-effect event, a mild negative event and a hard-safety-degrading event once all are labeled nonbeneficial.

The strongest empirical symptom is QPE+DOSE-X: aggregate score-sum improves slightly over frozen RSMR (`-4.143927` vs `-4.240928`) and hard-harm count drops from 25 to 23, yet NegRMS worsens (`0.154855` -> `0.157927`) and fold nonharm fails. Better binary ranking is not enough to control the severity/order of retained negatives.

## 6. Updated dominant bottleneck

After V50 the bottleneck was operator-relative selected-outcome state sufficiency. V51 resolves that layer enough to move on.

The new bottleneck is:

**structured paired selected-outcome functional sufficiency**, specifically the need to represent (a) whether the intervention has any causal/effect support and (b) the safety/utility partial order among effectful outcomes without collapsing them to a single sign.

This is a genuine bottleneck migration, not another feature refinement.

## 7. Mechanisms promoted / retained / closed

### Retain as paper/backbone evidence
- frozen full-set RSMR extremal selection and no-fallback containment;
- V44 ungated prospective interaction support;
- V45 agent-local longitudinal response;
- V47 EGO-REF supporting consequence coordinate;
- V50 metric-safe paired one-shot selected-outcome supervision;
- **V51 minimal additive operator-relative state `[Q,P-Q,E-P,D]` as an identified supporting state.**

### Do not promote as deployment mechanism
- QPE+DOSE retention policy;
- QPE+DOSE-X retention policy.

### Newly close as deployment-sufficient solutions
- **single sign-only paired retention over the V51 operator-relative state**;
- `D*QPE` interaction state as a necessary next-step mechanism (it is not needed for the next causal test).

This does **not** close D itself or paired outcome supervision.

## 8. V52 algorithm: HODR

V64.3.52 is **Hurdle Outcome-Dominance Retention (HODR)**.

It freezes the state to `[Q,P-Q,E-P,D]` and changes only the deployment functional.

### Arm A — HURDLE-SIGN

Factor the selected outcome into:

`effect support` -> `conditional beneficial/nonbeneficial sign`.

Two zero-bias pairwise rankers (fixed lambda=1) share the same state. Runtime risk is the fixed maximum of the two normalized risks. There is still exactly one split-conformal threshold calibrated on beneficial outcomes with the same alpha; there is no alpha split and no threshold sweep.

This tests whether the large structural-null population was corrupting the old global sign ranker.

### Arm B — HURDLE-PARETO

Keep the identical effect-support hurdle. Replace the conditional sign pair construction with a **Pareto deployment order** over the paired official-score delta and every hard-safety delta.

A bad event is paired against a good event only when it is no better on every deployment coordinate and strictly worse on at least one. Ambiguous utility/safety trade-offs are omitted rather than scalarized.

Therefore V52 adds:
- no safety weights;
- no catastrophe class weight;
- no standalone binary safety veto;
- no new runtime feature;
- no MLP/attention;
- no second threshold.

The scientific test is whether **constraint-aligned outcome ordering**, rather than another state block, is what the frozen extremal operator actually needs.

## 9. Preregistered promotion / STOP logic

1. HURDLE-SIGN must identify effect support and improve conditional effectful sign ranking over exact V51 control, then pass the unchanged V50/V51 deployment gate.
2. Only if it fails is HURDLE-PARETO considered.
3. HURDLE-PARETO must improve held-out Pareto-pair concordance over the exact V51 sign-risk control and pass the same deployment gate.
4. If either passes, freeze immediately and move to untouched paired validation; no TRAIN tuning.
5. If the structured functional is identifiable but deployment still fails, the static state+functional family is exhausted and the next evidence must be a paired **temporal closed-loop outcome process**, not an offline feature block.

## 10. CCF-A paper line

The paper-level claim should remain centered on **Selection–Valuation–Outcome Sufficiency under a Bounded Auditable Planner Interface**, with the new refinement:

> outcome sufficiency is both **intervention-relative** and **operator-functional**: even when the paired outcome source and operator-relative state are identifiable, collapsing a zero-inflated, multi-constraint potential outcome into one binary sign can remain deployment-insufficient.

V52 is designed as a falsifiable causal experiment for that statement, not as a hyperparameter search.
