# V64.3.52 result postmortem and V64.3.53 POTR design

## 1. Reliability verdict

V64.3.52 is engineering/provenance valid and may be scientifically attributed.

The uploaded code ZIP SHA256 is `c75ffd833a62c57d2ea3132aeec90950e4307c2142129285e62d9222bb45f9e2`, exactly the preregistered V52 delivery. The uploaded result ZIP SHA256 is `387397d1d60b08f101259000ef54d44b464a25fae69c88e358aa64614feb8b9e`. The 10-file curated science manifest passes. The V51 parent fit is exact (`54d366...23049`). The metric-safe paired evidence is exact 502/502 with SHA `d592baa3...39d43`, 121 beneficial and 25 hard-harm events. Server targeted regression is 57/57 PASS and the exact uploaded code independently replays 57/57. V52 terminates at the preregistered scientific STOP and untouched validation is unconsumed.

Therefore no V52 engineering repair is warranted.

## 2. V52 preregistered branches

### Shared effect-support hurdle: GO as a supporting mechanism

The effect-support risk reaches AUC `0.6516244589` with all 5/5 folds above random. This is the strongest stable mechanism in V52. It independently confirms the V51 scalar-D diagnostic after explicit cross-fitted factorization: QPE+D can distinguish structural null from interventions that actually change paired closed-loop outcome.

This mechanism is retained, but it is not by itself a deployment policy.

### HURDLE-SIGN: STOP

Conditional on an effect, AUC is `0.4802924348`, below the exact V51 conditional control `0.4964580874`; only 2/5 folds improve. Thus the conditional sign functional is not identified.

OOF deployment nevertheless produces a useful diagnostic rotation: hard harm drops 25->22, nonbeneficial 381->336, score sum improves -4.2409->-3.2021 and NegRMS improves .15486->.14609. But it retains only 110/121 beneficial events (90.91%), below the frozen conformal recall requirement, and fails fold-wise nonharm. These aggregate gains cannot override the preregistered identification/recall gates.

Interpretation: the support hurdle removes genuinely inactive interventions, but the conditional ranker still cannot distinguish valuable from harmful effectful interventions.

### HURDLE-PARETO: STOP

The unweighted Pareto concordance is `0.4977285319`; only 3/5 folds beat the V51 control. It is therefore not identified. Deployment is also worse: 111/121 beneficial retained, 24 hard harms, score sum `-5.3254`, utility and fold-wise nonharm failures.

This falsifies the specific hypothesis that changing only the order relation on the same scalar QPE+D state is sufficient. It does not justify tuning safety weights; the experiment was intentionally weight-free, and the state is now shown to be the more immediate missing factor.

## 3. Key V52 mechanism conclusion

V52 decomposes selected-outcome sufficiency into two distinct questions:

1. **Effect support:** does executing the selected proposal instead of incumbent change the closed-loop outcome at all?
2. **Conditional outcome order:** if it changes the outcome, is the change beneficial, harmful, or safety dominated?

The same scalar state is sufficient for (1) and insufficient for (2). Hence:

`effect-support sufficiency != conditional-outcome-order sufficiency`.

This also refines the V51 conclusion. V51's operator-relative state signal was real, but a substantial part of it was effect-support structure. Once nulls are removed, scalar D erases directional and temporal treatment geometry that can separate opposite outcome directions at similar intervention magnitude.

## 4. Evidence chain through V52

The cross-version chain is now:

- V34: ordinal extremal selection becomes the mature learned selector.
- V40: pure generic value-head expansion is falsified.
- V41-V45: endpoint/current/prospective interaction mediators are progressively identified; V44 ungated prospective support and V45 agent-local response are retained.
- V46: better ordinary prediction can worsen extremal deployment decisions: prediction sufficiency != decision sufficiency.
- V47: identifiable future-state features need not be deployment sufficient; representation expansion is stopped.
- V48: in-domain selected-risk signal need not transport to fresh selection regimes.
- V49: changing offline selection measure does not identify the deployed selected-outcome law.
- V50: deployment-aligned paired outcome evidence does not make QPE-only state sufficient.
- V51: scalar operator contrast is an identified relational state, but sign-only retention remains insufficient.
- **V52: scalar relational state identifies effect support but not conditional effect direction/order.**

The paper line therefore remains stable as **Selection-Valuation-Outcome Sufficiency under a Bounded Auditable Planner Interface**, with a sharper Outcome Sufficiency hierarchy.

## 5. Dominant bottleneck

The dominant bottleneck is now:

**effectful selected-outcome state sufficiency / directional-temporal operator-contrast sufficiency.**

The model has learned whether the frozen operator is likely to matter, but not what signed treatment it applies relative to incumbent when it matters.

## 6. What is learned and what remains missing

Retain/freeze:

- bounded EAF interface, exact attribution, support/admissibility;
- full-set frozen RSMR ordinal selector;
- incumbent/no-fallback containment;
- V44 ungated prospective interaction support;
- V45 agent-local longitudinal response;
- V47 EGO-REF supporting consequence coordinate;
- V50 metric-safe paired one-shot outcome evidence;
- V51/V52 QPE+D **effect-support** factorization.

Still missing:

- stable ordering of beneficial vs harmful **effectful** interventions;
- signed proposal-vs-incumbent treatment channels rather than only `||contrast||_inf`;
- if endpoint sign is insufficient, minimal temporal shape of the treatment itself.

## 7. Newly closed directions

Close static conditional sign and static unweighted Pareto ordering **on QPE+D** as sufficient deployment solutions. Do not rescue them by threshold, lambda, weighting, bigger MLP, safety scalarization, or post-hoc policy union.

Do not close all structured/Pareto functionals globally: V52 simultaneously falsifies the conditional state, so more expressive functionals on the same state cannot be cleanly attributed.

All prior closures remain active.

## 8. V64.3.53 POTR

V53 changes one scientific object only: the conditional-outcome state.

The V52 QPE+D effect-support model is frozen/replayed exactly. The paired labels remain the exact V50.5 metric-safe 502/502 outcomes. Runtime remains frozen RSMR winner or incumbent only.

### State-only acquisition

Old artifacts contain scalar D but not the full runtime incumbent trajectory. V53 therefore runs a treatment-only, label-free state replay at the exact V50 intervention anchor. A V53-specific process-local nuPlan entrypoint wraps the historical byte-identical planner and writes a sidecar before treatment execution. It records the cached frozen proposal vs the actual runtime incumbent trajectory. It never changes the action and never reads outcome labels.

This avoids touching historical science-critical `nuplan_planner.py` and avoids recollecting the 502x2 outcome experiment.

### ENDPOINT arm

State:

`[Q, P-Q, E-P, D, dx_T, dy_T, wrap(dyaw_T), dv_T]`.

This is the minimum signed/channel factorization. If it passes, direction rather than temporal shape was the missing statistic and the simpler arm is promoted.

### TEMPORAL arm

If ENDPOINT does not yield a complete promotion result, add fixed orthonormal DCT-II modes k=1,2 for each signed channel `dx(t),dy(t),dyaw(t),dv(t)`, adding exactly eight coordinates.

There is no basis/horizon/mode sweep, no attention, no peak/early statistic. This is a minimal falsification of whether treatment temporal shape is needed.

### Why this is not V46 temporal-profile resurrection

V46's closed direction used handcrafted temporal statistics of predicted agent interaction/occupancy consequence. V53 instead represents the **pre-execution operator treatment**: proposal trajectory minus incumbent trajectory, learned against real paired selected outcomes. The statistical object, evidence source and causal question are different.

### Identification and promotion

ENDPOINT must beat random, exact V52 scalar HURDLE-SIGN conditional risk, and exact V51 conditional control in aggregate and >=4/5 folds, then pass the unchanged deployment gate.

TEMPORAL must satisfy those gates and additionally beat ENDPOINT in aggregate and >=4/5 folds to support temporal necessity.

Promotion order is ENDPOINT then TEMPORAL.

If state identifies but deployment fails, the state bottleneck is considered solved and the next version may revisit a richer paired functional without more state expansion. If both state arms fail identification, pre-execution operator trajectory geometry is closed as sufficient and the next evidence must be a post-intervention paired dynamic response/outcome process.
