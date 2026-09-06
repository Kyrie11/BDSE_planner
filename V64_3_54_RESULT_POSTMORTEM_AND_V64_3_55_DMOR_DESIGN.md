# V64.3.54 Result Postmortem and V64.3.55 DMOR Design

## 1. Reliability gate

V64.3.54.1 is valid for TRAIN-level scientific attribution.

- uploaded repaired code ZIP SHA256: `b841cd90ce8cb93f58404be1ecb98df6a99fc12b05e68578fe3a14efc5f8fe14`;
- uploaded result ZIP SHA256: `76c6b81b8bf007222ee08ce598aaac195d62eff3453815ade1d999bbc0c6b4f7`;
- result-defining science manifest: 20/20 PASS;
- server targeted regression: 74/74 PASS;
- V50.5 paired-outcome SHA: exact `d592baa3508ae9dc084eebcbb3505d9accadc724601ab35a1253d3536af39d43`;
- V53 parent fit SHA: exact `9174ffeac064a85bef6c1727915d93903271f9afe1770f5e5ba3e3e51efe1b6e`;
- treatment/control short-horizon replay: 502/502 each, zero failures, one probe/event/scene;
- raw sidecars: 3012 events per arm = 502 scenes x six synchronized samples (iterations 0..5);
- treatment/control scenario token sets, iteration indices and timestamps are exactly paired;
- all 502 initial paired states agree; all 38 planned-physical-equality scenes produce zero realized response;
- outcome labels were not recollected and nuPlan final metrics were not recomputed;
- untouched validation was not consumed.

Therefore V54 is a preregistered scientific result, not an engineering/data STOP.

## 2. Preregistered V54 verdict

The branch order was `REALIZED-ENDPOINT -> REALIZED-TEMPORAL`.

### Frozen V52 effect-support hurdle: GO / retain

The already-frozen QPE+D effect-support AUC remains `0.6516244589`, 5/5 folds above random. V54 does not modify this mechanism.

### REALIZED-ENDPOINT: mediator GO, deployment policy STOP

Effectful conditional outcome AUC:

`0.6117518845`.

Preregistered comparisons:

- > random: 5/5 folds;
- > exact V51 scalar control: 5/5;
- > exact V52 scalar control: 5/5;
- > exact V53 pre-execution temporal state: 4/5.

Thus the realized one-replan endpoint state is independently identified as an outcome mediator.

The retrospective sign-retention diagnostic is nevertheless not a deployable promotion:

- selected: 446/502;
- beneficial retained: 113/121 = 93.39%;
- nonbeneficial: 381 -> 333;
- hard harm: 25 -> 17;
- closed-loop score sum: -4.2409 -> -2.2731;
- negative RMS: 0.15486 -> 0.14363;
- historical paired gate: FAIL only on all-fold nonharm.

Fold 2 is the decisive counterexample: conditional AUC is still >0.5 (`0.54074`), yet selected score sum moves from roughly `-0.0132` to `-0.8056` with no hard-harm reduction. This directly falsifies the idea that better binary sign ordering is enough for deployment ordering.

### REALIZED-TEMPORAL: STOP as temporal necessity

Aggregate AUC is `0.6090727454`, with 5/5 folds above random and 4/5 beating V53 pre-execution TEMPORAL. But it beats REALIZED-ENDPOINT in only 2/5 folds and is slightly worse in aggregate (`0.60907 < 0.61175`). Therefore the preregistered temporal-necessity claim fails.

The same retrospective deployment gate also fails all-fold nonharm. No additional DCT mode, horizon, early/peak statistic, attention or basis sweep is permitted.

## 3. Mechanism interpretation

V52 showed that scalar operator dose identifies whether an intervention has any effect but not its direction. V53 showed that planned signed endpoint and planned low-order temporal shape do not identify effectful outcome order. V54 now shows that the *realized* treatment-control transition does.

The most important distinction is:

`planned operator geometry != realized treatment mediation`.

A consumed-TRAIN design diagnostic further shows that realized response magnitude alone is not the mechanism: planned D and realized L-infinity are near-random for good/bad ordering, whereas the signed realized longitudinal displacement/speed channels carry much stronger directionality. In other words, the missing object is not “how big the intervention is” but “how the executed treatment actually changes ego state.”

The failure of the retrospective sign gate adds a second conclusion:

`realized mediator identification != deployment outcome-order sufficiency`.

The binary beneficial/nonbeneficial ranker cannot preserve negative severity and hard-tail order. V54 fold 2 is the cleanest counterexample.

## 4. Evidence-chain update

The cumulative chain is now:

- ordinal extremal selection can be solved independently from absolute execution valuation;
- predictive accuracy is not decision sufficiency;
- future-state identifiability is not deployment sufficiency;
- observational post-selection risk is not fresh transport sufficiency;
- changing the offline selection measure is not deployed outcome identification;
- deployment-aligned paired outcome evidence is not QPE state sufficiency;
- effect-support sufficiency is not conditional effect-order sufficiency;
- planned treatment geometry is not realized outcome mediation sufficiency;
- V54: realized one-replan mediation is identifiable, but binary sign ordering still is not deployment-order sufficient.

This supports the stable paper line **Selection–Valuation–Outcome Sufficiency under a Bounded Auditable Planner Interface**. The algorithmic novelty should be expressed as identifying distinct sufficient statistical objects for selection, effect support, realized mediation and constrained outcome ordering, all under a frozen monotone no-fallback operator.

## 5. Dominant bottleneck after V54

The dominant bottleneck becomes:

**operator-aligned structured outcome order over an identified realized mediator, followed by a t0 mediator bridge.**

Do not add more ego state before answering the functional question. A perfect post-intervention mediator already fails the old sign functional on the all-fold deployment criterion; predicting that mediator first would only reproduce a known functional failure.

## 6. Newly retained mechanisms

Retain/freeze:

- EAF bounded auditable interface;
- support/admissibility and full-set RSMR winner;
- same-winner-or-incumbent / no-fallback containment;
- V44 prospective support and V45 response supporting layers;
- V47 EGO-REF supporting consequence coordinate;
- V50.5 metric-safe paired one-shot outcome evidence;
- V52 structural-null/effect-support hurdle;
- V54 **realized one-replan endpoint as an identified mediator**.

Do not promote V54 itself as a t0 runtime policy.

## 7. Newly closed directions

Close as unnecessary/insufficient:

- realized temporal DCT k=1,2 as a necessary mediator beyond realized endpoint;
- any V54 horizon/basis/mode/peak/early sweep;
- rescue of the V54 realized mediator with the same binary sign-only conditional ranker;
- direct use of post-intervention mediator as if available at t0.

All historical closures remain active: no RSMR/B/M/top-K/candidate-count changes, no rerank/second-best/fallback, no K/logK, no random-prefix SIIR, no CVaR, no selected translation, no class/focal/catastrophe weighting, no safety scalarization, no large MLP/attention, no new offline future-observable expansion.

## 8. V64.3.55 DMOR

**Dynamic-Mediator Outcome Retention** is a fit-only causal decomposition with no new simulation.

### Arm A: REALIZED-DOMINANCE

Freeze the V54 realized endpoint state:

`[Q, P-Q, E-P, D, realized_dx_end, realized_dy_end, realized_dyaw_end, realized_dv_end]`.

Change only the conditional outcome functional. On effectful TRAIN events, form pairwise constraints only when one paired outcome is Pareto-worse than another over the already-existing vector:

`[closed_loop_score_delta, hard-safety delta 1, ...]`,

with every coordinate “larger is better.” Ambiguous trade-off pairs are omitted; no safety weight is introduced.

Identification requires held-out Pareto concordance >0.5 and better than the exact V52 static QPE+D Pareto control in aggregate and >=4/5 folds. It must also pass the unchanged paired deployment gate.

This arm is diagnostic/oracle only because it consumes a post-intervention mediator.

### Arm B: PREDICTED-DOMINANCE

This arm is scientifically eligible only if REALIZED-DOMINANCE fully passes. The fit code is sequential fail-closed: if Arm A fails either identification or the unchanged deployment gate, Arm B is not fit, scored, or emitted at all.

The mediator target stays exactly the V54 realized endpoint. The input is only the already-frozen V53 pre-execution signed operator profile: four planned terminal channels plus the fixed DCT-II k=1,2 channels. A zero-bias multi-output ridge with lambda=1 is fit after RMS scaling without centering, so physical equality maps exactly to zero predicted response.

For every outer fold, mediator prediction excludes both the outer test and calibration folds. Outcome-ranker training uses inner cross-fitted mediator predictions so no fit-row realized mediator leaks into its state. Calibration/test use a predictor trained only on the outer fitting folds.

Mediator identification requires standardized prediction MSE below a zero-response baseline in aggregate and >=4/5 folds. The *same* Pareto functional and unchanged deployment gate then apply to QPE+D plus the predicted mediator.

If this arm passes, it is a t0-available TRAIN mechanism and triggers immediate freeze.

## 9. Convergence rule

Internal algorithm convergence is reached when a t0-deployable arm passes both nested identification and the unchanged paired deployment gate. At that point:

1. freeze all model/state/threshold definitions immediately;
2. perform only engineering runtime integration;
3. run untouched paired validation without TRAIN tuning;
4. if untouched validation independently passes, stop internal algorithm search and move to external baselines + official closed-loop benchmarking.

If REALIZED-DOMINANCE fails, close the **combination** of the static one-replan ego mediator with the already-tested static sign/Pareto functionals as deployment-sufficient; retain the V54 endpoint only as an identified diagnostic mediator. The next and final eligible internal state family is a realized interaction/safety consequence process. This prevents indefinite ego-feature iteration.

## 10. Compute policy

V55 is fit-only. It reuses V50.5 paired outcomes, V53 pre-execution operator profiles and V54 short-horizon mediator profiles. No GPU simulation, checkpoint or nuPlan metric computation is required. This is the fastest scientifically valid next experiment.
