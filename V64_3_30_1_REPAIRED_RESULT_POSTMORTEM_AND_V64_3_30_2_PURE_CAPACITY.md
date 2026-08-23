# V64.3.30.1 Repaired Result Postmortem and V64.3.30.2 Pure-FBIC Design

## Executive conclusion

The repaired V64.3.30.1 run is **engineering-valid through the paired B16/B24 TRAIN capacity experiment and the B24 DRC TRAIN fit**. The previous B24 attribution-spectrum crash is fixed: both capacity arms complete all 3000 frozen TRAIN scenes, paired identity/query accounting is exact, and the FBIC capacity-only contract holds. The run then stops **by design** because the unchanged B24 aggregate-downside DRC recipe is only 4/5 fold-safe. That stop is an algorithmic/development gate, not a runtime error.

However, this run is **scientifically incomplete for the central V30 mediator question**. The launcher places the B24 DRC TRAIN gate before fresh selection, so the valid DRC failure also prevents the pure `raw / B16-V20 / B24-V20` untouched A/B capacity control from running. Therefore V30.1 can support TRAIN-side development attribution, but it cannot yet establish on independent data whether B=16 retained capacity is a first-order missing mediator.

V64.3.30.2 corrects only this experimental orchestration: it preserves the failed B24 DRC gate and never evaluates B24 DRC on fresh data, while allowing the pure capacity control to spend the still-untouched original V30 seed.

## 1. Engineering validity of the repaired V30.1 run

The repaired run satisfies all capacity-isolation contracts:

- B16 and B24 arms contain the exact same 3000 frozen TRAIN scenes.
- Upstream configured evidence/selector budget remains B=16.
- The queried decision bank remains M=24.
- FBIC is post-selector only: the safe-domain final retained interface is a strict superset of the B16 set, with no removed baseline atom.
- Safe-domain mean retained count grows from 15.9496 to 23.7560; mean added atoms = 7.8064.
- All-flagged structural scenes = 262; FBIC application rate there = 0.
- Per-scene `action_atom_query_count`, `proposal_candidate_atom_count`, and `effective_query_action_count` are exactly identical between B16 and B24.
- `upstream_configured_decision_budget_atom_count=16`, retained-interface ceiling=24, retained-budget pass rate=1, and no-new-query rate=1.
- Historical B16 V25 fit is exactly reproduced: 75,133 frontier rows / 1,455 eligible edges / 310 replacement scenes / 5-of-5 fold-safe / 71 selected / +5.527642 teacher-improvement sum.

No second runtime/shape/accounting blocker is present in the uploaded repaired execution.

## 2. What stopped the run

The B24 DRC fit is a real frozen-TRAIN algorithmic failure:

- eligible edges: 1,455 -> 3,821;
- replacement scenes: 310 -> 483;
- teacher-positive edge fraction: 62.13% -> 40.23%;
- eligible-population teacher-improvement sum: -52.589 -> -590.668;
- aggregate-downside selected count: 71 -> 84;
- aggregate-downside fold safety: 5/5 -> 4/5.

The failing fold is fold 2: 22 selected proposals, teacher-improvement sum -0.041637, worst -1.051350, NegRMS 0.306225. The global selected sum remains positive (+9.308571), but the pre-registered contract requires every fixed scene fold to be selected-path non-harmful. The launcher therefore correctly stops and must not lower K, the DRC threshold, the downside multiplier, or any support/scalar boundary.

This is valuable development evidence: opening the retained interface substantially changes the eligible replacement distribution, and the frozen edge-local aggregate consumer no longer satisfies the same cross-fold safety contract.

## 3. What TRAIN says about the capacity-vs-consumer question

The evidence-transmission side clearly improves under B24:

| Diagnostic | B16 | B24 |
|---|---:|---:|
| selected decisive-atom recall | 0.5305 | 0.7158 |
| effective selected decisive-atom recall | 0.7580 | 0.9433 |
| selected interaction-decisive recall | 0.4823 | 0.6904 |
| selected soft-interaction decisive recall | 0.6332 | 0.9068 |
| evidence-certificate fraction | 0.8753 | 0.9823 |
| budget-vs-full action match | 0.2137 | 0.2453 |

The same TRAIN population also shows more downstream preconditions becoming positive:

- B16 positive direct opportunities: `803 -> support 552 -> scalar 248 -> support+scalar 229 -> selected-positive 206` (25.65% capture).
- B24 positive direct opportunities: `787 -> support 551 -> scalar 327 -> support+scalar 305 -> selected-positive 141` (17.92% capture).

Thus B24 does not fail because it simply “sees nothing new.” It transmits substantially more decisive/interaction evidence and creates many more scalar/support+scalar-positive opportunities. The break occurs when those newly exposed signals are converted into the final extremal recovery action.

The pure V20 endpoint moves in the wrong direction on frozen TRAIN:

- teacher action match: 22.47% -> 20.57%;
- mean teacher regret: 15,405.18 -> 16,323.01;
- beneficial rate: 7.07% -> 4.93%;
- direct positive-opportunity capture: 25.65% -> 17.92%.

There are 742 safe-domain scenes where B24 changes the final action relative to B16. On exactly these paired action changes, `B16 teacher regret - B24 teacher regret` sums to -2,753,490.91: 337 changes are beneficial for B24, 403 harmful, 2 equal. The largest B24 regret increase is 270,855.33.

This is strong **development-direction** evidence for a downstream semantics/operator mismatch: more useful evidence is visible, yet the frozen intervention operator converts it into worse decisions, and the DRC reliability population becomes much less favorable. But these 3000 scenes are the frozen development population. They cannot resolve the publication-level causal question by themselves.

## 4. Why the central V30 question remains unresolved

The scientific V30 question is:

> Is useful candidate-specific signal missing because B=16 discards already-queried M=24 evidence, or is sufficient signal already available while the downstream recovery semantics/operator cannot use it stably?

V30.1 stops before untouched token selection. Therefore there is no independent paired `B16-V20 vs B24-V20` evidence. The correct current statement is:

> TRAIN strongly shifts the hypothesis toward downstream consumer/operator mismatch, but B=16 retained capacity has not yet been independently ruled in or ruled out as a first-order mediator.

It would be a protocol error to promote the TRAIN direction into the final paper claim or to begin V31 before this mediator test is completed.

## 5. V29 -> V30.1 evidence chain

1. Earlier acquisition/coreset/teacher-shaped selector work became exhausted; repeated same-bank selector tuning was demoted.
2. EAF showed that complete action-local instrumentation and exact evidence attribution are useful for diagnosis.
3. V24 showed that richer attribution geometry can destroy local reliability neighborhoods; more representation is not automatically a better risk geometry.
4. V25 aggregate DRC recovered useful direct actions but left severe split-dependent catastrophic tails.
5. V26/V27 semantic/local confirmation variants did not provide stable high-retention reliability.
6. V28 PTMC was strong on TRAIN but had zero fresh harmful-veto specificity and removed beneficial proposals; classifier/threshold rescue was falsified.
7. V29 FCR improved its full-frontier compression proxy on ~73% of fresh scenes but produced almost no pure recovery gain and destabilized DRC/catastrophic behavior. Therefore global frontier fidelity is not decision sufficiency.
8. V30.1 now shows on frozen TRAIN that opening the retained interface from the B16 operating point to the complete already-queried M24 bank materially increases decisive-evidence transmission, yet the pure endpoint worsens and unchanged DRC loses cross-fold safety.

The cumulative evidence has therefore narrowed the main unresolved variable from “which selector/classifier?” to “is retained capacity itself a causal mediator on untouched data, or is the remaining bottleneck the intervention consumer?”

## 6. Paper thesis after V30.1

`B=16` must remain an **operating point/control variable**, not a novelty. B=24 is also not a proposed paper operating point. The paper problem should be expressed over a bounded/fixed auditable planner interface.

The strongest current thesis is:

> **Intervention-conditioned decision sufficiency under a bounded auditable planner interface:** evidence is sufficient not when it reconstructs the world or the complete frontier faithfully, but when it provides stable support for accepting or rejecting a concrete deployment-admissible incumbent-to-challenger intervention under extremal selection.

A candidate mechanism-level novelty is:

> **Evidence-attributed, deployment-admissible incumbent-contrastive recovery with monotone intervention containment, where evidence sufficiency and reliability are defined at the selected intervention boundary rather than by global reconstruction or edge-average prediction.**

The pieces supported by the history and worth freezing are: bounded/auditable interface; EAF instrumentation and exact attribution; deployment-admissible domain separation; incumbent-contrastive direct recovery; asymmetric admissible-incumbent preservation; all-flagged structural delegation; incumbent-default/no-fallback containment; fail-closed independent evaluation.

Do not make PTMC, FCR, a particular KNN/Gaussian/DRC, a literal evidence count, or a threshold the novelty headline.

This is a plausible CCF-A-level research direction only if a concrete final mechanism subsequently survives double-fresh, independent full validation, strong planning baselines/ablations, budget sensitivity, and closed-loop evaluation. No design can guarantee venue acceptance from theory alone.

## 7. Provisional dominant bottleneck

The dominant bottleneck can be tightened provisionally to:

> **selection-conditioned counterfactual reliability at the direct incumbent-replacement boundary: converting candidate-specific evidence into a high-coverage action replacement that remains stable after extremal selection and catastrophic-tail constraints.**

The word “provisionally” matters until V30.2. If untouched B24 pure capacity produces a reproducible safe gain, retained capacity is also a real mediator and the final consumer should be built to exploit a capacity-complete signal. If it does not, capacity/same-bank allocation can be terminally demoted and the operator mismatch becomes the first-order bottleneck.

For under-fixed-budget closed-loop SOTA, the current gap is no longer merely evidence recall. It is **useful recovery coverage under reliability**: recover materially more teacher-positive/closed-loop-improving direct alternatives while preserving incumbent/structural behavior and keeping the selected catastrophic tail controlled across independent populations.

## 8. V64.3.30.2: complete the missing pure-capacity causal test

V30.2 is intentionally not V31 and introduces no new planning algorithm. It fixes experiment orchestration only.

TRAIN:

- replay B16-V20 and B24-V20 with the exact frozen 3000 scenes;
- exactly reproduce historical B16 V25 fit;
- rerun the B24 V30 DRC fitter and **require** its non-zero fold-safety failure to reproduce;
- never lower or tune that gate;
- run a lightweight paired TRAIN audit to verify capacity isolation and failure identity.

Untouched A/B:

- use the same original V30 seed `v64.3.30-eaf-icer-fbic-double-fresh-v1`, because V30.1 stopped before fresh-token selection;
- evaluate exactly `raw / B16-V20 / B24-V20` on each 500-scene split;
- do not run any B24 DRC fresh arm.

The pure-capacity checker measures three different causal views:

1. direct positive-opportunity capture gain (pre-registered signal threshold +3 percentage points);
2. all safe-domain scenes where B24 actually changes the action, with paired teacher-regret net effect;
3. the **same B16-defined positive-opportunity scenes**, comparing B16 and B24 final teacher regret, so a B24-induced change in anchor/admissible population cannot manufacture a favorable denominator.

It additionally requires exact query parity, capacity superset accounting, structural no-op, and endpoint non-inferiority.

### Branches

**A. Both fresh blocks show useful, non-harmful pure B24 signal.** Retained capacity is a real mediator. Because B24 DRC is already TRAIN-falsified, do not make B24 the method and do not restore DRC. The next algorithm should be a selection-aware candidate-conditioned consumer, followed only later by an adaptive bounded completion mechanism if needed.

**B. Capture rises but paired action/opportunity effects or endpoint are harmful.** More evidence exposes opportunity but the frozen action operator converts it unsafely. The consumer/operator is the dominant bottleneck; stop capacity tuning and redesign selection-conditioned counterfactual reliability.

**C. Pure B24 does not reproducibly improve both blocks.** Capacity-only transmission is not a first-order solution under the frozen consumer. Stop B sweeps and all same-bank selector/rebinding work; focus directly on intervention-conditioned recovery semantics/operator mismatch on the bounded interface.

No pooled rescue is allowed.

## 9. Historical routes that remain prohibited

Do not repeat: learned/teacher-shaped acquisition objectives; DACC/beam/swap/same-bank coreset rescue; FCR/global frontier compression tuning; transition/signed-profile/full-attribution KNN geometry; semantic-family/type-KNN thresholds or weights; KNN-radius/OOD rescue; mean-SE/DRC mixing or DRC K/threshold/downside sweeps; support/scalar threshold rescue; action blacklists; PTMC/classifier v2/v3; failed-view AND stacking; naive aggregate+type concat; learned admissible-incumbent-to-anchor veto; broad B sweeps before the single capacity mediator is resolved.

## 10. Engineering validation of V30.2

Local validation after adding the pure-capacity orchestration and paired audit tools:

- Python compile: PASS;
- V30.2 launcher `bash -n`: PASS;
- new V30.2 + FBIC focused tests: 16/16 PASS;
- V13--V30.2 targeted regression: **123/123 PASS**;
- repository full regression: **453/453 PASS**;
- warnings: **36**, all pre-existing PyTorch Transformer `nested_tensor/norm_first`; no new warning class.
