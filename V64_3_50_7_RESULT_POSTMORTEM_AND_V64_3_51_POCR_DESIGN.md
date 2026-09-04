# V64.3.50.7 reliable scientific result and V64.3.51 POCR design

## 1. Reliability verdict

V64.3.50.7 closes the remaining V50.6 provenance gap. The uploaded result is valid for TRAIN-level algorithm attribution.

- result-defining V50.6 fit source, test and runner: expected SHA256 == server SHA256 for all three;
- V50.6 and V50.7 `v64_3_50_6_pior_fit.json`: byte-identical;
- paired evidence hash unchanged: `d592baa3508ae9dc084eebcbb3505d9accadc724601ab35a1253d3536af39d43`;
- control/treatment metric-safe certificates: 502/502 each, zero failures;
- targeted regression: 45/45 PASS;
- V50.7 termination: preregistered TRAIN scientific STOP, not an exception;
- untouched validation: unconsumed.

Therefore V50 can finally receive a scientific GO/STOP judgment.

## 2. Preregistered V50 verdict

V50 required PIOR risk to beat both frozen OBS and EGO baselines in aggregate and in at least 4/5 outer folds, then also pass the paired causal deployment gate.

Observed selected-outcome AUC:

| Risk view | AUC |
|---|---:|
| EGO-REF baseline | 0.474610 |
| offline OBS | 0.499425 |
| PIOR QPE | 0.508709 |

PIOR beats EGO in 3/5 folds and OBS in 3/5 folds. Identification therefore fails by preregistration.

The causal-retention gate also fails:

| Metric | Frozen RSMR paired baseline | PIOR QPE |
|---|---:|---:|
| selected | 502 | 460 |
| beneficial retained | 121 | 115 |
| beneficial recall | 1.0000 | 0.9504 |
| nonbeneficial retained | 381 | 345 |
| hard harm | 25 | 24 |
| closed-loop score delta sum | -4.2409 | -5.4262 |
| negative RMS | 0.15486 | 0.15627 |

The beneficial-retention and nonbeneficial-reduction subgates pass, but hard-tail, utility-nonharm and all-fold nonharm fail. Fold 2 is the strongest counterexample: OBS AUC 0.6604 versus PIOR AUC 0.3865, with PIOR score-sum worse than the frozen baseline.

Formal V50 verdict:

`V50 PIOR = scientific STOP / promotion failure`.

Because identification fails first, the preregistered next branch is the V50 identification-failure branch: `[Q, P-Q, E-P]` is not sufficient for real paired selected-outcome discrimination. This does **not** license changing loss, lambda, class weights, threshold, MLP capacity or offline feature families.

## 3. What V50 learned and did not learn

V50 learned a weak coarse retention direction. It vetoes 42/502 selected proposals while preserving 115/121 beneficial outcomes and removing 36/381 nonbeneficial outcomes. That is evidence that paired outcome labels are usable and that the conformal false-veto budget controls coverage.

What it does **not** learn is a stable selected-outcome ordering. Only one of 25 hard-harm events is removed, negative RMS worsens, and the retained score sum is 1.1853 lower than the frozen RSMR paired baseline. The fold reversal shows this is not a threshold placement issue: the underlying QPE risk ordering itself is unstable with respect to actual one-shot closed-loop outcomes.

The correct mechanistic interpretation is therefore:

`deployment-aligned outcome supervision != outcome-state sufficiency`.

Paired evidence should be retained; QPE-only paired state should not be promoted as a sufficient deployment state.

## 4. Postmortem diagnostic: treatment/control execution contrast

This section is design-only and must not be used to retroactively promote V50.

The V50 treatment probe already audits the exact physical difference between the frozen RSMR proposal and the runtime incumbent:

`D = || xi_proposal - xi_incumbent ||_infinity`.

This is available before execution and is part of the bounded planner interface; it is not teacher information, logged future, candidate multiplicity or an offline feature sweep.

On the consumed 502 TRAIN pairs:

- AUC of D for **any closed-loop effect**: 0.69755, 5/5 folds > 0.5;
- AUC of D for **beneficial outcome**: 0.61209, 5/5 folds > 0.5;
- AUC of D for **hard harm**: 0.73618, 5/5 folds > 0.5;
- 38/502 physically identical proposal/incumbent pairs have D=0 and all 38 have null effect;
- 209/502 have exactly zero score delta; 171 of those are physically different, showing that the outcome functional contains a large structural null region beyond exact physical equality.

Importantly, larger D increases both benefit probability and hard-harm probability. Therefore D must **not** be interpreted as a monotone risk score. Its correct role is an operator-relative intervention dose that conditions how the existing consequence coordinates Q/P/E should map to a realized outcome.

This motivates V51 and also explains why simply adding another generic offline value feature would be the wrong branch.

## 5. Evidence chain V34 -> V50

The current chain can be stated compactly:

1. V34: ordinal extremal proposal selection becomes reliable enough to freeze RSMR.
2. V37-V40: selected residual exists, but generic value/head/19-D distribution routes are not sufficient.
3. V41-V43: endpoint/current/prospective consequence views are real partial mediators.
4. V44: ungated full-horizon prospective interaction support is a strong selected-value mediator.
5. V45: agent-local continuous response is identifiable and useful; plan conditioning is real but incremental.
6. V46: additional predictive information can improve regression while worsening the deployed extremal decision; prediction sufficiency != decision sufficiency.
7. V47: EGO-REF is a strong supporting consequence mediator, but identifiable representation != deployment sufficiency; representation expansion is stopped.
8. V48: multiplicity/logK is fresh-falsified; in-domain post-selection risk != transport sufficiency.
9. V49: changing the offline selected-event measure does not identify the deployed selected-outcome law; the offline selected-risk family is closed.
10. V50: actual paired selected-outcome evidence is obtained, but QPE-only state still fails to identify the outcome law.

The new paper-level conclusion is:

`selection sufficiency != prediction sufficiency != representation identifiability != observational post-selection sufficiency != outcome-source sufficiency != selected-outcome state sufficiency`.

## 6. Dominant bottleneck after V50

The dominant bottleneck is no longer evidence source. V50 has finally supplied deployment-aligned paired evidence.

It is now:

**operator-relative selected-outcome state sufficiency** — for the frozen RSMR proposal, the model must condition the outcome law on what is actually changed relative to the incumbent, not only on compressed proposal consequence coordinates.

A precise target is:

`P(Y_deploy | Q, P-Q, E-P, C(xi_b, xi_i))`,

where `C` is a bounded, runtime-observable treatment/control contrast.

## 7. Newly closed / retained families

Newly closed as a deployment-sufficient solution:

- QPE-only PIOR selected-outcome state;
- any attempt to rescue QPE-only PIOR by threshold, lambda, class/focal/catastrophe weighting or a larger MLP.

Not closed:

- Q, P and E as supporting consequence coordinates;
- paired one-shot closed-loop selected-outcome supervision;
- low-capacity pairwise sign-risk as a causal control for the next state experiment (its sufficiency cannot be judged before the state is identifiable).

All previously closed families remain closed: V46 low-order variance/handcrafted temporal profile, V47 constant-drift AGENT-2D, V48 K/logK/multiplicity, V49 random-prefix SIIR/offline selected-risk, selected translation, CVaR tuning, binary catastrophe veto, RSMR/B/M/top-K/candidate-count changes, reranking, second-best and fallback.

## 8. V64.3.51 POCR — Paired Operator-Contrast Retention

V51 changes exactly one scientific factor: the selected-outcome state.

RSMR first freezes the same proposal. Q/P/E are reconstructed exactly as in V50. The new coordinate is:

`D = ||xi_b - xi_i||_infinity`,

computed directly from the frozen proposal and incumbent trajectory tensors before execution.

The same zero-bias pairwise sign-risk, lambda=1, frozen alpha, finite-rank split-conformal threshold, paired outcome labels and same-winner/incumbent-only policy are retained.

### Arm A — QPE+DOSE

State:

`[Q, P-Q, E-P, D]`.

Question: is explicit treatment/control intervention magnitude the missing selected-outcome state?

### Arm B — QPE+DOSE-X

State:

`[Q, P-Q, E-P, D, D*Q, D*(P-Q), D*(E-P)]`.

Question: because D predicts both benefit and harm, must treatment dose modulate the Q/P/E consequence slopes rather than act as an additive monotone risk feature?

No MLP, attention, learned threshold, K, class weighting or new offline observable is introduced.

## 9. V51 preregistered gates and branch order

Exact V50 QPE is replayed first as a hard causal control. V51 aborts as ENGINEERING STOP if the V50 control AUC/fold signatures drift.

For each V51 arm, outcome identification requires:

- arm AUC > QPE-control AUC, EGO AUC and OBS AUC in aggregate;
- arm > QPE control in at least 4/5 folds;
- arm > EGO in at least 4/5 folds;
- arm > OBS in at least 4/5 folds.

The original V50 causal deployment gate is then applied unchanged.

Promotion order is fixed before running:

1. if QPE+DOSE passes identification + deployment, promote the simpler additive state;
2. else, if QPE+DOSE-X passes both, promote the interaction state;
3. if state identification succeeds but deployment fails, the next branch is a structured paired closed-loop outcome functional; do not add more offline state;
4. if scalar D identifies effect support but neither state identifies benefit/harm, close scalar dose as a sufficient sign-state and move to a richer **temporal treatment/control contrast profile** under the same paired evidence;
5. if D itself does not identify effect support, close the scalar operator-contrast family and reassess paired outcome state acquisition/measurement.

An independent nuisance diagnostic reports whether D identifies *any effect* versus the structural null. This diagnostic is not itself a deployment promotion gate.

## 10. Paper mainline

The recommended headline remains:

**Selection–Valuation–Outcome Sufficiency under a Bounded Auditable Planner Interface**.

V51 makes the outcome layer relational rather than proposal-only:

`bounded evidence -> extremal selection -> proposal freeze -> prospective valuation -> paired selected-outcome evidence -> operator-relative contrast-conditioned outcome identification -> monotone retention`.

The novelty is not the scalar D by itself. The paper-level mechanism is that, under an extremal no-fallback planner interface, an actual treatment effect is a **proposal-versus-incumbent relational object**. Aligning labels to deployment is insufficient if the state omits the deployed operator contrast.

## 11. Experiment size and speed

V51 TRAIN reuses the already metric-safe 502x2 V50.5 paired outcomes and the already-recorded treatment probe contrast. It does not rerun the 22-hour paired simulation and does not consume untouched validation.

This is both faster and scientifically cleaner: the only changed factor is the state supplied to the same cross-fitted risk/retention operator.

If V51 TRAIN passes, freeze the preferred arm immediately and then collect a new untouched paired validation set. No TRAIN tuning is permitted after PASS.
