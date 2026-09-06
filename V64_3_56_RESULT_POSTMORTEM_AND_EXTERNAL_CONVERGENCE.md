# V64.3.56 result postmortem and internal-search convergence

## Verdict

V56 is reliable for scientific attribution. The result lands in preregistered **case B**: the realized constraint-process oracle does not fully pass its mechanism-identification gate, so internal algorithm search converges by falsification and the predicted t0 branch is not evaluated.

## Why this is not a V56 promotion

The oracle deployment metrics are unusually strong: 115/121 beneficial retained, hard harm 25->15, nonbeneficial 381->326, score sum -4.2409->+0.8408, NegRMS .15486->.12989, and all-fold nonharm passes. But the scientific question was not "can this consumed-TRAIN oracle produce a good subset?" It was whether the *new constraint-process state* is a stable incremental mechanism over exact V55. Pareto concordance is 0.576731 vs 0.567756 for V55, but only 2/5 folds improve over V55 (pre-registered minimum 4/5). Thus the incremental state-family claim is not identified.

## Mechanism interpretation

V54 established that realized ego response is a real mediator. V55 established that an unweighted paired-outcome Pareto order is a real supporting functional. V56 shows that adding direct realized occupancy/TTC/off-route consequence can yield an excellent aggregate deployment rotation, yet its incremental ranking contribution is not fold-stable. This is evidence of substantial scene-regime heterogeneity: the state is useful in some regimes (folds 3/4) but does not improve or even weakens ranking in others (folds 0/1/2). Continuing to tune the same family would convert a hypothesis-driven mechanism study into post-hoc optimization.

## Evidence-chain extension

prediction sufficiency != decision sufficiency
-> representation identifiability != deployment sufficiency
-> observational post-selection identification != fresh transportability
-> paired outcome source != selected-outcome state sufficiency
-> effect support != conditional effect order
-> planned treatment geometry != realized treatment mediation
-> realized mediation != deployment outcome-order sufficiency
-> structured functional identifiability != fold-stable deployment sufficiency under ego-only state
-> **strong deployment rotation != stable incremental mechanism identification for realized constraint-process state**.

## Internal convergence

There is no V57 feature/state branch. The convergence target required one t0-deployable arm to pass identification + paired deployment. V56 does not achieve success convergence because the oracle itself fails identification and therefore the t0 branch is ineligible. It does, however, satisfy the predeclared falsification-convergence condition. The next phase is external baselines and full closed-loop benchmarking.

## What is benchmarked

The primary own method is the strongest fully t0-deployable frozen backbone: EAF + full-set RSMR. Post-intervention V54/V55/V56 oracle states are diagnostics only and are excluded from runtime benchmarking.

Use B=16 as the primary matched-interface comparison. A B=8/B=16/B=24 run is also provided, but own-model B=8/B=24 are explicitly frozen-policy cross-budget robustness ablations unless independently refit by a separately preregistered protocol. The trainable external adapters are budget-specific.
