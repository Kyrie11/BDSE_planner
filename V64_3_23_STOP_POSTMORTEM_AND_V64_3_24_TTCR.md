# V64.3.23 STOP Postmortem and V64.3.24 TTCR Plan

## Executive conclusion

The paper motivation should be retained: BDSE's central problem is a **planner-interface information budget**, and the right objective is downstream decision sufficiency rather than world-model reconstruction fidelity. The uploaded V64.3.23 code/result, however, shows that the current intervention layer is not yet decision-sufficient in the stronger sense needed by the final algorithm.

The V23 STOP is algorithmic rather than engineering. Split A shows that evidence-local recovery can be useful; split B shows that the same mechanism can select a handful of O(1) negative candidate-vs-incumbent replacements that dominate the block. The frozen interface, support, dominance, and structural delegation are healthy. The dominant bottleneck is **representation-level identifiability of the material tail for the selected incumbent replacement**. Mean-LCB mismatch and dominance-first extremal ranking are secondary mechanisms.

The proposed V64.3.24 algorithm is **Typed Tail-Coherent Incumbent-Contrastive Recovery (TTCR)**. It keeps B/M, acquisition, selector, EAF, support/dominance, incumbent default, and final/structural guards frozen. It exposes type-aware contrasts from the already-selected evidence and penalizes the local lower partial moment of material replacement loss. A seven-arm C/D fresh experiment isolates representation, tail objective, and extremal rank alignment separately.

## Source-derived paper idea

The paper's main motivation is internally coherent: a real-time planner cannot explicitly consume a full predictive world model, so compression should preserve information that changes candidate-action margins rather than maximize distributional fidelity. Rare evidence can be decisive if removing it flips a feasible-action ordering. BDSE makes that interface auditable via evidence atoms and a fixed query budget, decomposes teacher margins into base plus evidence terms, selects evidence to protect decisive winner-versus-rival margins, and chooses an action through a candidate tournament.

For the final paper, the code should remain the algorithm source of truth. The paper's old generic risk/tournament description should eventually be synchronized only after the V24 mechanism is validated.

## V23 result that matters

| Quantity | Split A | Split B |
|---|---:|---:|
| V23 signed-RCR direct incumbent replacements | 45 | 29 |
| direct-path teacher-regret delta sum | -86669.17 | +57895.19 |
| selected teacher-improvement sum | +4.33346 | -2.89476 |
| worst selected teacher improvement | -1.23146 | -0.98990 |
| screen result | fail: signed-profile not incremental | fail: local regret coherence harmful |

B is dominated by four material negatives around `-0.99, -0.99, -0.93, -0.93`, while a `+0.98` beneficial replacement also exists. This is a mixed near-zero/heavy-tail problem, not a uniformly wrong edge classifier.

V23's 5-fold TRAIN gate was too weak because it tested aggregate fold sum. The signed main could pass all folds while fold 4 still selected a `-0.255` replacement; scalar cross-fit had selected losses around `-1.70` and `-0.99` in other folds.

## Bottleneck hierarchy

1. **Primary — missing typed intervention observables.** Current local features can make catastrophic and benign replacements nearest neighbors. The lost information is which selected evidence type supports/contradicts the incumbent change, especially occupancy/collision/TTC and hard-rule evidence.
2. **Secondary — average objective versus material tail.** Mean minus one SE can stay positive despite a low-probability O(1) downside. The target should penalize probability times magnitude beyond the existing normalized action-change resolution.
3. **Tertiary — gate/ranking mismatch.** V23 gates with local regret but then maximizes dominance. This can amplify winner's curse. It must be ablated separately because old-feature risk-first replay does not solve B.
4. **Not dominant — selector/B/M/EAF/support/structural guards.** These were healthy on A/B and should remain frozen.

## V24 TTCR mechanism

The runtime representation adds 49 deterministic typed selected-evidence statistics to the frozen 18 evidence reliability features. Seven fixed type groups each expose selected fraction, candidate cost, incumbent cost, signed improvement, upside mass, downside mass, and maximum downside. These are computed only from selected atom type and frozen predicted per-atom action cost; no new evidence is queried.

For candidate `b` against admissible incumbent `l`, TRAIN target `Delta` is the normalized teacher improvement of `b` over `l`. For K=32 and K=64 fixed local neighborhoods:

`mean_LB = weighted_mean(Delta) - SE(Delta)`

`downside_UCB = weighted_mean(max(-Delta-tau,0)) + SE(max(-Delta-tau,0))`

`R_TTCR = min_K(mean_LB - downside_UCB)`

with frozen `tau = fallback.tau_delta_normalized = 0.004`. Replacement requires support>0, scalar dominance>0, and `R_TTCR>0`. The full main ranks eligible alternatives by `R_TTCR` first. The admissible incumbent stays the default.

## Experiment logic for the next round

The next screen is intentionally mechanistic rather than another monolithic main-vs-baseline run. Fresh blocks C and D each evaluate raw, V20, evidence-LCB, evidence-tail, typed-LCB, typed-tail/dominance-first, and typed-tail/risk-first.

Interpretation is pre-registered:

- `typed-tail/dom > evidence-tail/dom` on both C/D: typed representation contributes in the actual tail context;
- `typed-tail/dom > typed-LCB/dom` on both C/D: the tail objective contributes under the typed representation;
- `typed-tail/risk-first > typed-tail/dom` on both C/D: extremal rank alignment contributes;
- any failed matched comparison removes that component from the paper mainline instead of triggering a weight/threshold sweep;
- any selected `Delta < -0.004` in the full main is a tail failure and points to a still-missing runtime observable, not to K/tau tuning.

Before fresh evaluation, the 3000-scene TRAIN frontier is replayed because the V23 edges do not contain typed features. The strict main cross-fit requires at least 8 replacements per fold, nonnegative sum per fold, and zero material negatives per fold. Failure stops before fresh token selection.

The V23 A/B 1000 tokens have been added to the permanent design exclusion, which is now 5700 unique validation identities. C and D are two new independent 500-scene blocks and cannot be pooled.

## CCF-A standard decision

The current broad V23 mechanism stack should not be frozen. The stronger and cleaner paper line is:

**budgeted decision-sufficient evidence -> typed intervention-sufficient evidence -> material-tail-coherent incumbent recovery -> decision preservation**.

This is stronger scientifically because every new term answers a failure directly and is separately falsifiable. The KNN estimator is implementation, not novelty. If V24 passes double-fresh plus one frozen full-validation reproduction, the paper can freeze this line and add a theory statement around **tail decision sufficiency / intervention sufficiency** under local regularity and support assumptions. It should not claim distribution-free guarantees for the current heuristic local score.

If typed features fail, the next representation audit should become finer (e.g. actor-conditioned selected hard-interaction ownership/geometry) rather than larger. If the tail penalty fails, drop it. If risk-first ranking fails, revert it. This is the mechanism-pruning rule that keeps the final claim reproducible.

## Engineering status

The implementation has been applied in the repository. Final local checks: Python compile PASS; launcher syntax PASS; V64.3.6--V64.3.24 targeted regression 118/118 PASS; V24 synthetic TRAIN fitter PASS; five generated synthetic configs pass the V24 contract checker. The actual nuPlan V24 experiment has not been executed here because the required dataset/GPU environment is external to this analysis runtime.

## Next command

```bash
bash RUN_V64_3_24_EAF_ICER_TYPED_TAIL_SCREEN_2GPU.sh
```

Output root: `outputs_v64_3_24_eaf_icer_typed_tail_screen_2gpu_v1`.
