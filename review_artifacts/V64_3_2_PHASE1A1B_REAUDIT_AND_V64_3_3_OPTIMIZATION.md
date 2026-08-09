# V64.3.2 Phase-1A/1B Re-audit and V64.3.3 Optimization

## Executive conclusion

The uploaded V64.3.2 Phase-1A AP-WCCA and Phase-1B AP-WRCCA screens do **not** establish that either acquisition representation is ineffective. Two independent engineering defects invalidate the central `critical Top-M delta=0` interpretation:

1. Training validation used the HAB Top-M mask itself as the active evidence universe when defining literal winner-flip criticality. Therefore every critical atom under that restricted universe is necessarily in Top-M and `critical Top-M recall=1.0` by construction. `1.0 -> 1.0` / delta `0` is tautological rather than an acquisition measurement.
2. `encode_context()` produced `critical_proposal_residual_logits`, but `BDSEModel.forward()` did not export it. The ACRA branch queried this output key and therefore never received the intended residual tensor. Adapter parameters still moved through other losses, which explains nonzero parameter delta together with exactly-zero direct residual/ACRA diagnostics.

A third provenance defect was found in Phase-1B: AP-WRCCA training used winner-rival conditioning while the screen launcher named the AP-WCCA eval config. Train-only screen mode prevented direct weight corruption, but the old contract failed to reject the mismatch.

Accordingly, the immediate next step is **not** to alter B, M, selector, or certificate. It is also premature to move directly to downstream atom-to-action value learning. First obtain a clean full-support acquisition experiment.

## Uploaded screen evidence

### Phase-1A AP-WCCA

- Adapter parameter delta RMS max: `0.0052386` — parameters did move.
- Proposal decisive recall: `0.79151 -> 0.77514` (`-0.01637`).
- Teacher action match: approximately `0.264 -> 0.258`.
- Reported validation critical Top-M: `1.0 -> 1.0` — invalid because of Top-M-as-support.
- Reported selected critical recall: `0.74861 -> 0.72083` — also defined on the restricted support and not comparable to the formal full-support critical recall.
- Training-side literal critical Top-M diagnostic is around `0.3653 -> 0.3685 -> 0.3682`, but it has no same-subset step-zero train anchor and therefore is not a clean causal gain estimate.

### Phase-1B AP-WRCCA

- Adapter parameter delta RMS max: `0.0050095`.
- Proposal decisive recall: `0.79151 -> 0.76280` (`-0.02871`).
- Teacher action match: approximately `0.264 -> 0.258`.
- Reported validation critical Top-M: `1.0 -> 1.0` — equally invalid.
- Training-side literal critical Top-M peaks around `0.3689`, nearly indistinguishable from AP-WCCA without a step-zero anchor.

AP-WRCCA therefore has a **provisional negative broad-recall signal**, but it is not enough to reject the representation because the direct ACRA objective was disconnected and the target acquisition metric was invalid.

## What the model history supports

### Components with positive evidence and should be retained

1. **Fixed B=16 planner-interface budget.** Historical formal runs show high winner preservation after compression; selector loss is much smaller than the acquisition loss once Top-M is formed.
2. **Auditable evidence atoms + literal removal-induced winner-flip criticality.** This remains the cleanest semantic core of the paper and should not be replaced by a margin-deficit proxy.
3. **Legacy HAB family-aware proposal as an immutable anchor.** Historical broad proposal decisive recall near ~0.80 and dense-to-HAB winner preservation show that the legacy acquisition prior is useful.
4. **DA-EPC exact downstream preservation certificate.** Recent screen branches show high fixed-budget exact-preservation coverage and essentially zero fallback, while the old pairwise AOCC bound was severely misaligned with actual winner preservation.
5. **Strict checkpoint/config/source provenance.** Multiple prior apparent algorithm failures were caused by configuration/support drift; SHA-bound provenance is necessary for valid ablation conclusions.
6. **Sparse exact-selector supervision cadence.** V64.3.2 correctly reduced exact-selector supervision frequency after the V64.3.1 launcher override bug. The uploaded screens show `selector_exact_fraction` near `0.0157` for early epochs instead of 1.0, materially reducing loss-stage overhead and preserving intended training semantics.

### Designs with negative evidence / should not be repeated as the next move

1. **Full proposal/family unfreezing and broad global ranking pressure (V64.2 style):** broad decisive recall fell without breaking the literal-critical acquisition plateau.
2. **BCC/HCBE/global hardest-negative weight escalation as configured previously:** did not establish a sustained improvement beyond the historical ~0.35 full-support critical Top-M regime and can conflict with broad coverage.
3. **V40-V43 beam/swap/repair deployment search:** substantially increased CPU/open-loop cost and did not solve teacher-action quality. Do not revive it to address current acquisition.
4. **Increasing B or M:** not justified by current evidence; B=16 preserves most of what Top-M already contains.
5. **Certificate relaxation:** DA-EPC already addresses the previous certificate mismatch; relaxing it cannot create missing critical evidence.
6. **6-D query extension in nominal mode / opaque runtime base prior:** previous results showed dense-interface drift or harmful winner replacement. Keep them disabled in the mainline.
7. **Lowering residual conformal epsilon merely to manufacture flips:** calibration should follow learned error, not be relaxed to force deployment changes.

### Designs that are still inconclusive

- **AP-WCCA:** not yet cleanly tested with full-support validation and wired ACRA.
- **AP-WRCCA:** same; current broad-recall drop is a warning, not a proof of failure.
- **ACRA:** not tested at all in V64.3.2 because the forward output contract disconnected it.

## Is acquisition/value representation really the next direction?

**Yes at the module level, but with an important distinction.** Historical formal data still identify missing literal critical evidence before Top-M as the dominant upstream loss. Therefore the next controlled work should stay in **acquisition representation / acquisition value target**, not selector/B/M/certificate.

However, downstream **atom-to-action value** is not yet the next module. It becomes the next bottleneck only after a clean experiment shows:

- full-support literal critical Top-M recall improves;
- selected critical recall improves or at least follows;
- but teacher action match / paired regret does not improve.

That pattern would mean the right evidence was acquired and retained but its action-dependent value was wrong.

## V64.3.3 changes

### 1. Full-support literal criticality

Training validation now defines criticality over `sample.evidence_bank.active_mask`, matching formal open-loop support semantics. Top-M and selected masks are evaluated against that full universe rather than being used to define the universe.

`_criticality_metrics()` additionally exports critical counts and hit counts, allowing stable micro recall:

`sum(critical & TopM) / sum(critical)`.

A regression test constructs a literal critical atom outside Top-M and requires measured Top-M recall to be zero, preventing the tautological implementation from returning.

### 2. ACRA is actually wired

`BDSEModel.forward()` now exports `critical_proposal_residual_logits`. The loss receives the residual tensor and logs `L_critical_adapter_residual_alignment`. The screen requires all of:

- parameter delta > 0;
- residual forward RMS > 0;
- ACRA alignment loss > 0.

Parameter movement alone is no longer considered sufficient instrumentation.

### 3. Matched train/eval conditioning contract

V64.3.3 validates that train and eval acquisition conditioning agree. AP-WRCCA train + AP-WCCA eval is now a hard error. The AP-WRCCA and value-probe launchers use matched winner-rival eval configs.

### 4. Three-stage acquisition experiment

**Stage A — AP-WCCA + binary literal-critical ACRA.** This is the minimal corrected re-test and should run first.

**Stage B — AP-WRCCA + binary literal-critical ACRA.** Run only if Stage A is fully instrumented but critical Top-M micro recall fails to improve.

**Stage C — Literal-Critical Value (LCV) probe.** Run only if A and B are both valid and non-improving. Non-critical atoms retain exactly zero target. Only atoms already proven literal winner-flip critical receive a continuous severity scaling based on the post-removal winner gap. Thus LCV tests whether binary labels discard useful within-critical value ordering without redefining criticality.

This is deliberately different from the older DBCE/margin-deficit direction: an atom that does not literally flip the winner receives zero LCV target regardless of its margin effect.

## Decision table after V64.3.3 screens

- AP-WCCA clean + critical Top-M micro delta > 0, broad decisive delta >= -0.02: promote AP-WCCA to full pipeline; AP-WRCCA becomes an ablation only.
- AP-WCCA clean but no critical gain: run AP-WRCCA.
- AP-WRCCA clean + positive critical gain: promote AP-WRCCA; retain AP-WCCA as winner-only ablation.
- Both binary screens clean and flat: run LCV value probe.
- LCV positive: absorb literal-critical severity alignment into the winning acquisition representation and then run full causal open-loop evaluation.
- AP-WCCA/AP-WRCCA/LCV all clean and flat: acquisition **representation/conditioning** is the bottleneck. The next paper-level algorithm should expose richer deployment-available boundary state (e.g. multi-rival action context), not alter B/M/certificate.
- Critical Top-M and selected both improve but teacher action/regret stays flat: move to downstream **atom-to-action / pair-margin value representation**.
- Open-loop improves but paired diagnostic closed-loop does not: move to candidate dynamics, reactive interaction, and replanning.

## CCF-A trajectory

For a CCF-A submission, the strongest path is not to accumulate more modules. It is to establish a clean causal chain around the fixed-budget novelty:

1. exact literal criticality is measurable over the full evidence support;
2. a learned acquisition mechanism improves critical evidence coverage under a fixed Top-M acquisition budget;
3. deterministic/exact B=16 selection preserves the acquired decision;
4. the improvement transfers to teacher-aligned open-loop metrics and finally paired closed-loop behavior;
5. each stage has a targeted ablation and no hidden support/provenance change.

V64.3.3 is designed to answer step 2 cleanly before any further main-algorithm expansion.
