# V64.3.5 Result Audit and V64.3.6 Optimization

## Executive decision

V64.3.5 CCBR+LEA fixed the **finite frontier support** defect but did not improve the learned evidence admission. The downstream pair-value bottleneck is now directly confirmed. V64.3.6 therefore keeps the paper spine unchanged and introduces two orthogonal, zero-init residual mechanisms: BCHA for hierarchical evidence admission and LBPR for literal decisive-boundary value.

The paper spine remains:

`fixed planner-interface budget -> auditable evidence atoms -> decisive action-margin preservation -> budgeted evidence -> final decision preservation`.

No experiment in this delivery increases B=16 or proposal M, redefines criticality, relaxes DA-EPC, or introduces teacher information at deployment.

## What V64.3.5 actually proved

### CCBR solved support, not acquisition

| Metric | Anchor | CCBR-noLEA | CCBR+LEA |
|---|---:|---:|---:|
| literal critical Top-M micro | 0.36015 | 0.36015 | 0.36015 |
| selected critical micro | 0.26054 | 0.25287 | 0.25287 |
| proposal decisive recall | 0.79151 | 0.77858 | 0.77632 |
| teacher action match | 0.264 | 0.260 | 0.260 |
| endpoint representability | n/a | n/a | ~1.000 |
| max LEA loss | n/a | 0 | 3.2463 |
| max CCBR residual RMS | 0 | 0.4183 | 0.4676 |

The crucial result is semantic: FPCCA F6/F8 could not represent most literal boundaries, whereas CCBR+LEA reaches essentially complete endpoint support and learns non-zero endpoint attribution. But Top-M remains exactly unchanged, so support was a real intermediate defect rather than the final admission bottleneck.

### The downstream value ceiling is real

The full-evidence pair pathway remains near 26% teacher action match across the screen:

- pair-full: ~0.256--0.262;
- local pair-full: ~0.258--0.264;
- budget-vs-pair-full: ~0.928--0.948;
- pair-full -> budget flip: only ~0.052--0.072.

If the B=16 interface were the dominant problem, pair-full would be much stronger and budget-vs-pair-full would be poor. The opposite is observed: budgeted and pair-full decisions usually agree, but both disagree with the teacher. Therefore action/evidence value representation is a proven limiting factor.

## Does V64.3.5 prove frozen family slots are the admission bottleneck?

No. The intended frozen-family-slot oracle is null in both uploaded screens. Code audit found an instrumentation placement error: the oracle existed in the optional dense evaluator, but the short training screen never executed that branch. V64.3.6 fixes the location and adds a global-oracle comparator.

Interpretation next run:

- frozen oracle >=0.90: family slots are not the primary admission limit; do not use/tune BCHA;
- frozen oracle <0.90 and global-frozen >=0.05: family admission is a measured ceiling; BCHA is causally justified;
- both frozen and global oracle low: the limit is not family allocation; investigate M/cost feasibility/label distribution before architecture changes (without increasing M for the main result).

## V64.3.6 mechanism 1: BCHA

CCBR proposal residual appears after the frozen family gate. BCHA sends a bounded summary of that exact same boundary residual upstream into family logits, without unfreezing the family network. This allows a literal-boundary signal to influence the hierarchical admission allocation while keeping fixed B/M and auditable atoms.

BCHA is deliberately conditional. It is not claimed as the next main algorithm unless the oracle says family slots are limiting.

## V64.3.6 mechanism 2: LBPR

LBPR is the primary next algorithm because the value ceiling is already proven. It is low-rank, evidence-attributable, exactly antisymmetric, and pair-conditioned. It is not a global action potential and not a broad arbitrary pair MLP.

Supervision focuses on the exact decision boundary supplied by the existing literal winner-flip label: teacher winner versus the leave-one-atom-out flip target. A reserved quota ensures these rare pair edges are not lost in the cached training pair subset. CCBR endpoint compatibility gates the residual at runtime but is detached from pair-value gradients.

The resulting story is aligned with the paper: the evidence atom that is literally decisive is not only acquired under budget; its contribution to the exact decisive action boundary is represented explicitly.

## Why not scratch-train now?

The uploaded screens used the intended V62 checkpoint. Keep it for V64.3.6. A full scratch retrain now would simultaneously change scene/action/evidence representations, proposal behavior, and pair value and would erase the clean causal comparison. Historical broad unfreezing/value-field attempts were negative.

If LBPR is active but pair-full cannot improve, that becomes evidence that frozen representations are insufficient; only then should a selective action/evidence encoder unfreeze be designed.

## Engineering findings and fixes

1. Moved frozen-family oracle into the mandatory screen validation path; added global-oracle and gap.
2. Found and fixed an LBPR antisymmetry bug before delivery: a trainable output bias would violate r(a,b)=-r(b,a). LBPR output is now bias-free and regression-tested after non-zero weights.
3. Added exact literal winner->LOO-flip reserved pair quota, while retaining the vectorized sampler.
4. Disabled unused historical residual-action/set-potential computation in V64.3.6 configs; these branches are not part of the direct pair tournament and have negative historical evidence.
5. Full training remains hard-gated by the screen comparison and an explicit existing foundation checkpoint.

## Recommended experiment order

1. LOCAL control — obtains the repaired family oracle on the same V62 anchor.
2. LBPR — always run, because value bottleneck is proven.
3. BCHA — run only if frozen-family oracle indicates a family ceiling (or `RUN_FULL_2X2=1`).
4. BCHA+LBPR — run only when BCHA is justified and one individual module shows a meaningful signal; force with `RUN_FULL_2X2=1` for a final 2x2 paper ablation.

A full 50k/8-epoch run is blocked unless teacher action match itself improves by >=0.5pp along with a meaningful mechanism gain.
