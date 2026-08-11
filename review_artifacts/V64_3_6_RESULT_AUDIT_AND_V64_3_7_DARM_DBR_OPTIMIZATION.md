# V64.3.6 Result Audit and V64.3.7 DARM+DBR Optimization

## Executive conclusion

The V64.3.6 `winner=null` is the correct outcome under the existing promotion rules. It does **not** mean every new mechanism was inert.

- **BCHA is ruled out.** Frozen-family-slot oracle Top-M recall = 1.0, global oracle Top-M recall = 1.0, gap = 0. Therefore family allocation is not the learned critical-admission ceiling on this subset.
- **LBPR has a weak positive value signal but is not promotable.** It trains, produces non-zero residuals, improves final pair-full teacher match from the same-epoch local pair-full control 0.174 to 0.176 (+0.2pp), lowers pair-full teacher regret from 11286.84 to 11030.27, has 0.2% beneficial and 0% harmful residual intervention, but final deployed teacher match remains 0.178 and the action-match delta is far below the +1pp value threshold.
- **CCBR/LEA solved representability, not admission or value.** Endpoint support is ~100%, but literal critical Top-M stays exactly 0.360153. The final LEA endpoint CE is still 2.940, only moderately below the ~3.30 uniform baseline for ~27 valid candidates, so using that posterior as an LBPR value gate is premature.
- **V64.3.6 contains an aggregation confound.** Its zero-residual `legacy_tournament` anchor is only ~0.18 teacher match / 0.172 pair-full, whereas the same V62 warm-start in V64.3.5 had a direct selected-local anchor of ~0.264 teacher match and ~0.264 local pair-full. Before concluding that pair residuals cannot help, the residual must be tested on the strongest fixed selected-local anchor.

The highest-priority next experiment is therefore **value/decision aggregation isolation**, not another acquisition architecture. Acquisition is still a real secondary bottleneck, but its remaining failure has already been localized to learned proposal-score generalization rather than family capacity or frontier support.

## V64.3.6 numerical audit

| Metric | LOCAL anchor | LOCAL final | LBPR anchor | LBPR final |
|---|---:|---:|---:|---:|
| teacher action match | 0.180 | 0.178 | 0.180 | 0.178 |
| pair-full interface match | 0.172 | 0.174 | 0.172 | **0.176** |
| local pair-full match | 0.172 | 0.174 | 0.172 | 0.174 |
| pair-full over local | 0.000 | 0.000 | 0.000 | **+0.002** |
| budget vs pair-full | 0.952 | 0.954 | 0.946 | 0.938 |
| literal critical Top-M | 0.360153 | 0.360153 | 0.360153 | 0.360153 |
| selected literal critical | 0.260536 | 0.245211 | 0.260536 | 0.252874 |
| proposal decisive recall | 0.791509 | 0.780862 | 0.791509 | 0.777787 |
| frozen family oracle Top-M | 1.000 | 1.000 | 1.000 | 1.000 |
| global oracle Top-M | 1.000 | 1.000 | 1.000 | 1.000 |

LBPR activation:

- max adapter parameter delta RMS: 0.00583805;
- max residual RMS: 0.03097836;
- final pair-full regret: 11030.27 vs same-run local-pair-full regret 11286.84;
- final beneficial residual intervention: 0.002;
- final harmful residual intervention: 0.000;
- endpoint representability: ~0.997--1.0;
- final endpoint attribution CE: 2.94009;
- literal atom-pair fraction: ~7.30e-5.

These numbers support the interpretation “LBPR learned a direction that sometimes helps, but its effect is too sparse/weak and is carried by a poor global aggregation anchor.” They do not support adding V64.3.6 LBPR as-is to the main algorithm.

## What enters the main algorithm from the V64.3.6 ablation family?

### Do not promote

1. **BCHA** — definitive negative; family oracle proves no admission ceiling.
2. **CCBR proposal residual / LEA proposal supervision** — representation-support mechanism is useful scientifically, but no validation Top-M gain after multiple clean screens; keep as negative/mechanism ablation, not runtime main path.
3. **LBPR endpoint gate** — endpoint posterior is not sharp enough to be a reliable multiplicative value gate.
4. **legacy global tournament aggregation** — weak zero-residual baseline creates a confounded evaluation of the new residual.

### Retain as a building block

The part of LBPR worth preserving is the **low-rank, evidence-attributable, exactly antisymmetric pair residual**. It is parameter-efficient, produces a small beneficial signal, and directly models the paper's atom contribution to a decisive action margin. V64.3.7 turns this building block into DBR and tests it without endpoint gating or global-pair aggregation.

## Current bottleneck ordering

### Priority 1: decisive pair value / final decision aggregation

The downstream ceiling is now experimentally real. Even pair-full evidence remains far from teacher. The immediate question is not whether to add more evidence; it is whether an auditable atom can produce the correct signed correction on the specific margin that could change the selected-local decision.

### Priority 2: learned proposal-score generalization

Critical Top-M remains ~36%. However:

- CCBR removed the FPCCA boundary-support ceiling;
- frozen family oracle = 1.0;
- global oracle = 1.0.

So the upstream residual failure is specifically a **learned score/generalization** problem under extremely rare literal-critical targets. It is not a reason to increase B/M, expand the family stack, or build a larger frontier. This should be revisited only after a stronger downstream margin model exists; then realized decisive-margin utility can become a better acquisition supervision target.

## V64.3.7 algorithm: DARM + DBR

### DARM: Decisive Anchor-Relative Margin Refinement

Let the fixed budgeted local cost be

`J_B^L(a) = J0(a) + sum_{i in S_B} g_i(a)`

and `a0 = argmin J_B^L(a)`.

DARM refines only margins incident to `a0`:

`M(a0,b) = J_B^L(b) - J_B^L(a0) + sum_{i in S_B} r_i(a0,b)`.

A non-anchor pair residual has no effect on the action. If a learned anchor edge is absent, the margin falls back to the local selected-evidence margin. At zero DBR residual, DARM is exactly the selected-local planner before the same runtime safety/utility post-processing.

This is intentionally a one-sided decision correction rather than a globally reconstructed pair field. It directly mirrors the paper guarantee: preserve the decisive winner-vs-rival margins needed for the final action.

### DBR: Decisive Boundary Residual

DBR removes LBPR's endpoint gate and predicts the evidence-attributable correction for the **runtime pair itself**. It is low-rank and bias-free, and uses action/query difference factors so

`r_i(a,b) = -r_i(b,a)`

holds after training. The output is zero initialized.

Two screens are intentionally small:

- **BROAD:** normal decision-weighted winner/hard/near pair support;
- **LITERAL:** BROAD plus a bounded exact winner -> leave-one-atom-out flip pair quota and corresponding literal atom upweight.

This directly tests whether exact literal-boundary emphasis improves pair value once the immature endpoint posterior is no longer a gate.

## Why V62 warm start remains correct

Use the same checkpoint:

`outputs_v62_dcab_ewfc_fast_2gpu_v1/train/bdse_v62_dcab_ewfc.best.pt`

The question is causal: can DARM+DBR improve a fixed representation and a strong selected-local anchor? Full scratch would simultaneously change J0, action/evidence embeddings, proposal ranking and local value, destroying attribution. Scratch or broad unfreezing is not justified yet.

A representation-capacity experiment is allowed only if all of the following are true in V64.3.7:

1. strong epoch -1 selected-local anchor is restored;
2. DBR parameter movement and residual RMS are non-zero;
3. anchor-pair coverage is adequate;
4. pair-full fails to improve.

Then the next experiment should be a **small selective pair-feature/action-evidence adapter**, not full scratch training.

## V64.3.7 promotion rules

The screen checker first requires `anchor teacher match >= 0.24`. This detects accidental regression to the weak V64.3.6 aggregation path.

Mechanism value signal:

- pair-full gain >= +1.0pp vs epoch -1;
- pair-full - local pair-full >= +0.5pp at the same epoch;
- teacher action match delta >= -0.5pp;
- budget-vs-pair-full delta >= -2pp;
- beneficial minus harmful residual intervention >= 0.

Full promotion additionally requires deployed teacher action match >= +0.5pp vs anchor. Thus a pure pair-full diagnostic gain cannot automatically trigger expensive full training.

## Engineering audit

The implementation adds explicit tests/contracts for:

- exact DBR antisymmetry after non-zero weights;
- zero-residual DARM exact selected-local no-op;
- non-anchor residual edges cannot alter action;
- anchor margin crossing can alter action;
- differentiable DARM and runtime DARM semantics agree;
- duplicate directed anchor-pair observations are averaged rather than double-counted;
- safety atoms are included consistently in pair regression and pair-action loss;
- V62 warm start may omit only the new DBR head, while missing core foundation tensors remain fatal;
- V64.3.7 screen/full provenance flags match metadata;
- V64.3.7 is included in strict cross-config signature checking;
- full training remains blocked unless a screen report has a non-null winner.

Regression status: **299 tests collected after final additions; all executed test groups pass with 0 failures**. Six V64.3.7 YAMLs parse, four launchers pass `bash -n`, broad/literal screen+full config contracts pass, and Python `compileall` passes. The delivery environment did not execute nuPlan GPU training, so no V64.3.7 performance claim is made here.

## Efficiency

The V64.3.6 uploaded logs show pair sampling is already solved (~1.5--2.3 s/epoch). The costly regions are data wait and LBPR/CCBR loss construction. V64.3.7 removes CCBR/LEA/BCHA from the trainable value screen and trains only a ~35k-parameter DBR. Runtime aggregation is anchor-star O(K) after existing sparse pair scoring; it does not construct a new O(K^2) global tournament or enlarge B/M. No supervision quota is removed merely for speed.
