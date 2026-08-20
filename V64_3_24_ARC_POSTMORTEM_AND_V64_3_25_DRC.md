# V64.3.24 ARC uploaded-result postmortem and V64.3.25 EAF-ICER-DRC design

## Executive conclusion

The uploaded V64.3.24 run is **scientifically valid but intentionally incomplete beyond TRAIN**. It correctly stopped at the pre-registered TRAIN ARC gate before selecting any fresh validation token. This is not an engineering false STOP: independent reproduction from the 747,232,170-byte TRAIN frontier gives the same decisive result. `aggregate-downside` is the only 5/5 fixed-fold selected-path-safe branch (`71` replacements, `+5.527642` teacher-improvement), while the V24 main `attribution-downside` is only 3/5 (`84`, `+4.526551`; folds 2 and 4 fail).

Therefore the right next action is not to repair the gate or tune the attribution representation. The right action is the branch that V24 itself pre-registered: **drop the full attribution spectrum from the main risk geometry, retain the aggregate evidence-local downside certificate, and run it against the aggregate mean-SE control on two new untouched 500-scene blocks.**

## 1. Result completeness and execution validity

The output archive has eight files: prerequisite re-audit, targeted-test log, TRAIN replay log/metrics/frontier, ARC fitter log, and stage timing. The stage chain reaches only:

`prerequisites -> frozen 3000-scene TRAIN full-attribution replay -> ARC TRAIN fit -> STOP`.

There is no `fresh_1000` manifest, no A/B token file, no fresh per-sample row file, no fresh frontier, and no split/double-fresh checker output. The fitter log contains the explicit fail-closed message that attribution-resolved downside was not path-safe in all fixed folds. The launcher uses `set -e`, so this non-zero fitter exit prevented fresh selection exactly as intended.

The TRAIN frontier is complete: 75,133 rows, 3,000 unique scenes, no missing full-attribution features in the eligible replacement population. The frozen replacement population is 1,455 alternatives from 310 scenes. Its 62.13% positive sign rate but negative overall mean (-0.03614) re-confirms the heavy-tail motivation inherited from V23.

A small engineering defect made the STOP less auditable than it should have been: V24 raised before writing `v64_3_24_train_arc_fit.json` and the TRAIN token manifest. Because the exact fit is reproducible from the complete frontier, this is a diagnostics/provenance bug rather than a scientific execution error. The V25 delivery fixes that lifecycle and also patches the historical V24 fitter diagnostically.

## 2. Exact fixed-fold causal attribution

| variant | safe folds | selected | selected improvement sum |
|---|---:|---:|---:|
| aggregate mean-SE | 4/5 | 111 | +9.096630 |
| **aggregate downside** | **5/5** | **71** | **+5.527642** |
| attribution mean-SE | 3/5 | 119 | +6.803271 |
| attribution downside | **3/5** | 84 | +4.526551 |

This orthogonal table answers both V24 causal questions without a fresh-data rescue.

First, **mean confidence is not enough**. With the same conservative 18-D aggregate evidence geometry, replacing mean-SE by downside RMS repairs the single unsafe aggregate fold. Aggregate mean-SE fold 2 has sum -0.6953 and a selected outcome as low as -1.7026; aggregate downside fold 2 becomes +0.9949 with only tiny residual negatives. This supports the V23 hypothesis that rare negative magnitude, not average correctness, must enter the certificate directly.

Second, **the current full attribution spectrum does not resolve hidden modes**. It makes the outcome worse: attribution mean-SE and attribution downside are each only 3/5. Thus the current full spectrum is not a novelty to salvage through metric tuning.

## 3. Why the attribution spectrum is harmful

The strongest evidence is same-edge counterfactual scoring. Four catastrophic replacements selected by attribution downside have true normalized teacher improvements around -0.93 to -0.99, yet positive attribution-downside scores. The same exact candidate edges have negative aggregate-downside scores and would be rejected.

| scene | action | true improvement | attribution score | aggregate score |
|---|---:|---:|---:|---:|
| 67a57ae417045162 | 2 | -0.990634 | +0.028570 | -0.070323 |
| c34206b68ee6576f | 21 | -0.989780 | +0.052723 | -0.128165 |
| 441987f47a6f5784 | 2 | -0.928699 | +0.036416 | -0.175912 |
| 479da12f5f165e29 | 5 | -0.927054 | +0.061433 | -1.060837 |

At exact token+action identity, the 31 replacements common to both downside arms are healthy (sum +5.1491, 74.2% positive, worst -0.00383). The 40 aggregate-only edges are slightly net positive (+0.3785). The **53 attribution-only edges are net harmful (-0.6225), only 54.7% positive, and include the -0.9906 disaster**. This is direct causal evidence that the representation intervention creates the harmful population.

The geometry is consistent with that behavior. The candidate-minus-incumbent 16-D spectrum has mean absolute inter-dimension correlation ~0.970; its first standardized principal direction explains 97.16% of variance. The candidate spectrum is also redundant (top-1 62.89%, top-3 89.36%). Yet V24 z-scores each dimension and lets each 16-D group contribute one complete group of distance. Because both spectra are independently abs-sorted and L1-normalized, the metric loses semantic atom identity, candidate-incumbent atom correspondence, and scale while amplifying residual shape noise.

The correct bottleneck is therefore **representation-induced risk-neighborhood fragmentation**, not “we still need more attribution capacity.” The simple aggregate geometry actually exposes negative modes that the full-spectrum geometry hides.

## 4. Relation to the paper and main story

The uploaded TeX still presents the earlier BDSE formulation: fixed budgeted decision-sufficient evidence, candidate-set teacher, evidence atoms, hierarchical budgeted selection, risk-aware tournament, and a decision-preservation theorem. That conceptual root remains useful, especially the fixed auditable evidence interface and decision preservation motivation, but the implementation has evolved substantially beyond the paper text.

The current code-defined scientific line is now:

`fixed B<=16 evidence interface -> auditable selected evidence / exact EAF attribution -> frozen complete DARM-anchor value frontier -> final-guard-admissible alternatives -> frozen support + scalar incumbent dominance -> local selected-path regret certificate -> incumbent-default extremal replacement -> unchanged evidence/structural/final guards -> preservation + endpoint`.

V24 says to **tighten** the claim. The paper should not say the certificate is “attribution-resolved.” Exact attribution remains an upstream EAF/audit mechanism, but the reliable certificate candidate is the frozen 18-D aggregate evidence-local representation.

A defensible candidate headline, conditional on future validation, is:

> **Evidence-attributed incumbent-contrastive downside-regret certification for deployment-admissible extremal recovery under a fixed planner-interface evidence budget.**

The novelty is not “a KNN risk score.” It is the alignment of the learned certificate with the actual extremal deployment path: replacement is incumbent-relative, restricted to the complete deployment-admissible frontier, asymmetric (admissible incumbent is default-preserved), and certified against downside magnitude rather than average edge correctness. This story is coherent with a CCF-A target, but current evidence is not yet enough to claim CCF-A-level contribution: no V25 fresh/full-val/closed-loop evidence exists yet.

## 5. V64.3.25 mechanism

V64.3.25 is named **EAF-ICER-DRC (Downside Regret Certification)**. It deliberately makes no new representation change after observing V24.

Main: 18-D aggregate evidence-local KNN neighborhoods, fixed K=32/64, inverse-distance weighting, per-scale score `mu - RMS(min(delta,0))`, minimum over scales, zero boundary. Only final-guard-admissible, support-positive, scalar-dominance-positive alternatives are eligible. Accepted candidates are ranked by the frozen scalar dominance; the admissible incumbent remains the default action.

Control: identical population, geometry, K, ranking, and operator, but V23-style `mu - SE`. This makes the fresh experiment a direct test of downside sensitivity.

The fitter is memory-efficient and streams only the fields needed for the 18-D certificate. It reuses the already-computed V24 TRAIN frontier if available. Running the new fitter on the uploaded frontier reproduces 3,000 scenes / 75,133 rows / 1,455 replacement edges and the exact 5/5 aggregate-downside gate.

## 6. Next experiment and decision tree

V24 consumed no fresh scenes, so the design exclusion remains 5,700 previously inspected validation tokens. V25 selects 1,000 new untouched tokens with a new hash seed, splits A/B 500/500, and runs only four arms per block: raw, frozen V20, aggregate mean-SE, aggregate downside.

Both blocks must independently establish the entire causal chain. In addition to the existing instrumentation/frontier/support/recovery/preservation/endpoint gates, V25 explicitly reports the selected replacement tail: actual positive regret RMS / worst regret increase and normalized teacher-improvement negative RMS / worst outcome. Downside must add path-level value over mean-SE; an AUC-only improvement is not sufficient.

Interpretation is pre-registered:

- **A+B pass:** freeze V25 and run one independent full-validation reproduction only.
- **path-safe but no increment over mean-SE:** do not claim DRC novelty; do not tune the downside multiplier.
- **fresh catastrophic tail remains:** only then test whether aggregate evidence misses a semantic latent state. The next representation should preserve atom family/type/identity correspondence; do not return to sorted normalized spectra.
- **recovery becomes too conservative:** audit support/capture and candidate coverage; do not lower zero/K on fresh.
- **one block fails:** stop; no pooled rescue.

## 7. Engineering validation

The delivered repository passes Python compileall and launcher `bash -n`. V25 unit tests are 5/5, the targeted V64.3.13-V64.3.25 stack is 82/82, and the full repository is **412/412 PASS** with the same 36 existing Transformer warnings. The real uploaded 747MB frontier passes end-to-end V25 fit and both config/memory contracts. The fitter's measured peak RSS for this artifact is approximately 444 MB.

See `V64_3_24_UPLOADED_RESULT_CAUSAL_AUDIT.json` for machine-readable fold/selection/geometry details and `V64_3_25_ENGINEERING_VALIDATION.txt` for the engineering record.
