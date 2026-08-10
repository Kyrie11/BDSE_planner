# V64.3.3 Results Audit and V64.3.4 FPCCA+LBA Optimization

## 1. Executive conclusion

The uploaded V64.3.3 corrected experiments are now strong enough to reject both **AP-WCCA** and **AP-WRCCA** as the next main acquisition algorithm. The remaining `acra_wired=false` flag is a logging defect, not a loss-routing defect: ACRA is included in `L_exact_winner_flip_critical_proposal`, but V64.3.3 omitted the standalone ACRA value from the returned diagnostic dictionary. The adapters moved substantially and their forward residuals are large, yet literal critical Top-M recall is exactly flat.

No uploaded V64.3.3 ablation should therefore be folded directly into the main algorithm. The components that should remain are the immutable HAB anchor, fixed `B=16`, auditable evidence atoms, literal removal-induced winner-flip criticality, DA-EPC, strict provenance, and sparse exact-selector supervision. **LCV has not yet been run in the uploaded outputs and cannot be claimed positive.**

The most valuable next algorithmic move is **Frontier-Pair Conditioned Critical Acquisition (FPCCA)** with an ablation for **Literal Boundary Attribution (LBA)**. This directly addresses a semantic mismatch exposed by the current results: AP-WCCA/AP-WRCCA condition acquisition around the frozen foundation winner, while the matched foundation agrees with the teacher action on only `0.227` of formal scenes. The acquisition residual is therefore centered on the wrong decision boundary in most scenes.

## 2. Alignment with the paper and novelty

The paper's core idea is not generic evidence compression. It defines auditable evidence atoms and seeks a fixed-budget subset that preserves **decision-sufficient teacher action margins** over a finite candidate bank. The theoretical/algorithmic framing is inherently pairwise: decisive rival pairs determine whether removing or retaining evidence can change the winner.

FPCCA is a natural extension of that framing rather than a departure from it. It keeps the entire bounded interface unchanged but replaces the assumption that one base top-1 action is the universal acquisition anchor. The proposal residual sees a small deployment-available frontier of action-pair boundary tokens. LBA then uses teacher information only during training to identify which represented boundary a *literal* winner-flip atom belongs to. Criticality itself remains exactly literal removal-induced winner flip.

This preserves the strongest novelty axis for a CCF-A submission:

- **auditable atom semantics**, not latent-token evidence;
- **fixed planner-interface evidence budget** `B=16`;
- **literal causal criticality** by leave-one-atom-out winner flip;
- **decision-boundary-conditioned sparse acquisition**, with no teacher at deployment;
- **exact downstream decision certificate** retained through DA-EPC.

## 3. V64.3.3 screen audit

### AP-WCCA

- adapter parameter delta RMS max: `0.0051577`
- residual RMS max: `0.96157`
- literal critical Top-M micro: `0.360153 -> 0.360153` (`+0.000000`)
- literal critical selected micro: `0.260536 -> 0.252874` (`-0.007663`)
- proposal decisive recall: `0.791509 -> 0.767043` (`-0.024466`)
- last teacher action match: `0.258`

### AP-WRCCA

- adapter parameter delta RMS max: `0.0048339`
- residual RMS max: `1.02724`
- literal critical Top-M micro: `0.360153 -> 0.360153` (`+0.000000`)
- literal critical selected micro: `0.260536 -> 0.256705` (`-0.003831`)
- proposal decisive recall: `0.791509 -> 0.773034` (`-0.018475`)
- last teacher action match: `0.260`

The residual is not under-trained: it is large enough to noticeably change broad proposal ranking. The failure is that those changes do not target literal critical atoms. The strongest-single-rival extension reduces broad-recall damage slightly but still gives zero critical Top-M gain.

## 4. Formal open-loop audit

On the uploaded 1000-scene full-support evaluations:

| Metric | AP-WCCA | AP-WRCCA |
|---|---:|---:|
| candidate teacher action match | 0.224 | 0.224 |
| matched foundation teacher action match | 0.227 | 0.227 |
| pair-full interface teacher action match | 0.236 | 0.236 |
| proposal decisive recall | 0.78410 | 0.78052 |
| literal critical Top-M recall | 0.35476 | 0.35476 |
| literal critical selected recall | 0.28048 | 0.28201 |
| dense -> HAB Top-M dense-value winner preservation | 0.970 | 0.968 |
| dense -> selected B16 dense-value winner preservation | 0.936 | 0.934 |
| budget vs pair-full winner preservation | 0.899 | 0.902 |
| DA-EPC evidence certificate fraction | 0.899 | 0.902 |
| p95 planner latency (ms) | 557.73 | 557.83 |
| deployed residual flips | 0 | 0 |

The candidate actually loses `0.003` teacher-action match versus its matched foundation control. Pair-full only reaches `0.236`, so the system has a **dual bottleneck**:

1. **Immediate bottleneck: acquisition representation/conditioning.** Only ~35.5% of literal critical evidence reaches Top-M.
2. **Secondary bottleneck: atom-to-action / pair-value representation.** Even the pair-full interface barely improves over foundation.

Causal priority still requires fixing acquisition first. If FPCCA raises critical Top-M and selected recall while pair-full/candidate teacher match stays flat, the next version should move directly to atom-to-action pair-margin value modeling instead of adding further acquisition losses.

## 5. Why selector/B/M/certificate should not be the next target

The current numbers do not support increasing evidence budget or relaxing the certificate. Dense-value action preservation remains `~0.97` through Top-M and `~0.934-0.936` through B16; budget-vs-pair-full/certificate is `~0.90`. Literal critical acquisition, by contrast, is only `0.355` before the selector. This is too large an upstream loss to justify hiding it by increasing `M` or `B`.

Selected literal critical recall (~0.28) means the selector is not perfect, but it is not the first causal failure. Revisit B16 allocation only after acquisition improves and the selected-recall gap becomes the dominant loss.

## 6. New algorithm: FPCCA

### 6.1 Frontier-Pair Conditioned Critical Acquisition

FPCCA keeps the legacy HAB proposal as an immutable anchor and adds a zero-initialized residual adapter.

For each scene:

1. obtain the top-`F` valid actions under frozen deployment-available base cost `J0`;
2. construct all unordered pairs in that action frontier (`F=6` gives 15 pairs);
3. encode each pair using both action embeddings, relative/product interaction features, and normalized base-cost gap;
4. let each evidence atom attend to the set of pair-boundary tokens;
5. predict an atom-specific zero-initialized residual proposal score.

Unlike AP-WCCA/AP-WRCCA, no single base winner must be semantically correct for the representation to expose the relevant boundary.

Default `F=6` is deliberately small. A prepared `F=8` screen is allowed only when the new literal-boundary representability diagnostic says top-6 support is inadequate.

### 6.2 Literal Boundary Attribution (LBA)

For a literal critical atom, exact leave-one-out evaluation gives:

- teacher winner with all evidence;
- winner after removing that atom.

If that unordered action pair exists in the base frontier, LBA supervises the atom's pair-attention distribution toward that boundary. Non-critical atoms receive no LBA target. Therefore LBA **does not redefine criticality** and is not another soft-margin proxy.

Main weight: `boundary_attribution_weight=0.25`, intentionally conservative so LBA augments rather than overwhelms the established exact-critical objective.

## 7. Next experiment matrix

Run only short same-subset 2-GPU screens first:

1. **AP-WRCCA + LCV** — finish the previously unrun target-granularity hypothesis.
2. **FPCCA without LBA** — isolate the representation change.
3. **FPCCA + LBA** — test whether exact boundary identity adds value beyond representation.
4. **FPCCA-F8 + LBA** — only if no candidate passes and anchor top-6 literal-boundary representability is below `0.70`.

A candidate is eligible for promotion only if all instrumentation is valid and:

- literal critical Top-M micro delta `>= +0.01` absolute;
- selected literal critical delta `>= -0.005`;
- broad proposal decisive delta `>= -0.02`;
- teacher action-match delta `>= -0.005`.

The comparator ranks eligible candidates by critical Top-M gain first. Only one winner proceeds to the 50k/8-epoch full pipeline.

This is intentionally stricter than “delta > 0”: a sub-percentage-point fluctuation is not enough evidence for a CCF-A contribution.

## 8. Closed-loop interpretation

The uploaded full runs cannot provide a valid formal closed-loop comparison because `dense_runtime_query_decision_match=0.997 < 0.999` blocks the protocol. Re-auditing the three AP-WCCA mismatched scenes found raw dense/runtime query features exactly equal, while neural score differences are <`0.001`; the action differences are near-tie batch-shape numerical sensitivity, not cache/provenance drift.

Do **not** lower the protocol threshold. First choose the acquisition winner. If 0.997 repeats, isolate strict matmul / near-tie numerical reproducibility as a separate engineering experiment. Mixing that change into the acquisition ablation would damage causal attribution.

Once protocol passes, run diagnostic CL20 first. If open-loop acquisition/action metrics improve but CL20 does not, the next research bottleneck is candidate dynamics, interactive prediction and replanning—not B/M/certificate.

## 9. Efficiency audit and safe optimizations

Uploaded V64.3.3 average epoch timing:

| Component | AP-WCCA | AP-WRCCA |
|---|---:|---:|
| epoch wall time | 1564.2 s | 1553.6 s |
| samples/s | 32.08 | 32.36 |
| data wait | 283.75 s | 265.18 s |
| pair sampling | 251.13 s | 250.41 s |
| forward | 58.91 s | 59.58 s |
| loss construction | 394.06 s | 393.50 s |
| backward/step | 42.63 s | 42.27 s |
| exact selector | 0.10 s | 0.10 s |

Thus exact selector is no longer a meaningful runtime bottleneck. V64.3.4 makes only semantics-preserving speed changes:

- vectorized boundary-focused pair sampling with batched quota-preserving top-k, removing per-row CUDA `.item()/nonzero/topk` synchronizations;
- `torch.topk` for frontier extraction rather than sorting all K actions;
- vectorized pair construction with `torch.triu_indices`;
- keep two-GPU DDP, `12` train workers/GPU and `4` val workers/GPU;
- preserve exact-selector cadence, all pair quotas, losses, evidence budget and training labels.

A regression test compares the vectorized sampler's selected pair tensors exactly against the V64.3.3 reference rule. Local CPU microbenchmark is ~1.2 ms/call for B=16/P=112, but server CUDA speedup is **not claimed until measured in the next run**.

## 10. Engineering corrections included

- restored standalone `L_critical_adapter_residual_alignment` logging, eliminating the false ACRA screen failure;
- added FPCCA/LBA activation diagnostics and frontier representability metrics;
- strict train/eval critical-adapter signature binding now covers conditioning, rank/scale, frontier count/size and gap-bias configuration;
- AP-WRCCA+LCV now has V64.3.4-specific candidate/control configs so train/eval provenance versions no longer silently differ;
- generic pipeline training contract reads the current YAML algorithm version instead of a stale hardcoded version;
- backward-compatible model context reads avoid breaking older mocked/test paths;
- vectorized pair sampler regression protection added.

## 11. No-repeat constraints

Do not repeat the previously negative directions:

- AP-WCCA/AP-WRCCA binary as candidate main algorithms;
- full proposal/family unfreezing;
- brute-force BCC/HCBE/hardest-negative weight escalation;
- V40-V43 beam/swap/repair;
- increasing B/M to mask acquisition failure;
- certificate relaxation or hand-lowering residual conformal epsilon;
- nominal 6-D query extension or opaque runtime base prior;
- expanding FPCCA frontier when F6 boundary support is already adequate.

LCV remains an untested ablation and should be treated as such until its screen runs.

## 12. Decision tree after the next screens

- **FPCCA representation-only passes, LBA adds little:** promote FPCCA; keep LBA as negative/neutral ablation.
- **FPCCA+LBA clearly beats representation-only:** promote FPCCA+LBA and use literal boundary attribution as the main new acquisition contribution.
- **AP-WRCCA+LCV passes while FPCCA does not:** target granularity, not pair representation, was the key issue; promote LCV conditionally.
- **F6 fails with low boundary representability and F8 passes:** support size was limiting; report F sensitivity explicitly.
- **all clean screens fail despite high boundary representability:** acquisition residualization over frozen HAB is likely exhausted; next redesign should be a stronger pair-conditioned proposal representation rather than more loss weights.
- **critical Top-M/selected improve but action metrics stay flat:** immediately shift to atom-to-action/pair-margin value representation.
- **open-loop improves but CL20 stays flat:** shift to interactive candidate dynamics / prediction / replanning.
