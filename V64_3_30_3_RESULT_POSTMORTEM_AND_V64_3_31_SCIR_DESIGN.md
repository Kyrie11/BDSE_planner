# V64.3.30.3 result closure and V64.3.31 EAF-ICER-SCIR design

## Executive verdict

V64.3.30.3 is **engineering- and protocol-valid for algorithm attribution**. The corrected pointwise FBIC contract passes on both new untouched 500-scene blocks, capacity exposure is adequate, query accounting is exact, structural scenes are exact no-ops, and the official screen closes the pure-capacity branch as:

> `retained_capacity_is_not_the_reproducible_first_order_missing_mediator_under_current_frozen_consumer`

One packaging-only defect is present: the V30.3 launcher hashes `v64_3_30_3_stage_timing.tsv` and then appends its `TOTAL` row, so exactly that timing file mismatches the generated SHA256 manifest. All planner/config/metrics/rows/edges/screen artifacts match. This does not invalidate the scientific run. The delivery copy fixes the order: append `TOTAL` first, hash second.

The uploaded root code archive also omits historical `V64_SAQA_BCC_NEXT_COMMANDS.sh` while retaining a test that requires it. V31 restores a fail-closed compatibility entrypoint rather than silently synthesizing unavailable historical experiment semantics.

## 1. V30.3 engineering closure

Both blocks have 500 exact paired scenes. Each has 479 safe-domain scenes and 21 all-flagged structural scenes.

| Pointwise FBIC contract | A | B |
|---|---:|---:|
| safe expandable scenes | 420 | 417 |
| safe no-expansion scenes | 59 | 62 |
| applied given expandable | 100% | 100% |
| reason-6 exact no-op | 100% | 100% |
| structural reason-1 exact no-op | 100% | 100% |
| global accounting violations | 0 | 0 |

B16 atoms are never deleted, the retained ceiling is 24 only in the FBIC arm, and the already-queried bank is reused without any evidence/model query increase. The V30.2 TRAIN causal prerequisite remains valid: historical B16/V25 reproduces while unchanged B24 DRC fails 4/5 selected-path fold safety for algorithmic—not runtime—reasons.

## 2. What the independent capacity experiment actually says

The strongest result is not an endpoint score. It is the combination of **more visible evidence** with **worse direct conversion** on both independent blocks.

### 2.1 Evidence visibility increases strongly

| Diagnostic | A B16 | A B24 | Δ A | B B16 | B B24 | Δ B |
|---|---:|---:|---:|---:|---:|---:|
| selected decisive recall | 0.5501 | 0.7344 | +18.43pp | 0.5644 | 0.7486 | +18.42pp |
| effective selected decisive recall | 0.7772 | 0.9615 | +18.43pp | 0.7826 | 0.9668 | +18.42pp |
| selected interaction recall | 0.5000 | 0.7044 | +20.44pp | 0.5157 | 0.7239 | +20.82pp |
| effective interaction recall | 0.7440 | 0.9484 | +20.44pp | 0.7395 | 0.9476 | +20.82pp |
| soft-interaction recall | 0.6685 | 0.9325 | +26.40pp | 0.6703 | 0.9372 | +26.69pp |
| evidence certificate | 0.928 | 1.000 | +7.2pp | 0.940 | 0.996 | +5.6pp |
| budget-vs-full action match | 0.232 | 0.262 | +3.0pp | 0.266 | 0.282 | +1.6pp |

Therefore V30.3 independently rejects the simple explanation that the extra M24 atoms are inert or never expose decision-relevant evidence.

### 2.2 Support remains stable while incumbent-dominance/order degrades

| Direct boundary metric | A B16 | A B24 | B B16 | B B24 |
|---|---:|---:|---:|---:|
| support AUC | 0.8109 | 0.8080 | 0.8106 | 0.8174 |
| direct incumbent-dominance AUC | 0.7708 | 0.7259 | 0.7703 | 0.7470 |
| direct replacement precision | 57.89% | 42.22% | 62.96% | 53.25% |
| positive-opportunity capture | 32.74% | 23.17% | 33.55% | 27.15% |

This is the decisive localization. Anchor support does not collapse under the representation expansion. The deterioration occurs in the incumbent-contrastive decision semantics/order and the resulting extremal action conversion.

### 2.3 Gate decomposition makes the consumer failure explicit

A:

- B16: `168 opportunity -> 118 support -> 74 scalar -> 64 both -> 55 useful selected` = 32.74% capture.
- B24: `164 -> 114 -> 77 -> 69 -> 38` = 23.17% capture.

B:

- B16: `152 -> 99 -> 71 -> 61 -> 51` = 33.55% capture.
- B24: `151 -> 107 -> 78 -> 67 -> 41` = 27.15% capture.

The extra evidence does **not** uniformly destroy candidate prerequisites. In B, support/scalar-positive counts increase. Yet useful selected recovery decreases. This is an operator/selection conversion failure rather than an acquisition-recall failure.

### 2.4 Fixed-opportunity scene transitions are even stronger evidence

Using the B16 positive-opportunity set as a frozen denominator:

- A: `0->1 = 4`, `1->0 = 21`, `1->1 = 34`, `0->0 = 109`.
- B: `0->1 = 5`, `1->0 = 16`, `1->1 = 35`, `0->0 = 96`.

Thus capacity expansion loses many more previously correct captures than it newly recovers in **both** blocks. This cannot be explained by a changed opportunity denominator.

### 2.5 Total endpoint effects are split-unstable and must not rescue the mechanism

A capacity-induced action switches are strongly harmful overall: 116 changes, 45 beneficial vs 71 harmful, `sum(Regret_B16-Regret_B24) = -405,621`.

B switches happen to be net beneficial: 90 changes, 45 beneficial vs 44 harmful, sum `+437,189`.

The fixed-B16-opportunity paired effect has the same sign reversal (A `-380,095`, B `+109,885`). Therefore total endpoint sign is not a stable capacity mechanism. The cross-split invariant is the direct-boundary result: evidence visibility rises while dominance/precision/capture fall.

Representative direct failures include:

- A `767b99cfafdf558f`: direct 9 -> direct 4, selected teacher improvement `+0.00484 -> -3.38534`, regret effect `-67,803`.
- A `170f8feb6b8652eb`: direct 18 -> direct 28, `-0.92877 -> -2.76962`, regret effect `-55,393`.
- A `5089c443c218510a`: direct 24 -> direct 23, `+0.00073 -> -0.98930`.
- B `e43a73c1ff4e55b1`: direct 15 -> direct 6, `-0.00013 -> -1.11613`, regret effect `-22,320`.
- B `0205ad01b7b755b3`: direct 2 -> direct 10, `+0.98001 -> -0.00059`, regret effect `-19,612`.
- B `10d353b57e7754ba`: incumbent keep -> direct 5, selected teacher improvement `-1.11317`.

The complete 1000-scene attribution is delivered as `V64_3_30_3_SCENE_LEVEL_CAUSAL_AUDIT.csv`.

## 3. Formal paper-level / independent answer

This question can now be answered formally, with a precise scope:

> **Under the current frozen downstream consumer and the already-queried M=24 bank, simple retained-interface expansion from B16 to B24 is not a reproducible first-order missing mediator. Candidate-specific/decisive evidence is observably present and becomes substantially more visible, but the incumbent-replacement consumer converts it less reliably at the direct intervention boundary on both independent blocks.**

This does **not** claim that B16 never discards useful latent information, that capacity is universally irrelevant, or that another acquisition process could never matter. What is falsified is the first-order causal proposal “expose more of the same already-queried bank to the frozen downstream semantics and recovery will become reliably better.”

Therefore the previous development-level conclusion—“consumer/operator mismatch is more likely than B16 simply erasing the signal”—is now promoted to an independent conclusion for this frozen capacity intervention.

## 4. Mechanism chain after V30.3

The evidence chain can now be tightened to:

1. **Acquisition complexity is not the main lever.** Learned subset objectives, beam/swap/coreset and teacher-shaped acquisition repeatedly improved proxies without stable endpoint conversion.
2. **EAF/exact attribution is valuable instrumentation.** It exposes the complete action-local frontier, exact selected-evidence attribution, candidate, intervention and gate responsible for a decision change.
3. **Richer representation is not reliability geometry.** Attribution-spectrum/transition/local semantic geometry can be informative yet fragment or alias the selected tail.
4. **Proposal signal exists, but extremal reliability is unstable.** DRC can find useful direct replacements, while rare selected negatives dominate fresh risk.
5. **Post-hoc local/classifier confirmation does not solve the selected tail.** Semantic KNN and PTMC fail independently; classifier/threshold iteration is terminated.
6. **Global frontier fidelity is not decision sufficiency.** FCR improves its reconstruction objective but not recovery and can create new catastrophes.
7. **More evidence visibility is not decision sufficiency.** FBIC strongly raises decisive/interaction visibility and certificate coverage.
8. **The stable failure is conversion at the direct incumbent boundary.** Support remains stable, while incumbent-dominance AUC, direct precision and useful capture decrease under B24 on both fresh blocks.
9. **Decision sufficiency is therefore a joint property of evidence and the selected-action operator.** In an extremal recovery problem, the evidence representation cannot be called sufficient independently of the intervention-conditioned ordering/calibration rule that consumes it.

The paper mainline should now be expressed as:

> **Selection-Conditioned Intervention Sufficiency for Autonomous Planning under a Bounded Auditable Evidence Interface.**

A more implementation-continuous variant is:

> **Intervention-Conditioned Decision Sufficiency with Selection-Stable Extremal Recovery under a Bounded Auditable Planner Interface.**

The mechanism chain is:

`bounded auditable evidence interface`
`-> exact EAF action-local attribution`
`-> deployment-admissible incumbent/challenger intervention`
`-> evidence visibility / intervention-sufficiency diagnostic`
`-> same-scene incumbent-contrastive ordering`
`-> extremal-selection-aware reliability calibration`
`-> incumbent-default / monotone intervention containment`
`-> structural + catastrophic-tail safety contracts`
`-> double-fresh -> independent full-validation -> closed-loop validation`.

## 5. Tightest dominant bottleneck

The previous “downstream consumer/operator mismatch” can now be narrowed to:

> **selection-conditioned incumbent-contrastive ordering reliability at the direct replacement boundary, plus post-selection calibration of the extremal proposal.**

The model already sees candidate signal: support AUC is approximately 0.81 and remains stable when the interface is expanded. The weak link is mapping candidate-vs-incumbent evidence to a correctly ordered intervention *after* an argmax/extremal selection. This is a winner's-curse / selection-shift problem: small edge-level semantic errors become dominant because deployment chooses an extreme score rather than an average edge.

Current model state:

- mature/stable: bounded interface, queried-bank accounting, EAF complete instrumentation, deployment admissibility, structural delegation, exact attribution, incumbent-default/no-fallback structural ideas;
- useful but not sufficient: anchor support and candidate opportunity signal;
- current weakest component: incumbent-relative dominance/order and selected-proposal reliability;
- endpoint interpretation remains downstream of path composition and must not override a failed direct selected-path mechanism.

## 6. No-repeat set after V30.3

Continue to freeze all historical dead ends: acquisition/selector complexity, beam/swap/coreset, FCR/global reconstruction, broad B sweeps, same-bank rebinding, attribution-spectrum/transition/semantic-family/type KNN geometries, K/radius/OOD variants, DRC K/threshold/downside tuning, support/scalar threshold rescue, PTMC/classifier v2/v3, naive feature concatenation, action blacklist, failed-view AND stacking, and learned admissible-incumbent->anchor veto.

Add the following terminal constraints:

- do not treat B24 or any particular B as novelty;
- do not equate higher decisive recall/certificate rate with decision sufficiency;
- do not use the favorable B endpoint to pool/rescue the failed direct capacity mechanism;
- do not continue capacity-only or same-bank selector/rebinding work as the first-order solution;
- do not reuse the V30.3 1000 fresh scenes for V31 calibration or promotion;
- do not rebuild a generic shared listwise/BCE model identical to RAER/DALER/DACER history;
- do not rebuild V22 as a magnitude-weighted sign classifier after frozen scalar selection;
- if continuous incumbent-conditioned ordering itself fails fresh, do not rescue it by conformal alpha/threshold sweep.

## 7. V64.3.31: EAF-ICER-SCIR

SCIR = **Selection-Conditioned Intervention Recovery**.

### 7.1 What remains frozen

- configured planner evidence budget B=16;
- queried proposal bank M=24;
- acquisition/selector;
- EAF checkpoint/value arithmetic/exact attribution;
- anchor-support head;
- deployment-admissibility contract;
- evidence/model query count;
- evidence certificate / final execution guard;
- all-flagged structural delegation;
- no FBIC, FCR, DRC, PTMC or new KNN/classifier.

### 7.2 PRESERVE control

SCIR explicitly separates an earlier resolved branch. `PRESERVE` uses frozen V20 direct dominance semantics but preserves a deployment-admissible incumbent by default. This control isolates removal of the historically unstable learned incumbent->anchor veto from the new direct ordering mechanism.

Untouched A/B therefore run:

1. Raw;
2. frozen V20;
3. PRESERVE control;
4. SCIR-RANK;
5. SCIR-MAIN.

### 7.3 Same-scene continuous intervention semantics

For each support-positive, deployment-admissible alternative `b` and same-scene admissible incumbent `i`, SCIR uses the frozen 18-D evidence view as a **contrast**, not as an independent candidate descriptor:

`x(b,i) = [phi_18(b)-phi_18(i), support(b)-support(i)]` (19 dimensions).

A low-capacity TRAIN-only scene-equal linear ridge (`lambda=1`, frozen) predicts the continuous target

`Delta_T(b;i) = teacher_margin(b)-teacher_margin(i)`.

Every scene receives total training weight 1, preventing scenes with many alternatives from dominating the fit. Runtime has no teacher/future input and makes no extra evidence query.

The incumbent is an implicit semantic zero boundary. An alternative is proposed only if predicted improvement `mu(b;i)>0`; the rank arm chooses the largest `mu`, with frozen support/margin/utility/action deterministic tie breaks.

This specifically differs from earlier failed mechanisms:

- not anchor-relative shared RAER/DALER scoring;
- not generic listwise CE/BCE;
- not V22 magnitude-weighted sign veto behind frozen scalar dominance;
- not DRC local-neighbor certification;
- not PTMC catastrophic classification;
- not evidence rebinding/capacity change.

It directly replaces the semantic component V30.3 localized as unstable: candidate-vs-incumbent ordering under extremal selection.

### 7.4 TRAIN fail-close before spending new validation

Five deterministic scene-disjoint folds audit the *selected path*, not only edge AUC. Required before calibration/fresh:

- all 5 fold selected teacher-improvement sums >=0;
- aggregate selected proposals >=64;
- aggregate teacher-positive selected proposals >=32.

Failure stops V31 before selecting/calibrating fresh data. No ridge/feature/threshold sweep is allowed.

### 7.5 Independent CAL500 selected-path conformal certificate

After the TRAIN model/proposal operator is frozen, V31 selects 1500 new labels-unseen scenes by fixed hash after excluding all 10,700 spent design scenes and the frozen TRAIN population. It partitions them **before evaluation** as:

- CAL500: calibration/design only;
- fresh A500: untouched promotion block;
- fresh B500: untouched promotion block.

Run only SCIR-RANK on CAL500. For its selected proposals define residual

`r_j = mu_j - Delta_T,j`.

With fixed `alpha=0.05`, choose the standard one-sided finite-sample split-conformal order statistic

`q = max(0, r_(ceil((n+1)(1-alpha))))`.

The main lower bound is

`LCB(b*) = mu(b*) - q`.

SCIR-MAIN executes exactly the rank proposal `b*` only when `LCB(b*)>0`; otherwise it returns directly to the incumbent. It cannot re-rank or fall through to a second alternative.

**Theorem scope.** If the calibration and future selected-proposal population are exchangeable and the predictor/proposal rule was frozen before calibration, the one-sided split-conformal lower bound has marginal coverage at least `1-alpha` under the usual finite-sample rank construction. Since main executes only when `LCB>0`, any accepted harmful intervention with true improvement `<=0` is necessarily a lower-bound miscoverage event. This is a proposal-population marginal guarantee, not a per-scene or closed-loop absolute-safety theorem.

### 7.6 Double-fresh promotion gates

A and B are judged independently; no pooling.

Per block:

- exact five-arm identity, query parity and structural identity;
- SCIR rank proposal identical between rank/main, main accept is same proposal, veto returns incumbent/default, zero fallback violation;
- SCIR-RANK direct positive-opportunity capture >= PRESERVE +3pp;
- SCIR-MAIN capture >= PRESERVE;
- MAIN direct replacement precision >= PRESERVE;
- MAIN selected direct count >=8, teacher-improvement sum >=0, worst >-0.5, NegRMS <= PRESERVE;
- conformal MAIN does not worsen rank precision/NegRMS/worst;
- MAIN endpoint non-inferior to both PRESERVE and frozen V20 (match tolerance -0.2pp; regret <=1.005x).

Passing both blocks authorizes exactly one frozen independent full-validation reproduction. Failure does not authorize alpha, ridge, feature or threshold sweeps.

## 8. CCF-A assessment

No algorithmic analysis can guarantee CCF-A acceptance. V30.3 nevertheless materially improves the *scientific shape* of the work: it closes a meaningful causal ambiguity with double-fresh independent evidence and turns a broad “better evidence selector” story into a sharper intervention/selection reliability problem.

To reach a credible CCF-A standard, the eventual frozen mechanism still needs:

1. both fresh A/B mechanism blocks to pass independently;
2. exactly one independent full-validation reproduction after freeze;
3. nuPlan closed-loop evidence that the direct mechanism converts into planning quality;
4. strong current planning baselines under matched candidate/query/runtime conditions;
5. compact concept-level ablations (PRESERVE, RANK, MAIN), not threshold zoos;
6. budget sensitivity only *after* the mechanism is frozen, with B treated as robustness, not novelty;
7. latency/query/memory cost;
8. selected-tail and failure-mode analysis;
9. paired bootstrap/confidence intervals and preferably multiple seeds for endpoint claims;
10. exact code/provenance/claim alignment.

If SCIR succeeds, the most defensible novelty is not the linear ridge or conformal method alone. It is the formulation and operator decomposition:

> **evidence-attributed deployment-admissible incumbent-contrastive recovery in which decision sufficiency is defined at the selected intervention boundary, continuous same-scene ordering is separated from post-selection reliability calibration, and learned intervention is structurally contained by incumbent-default/no-fallback semantics under a bounded auditable evidence interface.**

If SCIR fails, the failure is also clean: it would falsify a much more specific hypothesis—same-scene continuous evidence semantics plus independent post-selection calibration—without reopening acquisition, B, KNN, classifier or global reconstruction loops.
