# V64.3.39 result postmortem and V64.3.40 SDFR design

## Executive decision

The uploaded V64.3.39 code is byte-identical to the preregistered package (SHA256 `73debeb3b9348309b78fe7195fad478818b7b650c5723d91e5a10148331c0edb`). The run completed all five nested outer folds, all inner OOF residual populations, and all independent value-calibration folds. Server targeted regression is **184/184 PASS**; the TRAIN scene audit contains **782/782 unique direct scenes**. The frozen-RSMR action identity/monotone-subset contract is valid. V39 stopped at its preregistered TRAIN scientific gate and created no non-empty CAL500/A500/B500 manifests. Therefore V39 is engineering-valid for TRAIN-level algorithm attribution and no V39.1 hotfix is required.

V39 is **not promotable**, but it is a meaningful partial mechanism success. `CFSR-MAIN` halves catastrophes (`28 -> 14`), reduces no-positive-opportunity false interventions (`107 -> 67`), improves NegRMS (`0.3557 -> 0.3073`), preserves nonnegative selected sum on all five folds, and slightly improves aggregate selected teacher sum (`+43.2941 -> +43.5752`). However useful positive-opportunity capture falls from `38.50%` to `25.96%`, far below the preregistered floor `35.50%`. The failure is therefore **zero-crossing/capture**, not absence of selected-tail signal.

## 1. Reliability/completeness audit

- Code ZIP SHA256: `73debeb3b9348309b78fe7195fad478818b7b650c5723d91e5a10148331c0edb`.
- Result ZIP SHA256: `d80f30706e7f676a96d1f61b37a6497230e314491092ffc9d26331b4e3c74a6a`.
- Server targeted regression: **184/184 PASS** (two historical transformer warnings only).
- TRAIN audit: **782 rows / 782 unique scene tokens**, all five outer test folds present.
- Selected-policy calibration proposal counts: `97 / 100 / 98 / 86 / 110`, all >=64.
- Inner OOF selected residual populations: `347 / 318 / 266 / 260 / 285`, all >=192.
- For every inner holdout, the RSMR/DENSE models producing the proposal/residual are fitted on the other two outer-fit folds; held-out scenes are not used to fit those models.
- Final CFSR residual model uses TRAIN fit folds only; outer-test scenes remain untouched.
- Residual heads are orthogonal to the removed RSMR/DENSE span (maximum absolute cosine around numerical zero in all folds).
- DENSE/DENSE-SHIFT/CFSR-RAW/CFSR-MAIN can only accept the exact frozen RSMR winner or return incumbent. No reranking, second-best, or proposal creation is possible.
- Stage timing terminates at `train_nested_cfsr_gate`; no CAL500/A500/B500 token files were generated.

Conclusion: **PASS for TRAIN-level causal/mechanistic attribution.**

## 2. V39 preregistered result

| TRAIN cross-fit | RSMR | DENSE-SHIFT | CFSR-RAW | CFSR-MAIN |
|---|---:|---:|---:|---:|
| selected | 502 | 376 | 316 | 354 |
| positive | 221 | 175 | 143 | 149 |
| useful capture | **38.50%** | 30.49% | 24.91% | **25.96%** |
| selected teacher sum | +43.2941 | +28.3339 | +41.8810 | **+43.5752** |
| catastrophe <= -0.5 | 28 | 21 | 15 | **14** |
| no-op false intervention | 107 | 80 | **55** | 67 |
| NegRMS | 0.3557 | 0.3702 | **0.2872** | 0.3073 |
| 5/5 fold selected sum >=0 | yes | no | **yes** | **yes** |

The main preregistered gate requires all of:

- >=20% no-op false reduction vs RSMR: **PASS** (`107 -> 67`, 37.4% reduction).
- useful capture >= RSMR - 3 pp: **FAIL** (`25.96% < 35.50%`).
- >=25% catastrophe reduction: **PASS** (`28 -> 14`, 50% reduction).
- NegRMS not worse: **PASS**.
- aggregate sum nonnegative: **PASS**.
- 5/5 outer-test sums nonnegative: **PASS**.
- selected >=64 and positive >=32: **PASS**.
- exact frozen-winner containment: **PASS**.

So V39 is a **scientific promotion failure with a real selected-tail mechanism gain**.

## 3. What DENSE-SHIFT says

`DENSE-SHIFT` fails the same capture/tail contract and even has a negative selected-sum outer fold. Therefore a scalar unit-slope selected-policy zero-point translation is not sufficient. V38's ordinary edge cardinal signal cannot be repaired by a single selected-policy offset.

This closes the minimal explanation:

> population-edge cardinal value is correct up to only a selected-policy intercept shift.

## 4. What CFSR says

CFSR's inner OOF residual fit is not empty or purely overfit. In every outer fit, honest inner-OOF dense residual MSE decreases after the feature-dependent correction:

- fold 0: `0.6929 -> 0.5678`
- fold 1: `0.7144 -> 0.5912`
- fold 2: `0.5833 -> 0.4852`
- fold 3: `0.6002 -> 0.4215`
- fold 4: `0.4894 -> 0.4310`

Outer-test behavior also shows consistent tail signal: CFSR-RAW reaches 15 catastrophes and NegRMS 0.2872; CFSR-MAIN reaches 14 catastrophes with all five fold sums nonnegative.

The most informative transition is on the 502 frozen RSMR proposals. CFSR-MAIN deletes 148 proposals, but the **true teacher-improvement sum of those deleted proposals is only -0.2811**. In other words, it almost perfectly preserves aggregate value while removing 14/28 catastrophes. This is qualitatively more selective than the blanket-abstention behavior seen in V36.

Therefore the V39 hypothesis must be split:

1. **Feature-dependent operator-conditioned selection residual exists:** supported.
2. **A linear signed-mean residual on the current 19-D evidence is sufficient to solve the incumbent-exit zero crossing:** falsified.

## 5. Why capture still fails

CFSR-MAIN deletes 72 true-positive RSMR proposals. But the count-level failure is strongly concentrated near the zero boundary:

- 47/72 have true improvement <=0.001, total teacher gain only `+0.0133`.
- 56/72 have true improvement <=0.01, total only `+0.0383`.
- 58/72 have true improvement <=0.2, total only `+0.2849`.

Thus a squared-error/signed-mean objective can achieve excellent aggregate value and tail behavior while still failing the promotion capture metric, because a tiny positive and a large positive are very different in MSE but equally count as one captured positive opportunity.

A small number of material false abstentions remain, so this is not only a metric artifact. Representative missed high-value RSMR winners include:

- `de3c0c52e7815332`: true `+3.1496`, CFSR-MAIN value `-2.9823`.
- `bbc08519add25675`: true `+2.2840`, value `-0.0306`.
- `c4b429aa06c65c18`: true `+2.0200`, value `-0.7538`.
- `3a223e65af13545c`: true `+1.6617`, value `-0.6497`.

CFSR also still accepts 14 catastrophes, including `2b32a9f406845f75` (true `-3.8065`, selected value `+0.0006`) and `c70954fab4a650c7` (true `-1.2330`, value `+1.6182`). So zero-boundary/tail identification is still not mature.

## 6. Evidence chain V32.1 -> V39

- **V32.1:** corrected scene-equal pointwise mean has real intervention signal, but using it as an extremal selector creates no-op errors and heavy selected tail.
- **V33:** explicit incumbent/null action is structurally necessary; all-rivals pair averaging suppresses no-op errors but destroys opportunity capture through abstention/dilution.
- **V34:** scene-level regret alignment produces the strongest challenger ordering signal to date: 5/5 positive fold sums, aggregate `+43.29`, capture `38.50%`; however incumbent-exit/no-op discrimination remains weak.
- **V35:** loss factorization alone fails; absolute incumbent basepoint has weak consistent signal but is not a first-order mediator.
- **V36:** frozen-order basepoint/selection-geometry reservation reduces tail mainly by blanket abstention; neither is the dominant solution.
- **V37:** post-selection value is the correct unresolved layer; sparse selected-only 18-D residual has tail signal but unstable cardinal generalization.
- **V38:** dense all-edge supervision proves the same 19-D evidence contains ordinary edge cardinal sign information (positive AUC/sign accuracy improve), but this pointwise value reverses on the extremal selected tail.
- **V39:** honest cross-fitted selected residual populations prove that feature-dependent selection distortion is real, not merely selected-sample scarcity. CFSR nearly preserves aggregate value while halving catastrophe, yet still loses count-level capture around zero.

The mechanism stack is therefore now better represented as:

`ordinal challenger ranking != population-edge cardinal mean != operator-conditioned selected outcome distribution`.

## 7. Current layer status

Mature/frozen:

- bounded B16/M24 evidence/query interface;
- EAF complete frontier and exact action-local attribution;
- support/admissibility and structural delegation;
- incumbent default/no fallback;
- V34 RSMR as the current best ordinal challenger selector.

Partially mature:

- ordinary population-edge cardinal sign/value (V38 signal exists);
- feature-dependent selected-policy residual/tail correction (V39 signal exists).

Still immature / current bottleneck:

- **selected-policy zero crossing**, especially sign-frequency identification near zero;
- joint control of near-zero positive recovery and negative/catastrophic magnitude;
- a deployment-relevant selected outcome distribution, not just a signed conditional mean.

Updated dominant bottleneck:

> **operator-conditioned selected-outcome distribution at the incumbent boundary: the current signed-mean objective conflates beneficial-event frequency with positive/negative conditional magnitude, so it can optimize aggregate value/tail while misclassifying many near-zero positive opportunities.**

## 8. Directions to stop

Continue all prior prohibitions: acquisition/coreset/beam/swap, B/M broad sweep, capacity-only, FCR/DRC/KNN/PTMC revisits, candidate-count gate, support/score threshold rescue, action blacklist, basepoint/geometry expansion, conformal alpha/q tuning, and A/B pooling.

New V39-specific prohibitions:

- do not enlarge CFSR/OPVR with an MLP or nonlinear/high-dimensional selected head;
- do not tune the capture threshold or redefine `positive` post hoc to hide near-zero misses;
- do not relax the 3 pp capture gate because CFSR aggregate sum looks good;
- do not re-fit or alter RSMR while claiming a value-layer test;
- do not let DENSE/hurdle/value heads rerank challengers;
- do not add a generic catastrophe classifier as a separate fallback/veto module;
- do not restore an unconstrained affine selected calibration that can reverse the value axis;
- do not tune lambda, alpha, q, top-K, candidate count, temperature, or value threshold.

## 9. V64.3.40 SDFR: Selection-Distribution Factorized Recovery

V40 does **not** enlarge the selected-policy residual model. It keeps the same 19-D evidence, fixed linear/ridge family, fixed `lambda=1`, and frozen RSMR winner. It changes the target from one signed conditional mean to a factorized outcome distribution.

For selected outcome `Y = Delta_T(bhat;i)`, use the exact identity

`E[Y|X,S] = pi(X,S) m_plus(X,S) - (1-pi(X,S)) m_minus(X,S)`

where

- `pi = P(Y>0 | X,S)`;
- `m_plus = E[Y | Y>0, X,S]`;
- `m_minus = E[-Y | Y<=0, X,S]`.

There is no tuned loss trade-off coefficient in this identity.

### 9.1 Dense population distribution base

Using the repaired V32.1 scene-equal objective and all fit candidate edges, fit three fixed-lambda linear heads:

1. Brier-ridge beneficial-event probability target `1[Y>0]`;
2. positive conditional magnitude on positive edges, target `Y`;
3. non-positive conditional magnitude on `Y<=0`, target `-Y`.

The HURDLE arm reconstructs the expected intervention value from these three dense population components, but it is evaluated only on the already frozen RSMR winner.

### 9.2 Honest selected-policy scalar component adaptation

Within each outer three-fit-fold population, perform inner OOF frozen-RSMR selection exactly as V39. For the OOF selected proposals, require:

- total selected >=192;
- positive selected >=64;
- non-positive selected >=64.

Fit only scalar selected-policy adaptation:

- a logit intercept shift for `pi`, preserving probability ordering;
- a nonnegative scalar scale for positive magnitude;
- a nonnegative scalar scale for negative magnitude.

No selected-policy feature vector/head is fitted.

### 9.3 Causal arms

- `RSMR`: frozen ordinal ranker.
- `DENSE`: V38 signed-mean control.
- `HURDLE`: dense factorized distribution without selected adaptation.
- `SIGN-SHIFT`: only selected beneficial-event logit intercept shift; magnitude components remain unchanged.
- `SDFR-RAW`: sign shift plus nonnegative positive/negative magnitude scalar adaptation.
- `SDFR-MAIN`: same SDFR-RAW output plus independent calibration-fold **unit-slope translation only**.

This factorization makes the next result mechanistically interpretable:

- HURDLE succeeds -> population distribution target is sufficient; discard selected adaptation.
- SIGN-SHIFT succeeds but full SDFR does not -> beneficial-event frequency shift is the primary mediator; discard magnitude adaptation.
- SDFR succeeds -> selected sign/magnitude factorization closes the zero/tail tradeoff without representation expansion.
- Sign/capture improves but tail remains -> next missing information is value-specific downside representation, not another target factorization.
- All arms fail despite adequate selected sign classes -> **close the current 19-D selected-value distribution route**; next work must introduce a value-specific representation/observable, not a larger head on the same evidence.

## 10. V40 gate and fresh protocol

Use the unchanged historical nested split: `3 fit + 1 independent value calibration + 1 test`.

SDFR-MAIN must satisfy before any fresh population is selected:

- exact frozen-winner subset/identity containment;
- >=20% no-op false reduction vs RSMR;
- useful capture no more than 3 pp below RSMR;
- >=25% catastrophe reduction;
- NegRMS not worse;
- aggregate selected sum >=0 and 5/5 fold sums >=0;
- selected >=64 and positive >=32;
- inner OOF selected >=192 and each selected sign class >=64.

No threshold/lambda/alpha/q/feature/candidate-count/temperature sweep.

V39 spent no fresh data. Permanent exclusion remains **10700 tokens**. New seed:

`v64.3.40-eaf-icer-sdfr-cal500-double-fresh-v1`.

Only after nested TRAIN pass may V40 select independent `CAL500+A500+B500`. A/B remain strictly unpooled. Fresh arms are:

`RAW / V20 / PRESERVE / RSMR / DENSE / HURDLE / SIGN-SHIFT / SDFR-RAW / SDFR-MAIN`.

## 11. Paper-level implication

If V40 is supported by double fresh and later frozen full-validation/closed-loop reproduction, the contribution is not “three heads.” The mechanism is:

> **operator-conditioned distributional decision sufficiency:** ordinal extremal ranking, population cardinal outcome decomposition, and selected-policy sign/magnitude alignment are distinct estimands/supervision units under a deterministic frozen-winner containment contract.

The immediate falsifiable paper claim is that the remaining incumbent-exit failure is not solved by increasing model capacity, but by representing the selected outcome distribution rather than collapsing sign frequency and magnitude into one mean-regression target.
