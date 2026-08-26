# V64.3.47 Result Postmortem and V64.3.48 OCRR Design

## Executive decision

V64.3.47 is **engineering-valid for TRAIN-level mechanism attribution**, with one non-blocking provenance-hygiene warning: the server targeted log collected 227 tests while the exact uploaded preregistered code package collects 225. Both suites pass, all V47 scientific hard signatures are exact, and the result stopped before fresh selection. Because the V47 launcher did not record a source manifest, server worktree byte identity cannot be proven retroactively. V64.3.48 closes this engineering gap with a hard source-manifest gate before any science.

Scientifically, V64.3.47 is a **promotion failure that completes the preregistered representation-stop experiment**. Both future-state nuisance families are identifiable, but no V47 deployment arm reaches the unchanged capture floor. The low-order AGENT-2D branch is actively harmful and is closed. EGO-REF is a genuine supporting mediator: it is strongly identifiable, adds a positive/zero-cat selected-set rotation and improves several value diagnostics, but it remains far from deployment sufficiency.

Therefore the dominant bottleneck moves exactly where V46 preregistered it should move:

> **operator-conditioned selected-policy zero/tail functional for the already frozen extremal RSMR proposal.**

V64.3.48 is **EAF-ICER-OCRR — Operator-Conditioned Risk Retention**. It freezes every mature representation layer and changes only the post-selection functional. Its central falsifiable hypothesis is that the conditional outcome law of an extremal proposal depends on the selection event, specifically the multiplicity of the existing admissible challenger set.

---

# 1. How the current code line relates to the paper idea

The uploaded manuscript starts from a fixed planner-interface budget and argues that evidence should be judged by **decision sufficiency**, not by reconstruction fidelity. It also recognizes that candidate replacement is an **extremal operation**, so average prediction quality is not sufficient to guarantee a reliable action-changing decision. The original PTMC manuscript then decomposes proposal generation from a veto-only confirmation stage and structurally forbids reranking/fallback.

The V34→V47 code line has made this idea sharper than the current manuscript formulation:

1. **Selection is already mostly solved.** EAF + RSMR provides a mature ordinal extremal proposal mechanism under the bounded auditable interface.
2. **Absolute deployment valuation is a different estimand.** Once the RSMR winner is frozen, the question is not which challenger ranks highest; it is whether the exact selected challenger should replace the incumbent.
3. **Prediction sufficiency is not decision sufficiency.** V46 directly showed a feature block can improve RMSE/Pearson while making catastrophe/zero-crossing deployment worse.
4. **Representation identifiability is not deployment sufficiency.** V47 now shows both new future-state nuisance families can be learned honestly while the deployment capture gate remains open.
5. **The remaining object is operator-conditioned.** The selected proposal is not an i.i.d. edge; it is an order statistic produced by a frozen extremal selection operator. The selected outcome law can therefore depend on the selection event itself.

For a CCF-A-standard paper, the evolving mainline should therefore not be “PTMC plus several feature heads.” A stronger candidate thesis is:

> **Selection–Valuation–Operator Sufficiency under a Bounded Auditable Planner Interface.**

The technical story is that a bounded representation may be sufficient for one deployed operator (ordinal extremal selection) while a different set of statistics and a different post-selection functional are required for another operator (absolute incumbent exit). The no-fallback containment remains a structural safety/auditability guarantee throughout.

---

# 2. V47 engineering reliability audit

## 2.1 Byte identity and result integrity

Uploaded V47 code ZIP SHA256:

`104c336c3b54bea59d123f53edc31a6433b1909e4ed9138e24010c290a5614a4`

This is exactly the SHA256 preregistered in the previous V46→V47 delivery.

Uploaded V47 output ZIP SHA256:

`68b59adacce4c3547c8cac560d49c09cd3bb09d87f986f1f089b5aad16cc51ed`

Key V47 output-file SHA256 values:

- `v64_3_47_fsfr_fit.json`: `1bf6cee0cbfd0c1b5e9c6445a68509e6b9cad7945f81cffdd4831d9f48f64be2`
- `v64_3_47_fsfr_train_scene_audit.csv`: `0335316e9e8dcd1cf411f2bab172ddc29bb67c3f6483fdb0f016c1df63bb06ce`
- `v64_3_47_plan_control.yaml`: `0587928a3baa8c0cdd6134ee54d3fb91145757adb3a49b4d425484306a5879c1`
- `v64_3_47_ego_ref.yaml`: `c4b32850637604cd8c2dafd464f44bafd8a81a19cb1173549fd673baf5c29ae5`

## 2.2 Scientific population / fold contract

- frozen TRAIN: 3000/3000;
- direct scientific scenes: 782/782 unique;
- nested outer folds: 5/5;
- exact V45 PLAN occupancy instrumentation replay: max absolute error 0;
- RSMR remains sole challenger selector;
- every value arm is same-winner containment only;
- no rerank, second best or fallback;
- response/reference nuisance supervision uses no teacher total/improvement;
- runtime uses no logged future;
- A500/B500 were not created or consumed;
- termination is the preregistered TRAIN scientific STOP.

## 2.3 Test-count provenance warning

The server log reports `227 passed, 2 warnings`. The uploaded V47 engineering-validation file records `225/225`, and the exact uploaded V47 package locally reproduces `225/225 PASS` for the same version-range test command.

This is not evidence of an algorithm failure: there are no failed tests and the scientific artifacts replay the expected hard signatures. But it is a reproducibility-process defect because the old launcher did not hash the server source tree. The safe verdict is:

> **PASS for TRAIN attribution, but not perfect source-provenance closure.**

V48 therefore adds a hard `V64_3_48_SOURCE_MANIFEST.sha256` gate. Any source/test/config drift causes an engineering stop before fitting or fresh selection.

---

# 3. Strict V47 preregistered GO decision

Frozen RSMR:

| metric | RSMR |
|---|---:|
| selected | 502 |
| positive | 221 |
| capture | 38.5017% |
| sum ΔT | +43.2941 |
| catastrophe | 28 |
| no-op false | 107 |
| NegRMS | 0.355688 |

Registered capture floor:

`38.5017% - 3pp = 35.5017%`.

V47:

| TRAIN arm | selected | positive | capture | sum ΔT | catastrophe | no-op false | NegRMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| QUALITY | 205 | 129 | 22.4739% | +43.9055 | 13 | 30 | 0.312658 |
| PLAN-CONTROL | 217 | 121 | 21.0801% | +56.5512 | 9 | 38 | 0.240173 |
| AGENT-2D | 213 | 118 | 20.5575% | +52.3056 | 10 | 36 | 0.256374 |
| EGO-REF | 251 | 136 | 23.6934% | +59.5327 | 9 | 45 | 0.223314 |
| FSFR-JOINT | 249 | 135 | 23.5192% | +57.0049 | 9 | 42 | 0.229206 |

All value arms satisfy tail, population and 5/5 nonnegative-sum checks. Every arm fails existence/capture. Therefore the formal verdict is:

> **V64.3.47 promotion failure; no fresh promotion arm.**

The high aggregate value of EGO-REF is not allowed to retroactively weaken the capture gate.

---

# 4. Branch A: AGENT-2D is identifiable but should be closed

The lateral nuisance gate technically passes:

`MSE_zero = 0.190161`

`MSE_LOCAL = 0.184763`

`MSE_PLAN = 0.184742`

LOCAL beats zero in 5/5 folds and PLAN beats LOCAL in 5/5 folds. But the PLAN-conditioned lateral increment is practically tiny: the aggregate LOCAL→PLAN MSE reduction is only about 0.0109%.

More importantly, deployment rotation is harmful:

| PLAN vs AGENT-2D subset | scenes | sum ΔT | material+ | catastrophe |
|---|---:|---:|---:|---:|
| common | 207 | +53.5290 | 33 | 9 |
| PLAN-only | 10 | +3.0222 | 2 | 0 |
| **AGENT-2D-only** | **6** | **-1.2234** | **0** | **1** |

Ordinary selected-value prediction also worsens:

- RMSE `0.6478 -> 0.6584`;
- Pearson `0.4171 -> 0.3887`.

Mechanism verdict:

> **Close the low-order constant local-normal drift family.**

Do not rescue it with larger lateral networks, drift clipping sweeps, trajectory MLPs or weights. Its nuisance signal exists, but it is not the missing first-order deployment statistic.

---

# 5. Branch B: EGO-REF is the only V47 mechanism worth retaining

The ego-reference nuisance is strongly identified:

`MSE_CV_proxy = 13.1037`

`MSE_predicted_reference = 4.16159`

This is about a 68.2% reduction, with improvement in 5/5 folds. Thus the earlier question “is this logged-future-dependent teacher component fundamentally unavailable at runtime?” is answered: **no**. A useful proxy can be predicted from runtime-available state/candidate information without deployment leakage.

PLAN→EGO-REF rotation:

| subset | scenes | sum ΔT | positive | material+ | catastrophe |
|---|---:|---:|---:|---:|---:|
| common | 211 | +56.5520 | 116 | 35 | 9 |
| PLAN-only | 6 | -0.0008 | 5 | 0 | 0 |
| **EGO-REF-only** | **40** | **+2.9807** | **20** | **2** | **0** |
| neither | 245 | -16.2378 | 80 | 13 | 19 |

This is a genuine selected-set rotation rather than blanket abstention. It adds 40 interventions whose total value is positive and whose negative cases are all extremely close to zero (worst about `-0.000867`), while recovering two material opportunities and introducing no catastrophe.

Material recovery among the 50 RSMR material positives (`ΔT>0.2`, total `+82.7336`):

- QUALITY: 30/50;
- PLAN: 35/50;
- AGENT-2D: 33/50;
- **EGO-REF: 37/50**;
- JOINT: 36/50.

Selected-value diagnostics PLAN→EGO-REF:

- RMSE `0.6478 -> 0.6434`;
- Pearson `0.4171 -> 0.4235`;
- positive AUC `0.6146 -> 0.6298`;
- non-catastrophe AUC `0.5892 -> 0.6157`.

But EGO-REF still misses 13 material positives totaling `+12.8356`, and capture remains only 23.69%. Several missed material positives have positive reference-improvement signal while the final signed mean remains negative. That is direct evidence that the information is present but the zero-crossing functional is wrong.

Mechanism verdict:

> **Retain EGO-REF as a supporting validated consequence coordinate; do not promote the EGO-REF signed-mean policy itself.**

---

# 6. FSFR-JOINT does not justify carrying AGENT-2D forward

EGO-REF→JOINT:

- common 241 scenes: `+58.2305`, 8 catastrophes;
- EGO-REF-only 10 scenes: `+1.3022`, one catastrophe;
- JOINT-only 8 scenes: `-1.2256`, one catastrophe.

The joint arm has lower aggregate value, lower capture and worse NegRMS than EGO-REF. There is no evidence that the lateral branch becomes useful only in combination with EGO-REF.

---

# 7. Why the dominant bottleneck must now change

V47 fulfills the exact V46 stopping condition:

1. two new future-state nuisance variables are honestly identifiable;
2. one of them (EGO-REF) adds real predictive and selected-set information;
3. all deployment arms still fail the absolute capture boundary;
4. additional future-state complexity (AGENT-2D) can hurt;
5. fresh data remain untouched.

Therefore another feature/observable block would violate the experimental logic. The primary bottleneck is now:

> **selected-policy decision functional under extremal post-selection.**

A useful way to quantify the challenge is to condition on frozen RSMR. There are 221 RSMR true-positive selected proposals. To meet the existing capture floor, a downstream veto must retain at least 204 of them (≈92.3%). At the same time it must reduce:

- no-op false interventions: `107 -> <=85`;
- catastrophes: `28 -> <=21`;
- and keep NegRMS non-worse and all fold sums nonnegative.

So the missing mechanism is not “abstain more.” It is **selectively remove harmful order-statistic winners while almost never removing useful winners**.

---

# 8. Layer maturity after V47

| layer | status after V47 | V48 rule |
|---|---|---|
| B16/M24 bounded interface | mature | freeze |
| EAF complete anchor frontier | mature / paper backbone | freeze |
| exact selected-evidence attribution | mature | freeze |
| evidence capacity / acquisition | first-order closed | no sweep |
| support/admissibility | mature | freeze |
| RSMR ordinal extremal selection | most mature learned layer | permanent freeze |
| incumbent/null/no-fallback containment | mature | permanent freeze |
| selected residual/tail existence | proven | no longer research existence |
| EPV endpoint geometry | real partial mediator | retain |
| QUALITY current consequence | real partial mediator | retain |
| prospective horizon | proven | retain |
| V44 ungated full-horizon occupancy | strong mechanism | retain |
| V45 agent-local longitudinal response | mature supporting layer | retain |
| plan-conditioned longitudinal mean | real but incremental | retain low capacity |
| V46 response second moment | identifiable, not decision-sufficient | closed |
| V46 handcrafted temporal profile | predictive but deployment harmful | closed |
| V47 AGENT-2D constant lateral drift | identifiable but deployment harmful | closed |
| V47 EGO-REF | strongly identifiable + useful mediator | retain as context, not policy |
| **selected-policy deployment functional** | **dominant immature layer** | **V48 target** |
| absolute zero / material recall | unresolved | final deployment gate |
| source provenance | V47 imperfect | V48 manifest hard gate |

What the model should learn next is therefore very specific:

> **risk ordering of already-selected extremal proposals conditional on validated consequence coordinates and the selection event itself.**

It should not learn another future trajectory, another response mode, or another scalar all-edge mean.

---

# 9. Why V48 may revisit a selected-policy learner without violating the V40 closure

V40 already tested a selected outcome distribution idea and failed. That failure must not be ignored.

The V40 closure was narrower and important: **do not keep changing targets/heads on the same pure 19-D selected EAF delta representation**. Its selected positive AUC was weak and its sign/magnitude decomposition did not solve the gate.

V41–V47 then established a sequence of new physical/decision-sufficient objects:

- endpoint/basepoint geometry;
- current QUALITY consequence;
- prospective horizon;
- ungated full-horizon interaction support;
- agent-local continuous response;
- runtime-predictable ego-reference consequence.

V48 therefore does not resurrect V40. It uses only a four-dimensional state constructed from validated consequence mechanisms and an explicit selection-operator variable. It changes **where the probability law is conditioned**, not merely the capacity of a head on old 19-D evidence.

---

# 10. V48 core mechanism: Operator-Conditioned Risk Retention

## 10.1 Freeze the proposal

As before:

`b_hat = argmax_b u_RSMR(b)`

under the unchanged admissibility/support rules.

No V48 variable participates in this argmax.

## 10.2 Build a minimal selected-operator state

Let:

- `Q` = frozen QUALITY absolute value of `b_hat`;
- `P` = frozen V45 PLAN-CONTROL absolute value;
- `E` = frozen V47 EGO-REF absolute value;
- `K` = number of existing deployment-admissible challenger alternatives at the moment RSMR chooses the extremal proposal.

Define:

`z = [Q, P-Q, E-P, log K]`.

The first three coordinates are a mechanism factorization rather than a redundant stack:

- `Q`: current/endpoint consequence;
- `P-Q`: prospective interaction/response increment;
- `E-P`: future ego-reference increment.

The fourth is the **operator state**. `K` is observed, not modified. V48 never sweeps candidate count/top-K and never changes support/admissibility.

Crucially, V48 does not assume larger K is always worse. The coefficient is unconstrained: after conditioning on physical consequences, multiplicity may represent winner's-curse pressure or, conversely, constrained-scene scarcity. The scientific claim is **dependence on the selection event**, not a hand-coded monotone penalty.

## 10.3 Learn selected sign-risk, not signed magnitude

On frozen RSMR winners only, define beneficial `y>0` and non-beneficial `y<=0` selected outcomes. Fit a zero-bias pairwise ranker:

`r(z)=w^T ((z-mu)/sigma)`

with fixed `lambda=1` and pairwise logistic loss over every fit-fold bad/good pair:

`L(w)=Σ log(1+exp(-(r(z_bad)-r(z_good)))) + 0.5*lambda*||w||^2`.

This is deliberately an ordinal selected-risk objective. It does not estimate teacher magnitude and cannot rerank candidate actions.

Why pairwise instead of a generic classifier:

1. the deployment task is retention ordering under a hard positive-retention budget;
2. pairwise ranking avoids introducing a free class-prior intercept;
3. the only score needed is relative selected-policy risk;
4. it cleanly supports the multiplicity ablation with the same functional.

## 10.4 Calibrate retention from the already registered capture budget

The deployment gate already states that absolute positive capture may fall by at most 3 percentage points relative to RSMR. Given frozen RSMR capture `0.3850174216`, the implied conditional false-veto budget among true-positive RSMR winners is:

`alpha_ret = 0.03 / 0.3850174216 = 0.0779185520`.

For each outer test fold:

- fit risk on the three folds excluding both test and calibration folds;
- use the held-out calibration fold's RSMR true positives only;
- set `tau` to the fixed `(1-alpha_ret)` split-calibration quantile of risk;
- accept `b_hat` iff `risk <= tau`; otherwise return to incumbent.

This is a **constraint calibration**, not a hyperparameter sweep. The same alpha is mathematically induced by the existing preregistered gate.

---

# 11. V48 causal ablation

Only two scientific arms are allowed.

### SIGN-NOMULT

`z0 = [Q, P-Q, E-P, 0]`

Same training objective, same folds, same split calibration, same frozen proposal.

Question: is a selected-policy functional sufficient once the validated physical consequence decomposition is available?

### SIGN-MULT / OCRR-MAIN

`z1 = [Q, P-Q, E-P, log K]`

The only difference is the observed extremal multiplicity.

Question: does post-selection risk require explicit conditioning on the extremal competition-set size?

Independent identification gate compares nonpositive-risk AUC to the directly preceding EGO-REF signed-value baseline. The new selected-risk score must improve aggregate AUC and at least 4/5 folds. This is separate from the deployment gate.

---

# 12. V48 design-only TRAIN replay

This section is intentionally labeled **post-hoc mechanism design on already consumed TRAIN**. It is not fresh evidence and must not be reported as independent performance.

## SIGN-NOMULT

| metric | value |
|---|---:|
| selected | 411 |
| positive | 187 |
| capture | 32.5784% **FAIL** |
| sum ΔT | +53.4956 |
| catastrophe | 18 |
| no-op false | 78 |
| NegRMS | 0.270655 |
| nonpositive-risk AUC | 0.613919 |
| EGO-REF baseline AUC | 0.629829 |
| folds with better AUC | 3/5 **FAIL** |

The selected functional alone is not enough.

## SIGN-MULT

| metric | value |
|---|---:|
| selected | 439 |
| positive | 204 |
| capture | **35.5401% PASS** |
| sum ΔT | **+62.6341** |
| catastrophe | **14 PASS** |
| no-op false | **74 PASS** |
| NegRMS | **0.229885 PASS** |
| all fold sums | **5/5 nonnegative** |
| nonpositive-risk AUC | **0.637413** |
| EGO-REF baseline AUC | 0.629829 |
| folds with better AUC | **4/5 PASS** |

Per-fold SIGN-MULT selected sums are all positive. Only fold 1 does not beat EGO-REF in risk AUC; folds 0/2/3/4 do.

The critical causal comparison is not that `+62.63` is large. It is:

> **under the same selected-policy objective and the same retention calibration, removing only the selection-multiplicity coordinate destroys both independent risk identification and the capture gate.**

That is the mechanism hypothesis to take to untouched A/B.

The capture margin is intentionally thin (`35.5401%` vs `35.5017%`). That is not a reason to tune it upward: the threshold is constrained by the existing capture tolerance. Fresh evaluation must decide whether the operator generalizes.

---

# 13. V48 preregistered GO/STOP logic

1. Reproduce exact V47 key-file SHA256 values and scientific stop.
2. Hard-pass the V48 source manifest.
3. Pass compile and V13→V48 targeted regression.
4. Run nested TRAIN OCRR.
5. Prefer the simplest passing arm in fixed order:
   - SIGN-NOMULT first;
   - SIGN-MULT second.
6. Only if TRAIN passes, create label-free A500+B500 with seed:

   `v64.3.48-eaf-icer-ocrr-double-fresh-v1`

7. Evaluate A and B independently. Never pool.
8. If either fresh split fails, STOP and do not tune multiplicity transform, lambda, threshold, class weights, consequence coordinates or operator features from the failed fresh evidence.
9. Only double-fresh pass permits full validation and official closed-loop/interventional evaluation.

---

# 14. Directions that are now explicitly forbidden

The historical no-repeat set remains binding. In particular V48 must not:

- modify RSMR ordering;
- sweep B/M/candidate count/top-K;
- change evidence acquisition or support/admissibility;
- reintroduce second-best fallback or reranking;
- revive V44 scene-global response classes;
- revive V46 response variance/CVaR/hand temporal profile;
- carry forward V47 AGENT-2D into OCRR-MAIN;
- add another future-state feature block;
- expand pure 19-D/endpoint selected MLPs;
- revive V40 SDFR on raw 19-D evidence;
- add a standalone catastrophe classifier/binary veto;
- tune a selected zero translation or threshold;
- use logged future/teacher value at deployment;
- tune on A and rescue B or vice versa.

Observing `K` does **not** reopen candidate-count tuning. V48 treats the existing admissible-set size as a state variable of the deployed selection operator while leaving the candidate process frozen.

---

# 15. What is now mature enough for the paper and what is not

## Candidate backbone

- fixed bounded auditable interface;
- EAF complete frontier and exact attribution;
- frozen RSMR ordinal extremal selection;
- deterministic incumbent containment / no fallback;
- prospective ungated interaction support.

## Strong supporting learned mechanisms

- agent-local continuous longitudinal response;
- runtime-predictable ego-reference consequence.

## Negative evidence worth publishing

- scene-global response classification can collapse to the majority and still appear useful through unrelated occupancy support;
- identifiable response variance need not be decision-sufficient;
- temporal features can improve regression while increasing catastrophes;
- identifiable future-state factors can still fail the deployed zero/capture gate;
- selected-policy functional without operator conditioning can fail even when it uses the same validated consequence coordinates.

## Not yet a headline claim

OCRR is not paper-ready based on TRAIN. It becomes a headline mechanism only if:

1. the exact preregistered SIGN-MULT arm passes A500 and B500 independently;
2. full validation confirms preservation/endpoint/tail behavior;
3. frozen official closed-loop evidence confirms real intervention benefit;
4. causal wording is limited appropriately—logged-future nuisance supervision supports predictive conditionals, while stronger interaction-causality claims need closed-loop/interventional evidence.

---

# 16. Recommended paper-level framing if V48 validates

A stronger title family than the current PTMC-specific manuscript would be something like:

> **Budgeted Decision-Sufficient Evidence for Extremal Autonomous Planning: Selection, Valuation, and Operator Sufficiency**

The core theorem/algorithm story could become:

1. bounded selected evidence induces an auditable complete frontier;
2. RSMR solves ordinal extremal proposal selection;
3. proposal identity is frozen;
4. physical consequence views estimate selected-action state without increasing evidence budget;
5. OCRR conditions retention risk on the extremal selection operator;
6. veto-only containment guarantees no new action path and no fallback.

The most research-relevant empirical thesis is:

> **For extremal planning, prediction sufficiency, representation identifiability, and post-selection decision sufficiency are distinct. A reliable bounded planner must match statistics and learning objectives to the deployed decision operator.**

This is a much stronger CCF-A-standard contribution than presenting each V4x mechanism as an accumulated trick.
