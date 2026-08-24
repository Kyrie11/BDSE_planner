# V64.3.33 SPCR result postmortem and V64.3.34 RSMR design

## 1. Attribution status

The uploaded V64.3.33 execution is engineering-valid for **TRAIN-only nested-crossfit algorithm attribution**. It is not a runtime crash and no second implementation bug was found in the frozen SPCR path. The run stops at the preregistered complete-mechanism TRAIN gate; no CAL500/A500/B500 manifests are non-empty, therefore no fresh population was consumed and no fresh generalization claim may be made.

Verified uploaded behavior:

- targeted regression: 146/146 PASS;
- uploaded SPCR fit reaches all 5 nested outer folds;
- corrected V32.1 MEAN is reproduced exactly: 476 selected / 260 positive / sum -18.0303795 / 47 catastrophes / 110 no-positive-opportunity false interventions;
- V33 PAIR produces 98 selected / 47 positive / sum -4.6676395 / 18 catastrophes / 15 no-positive-opportunity false interventions;
- nested calibration proposal counts are 15 / 12 / 8 / 28 / 20, below frozen minimum 32 in every fold, so MAIN fail-closes to zero selections;
- no validation/fresh token files were spent.

The current result therefore identifies the SPCR structured-selector behavior, but **does not experimentally identify calibration gain**. The selected-policy conformal method itself must not be reported as failed; the frozen V33 selector did not emit enough calibration outputs to support the preregistered nested calibration protocol.

## 2. Structured selector gain: what improved

V33 strongly improves the `should we intervene?` frequency error:

| TRAIN cross-fit diagnostic | corrected MEAN | V33 PAIR |
|---|---:|---:|
| selected | 476 | 98 |
| no-positive-opportunity false interventions | 110 | **15** |
| catastrophes <= -0.5 | 47 | **18** |
| selected teacher-improvement sum | -18.0304 | **-4.6676** |
| positive-opportunity capture | **45.30%** | 8.19% |
| precision | **54.62%** | 47.96% |

Of MEAN's 110 false interventions on no-opportunity scenes, PAIR removes **96 by abstaining**. This is a real structured-null-action gain: explicitly making the incumbent the teacher-optimal pseudo-item causes the selector to stop intervening in many scenes where all challengers are harmful.

However, the conditional severity of the residual no-opportunity mistakes is not solved. On no-opportunity scenes PAIR still selects 15 proposals, with sum -14.5035, 8 catastrophes, worst -3.5156 and NegRMS 1.4581. Thus V33 mainly reduces *frequency* through abstention rather than making the remaining false interventions reliable.

## 3. Which-intervention gain is negative

The V33 selector does not improve challenger ordering.

On the 96 scenes where both MEAN and PAIR emit proposals:

- PAIR yields a higher teacher improvement than MEAN on only **9** scenes;
- it yields a lower teacher improvement on **34** scenes;
- 53 are equal;
- 46/96 change winner identity.

On positive-opportunity scenes, PAIR selects only 83/574 scenes (14.46% proposal rate) and captures 47 positives. Corrected MEAN selects 366/574 scenes (63.76%) and captures 260 positives. The transition decomposition is especially diagnostic:

- **203** MEAN-positive selected scenes are lost because PAIR abstains;
- **13** MEAN-good proposals are changed into PAIR-nonpositive proposals;
- only **2** MEAN-nonpositive proposals are changed into PAIR-positive proposals.

So the aggregate V33 improvement comes from suppressing intervention frequency, not from learning a better conditional winner.

Representative harmful PAIR scenes include:

- `b16054ccc4bd54f9`: positive opportunity exists; MEAN y=-0.0757, PAIR changes to action 4 with y=-3.9925;
- `44ae005ad8b25d52`: no positive opportunity; PAIR changes winner and selects y=-3.5156;
- `7a1756cf8f7a566e`: positive opportunity exists; PAIR changes to action 4 with y=-3.3333;
- `5693aed0af7e548c`: no positive opportunity; same harmful action remains, y=-2.8394;
- `0bd45dc9e3a759c2`: positive opportunity exists; same selected action remains catastrophic at y=-1.9806.

This confirms that `should we intervene? + if yes, which intervention?` remains the correct decomposition, but V33 over-solves the first factor by abstention and does not solve the second factor.

## 4. Root cause inside the V33 structured objective

V33 minimizes teacher-best-vs-**all-rivals** pair-gap squared error with one total loss mass per scene. This looks scene-balanced globally, but it is **not balanced between the two deployment decisions**.

For an opportunity scene with `m` challengers, teacher-best is a challenger. Its rivals are the incumbent plus `m-1` challengers, so the one pair that teaches the critical challenger-vs-incumbent intervention boundary receives only

`1/m`

of scene loss mass. In the uploaded TRAIN population:

- 574 positive-opportunity scenes have mean candidate count 13.19;
- their mean incumbent-boundary loss mass is only **0.13390**;
- 208 no-opportunity scenes make the incumbent teacher-best, so **100%** of their pair loss is incumbent-vs-challenger separation.

Therefore the V33 objective gives the null-action boundary about

`1 / 0.13390 = 7.47x`

more per-scene mass in no-opportunity scenes than in opportunity scenes. This predicts the observed behavior: excellent intervention suppression, severe useful-recovery collapse.

The problem is not a ridge lambda, score threshold, alpha, or candidate-set setting. It is a mismatch between an all-pairs average surrogate and the actual scene-level argmax decision.

## 5. Dominant bottleneck after V33

Previous wording after V32.1:

> scene-structured intervention-existence plus selected-policy ordering reliability at the direct incumbent-replacement boundary.

Tightened after V33:

> **regret-aligned scene-level argmax reliability over the incumbent/null action and challengers, without candidate-count dilution, followed by policy-output calibration.**

The weak layer is now the decision operator itself: the evidence representation contains useful signal, but the surrogate used to train the joint null-action/challenger argmax does not match the cost of the final selected action.

## 6. V64.3.34 RSMR

Full name: **Evidence-Attributed Incumbent-Contrastive Regret-Structured Margin Recovery**.

V34 preserves the bounded B16 interface, queried M24 bank, EAF, 19-D candidate-minus-incumbent representation, deployment admissibility/support prerequisites, incumbent default, structural delegation, and no-fallback containment. It changes only the structured training objective and then reuses selected-policy calibration.

### 6.1 Regret-structured scene-max objective

Let candidate utilities be the teacher improvements `Delta_a`, with incumbent pseudo-item

`x_i=0, Delta_i=0, s_i=0`.

Let `t` be the teacher-best item over incumbent plus challengers. For every rival `r`, define

`d_tr = (x_t - x_r) / scale`,

`g_tr = Delta_t - Delta_r >= 0`.

V34 minimizes

`sum_scene [ max_r (g_tr - w^T d_tr) ]_+^2 + lambda ||w||^2`,

with frozen `lambda=1`.

The scale is a zero-preserving RMS computed from candidate features with **one total moment mass per scene**. No mean centering and no intercept are used, so incumbent score remains exactly zero.

Unlike V33, each scene contributes exactly one **worst cost-augmented decision violation**, regardless of candidate count. No pair weighting or top-K hyperparameter is introduced.

### 6.2 Decision-regret upper-bound property

Let runtime select

`a_hat = argmax_{a in {incumbent} U challengers} s_a`.

Because `s_a_hat >= s_t`, the per-scene structured hinge root satisfies

`max_r [Delta_t-Delta_r-(s_t-s_r)]_+ >= Delta_t-Delta_a_hat`.

Therefore the training surrogate directly upper-bounds the teacher regret gap of the **actually selected argmax action**. This is the key V34 mechanism-level change: the objective is aligned to final structured decision regret rather than average edge error or average pair error.

### 6.3 Causal controls

V34 retains four selector/certificate views:

- corrected V32.1 `MEAN`;
- exact V33 `PAIR` selector as a frozen ablation;
- V34 `RSMR-RANK`;
- V34 `RSMR-MAIN`, which may only accept the exact RANK proposal or return incumbent.

This separates V33 all-rivals averaging from V34 scene-max regret alignment and then separates selector gain from calibration gain.

### 6.4 Nested TRAIN diagnosis and conditional branches

The exact V32 fold hash remains frozen. Each outer fold uses 3 fit folds, 1 selected-policy calibration fold and 1 test fold.

RANK diagnostics separately test:

1. intervention-existence gain: no-opportunity false interventions must be below corrected MEAN;
2. opportunity-recovery gain: positive capture must exceed exact V33 PAIR;
3. selected-path direction: selected sum must exceed V33 PAIR and catastrophes must be below MEAN.

Selected-policy calibration still requires at least 32 proposals in every nested calibration fold. Full TRAIN promotion still requires all 5 MAIN test folds to have nonnegative selected sum and zero catastrophes, with aggregate MAIN selected >=64 and positive >=32.

The report emits a factorized `failure_diagnosis` before any fresh selection, distinguishing:

- intervention-existence failure;
- continued over-suppression of positive opportunities;
- which-intervention/tail failure;
- insufficient policy-output density for calibration;
- certificate failure;
- certificate-safe but insufficient recovery coverage.

No branch performs a lambda, alpha, feature, threshold, optimizer, or candidate-set sweep.

### 6.5 Independent protocol

V33 did not spend validation data, so V34 keeps the 10700-token permanent design exclusion and uses new label-free seed

`v64.3.34-eaf-icer-rsmr-cal500-double-fresh-v1`.

Only after nested TRAIN pass does it select CAL500+A500+B500. A/B each run:

`RAW / V20 / PRESERVE / MEAN / V33-PAIR / RSMR-RANK / RSMR-MAIN`.

The two fresh blocks are judged independently. PAIR is an ablation only and cannot become a rescue branch.

## 7. Paper-line implication

The CCF-A-oriented mainline should remain mechanism-level rather than estimator-level:

`bounded auditable evidence interface -> exact EAF action-local attribution -> incumbent-augmented regret-structured intervention decision -> policy-output reliability calibration -> monotone incumbent/no-fallback containment -> double-fresh/full-validation/closed-loop evidence`.

The potentially paper-level novelty is not squared hinge or conformal prediction alone. It is the joint formulation that (i) defines decision sufficiency at a deployment intervention boundary, (ii) treats no-intervention as a first-class structured action, (iii) trains a regret-aligned extremal operator whose surrogate upper-bounds final argmax teacher regret, (iv) calibrates only the frozen policy output, and (v) preserves deterministic monotone containment under a fixed auditable evidence budget.

