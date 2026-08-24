# V64.3.34 uploaded result postmortem and V64.3.35 FBCSR design

## 1. Attribution status

The uploaded V64.3.34 execution is engineering-valid for TRAIN-level algorithm attribution. The launcher reproduces the frozen prerequisite stack, targeted regression passes 154/154, the RSMR fitter completes all five nested `3-fit / 1-calibration / 1-test` folds, and the stop is the preregistered complete-mechanism TRAIN scientific gate. No CAL500/A500/B500 token manifest is non-empty, so no new fresh population was consumed.

No second V32-style objective-scaling bug, runtime/config schema mismatch, candidate-set mismatch, or selected-policy score-unit mismatch was found. The result must therefore be interpreted as a valid TRAIN-level mechanism failure, not an engineering-invalid run. It is not fresh evidence.

## 2. V34 RSMR rank result: objective alignment works in one important sense

Aggregates over the paired 782 direct TRAIN scenes:

| arm | selected | positive | precision | positive-opportunity capture | selected sum | catastrophes <= -0.5 | no-opportunity false interventions |
|---|---:|---:|---:|---:|---:|---:|---:|
| corrected MEAN | 476 | 260 | 54.62% | 45.30% | -18.0304 | 47 | 110 |
| V33 PAIR | 98 | 47 | 47.96% | 8.19% | -4.6676 | 18 | 15 |
| **V34 RSMR-RANK** | **502** | **221** | 44.02% | **38.50%** | **+43.2941** | **28** | **107** |

RSMR changes the qualitative behavior of the direct selector. All five outer-fold selected sums become positive: `+9.430 / +2.319 / +10.566 / +13.682 / +7.296`. It recovers much of the opportunity coverage lost by V33, and the opportunity-domain selected sum rises from MEAN `+42.1216` to RSMR `+71.3994`.

This is strong evidence that the V34 paper-level idea—training directly against scene-level selected-action regret rather than average edge/pair fidelity—is useful. RSMR is not a null result.

## 3. But V34 does not solve intervention existence

RSMR almost completely loses V33's no-intervention gain:

- no-positive-opportunity scenes: 208;
- MEAN false intervention: 110;
- V33 PAIR false intervention: 15;
- V34 RSMR false intervention: **107**.

The RSMR no-opportunity selected sum improves in magnitude (`-28.1054` versus MEAN `-60.1520`), and no-op catastrophes fall from 31 to 18, but the *frequency* of inappropriate intervention returns almost to the MEAN baseline.

Of RSMR's 28 catastrophes, **18 are in scenes where no positive challenger exists at all**. Thus the dominant selected-tail problem is still mostly an intervention-existence failure, not only wrong challenger ordering.

Representative no-opportunity catastrophes include:

- `2b32a9f406845f75`: RSMR action 2, `Delta_T=-3.806486`;
- `fd35b4e54a465e54`: action 7, `-2.020733`;
- `1c3f933be722564d`: action 7, `-2.020617`;
- `c70954fab4a650c7`: action 13, `-1.233024` with an extremely high RSMR score `3.51612`.

## 4. RSMR improves regret magnitude more than pointwise correctness

Compared with MEAN:

- both select: 361 scenes;
- same winner: 143;
- different winner: 218;
- RSMR-only proposals: 141, with total teacher improvement **+27.3295**;
- MEAN-only proposals: 115, with total teacher improvement **-7.4044**.

On the 218 common-but-different winners, RSMR is better in 88 and worse in 126, yet the net teacher-improvement delta is **+26.5905**. RSMR is therefore learning a magnitude-sensitive decision policy: it can trade more small mistakes for fewer/stronger large gains and still improve aggregate regret.

The score itself is not a tail-risk score:

- score vs teacher-improvement Pearson correlation: `-0.0144`;
- positive-vs-nonpositive AUC: `0.5928`;
- noncatastrophe-vs-catastrophe AUC: `0.4816`.

Therefore a RSMR-score threshold or alpha sweep is not a principled next step.

## 5. Calibration gain is negative under the current hard-tail objective, but not because conformal code is wrong

Nested selected-policy q values are approximately:

`1.881 / 0.702 / 1.093 / 1.010 / 1.609`.

RSMR-MAIN keeps only five proposals, one positive, with selected sum `-1.7466` and four catastrophes. The five accepted examples include four high-score overpredictions around `-1.22` and one positive `+3.1497`.

This does **not** violate the stated split-conformal marginal coverage claim. A harmful accepted intervention implies a lower-bound miscoverage event, but the finite-sample statement controls its *marginal probability over the proposal-emitting population*, not the conditional harmful fraction among the tiny accepted subset and not the zero-catastrophe requirement. Four harmful accepted proposals out of roughly 502 RANK proposals remain below a 5% marginal event rate.

The scientific mismatch is therefore:

> marginal selected-policy lower-bound coverage is weaker than the planner's hard selected-tail promotion requirement.

Do not fix this by lowering alpha or by claiming the conformal theorem supplies conditional safety.

## 6. New mechanistic conclusion: V33 and V34 expose a representation/surrogate ambiguity

The same frozen 19-D candidate-minus-incumbent representation shows a clear trade-off:

- V33 all-rivals PAIR strongly learns **do not intervene**, but collapses positive-opportunity recovery;
- V34 RSMR restores opportunity recovery and makes every fold aggregate selected sum positive, but reopens almost all no-opportunity interventions.

Two distinct explanations remain:

1. **surrogate factorization:** one scene-max structured violation can be dominated by challenger-ordering regret and fail to give the incumbent boundary an explicit optimization term;
2. **base-point observability:** the 19-D representation is intentionally translation-invariant (`candidate - incumbent`). Challenger ordering may be primarily contrastive, while whether it is worth leaving the incumbent can depend on the incumbent's absolute evidence state. For a nonlinear downstream cost relation, `U(candidate)-U(incumbent)` need not be a function of the feature difference alone; the base point can matter.

V35 is designed to separate these explanations instead of guessing.

## 7. V64.3.35 FBCSR

Full name: **Evidence-Attributed Factorized Basepoint-Conditioned Structured Recovery**.

The frozen layers remain unchanged: B16/M24, acquisition, EAF, admissibility/support prerequisites, structural delegation, incumbent default, no fallback, 19-D candidate contrast, and query accounting.

### 7.1 Factorized delta-only ablation (FDSR)

For positive-opportunity scenes, fit two explicit scene-level regret constraints:

- **existence:** teacher-best challenger versus incumbent;
- **ordering:** teacher-best challenger versus the worst competing challenger.

For no-opportunity scenes, fit one incumbent-versus-most-dangerous-challenger regret constraint.

The per-scene objective is the sum of the squared positive existence and ordering violations (ordering exists only in positive-opportunity scenes), plus fixed `lambda=1` L2. There is no pair count, top-K, tuned relative weight, or runtime threshold.

FDSR therefore tests whether V34 failed merely because one max operator let ordering dominate the incumbent boundary.

### 7.2 Basepoint-conditioned arm (FBCSR-RANK)

Add the absolute incumbent 18-D evidence view plus incumbent support logit as a scene context vector `c_i`, using only already-computed planner evidence.

For challenger `b`:

`score(b) = f_delta(x_b - x_i) + h_context(c_i)`.

The context term is **identical for every challenger in that scene**. Hence

`score(b1)-score(b2) = f_delta(delta_b1)-f_delta(delta_b2)`.

The context can change only `should we intervene?`; it mathematically cannot change `which challenger wins?`. This is not naive feature concatenation and does not create a new evidence query.

The incumbent remains the exact zero-score pseudo-action.

### 7.3 Causal interpretation

Nested TRAIN reports:

- V34 RSMR control;
- FDSR delta-only factorization;
- FBCSR basepoint context;
- FBCSR selected-policy MAIN.

If FDSR fixes the trade-off, the bottleneck was surrogate allocation, not missing context.

If FDSR does not, but FBCSR reduces no-op false interventions by at least 20% relative to FDSR while preserving FDSR opportunity capture within 3 pp, keeping aggregate/cross-fold direction nonharmful, and improving the V34 tail, then base-point context is a supported mediator.

If context fails this gate, stop absolute-context iteration; do not add richer context concatenations or nonlinear heads.

### 7.4 Calibration remains a downstream containment test, not a rescue branch

Only after the rank mechanism passes does nested selected-policy conformal run. It is still fixed `alpha=0.05`, at most one residual per frozen-policy proposal scene, no reranking and no second-best fallback. If the rank mechanism passes but the hard tail/coverage MAIN gate fails, the next bottleneck is calibration objective/guarantee alignment; do not modify the ranker to rescue it.

## 8. Updated dominant bottleneck and paper line

Current dominant bottleneck:

> **factorized scene-level intervention existence and challenger ordering under a contrastive evidence representation, with explicit testing of whether intervention existence requires absolute incumbent base-point context; after rank reliability is established, the remaining problem is hard-tail-aligned policy calibration.**

CCF-A-oriented mechanism line:

`bounded auditable interface -> exact EAF action-local attribution -> factorized incumbent/null-action structured intervention -> basepoint-conditioned existence with contrastive challenger ordering -> policy-aligned monotone calibration -> independent double-fresh/full-validation/closed-loop evidence`.

The novelty is not linear regression, LBFGS, an extra feature block, or conformal prediction in isolation. The research claim is the **invariance-aware decomposition of decision sufficiency**: challenger ordering is contrastive, intervention existence competes with an explicit null action and may require incumbent base-point state, while calibration acts only on the frozen deployment policy and cannot create a new action path.
