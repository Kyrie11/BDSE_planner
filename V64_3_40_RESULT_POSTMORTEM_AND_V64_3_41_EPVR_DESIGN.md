# V64.3.40 result audit and V64.3.41 EPVR design

## Reliability decision

V40 is engineering-valid for TRAIN-level attribution. Uploaded code SHA256 is `327569afef5ee5a78364710226929e7601442708f88fce24c98571c6a30778bc`, exactly the preregistered package. The run completed all five nested outer folds, produced 782 unique direct-scene audit rows, passed 190/190 targeted tests, and stopped at the preregistered TRAIN scientific gate. No CAL500/A500/B500 manifest was created. The V40 RSMR baseline also replays V39 scene-by-scene: fold, action and score agree exactly (teacher-improvement numerical difference <= 4.5e-16).

The five independent selected-policy calibration populations contain 97/100/98/86/110 frozen-RSMR proposals. Inner OOF selected-distribution populations contain 347/318/266/260/285 proposals, with positive/non-positive counts 148/199, 152/166, 138/128, 110/150, 106/179; all exceed preregistered minima. No leakage, reranking, fallback or raw-proposal/final-selection accounting mismatch was found.

## V40 scientific verdict

V40 fails promotion. RSMR remains 502 selected / 221 positive / 38.50% useful capture / 28 catastrophes / 107 no-op false interventions / +43.294 sum.

- DENSE: 263 / 138 / 24.04% / 23 / 50 / +12.218.
- HURDLE: 241 / 125 / 21.78% / 20 / 50 / +9.725.
- SIGN-SHIFT: 168 / 92 / 16.03% / 15 / 37 / +8.664.
- SDFR-RAW: 339 / 149 / 25.96% / 20 / 72 / +31.997.
- SDFR-MAIN: 403 / 172 / 29.97% / 22 / 83 / +32.665; one outer fold has negative selected sum.

All preregistered mechanism gates fail. The distribution decomposition therefore does not close the capture/tail tradeoff.

## Mechanism evidence

The selected positive-probability model has AUC 0.6039 and Brier 0.2405. Its mean selected probabilities on true-positive / mild-harm / catastrophe proposals are approximately 0.450 / 0.427 / 0.424, so the selected zero boundary remains weakly separated. More importantly, the predicted negative magnitude on catastrophes averages only 0.171 despite catastrophe truth being <= -0.5, while predicted positive magnitude is actually largest on catastrophes. The three distribution components are therefore not individually sufficient.

The scalar selected-distribution adaptation is not a hidden success: in most folds its full adapted MSE is equal to or worse than the raw hurdle MSE. SDFR-MAIN accepts catastrophe proposals at a high rate and its positive AUC falls to 0.466. This is not a threshold problem.

Combined with V37-V39, the evidence chain is now:

1. V34 RSMR has usable ordinal/high-value challenger ordering.
2. V37 selected-only residual has real tail information but high-dimensional selected fitting is unstable.
3. V38 dense all-edge 19-D supervision recovers ordinary cardinal sign but catastrophes remain badly misordered.
4. V39 honest cross-fitted selected residual proves operator-conditioned tail residual is real, yet zero-boundary capture remains poor.
5. V40 explicitly factorizes sign frequency and positive/negative magnitude on the same 19-D delta representation; all branches still fail with adequate selected populations.

Thus the preregistered V40 falsification branch is triggered: further selected heads or target re-factorizations on the pure 19-D delta representation should stop.

## New dominant bottleneck

`value-specific representation sufficiency for the frozen RSMR winner`.

The pure difference representation assumes intervention value is a globally stationary function of `q_b-q_i`. Yet for a nonlinear latent utility U, `U(q_b)-U(q_i)` depends on the endpoint/basepoint. V35/V36 only tested a scene-common incumbent shift/reservation; they did not test candidate-specific basepoint-dependent slopes. That missing interaction is not closed by the previous basepoint experiments.

## V41 EPVR

V41 freezes RSMR as the sole challenger selector and changes only the post-selection value representation. Let the 19-D endpoint evidence be `q=[18-D EAF evidence, support_logit]`, `delta=q_b-q_i`, and `m=(q_b+q_i)/2`.

### ZDELTA control

`phi_Z = delta` with zero-preserving RMS scaling, zero bias, scene-equal all-edge ridge, lambda=1. This controls whether the null-consistent solver itself matters relative to historical centered DENSE.

### DNLV control

`phi_D = [delta, delta*|delta|]`. This is antisymmetric and tests generic nonlinear contrast geometry without adding incumbent/basepoint information.

### EPV main representation

Assume a diagonal quadratic latent potential

`U(q)=a^T q + 1/2 b^T(q^2)`.

Then

`U(q_b)-U(q_i) = a^T delta + b^T(m*delta)`.

V41 therefore uses

`phi_E = [delta, m*delta]`.

This representation is exactly antisymmetric under endpoint swap and exactly zero when candidate equals incumbent. It is not naive candidate/incumbent concatenation and cannot change RSMR winner identity because it is evaluated only after winner freezing.

EPVR-MAIN adds only an independent selected-policy unit-slope translation to EPV. It cannot reverse value ordering.

## Causal branches

- ZDELTA improves: null-consistent cardinal model was the missing piece; endpoint expansion is unnecessary.
- DNLV improves but EPV does not: generic nonlinear delta geometry is primary, not basepoint-conditioned value.
- EPV/EPVR improves beyond ZDELTA/DNLV: evidence supports basepoint-conditioned local utility gradient as the missing representation mechanism.
- EPV improves sign but not tail: endpoint geometry helps value sign, but tail needs a genuinely new observable/uncertainty representation.
- None improves: close the current EAF-derived endpoint-value representation family and require a new value-specific observable; do not add MLP/head capacity.

The main TRAIN gate remains unchanged: >=20% no-op reduction, capture within 3 pp of RSMR, >=25% catastrophe reduction, non-worse NegRMS, nonnegative aggregate and 5/5 fold sums, selected>=64, positive>=32, exact frozen-winner containment. No threshold/lambda/alpha/top-K/candidate-count/temperature sweep.

Fresh is untouched. Permanent exclusion stays 10700 tokens. V41 uses `v64.3.41-eaf-icer-epvr-cal500-double-fresh-v1`; only nested TRAIN pass can select CAL500+A500+B500, and A/B remain independent and unpooled.
