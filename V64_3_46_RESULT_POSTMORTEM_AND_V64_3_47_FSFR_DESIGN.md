# V64.3.46 Result Postmortem and V64.3.47 FSFR Design

## Executive decision

**V64.3.46 is engineering-valid and supports TRAIN-level scientific attribution. No V46.1 engineering repair is required.** The uploaded code SHA256 is `8110dba70aaf52cef914d068fbb83031784a7f9597c715caa4edc84c0bfb92cc`, exactly matching the preregistered package. The uploaded outputs SHA256 is `f986d043a78df3e25fcdeffa2bdf52f9a3fe95519cc6dc90c49b15a32b056311`.

V46 is a **scientific failure as a promotion mechanism**, but it produces two strong diagnostic results:

1. the conditional longitudinal response second moment is honestly identifiable, yet `DIST-MEAN` does not improve the deployment zero/tail decision over V45 PLAN;
2. hand temporal-profile features strongly improve ordinary value regression while making the deployment tail substantially worse.

The first result closes “learn a low-order acceleration variance” as the first-order answer. The second result exposes a new secondary structural issue: **better ordinary signed-mean prediction is not equivalent to a better post-selection incumbent-exit functional**.

According to the V46 preregistered stop branch, V47 therefore tests two genuinely missing future-state families—**2-D agent response geometry** and **runtime-predictable ego-reference consequence**—without rescuing V46 second moments/profile features. If these nuisances are identifiable but again improve prediction without closing the deployment gate, the next version must stop adding representation and change the selected-policy deployment functional.

---

## 1. V46 engineering and protocol audit

The uploaded run satisfies the preregistered reliability contract:

- frozen code version is exact;
- `219/219` server targeted regression passes;
- 782/782 direct TRAIN scenes are unique and complete;
- five nested outer folds are complete;
- V45 RSMR / QUALITY / PLAN historical signatures are exactly replayed;
- the V46 instrumentation replays V45 PLAN occupancy with max absolute difference `0.0`;
- response second-moment nuisance fitting excludes the outer test fold and paired calibration fold;
- logged agent future is TRAIN-only nuisance supervision;
- no teacher value/improvement enters the response nuisance model;
- deployment uses no logged future;
- RSMR is the sole challenger selector and every value arm only accepts the same frozen RSMR winner or returns to incumbent;
- no rerank, second-best or fallback exists;
- the launcher stops at the TRAIN scientific gate;
- no A500/B500 fresh manifests were created.

Stage timing is consistent with a normal completed TRAIN experiment: prerequisite gate, regression, cross-fit response distribution, then nested TRAIN gate. No fresh stage was entered.

Therefore the experiment is not an engineering failure and can be interpreted scientifically.

---

## 2. Preregistered V46 mechanism verdict

The unchanged promotion capture floor is

`38.501742% - 3pp = 35.501742%`.

| TRAIN nested cross-fit | RSMR | QUALITY | V45 PLAN | DIST-MEAN | TEMPORAL-PROFILE | DIRP-JOINT |
|---|---:|---:|---:|---:|---:|---:|
| selected | 502 | 205 | 217 | 216 | 217 | 207 |
| positive | 221 | 129 | 121 | 118 | 119 | 115 |
| capture | 38.50% | 22.47% | 21.08% | 20.56% | 20.73% | 20.03% |
| teacher sum | +43.294 | +43.906 | +56.551 | +57.526 | +51.263 | +55.303 |
| catastrophe | 28 | 13 | 9 | 9 | 14 | 12 |
| no-op false | 107 | 30 | 38 | 39 | 41 | 39 |
| NegRMS | 0.3557 | 0.3127 | 0.2402 | 0.2413 | 0.3297 | 0.2730 |
| 5/5 fold sum nonnegative | yes | yes | yes | yes | yes | yes |

No new V46 arm reaches the capture floor, so `preferred_promotion_arm = null` and the preregistered TRAIN gate correctly stops before fresh evaluation.

**Verdict: V46 is a promotion failure.**

---

## 3. Second moment: identifiable nuisance, ineffective first-order decision mechanism

Honest OOF second-moment MSE:

- constant second moment: `0.7338943`;
- LOCAL second moment: `0.3918492`;
- PLAN second moment: `0.3844647`.

Both improvements are stable across all five folds, so the response second moment is not an unlearnable nuisance.

However, deployment behavior barely changes:

### PLAN vs DIST-MEAN

- both accept: 207 scenes, sum `+54.5651`, 9 catastrophes;
- PLAN-only: 10 scenes, sum `+1.9860`, zero catastrophes;
- DIST-only: 9 scenes, sum `+2.9604`, zero catastrophes.

DIST-MEAN leaves catastrophe at `9`, worsens NegRMS slightly (`0.2402 -> 0.2413`) and reduces capture. The small positive aggregate rotation does not constitute a first-order mechanism.

**Mechanism decision:** retain the V45 learned response mean; do not carry V46 second-moment quadrature into the next core mechanism.

---

## 4. Temporal profile: informative representation, wrong deployment use

TEMPORAL-PROFILE creates the most important V46 paradox.

Selected-proposal prediction improves substantially:

- RMSE: `0.6478 -> 0.6009`;
- Pearson: `0.4171 -> 0.5353`;
- MAE: `0.3302 -> 0.2878`.

Yet deployment deteriorates:

- catastrophe: `9 -> 14`;
- NegRMS: `0.2402 -> 0.3297`;
- sum: `+56.551 -> +51.263`.

The set rotation explains why:

- common PLAN/TEMP: 162 scenes, `+54.5423`, 8 catastrophes;
- PLAN-only: 55 scenes, `+2.0088`, 1 catastrophe;
- **TEMP-only: 55 scenes, `-3.2791`, 6 catastrophes**.

Six catastrophes that PLAN correctly rejected cross the temporal-profile zero boundary, including:

- `fd35b4e54a465e54`: teacher `-2.0207`, PLAN `-0.0190`, TEMP `+0.0451`;
- `1c3f933be722564d`: teacher `-2.0206`, PLAN `-0.0260`, TEMP `+0.0412`;
- `bce79a86a0b75d48`: teacher `-0.9901`, PLAN `-0.1916`, TEMP `+0.1178`.

This is not evidence that time structure contains no information. It is stronger evidence that **ordinary all-edge signed-mean residual fitting can exploit richer information for MSE/Pearson while moving the post-selection physical zero in the wrong direction**.

This becomes a secondary bottleneck to be acted on only after the V46-preregistered representation branches are exhausted.

---

## 5. DIRP-JOINT does not close the failure

The joint distributional + temporal representation reaches Pearson `0.5427`, even higher than TEMP alone, but deployment remains insufficient:

- capture `20.03%`;
- catastrophe `12`;
- NegRMS `0.2730`.

Relative to PLAN, JOINT-only has 47 scenes with only `+0.7626` total value and 4 catastrophes. Thus combining two individually informative nuisance summaries does not solve the absolute incumbent-exit decision.

The V46 preregistered falsification condition is therefore met: **close low-order longitudinal acceleration distribution plus handcrafted temporal-profile functionals as a deployment-sufficient family.**

---

## 6. Capture decomposition remains scientifically nontrivial

PLAN-CONTROL misses 100 RSMR positives. Of these:

- 69 are `<=0.001`;
- 85 are `<=0.01`;
- 15 are material positives `>0.2`;
- those 15 material misses total about `+15.796` teacher value.

Thus count-level capture is still dominated by near-zero positives, but there remains a meaningful material-opportunity failure. The capture gate must not be weakened or redefined post hoc.

None of the V46 arms removes this material problem: DIST still misses 15 material positives; TEMP and JOINT each miss 14.

---

## 7. What remains mature and what remains immature

| Layer | V46 status | Action |
|---|---|---|
| B16/M24 bounded interface | mature | freeze |
| EAF complete frontier | mature / paper backbone | freeze |
| exact attribution | mature | freeze |
| support/admissibility | mature | freeze |
| RSMR ordinal challenger selection | most mature learned layer | permanently freeze |
| incumbent/null + no fallback | mature | freeze |
| EPV endpoint geometry | real partial mediator | retain |
| current QUALITY | real partial mediator | retain |
| prospective horizon | validated in V43 | retain |
| V44 ungated occupancy support | strong validated mediator | retain |
| V45 agent-local longitudinal response mean | identifiable + selected-value mediator | retain |
| V45 plan-conditioned response mean | identifiable but incremental | retain low-capacity form |
| V46 longitudinal response second moment | identifiable but not decision-sufficient | close as first-order branch |
| V46 handcrafted temporal profile | predictive but deployment-harmful | close as first-order branch |
| 2-D agent response geometry | absent | V47 branch A |
| future ego-reference consequence | absent at runtime | V47 branch B |
| selected-policy zero/tail functional | increasingly suspect | secondary; becomes primary if V47 representations fail |

---

## 8. Dominant bottleneck after V46

The primary bottleneck is tightened to:

> **future-state factorization sufficiency for absolute valuation of the frozen extremal proposal: 2-D agent response geometry plus runtime-predictable future ego-reference consequence.**

Why these two rather than another scalar head:

1. V46 has already shown that low-order uncertainty and temporal functionals are present/learnable but insufficient.
2. The response model is still only longitudinal, despite autonomous-driving interactions being spatial.
3. Several remaining errors occur with current agent-risk support at zero, leaving a logically independent future non-agent/reference family open.
4. The teacher base cost includes a future demonstration/reference component that V42 deliberately excluded from deployment because it is not directly observable. A runtime predictor of that component is a valid nuisance-prediction question, distinct from consuming future labels at deployment.

The secondary bottleneck is:

> **selected-policy deployment functional mismatch.**

V46's temporal paradox is direct evidence that improved conditional-mean regression can worsen the deployment zero/tail decision. Therefore V47 is explicitly a final representation-family test before changing the deployment statistical functional.

---

## 9. V47 FSFR algorithm

### 9.1 Frozen proposal and mature backbone

V47 retains the exact operator:

`RSMR -> frozen winner b_hat -> post-selection value -> {b_hat, incumbent}`.

No new mechanism can rerank challengers, select second best, or create a proposal not selected by RSMR.

EPV and V42 QUALITY remain the frozen value backbone. V44 ungated occupancy support and V45 longitudinal response mean remain the interaction backbone.

### 9.2 AGENT-2D branch

V47 learns a TRAIN-only continuous **lateral drift** nuisance target for each valid agent, measured in the current local normal direction relative to current-state CV. It is fit with two stages:

- LOCAL lateral response from current/history agent state;
- PLAN residual from the existing V45 interaction-exposure conditioned candidate features, fixed `lambda=1`, zero bias.

The lateral target and runtime rollout are physically contained without a tunable hyperparameter: the local-normal drift magnitude cannot exceed the agent's measured current speed.

Runtime 2-D future combines:

- frozen V45 longitudinal acceleration response mean;
- learned lateral drift;
- V44 full-horizon ungated occupancy support.

No logged future is consumed at deployment.

### 9.3 EGO-REF branch

TRAIN logged ego future is used only to construct the teacher demonstration/reference **component** target for each candidate. The model does not receive teacher total cost, teacher improvement or a selected-action label.

Deployment predicts this reference consequence from current/runtime quantities only:

- candidate deviation from current-kinematic ego extrapolation;
- terminal and mean speed changes;
- terminal and mean heading changes;
- route deviation;
- progress deficit;
- comfort.

A deterministic current-kinematic CV proxy is the nuisance baseline. The learned EGO-REF model must beat this proxy in aggregate and in at least 4/5 outer folds before any value gain can be attributed to the reference mechanism.

### 9.4 Causal arms

- **PLAN-CONTROL:** exact V45 1-D response occupancy.
- **AGENT-2D:** replace only the 1-D response geometry by the separately identified 2-D trajectory response.
- **EGO-REF:** retain V45 1-D response and add predicted ego-reference consequence.
- **FSFR-JOINT:** combine AGENT-2D and EGO-REF.

All residual fits are scene-equal, zero-bias where structurally required, fixed `lambda=1`, with no threshold/translation/CVaR tuning.

### 9.5 Independent nuisance gates

AGENT-2D attribution requires:

- LOCAL lateral MSE < zero-drift baseline in aggregate;
- PLAN lateral MSE < LOCAL in aggregate;
- both improvements in at least 4/5 folds.

EGO-REF attribution requires:

- learned reference MSE < current-kinematic CV proxy in aggregate;
- improvement in at least 4/5 folds.

These gates prevent a value-arm movement from being mislabeled as future-state learning when the nuisance model itself learned nothing.

---

## 10. V47 preregistered branch decisions

1. **AGENT-2D passes + lateral gate passes:** promote 2-D response geometry; do not add EGO-REF.
2. **AGENT-2D fails, EGO-REF passes + reference gate passes:** promote future ego-reference prediction; keep V45 interaction geometry.
3. **Only FSFR-JOINT passes and both nuisance gates pass:** both future-state factors are jointly required.
4. **A nuisance gate fails:** do not attribute any value movement to that mechanism.
5. **Both nuisances are identifiable, regression improves, but all value arms fail:** stop representation expansion. The next version must explicitly model the selected-policy zero/tail deployment functional exposed by V46 rather than append another observable family.
6. **Neither nuisance is identifiable:** close this low-order factorization; do not rescue it with a larger MLP unless a substantially different physically motivated future-state target is introduced.

Promotion value gates remain unchanged, including the `RSMR - 3pp` capture floor.

---

## 11. Information contract and paper positioning

V47 deployment inputs are current ego/agent history, current map/route, and already-generated candidate trajectories plus frozen learned nuisance parameters. Logged agent/ego futures are TRAIN-only supervision.

This supports predictive conditional-response/reference claims. It does **not** by itself identify a causal ego-action effect on agent behavior; such a claim still requires frozen closed-loop/interventional evaluation after TRAIN and double-fresh promotion.

If validated, the broader paper line is better stated as **Selection–Valuation–Future-State Sufficiency under a bounded auditable planner interface** rather than as a collection of value tricks: ordinal extremal selection and absolute execution valuation require different sufficient statistics, and the latter may require separately identified future-state nuisance factors.

---

## 12. Explicit no-repeat list after V46

Do not:

- tune V46 variance, sigma points or temporal coefficients;
- add another low-order temporal functional;
- revive scene-global five-mode response classification;
- tune CVaR alpha/weights or response probabilities;
- enlarge the deterministic/second-moment value MLP;
- use selected translation or threshold sweeps;
- union policies post hoc;
- modify RSMR, B/M, candidate count, evidence acquisition, support/admissibility or no-fallback;
- add binary catastrophe vetoes;
- consume logged future at deployment;
- equate lower MSE/higher Pearson with deployment success;
- after an identifiable-but-failed V47, append still more feature blocks instead of changing the deployment functional.
