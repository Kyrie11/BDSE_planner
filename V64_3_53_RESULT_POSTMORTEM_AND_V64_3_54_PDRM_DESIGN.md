# V64.3.53 POTR Result Postmortem and V64.3.54 PDRM Design

## 0. Formal decision

V64.3.53 is **engineering/provenance valid for TRAIN-level scientific attribution**. This is not an engineering repair branch.

The preregistered scientific decisions are:

- frozen V52 structural-null / effect-support hurdle: **GO / retain**;
- V53 `ENDPOINT`: **STOP**;
- V53 `TEMPORAL`: **STOP**;
- no V53 runtime retention policy is promoted;
- untouched validation remains unconsumed.

The formal V53 failure diagnosis is:

`preexecution_operator_trajectory_contrast_does_not_identify_effectful_outcome_order`.

Therefore V54 follows the V53 preregistered branch into **post-intervention paired dynamic response / outcome-process identification**, rather than adding more pre-execution plan geometry.

---

## 1. Paper-level context

The current manuscript's durable conceptual backbone is the bounded planner interface and the distinction between **decision sufficiency** and predictive reconstruction. The planner exposes bounded auditable evidence, constructs an evidence-attributed frontier, chooses at most one extremal replacement, freezes that proposal, and then allows only `{same proposal, incumbent}` under a no-fallback monotone operator.

The current TeX still describes the older DRC/PTMC mechanism. The V44--V53 evidence chain has moved the algorithm well beyond that implementation. The manuscript should not be mechanically patched during every TRAIN iteration; once the outcome-sufficiency mechanism stabilizes, the abstract/method/experiments need a full claim-to-code rewrite. The stable paper-level thesis should remain:

**Selection--Valuation--Outcome Sufficiency under a Bounded Auditable Planner Interface.**

V53 refines the Outcome Sufficiency claim rather than changing the headline:

1. selected outcome is intervention-relative;
2. effect-support sufficiency is not conditional-outcome-order sufficiency;
3. pre-execution planned treatment geometry is not, by itself, a fold-stable sufficient state for realized effect direction.

---

## 2. V53 reliability audit

### 2.1 Source / provenance

- uploaded code zip SHA256: `37b0cc3ce184491fd013a8effe2f3c118ae73467e8db0bfa50e5fe1bed38885b`;
- uploaded result zip SHA256: `07505de85158f4ba84de3802eca202c2dd9278ed5fc23529d0c6b1af32ab0b99`;
- V53 fit JSON SHA256: `9174ffeac064a85bef6c1727915d93903271f9afe1770f5e5ba3e3e51efe1b6e`;
- exact reused V50.5 paired outcome SHA256: `d592baa3508ae9dc084eebcbb3505d9accadc724601ab35a1253d3536af39d43`.

The V53 result-defining science manifest verifies **15/15 files**. Server regression is **64/64 PASS**. An independent replay on the uploaded code is also **64/64 PASS**; Python compile and launcher syntax pass.

### 2.2 State replay / evidence identity

The V53 treatment-side state replay contains exactly **502/502 unique profiles**. It does not recollect paired outcome labels. The sidecar is recorded at the exact one-shot intervention anchor before execution, using the cached frozen RSMR proposal trajectory and actual runtime incumbent trajectory. Historical `nuplan_planner.py` remains byte-identical to the frozen V50.5 source.

The scalar execution contrast `D` replays V51/V52 token-by-token with error <= `1e-9`; the 38 physically identical proposal/incumbent pairs remain zero contrast. Nested fitting excludes both the outer test fold and the calibration fold.

### 2.3 Reliability verdict

This is a preregistered **scientific STOP**, not an ENGINEERING/DATA STOP. V53 can be attributed.

One reproducibility hygiene issue remains: the V53 result zip locks historical parent evidence by exact SHA but does not bundle the complete V49/V50.5 parent roots. That does not invalidate the executed result, but it makes independent third-party re-analysis less self-contained. V54 therefore emits a compact 502-row analysis-population snapshot in addition to hashes.

---

## 3. Strict preregistered branch decisions

### 3.1 Shared V52 effect support — GO / retain

The exact frozen QPE+D hurdle remains identified:

- aggregate effect-support AUC: **0.6516244589**;
- folds above random: **5/5**.

This mechanism is not retested or retuned in V54. It answers a first-stage question reliably: **will the frozen proposal-versus-incumbent intervention have any measurable closed-loop effect?**

### 3.2 ENDPOINT — STOP

V53 ENDPOINT state:

`[Q, P-Q, E-P, D, dx_T, dy_T, dyaw_T, dv_T]`.

Identification:

- aggregate conditional AUC: **0.5049495959**;
- folds > 0.5: **2/5**;
- folds > V52 scalar conditional control: **3/5**;
- folds > V51 scalar conditional control: **3/5**;
- preregistered identification: **FAIL**.

Diagnostic OOF retention nevertheless looks attractive in aggregate:

- selected: `461`;
- beneficial retained: `114/121 = 94.21%`;
- nonbeneficial: `381 -> 347`;
- hard harm: `25 -> 20`;
- score-delta sum: `-4.2409 -> -2.9866`;
- NegRMS: `0.15486 -> 0.14766`.

But `all-fold nonharm` fails. In folds 2 and 3, score sum reverses in the harmful direction. Because identification itself fails, these aggregate gains cannot justify threshold or functional rescue.

**Mechanism interpretation:** terminal signed direction/channel is not a stable sufficient statistic for effectful selected-outcome order.

### 3.3 TEMPORAL — STOP, but keep as diagnostic evidence

TEMPORAL adds fixed DCT-II `k=1,2` modes of planned `dx(t),dy(t),dyaw(t),dv(t)`.

Identification:

- aggregate conditional AUC: **0.5743801653**;
- folds > 0.5: **5/5**;
- folds > ENDPOINT: **4/5**;
- folds > V52 scalar: **3/5**;
- folds > V51 scalar: **3/5**;
- preregistered identification: **FAIL**;
- temporal-necessity claim: **FAIL**.

Diagnostic retention:

- selected: `454`;
- beneficial retained: `115/121 = 95.04%`;
- nonbeneficial: `339`;
- hard harm: `22`;
- score-delta sum: `-2.8500`;
- NegRMS: `0.15034`;
- `all-fold nonharm`: **FAIL**.

TEMPORAL is not noise: it beats endpoint in four folds and has substantially higher aggregate AUC. But it is not transport-stable relative to the already frozen scalar controls. Fold 3 is especially informative: TEMPORAL AUC `0.5955` ties V51 rather than exceeding it, while the retained score sum worsens from `-2.6427` to `-3.3636`.

Therefore V53 supports only the weaker diagnostic claim:

**planned temporal treatment shape contains outcome information, but it is not a preregistered stable/sufficient mediator of realized effect direction.**

It must not be promoted into the backbone or tuned by adding modes/horizon/basis.

---

## 4. What the model has learned / has not learned

### Learned and protected

1. bounded EAF/support/admissibility and frozen RSMR extremal proposal selection;
2. no-fallback incumbent containment;
3. V44 ungated prospective interaction support;
4. V45 agent-local longitudinal response as a supporting mediator;
5. V47 EGO-REF as a supporting consequence coordinate;
6. V50 metric-safe paired one-shot selected-outcome evidence;
7. V52 structural-null/effect-support factorization over QPE+D.

### Not learned

Given that an intervention is effectful, the current state still cannot stably answer:

**which realized intervention becomes beneficial versus harmful, and why?**

V53 now rules out a natural explanation: this is not fixed merely by preserving the frozen plan's signed terminal direction or two low-order temporal modes.

### What should be learned next

The next object is the **realized post-intervention treatment-control divergence** during the exact one-shot exposure window. Planned geometry describes what the planner intended to apply; the paired rollout can reveal what state transition actually occurred after that operator entered the closed loop.

This is a mediator-identification question. Because the state is post-treatment, it must not be used directly as the initial t=0 veto. If it is identified, a later version must make the mediator deployable by predicting it from pre-execution runtime state or by defining an auditable continuation operator.

---

## 5. Dominant bottleneck after V53

Previous bottleneck:

`effectful selected-outcome state sufficiency / directional-temporal operator contrast sufficiency`.

V53 narrows it to:

**post-intervention paired dynamic response / outcome-process sufficiency for effectful selected-outcome order.**

Equivalently, the key scientific question is:

> after the full-set RSMR proposal is frozen and known to be effectful, is final benefit/harm determined by a realized treatment-response mediator that is absent from the pre-execution plan geometry?

The paper headline is already stable and should not be renamed again.

---

## 6. Newly closed algorithm families

V53 closes the following as *sufficient conditional-outcome state families* under the current static pairwise ranker:

- scalar plan dose `D` alone (already known insufficient for sign);
- signed terminal proposal-incumbent operator geometry;
- fixed low-order pre-execution DCT shape of ego operator geometry.

Consequently, do not perform:

- endpoint transform / normalization sweep;
- more DCT modes;
- DCT basis sweep;
- temporal horizon sweep;
- peak/early handcrafted planned-trajectory statistics;
- larger MLP/attention over the same pre-execution plan-only geometry;
- threshold/alpha/lambda/loss/class/focal/catastrophe-weight rescue.

This closure is **not** a closure of Q/P/E, D as effect-support evidence, paired outcome supervision, post-intervention dynamics, or all structured outcome functionals.

All historical closures in `ALGORITHM_CHANGELOG.md` remain active.

---

## 7. Evidence chain through V53

The mechanism chain now reads:

- V34: ordinal extremal selection solved by RSMR;
- V37--V39: selected residual/tail is real;
- V40: pure value-head/19-D route falsified;
- V41: endpoint/basepoint geometry is a partial mediator;
- V42: current physical/quality consequence is a partial mediator;
- V43: prospective horizon adds necessary information;
- V44: ungated full-horizon prospective interaction support is a strong mediator; scene-global mode classifier is falsified;
- V45: agent-local continuous response is identifiable and improves selected value; a point response is not sufficient;
- V46: response second moment is identifiable but not decision-sufficient; temporal predictive features can improve regression while worsening deployment, directly establishing prediction sufficiency != decision sufficiency;
- V47: EGO-REF is a strong supporting consequence mediator; broader representation expansion stops;
- V48: multiplicity/logK is fresh-falsified; in-domain post-selection signal != fresh transportability;
- V49: random-prefix selection intervention fails; changing offline selection measure != selected-outcome-law identification;
- V50: metric-safe actual paired one-shot outcomes are obtained; paired outcome evidence != QPE-only selected-outcome state sufficiency;
- V51: QPE+D operator-relative state is identified but sign retention is deployment-insufficient;
- V52: QPE+D robustly identifies effect support, but static conditional sign and Pareto order fail; effect-support sufficiency != conditional-outcome-order sufficiency;
- **V53: planned signed endpoint/temporal operator geometry still does not give fold-stable conditional ordering; planned-treatment geometry != realized-outcome mediation sufficiency.**

This is a much stronger CCF-A-level mechanism story than a sequence of score-head tweaks: each failed family closes a specific sufficiency hypothesis under a bounded and auditable operator.

---

# V64.3.54 EAF-ICER-PDRM — Paired Dynamic Response Mediation

## 8. Scientific role

V54 is the exact preregistered V53 failure branch. It changes the *evidence source for the conditional mediator*, not RSMR, QPE, the paired labels, or the deployment threshold.

It asks whether the missing object is:

`planned treatment -> realized short-horizon paired state transition -> final paired outcome`.

V54 is intentionally **not yet a deployable t=0 retention mechanism**. That restriction is part of the scientific protocol, not a limitation to hide.

---

## 9. V54 state collection

Reuse the exact V50.5 population and outcome labels. For every one of the 502 full-set RSMR proposals, replay both original V50.5 arms:

- treatment: execute the frozen proposal once;
- control: execute the incumbent at the same anchor.

Collect only simulated ego rear-axle `(x,y,yaw,speed)` from iteration `0` through the first scheduled replan. The window length is the already frozen `planner.replan_interval_ticks` (currently 5), not a tunable horizon.

Construct paired realized channels in the shared t=0 control ego frame:

`[dx_real(t), dy_real(t), dyaw_real(t), dv_real(t)]`.

No new full-horizon score or safety metric is computed. The existing V50.5 full-horizon paired label remains the target.

### Engineering acceleration

V53 spent approximately `70025 s = 19.45 h` on a treatment-only full-scenario replay even though its state was available at the intervention event.

V54 instead:

1. runs only through the first replan window;
2. disables new metric computation (`run_metric=false`);
3. bypasses the expensive second BDSE replan at the terminal sample—the state is recorded immediately before that replan and the cached rollout is returned only to let the simulator exit;
4. runs treatment/control in parallel on two GPUs when available;
5. uses the first 8 scenes/arm as a fail-closed engineering sentinel via the existing batch mechanism;
6. reuses completed valid batches on resume.

This is a safer acceleration than reducing the 502 scientific population. The population/fold support is already marginal for some conditional outcomes, so sample shrinkage would weaken exactly the fold-stability question V53 exposed.

---

## 10. V54 causal arms

### Arm A — REALIZED-ENDPOINT

State:

`[Q, P-Q, E-P, planned_D, realized_dx_end, realized_dy_end, realized_dyaw_end, realized_dv_end]`.

Question:

> Is the actual treatment-control state transition at the end of the one-shot exposure window the missing mediator?

### Arm B — REALIZED-TEMPORAL

Only if REALIZED-ENDPOINT does not establish the minimal mediator, append fixed DCT-II `k=1,2` coefficients of the realized paired channels over the exact exposure window.

No horizon, mode-count or basis sweep is allowed.

Question:

> Is realized response shape necessary beyond the realized endpoint transition?

---

## 11. V54 identification gates

The gate is deliberately stricter than V53.

REALIZED-ENDPOINT must:

- aggregate AUC > 0.5;
- >=4/5 folds > 0.5;
- aggregate and >=4/5 folds beat exact V53 TEMPORAL;
- aggregate and >=4/5 folds beat exact V52 scalar conditional control;
- aggregate and >=4/5 folds beat exact V51 scalar conditional control.

REALIZED-TEMPORAL must satisfy all external comparator gates and also beat REALIZED-ENDPOINT in aggregate and >=4/5 folds before a temporal-necessity claim is allowed.

A retrospective retention gate is computed only as a diagnostic to ask whether an oracle-available mediator contains enough ordering information. It is **not promotion evidence**, because post-intervention state is unavailable at the initial t=0 decision.

---

## 12. V54 preregistered next branches

1. **REALIZED-ENDPOINT identified:** freeze it as the minimal outcome mediator. V55 must make it deployable via a pre-execution mediator predictor or auditable continuation operator. Do not use future state directly at t=0.
2. **Only REALIZED-TEMPORAL identified:** realized response shape is necessary; V55 predicts/uses that process without horizon/basis tuning.
3. **Mediator identified but retrospective retention still fails:** state contains outcome information but the static sign functional remains insufficient; next solve a structured paired dynamic outcome functional before adding more state.
4. **Both fail:** close short-horizon realized ego-response geometry as sufficient. The next paired process must represent realized interaction/safety consequence, not more ego trajectory geometry.

No untouched validation is allowed in V54 because V54 itself is not yet a deployable pre-action mechanism.

---

## 13. CCF-A novelty position

The novelty claim should not be “DCT features” or “another response head.” The stronger algorithmic contribution is the experimentally falsifiable hierarchy of sufficient objects under a bounded monotone planner interface:

`selection -> prospective valuation -> paired effect support -> conditional outcome mediation -> monotone retention`.

V54 specifically tests whether the gap between planned treatment and realized response is the missing causal/decision mediator. If positive, the next deployable algorithm can be justified by mediator evidence rather than by ad-hoc feature search. If negative, the result closes another whole family and points directly to realized interaction/safety process.

That discipline is important for a CCF-A submission: the paper should explain **why each deployed operator requires a different sufficient statistic**, not just report that a larger model improved a metric.
