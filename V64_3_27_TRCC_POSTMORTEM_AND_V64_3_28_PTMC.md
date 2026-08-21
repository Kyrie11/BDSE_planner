# V64.3.27 TRCC Result Audit, Causal Attribution, and V64.3.28 PTMC Design

## Executive conclusion

The uploaded V64.3.27 result is a **valid algorithmic TRAIN STOP**, not an engineering false stop. The uploaded code/result pair is internally consistent: the fitter can be independently replayed on the 3000-scene type-resolved TRAIN frontier and reproduces the official stop statistics.

V27 provides a clean causal result:

1. **No-fallback monotonicity worked structurally.** TRCC never created a replacement outside the V25 aggregate-DRC proposal set.
2. **The V27 local type-KNN confirmation signal failed.** It retained only 19/71 aggregate proposals, vetoed 35 teacher-positive proposals, and still retained the single meaningful -0.545756 catastrophic proposal.
3. Therefore the previous candidate headline, `monotone cross-view ... tail-regret confirmation`, is **not supported**.
4. The dominant bottleneck is now tightened to **proposal-conditioned rare catastrophic-mode detection under a high-retention constraint**.
5. A design diagnostic using the *same 24-D type representation* but replacing local KNN regret estimation with a global class-conditional catastrophic-mode contrast retains 68/71 proposals, is 5/5 fold-safe, improves worst selected improvement from -0.545756 to -0.009837, and improves selected negative RMS from 0.064782 to 0.001287. The same global estimator on 18-D aggregate or naive 42-D aggregate+type concatenation fails. This is TRAIN design evidence only, not independent paper evidence.

V64.3.28 therefore makes exactly one estimator-level change: **V25 aggregate DRC proposes one candidate; a global type-resolved tail-mode model may only confirm/veto that same candidate; failed confirmation returns to the incumbent with no fallback.**

---

## 1. V27 execution validity

Uploaded result execution chain:

`prerequisite re-audit -> targeted regression -> frozen 3000-TRAIN type-resolved instrumentation -> V27 fitter/gate -> STOP TRAIN`

No V27 fresh block was selected or evaluated.

Audited identity/provenance:

- TRAIN scenes: **3000 unique**
- frontier rows: **75,133**
- frozen replacement edges after final-guard/support/scalar-dominance filtering: **1,455**
- replacement scenes: **310**
- TRAIN token SHA256: `b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4`
- `fresh_validation_used=false`
- no traceback/runtime exception

Independent replay of the uploaded V27 fitter reproduces the official cross-fold statistics and `train_gate_pass=false`. Therefore V27 is a **scientific/algorithmic STOP**.

---

## 2. Official V27 TRAIN causal matrix

| arm | safe folds | selected | teacher-improvement sum | negative RMS | worst |
|---|---:|---:|---:|---:|---:|
| V25 aggregate DRC control | **5/5** | **71** | **+5.527642** | **0.064782** | **-0.545757** |
| type-only KNN diagnostic | 3/5 | 65 | +4.675357 | **0.311892** | **-1.959070** |
| V27 TRCC main | **0/5** | **19** | +1.442087 | **0.125205** | **-0.545757** |

TRCC formal failure is not just the per-fold minimum support. The main mechanism also fails the intended high-retention/tail-improvement purpose:

- aggregate proposals = 71
- retained = 19
- vetoed = 52
- vetoed teacher-positive = **35**
- vetoed teacher-nonpositive = 17
- worst V25 catastrophic proposal remains retained

Hence the correct action is STOP; do not relax `MIN_SELECTED`, the total 64-retained contract, or the V27 confirmation boundary.

---

## 3. Scene-level causal attribution

### 3.1 Catastrophe V27 fails to veto

Scene `b003a3bfbdb252e3`, action 19:

- true incumbent-relative teacher improvement: **-0.5457565672**
- support logit: **+0.292628**
- scalar dominance: **+1.040782**
- V25 aggregate DRC score: **+0.020969** -> proposed
- V27 type-KNN confirmation score: **+0.081467** -> confirmed

Thus the type confirmation is confidently on the wrong side for the only materially catastrophic aggregate proposal.

### 3.2 Large beneficial proposals V27 incorrectly vetoes

Examples:

| scene/action | teacher improvement | aggregate DRC | type-KNN confirmation |
|---|---:|---:|---:|
| `d97f7fc3e5a35f1e`, a3 | **+2.168968** | +0.043658 | **-0.550445** |
| `aaf8a6e715785915`, a17 | **+0.989941** | +0.004844 | **-0.094553** |
| `9ddade724a6a54e2`, a3 | **+0.929244** | +0.056742 | **-0.368176** |

So the failure is not simply “confirmation is conservative”. It is **mis-ranked with respect to the proposal-conditioned outcome**.

### 3.3 Proposal-conditioned discrimination quality

On the 71 aggregate proposals:

- type-KNN score Pearson correlation with actual teacher improvement: **0.0211**
- Spearman correlation: **0.2525**
- AUC for teacher-positive vs nonpositive proposal: **0.6210**

A post-hoc threshold sweep restricted to settings retaining at least 64/71 proposals finds **no threshold that improves either negative RMS or worst outcome**. The best such threshold is effectively “no veto”: 71 retained, negative RMS 0.064782, worst -0.545757.

Therefore the following strategy is now specifically falsified:

> “keep the V27 type-KNN score and tune its confirmation threshold/weight until retention is acceptable.”

---

## 4. What V27 says about type semantics and B<=16 capacity

V27 does **not** show that atom-type semantics contain no useful information.

The 24-D type representation has 8 constant-zero coordinates in this frozen population (candidate and delta for occupancy, drivable_area, wrong_way, red_light), leaving 16 active coordinates. This is an empirical property of the current frozen evidence generator, not an implementation bug; constant dimensions contribute zero discriminative information and do not corrupt the fixed runtime schema.

V27 *does* show that local KNN continuous-regret estimation is a poor estimator for the new problem after proposal generation.

The relevant task after V25 is no longer to estimate a smooth local mean regret for all candidate edges. It is:

> among a small, already selected aggregate proposal population, reject the very rare catastrophic mode while retaining most beneficial proposals.

This is a **rare-mode detection / high-retention confirmation** problem.

Therefore fixed B<=16 is still frozen for V28. V27 raises suspicion about interface observability, but does not identify interface capacity failure because the same existing type information has not yet been tested with an estimator aligned to rare-mode detection.

---

## 5. Dominant bottleneck after V27

Previous bottlenecks evolved as follows:

- V24: representation-induced neighborhood distortion
- V25: aggregate semantic outcome aliasing
- V26: representation-conditional neighborhood instability
- V27: local type-KNN confirmation cannot separate catastrophic proposals under the required high-retention regime

The dominant bottleneck is now:

> **proposal-conditioned rare catastrophic-mode detection under a high-retention constraint.**

More formally:

> **candidate-specific catastrophic-mode observability after the deployment-admissible extremal proposal has already been frozen.**

Proposal generation is no longer the next research question. The next mechanism must selectively veto a tiny harmful subset without reopening proposal geometry or deleting the recovery population.

---

## 6. Explicit no-repeat list through V27

Do not continue any of the following as the next primary mechanism:

1. V24 absolute-sorted/L1-normalized full attribution spectrum.
2. V26 family coordinates concatenated into the aggregate KNN geometry.
3. Any additional semantic/type dimensions flattened into a single KNN metric.
4. V27 type-KNN confirmation with threshold, K, type-weight, or group-weight tuning.
5. mean-SE + DRC weighted mixing.
6. downside multiplier sweeps.
7. zero-boundary sweeps.
8. support/scalar-dominance threshold tuning to remove known failures.
9. action/maneuver blacklists.
10. standalone KNN-radius/OOD rejection as the main solution.
11. transition geometry as the headline risk metric.
12. signed-profile ranking as the headline operator.
13. simple AND stacking of previously failed risk views.
14. pooled fold/fresh rescue.
15. naive aggregate+type concatenation under a new global classifier.
16. broad unfreezing of EAF/acquisition/selector/B/M **before V28's existing-interface observability test is completed**.

No-fallback monotonicity is retained as a **structural invariant**, not as a validated confirmation mechanism by itself.

---

## 7. V28 TRAIN design diagnostic: estimator versus representation

After inspecting V27 TRAIN, a design diagnostic was performed on the same frozen 3000 TRAIN population. It is explicitly **not independent evidence** and cannot support the paper claim by itself.

A catastrophic mode is frozen at teacher improvement `<= -0.5`. On V27 TRAIN there are 124 such edges. The threshold lies in an observed empty interval:

- nearest below: **-0.5457565672**
- nearest above: **-0.4778675650**

The new estimator is a transparent equal-prior diagonal-Gaussian class-conditional likelihood ratio:

`risk(x) = log p(x | catastrophic) - log p(x | non-catastrophic)`.

The confirmation threshold is calibrated on TRAIN aggregate proposals to preserve **95% of teacher-positive proposals**. It is frozen before V28 fresh selection.

Causal estimator/representation controls:

| confirmation estimator/view | fold safe | retained | teacher sum | negative RMS | worst |
|---|---:|---:|---:|---:|---:|
| no confirmation: V25 aggregate DRC | 5/5 | 71 | +5.527642 | 0.064782 | -0.545757 |
| V27 local type-KNN | 0/5 | 19 | +1.442087 | 0.125205 | -0.545757 |
| global tail-mode on 18-D aggregate | 4/5 | 68 | +5.524821 | 0.066195 | -0.545757 |
| global tail-mode on naive 42-D aggregate+type | 4/5 | 67 | +5.523452 | 0.066687 | -0.545757 |
| **global tail-mode on 24-D type, proposal-conditioned** | **5/5** | **68** | **+6.072558** | **0.001287** | **-0.009837** |

The V28 design diagnostic vetoes only three aggregate proposals:

- `66da55b392b45a5d`, action 17, improvement +0.000167
- `8e6c888a92425ff1`, action 20, improvement +0.000674
- `b003a3bfbdb252e3`, action 19, improvement **-0.545757**

This isolates a specific hypothesis:

> the current type evidence may contain catastrophic-mode information, but it is useful as a **global rare-mode discriminant conditioned on an already frozen proposal**, not as a local continuous-regret geometry.

---

## 8. V64.3.28 EAF-ICER-PTMC

Full name:

**Evidence-Attributed Incumbent-Contrastive Extremal Recovery with Proposal-Conditioned Tail-Mode Confirmation.**

### Stage 1: frozen V25 aggregate DRC proposal

No changes to:

- fixed planner-interface B<=16
- EAF acquisition / selector / attribution generation
- complete DARM-anchor frontier
- final-guard deployment admissibility
- support > 0
- scalar dominance > 0
- aggregate 18-D risk representation
- K={32,64}
- inverse-distance local downside certificate
- downside multiplier=1
- zero boundary
- scalar-dominance extremal ranking
- incumbent-default preservation

This stage produces at most one extremal alternative `b*`.

### Stage 2: global type-resolved catastrophic-mode confirmation

Use the same fixed 24-D atom-type representation from V27. Do not alter or concatenate it with aggregate features.

Offline TRAIN-only model:

1. standardize each type coordinate;
2. label frozen catastrophic edges by `Delta <= -0.5`;
3. fit diagonal Gaussian moments for catastrophic and non-catastrophic classes;
4. use **equal class priors** to avoid allowing base-rate rarity to trivially suppress tail risk;
5. compute a log-likelihood ratio risk score;
6. calibrate one TRAIN-only threshold at 95% teacher-positive *aggregate-proposal* coverage.

Runtime:

- aggregate DRC proposes `b*`;
- type tail model evaluates only `b*`;
- risk below the frozen threshold -> confirm same `b*`;
- otherwise -> preserve incumbent;
- **no fallback / no second alternative / no reselection**.

Structural invariant:

`PTMC selected replacements subseteq V25 aggregate-DRC selected replacements`.

---

## 9. V28 experiment protocol

### 9.1 TRAIN is a design/implementation gate, not paper evidence

Because V28 was designed after inspection of V27 TRAIN, its TRAIN cross-fit is only a fail-closed implementation/design check.

It must still satisfy:

- exact frozen TRAIN token SHA `b36a847e...47da4`;
- exactly 75,133 frontier rows;
- exactly 1,455 replacement edges / 310 replacement scenes;
- 5/5 fixed scene folds selected-path safe;
- >=64 retained replacements;
- teacher-improvement sum >=0;
- subset/no-fallback invariant;
- selected negative RMS and worst both non-worse than V25 aggregate DRC;
- at least one tail metric strictly better.

No V28 fresh may be used if this gate fails.

### 9.2 New untouched double-fresh is the actual mechanism test

V26 and V27 both TRAIN-stopped and consumed no fresh validation scenes. Therefore the frozen design exclusion remains **6700 unique validation tokens**.

Use a new hash seed to select 1000 untouched scenes:

- A = 500
- B = 500

Four arms per split:

1. raw
2. V20 preservation control
3. V25 aggregate DRC proposal control
4. V28 PTMC

Both A and B independently require:

- exact paired token identity;
- global-tail-mode runtime instrumentation active;
- structural guard/deployment invariants;
- PTMC replacement set/action is a subset of aggregate DRC with the same proposed action;
- direct incumbent->alternative replacement path non-harm;
- recovery/capture support;
- selected negative-tail non-worse and strictly improved over aggregate DRC;
- asymmetric preservation non-degradation;
- match/regret endpoint contract.

No pooled rescue.

### 9.3 Interpretation matrix

If PTMC passes both untouched blocks:

- existing B<=16 type information is sufficient for at least this rare-mode confirmation problem;
- V27 failed primarily because of estimator/objective mismatch, not because type information was absent;
- freeze V28 and allow exactly one independent full-val reproduction.

If TRAIN passes but either untouched A/B fails with residual catastrophic tail:

- do **not** tune `-0.5`, 95% coverage, K, downside, support/dominance, or type weights on validation;
- current within-interface tail-mode estimator is not fresh-stable;
- evidence for fixed-interface observability/capacity insufficiency becomes strong enough to stop KNN/confirmation-geometry iterations and reopen evidence-interface/acquisition research.

If PTMC is path-safe but fresh recovery collapses:

- the rare-mode model is over-rejecting under distribution shift;
- do not recover by validation threshold tuning;
- treat the fixed interface/representation as insufficiently calibrated for high-retention deployment.

---

## 10. Paper mainline after V27

The V27 candidate headline must be discarded as an established claim.

The code-faithful V28 mechanism chain is now:

`fixed B<=16 planner-interface evidence`
`-> auditable selected evidence / exact EAF attribution`
`-> frozen complete DARM-anchor frontier`
`-> complete deployment-admissible candidate population`
`-> frozen support + scalar incumbent dominance`
`-> aggregate downside-regret extremal proposal`
`-> proposal-conditioned global type-resolved catastrophic-mode confirmation of the same candidate`
`-> no-fallback incumbent preservation`
`-> unchanged evidence/one-sided certificate`
`-> unchanged structural-risk guard`
`-> final decision preservation`
`-> preservation + endpoint`.

The paper should remove/demote as headline mechanisms:

- full-spectrum attribution-resolved regret geometry;
- semantic-family flat KNN;
- local type-KNN confirmation;
- standalone DRC sufficiency;
- transition geometry;
- signed-profile ranking;
- generic AUC/binary reliability as the endpoint reliability definition.

### Novelty status

Do **not** currently claim the V27 headline.

If and only if V28 succeeds on both untouched blocks and one independent full-val reproduction, a candidate headline is:

> **evidence-attributed proposal-conditioned catastrophic-mode certification for monotone deployment-admissible extremal recovery under a fixed planner-interface evidence budget.**

The conceptual novelty is not the diagonal Gaussian model itself. The intended CCF-A-level mechanism claim is the **problem decomposition**:

1. deployment-admissible extremal proposal is learned/selected under aggregate downside evidence;
2. rare catastrophic-mode observability is treated as a separate proposal-conditioned problem;
3. an independent evidence view can only *shrink* the intervention set, never create/re-rank a new intervention;
4. preservation and endpoint gains must emerge under a fixed auditable evidence budget.

If V28 fresh fails, this headline is also rejected, and the paper should pivot the next research question to **evidence-interface observability/capacity**, not another confirmation estimator.

---

## 11. Engineering changes in V28

- added immutable global tail-mode model loader with SHA256 and schema checks;
- added runtime equal-prior diagonal-Gaussian likelihood-ratio confirmation;
- preserved aggregate DRC proposal and no-fallback selector helper;
- added exact 24-D type schema contract;
- froze catastrophic label threshold=-0.5 and positive-proposal coverage=0.95 in config/model contracts;
- added exact frozen TRAIN token SHA, frontier-row, replacement-edge and replacement-scene hard contracts;
- V28 can directly reuse the completed V27 768MB type-resolved TRAIN provenance via `V28_TRAIN_EDGES` or the standard V27 output path;
- added global tail-mode fresh instrumentation check;
- retained corrected paired identity semantics from V25 (exact token set + identical within-split arm order, not manifest order equality);
- fixed launcher version-name leftovers and restored V27 regression to the V28 targeted test stack;
- added V28 unit tests for tail-mode math, runtime/offline equality, SHA/schema, config freeze, no-fallback and launcher contracts.

No server nuPlan fresh result is fabricated in this package.
