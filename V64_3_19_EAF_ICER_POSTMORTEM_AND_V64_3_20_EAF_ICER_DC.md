# V64.3.19 EAF-ICER fresh-screen postmortem and V64.3.20 EAF-ICER-DC design

## Executive conclusion

V64.3.19 is a **mechanism-level true positive but not a full promotion**.

The support/dominance/operator decomposition genuinely repairs the V64.3.18 direct-incumbent replacement bottleneck on an untouched fresh 500-scene screen. The direct replacement precision increases from 35.29% (V18 profile) to 63.04% (V19 scalar) / 60.22% (V19 dual), and direct opportunity capture increases from 10.40% to 33.53% / 32.37%. Anchor recovery is 0%, so this cannot be explained by a looser anchor-recovery definition.

However, `full_promotion=false` because the regret endpoint remains outside the pre-registered 1.02x-raw tolerance. A scene-level audit shows that this endpoint failure is **not primarily caused by the newly successful direct-replacement mechanism**. It is dominated by an incomplete definition of deployment abstention in all-actions-safety-flagged scenes.

V64.3.20 therefore fixes deployment semantics only; it does not refit or expand the learned reliability model.

---

## 1. V64.3.19 fresh results

| arm | teacher match | teacher regret | beneficial | harmful | flip | final-guard block |
|---|---:|---:|---:|---:|---:|---:|
| DARM anchor | 16.8% | 24073.97 | - | - | - | - |
| raw EAF | 14.2% | **12960.05** | 7.8% | 10.4% | 61.0% | 31.8% |
| frozen V18 DACER-profile | 22.8% | 14030.53 | 6.2% | 0.2% | 37.8% | 0% |
| V19 ICER-scalar | **24.2%** | 14007.24 | **7.6%** | 0.2% | 37.8% | 0% |
| V19 ICER-dual | 23.8% | **13620.09** | 7.2% | 0.2% | 38.0% | 0% |

Screen verdict:

- instrumentation: PASS;
- candidate support: PASS;
- fresh support/dominance generalization: PASS;
- incumbent recovery mechanism: PASS;
- gain over V18: PASS;
- signed-profile composite causal support: PASS;
- final guard alignment: PASS;
- preservation: PASS;
- endpoint: FAIL;
- full promotion: FALSE.

The endpoint fail is specifically regret: V19-dual is ~5.09% above raw, while the registered tolerance is 2%.

---

## 2. Did support/dominance/operator decoupling fix incumbent replacement?

**Yes, at the mechanism level.**

| mechanism metric | V18 profile | V19 scalar | V19 dual |
|---|---:|---:|---:|
| support AUC | - | 0.7983 | 0.7983 |
| direct dominance AUC | - | 0.7703 | **0.7841** |
| direct replacement rate | 17.29% | 31.19% | 31.53% |
| direct replacement precision | 35.29% | **63.04%** | 60.22% |
| direct opportunity capture | 10.40% | **33.53%** | 32.37% |
| alternative precision | 86.27% | 91.30% | **93.55%** |
| anchor recovery | - | **0%** | **0%** |

This is the first version where the intended incumbent-relative causal link is supported on fresh data rather than merely by anchor preservation or fallback.

The main paper mechanism therefore should remain centered on **incumbent-contrastive reliability**, not return to selector/acquisition redesign.

---

## 3. Does signed selected-evidence attribution add value?

**It adds fresh information, but the precise claim must be narrower than “it improves direct replacement precision.”**

Dual relative to scalar:

- combined direct dominance AUC: 0.7703 -> **0.7841**;
- alternative precision: 91.30% -> **93.55%**;
- alternative teacher margin: 1.459 -> **1.512**;
- regret: 14007.24 -> **13620.09**;
- selected-nonanchor teacher-better: 89.95% -> **90.53%**.

But:

- direct replacement precision: 63.04% -> **60.22%**;
- direct capture: 33.53% -> **32.37%**;
- match: 24.2% -> **23.8%**.

Among the 39 scenes where scalar and dual deploy different actions, dual has lower teacher regret on 18 scenes and higher regret on 21 scenes, yet the total dual-minus-scalar regret is **-193576.31**. The signed profile is therefore behaving like an **extremal/tail-quality or improvement-magnitude signal**, not a clean binary replacement-precision booster.

Decision:

- retain the signed profile as a structured main-candidate view;
- retain scalar ICER as a mandatory ablation;
- do not make “signed attribution raises direct replacement precision” a paper claim unless later fresh/full-val evidence demonstrates it.

---

## 4. New primary bottleneck: deployment is not complete in the all-flagged domain

The crucial paired decomposition of V19-dual versus raw is:

| path | scenes | total dual - raw teacher regret |
|---|---:|---:|
| same final action | 292 | 0 |
| direct incumbent -> alternative | 93 | **-191056.43** |
| admissible incumbent -> anchor | 105 | **-96001.71** |
| all-flagged structural-domain divergence | **10** | **+617078.33** |

There are 18 all-actions-safety-flagged scenes total. All of the +617078.33 all-flagged excess is concentrated in 10 final-action divergences. On the remaining 482 non-all-flagged scenes, V19-dual improves raw by **-287058.14 total teacher regret**.

Thus the direct incumbent mechanism is not the endpoint bottleneck. The remaining issue is the definition of the deployment incumbent / abstention operator.

### Why V19 is semantically incomplete

V19 defines the learned deployment incumbent as:

- raw EAF incumbent if it passes the pre-structural final-guard-admissible mask;
- otherwise DARM anchor.

The mask deliberately has no learned-admissible challenger when all valid actions are safety-flagged. In those scenes V19 therefore selects the DARM anchor before the downstream structural guard.

But the frozen planner subsequently applies a continuous all-flagged structural-risk guard. The DARM anchor is not a neutral no-op before that stage: selecting anchor changes the proposal/tie-break context that enters the structural post-processing and can change the final deployed action.

Correct learned abstention in this domain is therefore:

> **preserve the frozen raw-EAF proposal and delegate the entire scene to the unchanged structural-risk deployment stack.**

A design-only hybrid replay on the already-inspected V19 500 scenes (raw final behavior in all-flagged scenes, V19 dual elsewhere) gives teacher match **23.4%** and regret **12385.93**. This cannot be used as validation or promotion evidence, but it establishes a highly specific next causal hypothesis.

---

## 5. Novelty decision

Keep the current headline novelty:

> **evidence-attributed incumbent-contrastive reliability for deployment-admissible extremal recovery under a fixed planner-interface evidence budget**

Do not inflate it again in V20. Instead make `deployment-admissible` technically exact across the full deployment operator.

The refined mechanism is domain-partitioned:

1. **safe-available domain**: complete final-guard-admissible frontier + frozen ICER support/dominance operator;
2. **all-flagged structural domain**: no learned intervention; preserve raw proposal and delegate to frozen continuous structural-risk guard.

This is a stronger CCF-A-style mechanism story than adding another generic classifier because every learned intervention is now aligned with the actual planner deployment stack, while abstention is explicitly defined as preservation of the downstream operator rather than replacement with an arbitrary anchor.

Signed attribution remains an important structured component, but it is not yet the headline claim independently of incumbent-contrastive recovery.

---

## 6. V64.3.20 EAF-ICER-DC

Name: **Evidence-Attributed Incumbent-Contrastive Extremal Recovery — Deployment Complete**.

V20 changes no learned parameters.

### Safe-available domain

Execute exactly V19:

- same B<=16/M=24 evidence interface;
- same frozen EAF value;
- same guard-admissible frontier;
- same V19 support head;
- same scalar/profile dominance heads;
- same fixed zero support/dominance thresholds;
- same selection/tie-break;
- same evidence / one-sided / structural guards.

### All-flagged domain

If every valid action is safety-flagged:

- skip learned ICER support/dominance selection;
- preserve the legacy raw-EAF proposal;
- expose explicit delegation diagnostics;
- let the unchanged downstream one-sided/evidence and continuous structural-risk stack determine the deployed action.

The old V19 behavior remains available only in the frozen V19 control config. V20 fitted configs explicitly set:

`all_flagged_policy: preserve_legacy_for_structural_guard`.

---

## 7. Why no magnitude objective in V20

The changelog allows teacher-improvement magnitude / robust extremal ordering when direct recovery and preservation pass but regret fails. V19 formally meets that condition.

However, the new decomposition shows that direct replacements and incumbent->anchor decisions jointly **reduce** regret relative to raw; a deterministic 10-scene structural-domain mismatch dominates the endpoint failure.

Adding a magnitude objective now would confound a semantic deployment bug with a representation/ordering change. Therefore:

- V20 = deployment semantics only;
- if untouched V20 fresh still passes mechanism/preservation but fails regret, then V21 may add a TRAIN-only teacher-improvement magnitude objective on the same frozen deployment-complete frontier.

---

## 8. V64.3.20 causal screen design

Fresh untouched 500 scenes, four independent replays:

1. raw EAF;
2. frozen V19 ICER-scalar control;
3. V20 ICER-DC-scalar;
4. V20 ICER-DC-dual.

No fit stage.

Causal comparisons:

- **V19 scalar -> V20 scalar**: only all-flagged deployment semantics;
- **V20 scalar -> V20 dual**: signed-attribution increment;
- **raw -> V20 dual**: final preservation/endpoint.

Structural-domain hard gates include:

- >=5 all-flagged fresh scenes;
- V20 structural delegation =100%;
- V20 pre-structural selected action = raw legacy proposal =100% in all-flagged scenes;
- V20 final action = raw final action =100% in all-flagged scenes;
- safe-domain V20 scalar selected/final action = frozen V19 scalar =100%.

Direct incumbent recovery, AUC, preservation, final-guard and endpoint gates remain frozen.

If signed-profile incremental support fails, the next action is to demote dual and keep scalar ICER-DC rather than retune the dual mixing weight.

If deployment-completeness/recovery/preservation all pass but endpoint still fails, only then enter teacher-improvement magnitude ordering.

---

## 9. Speed result

V19 speed work was successful:

| stage | seconds |
|---|---:|
| prerequisites | 15 |
| frozen train reuse | 10 |
| fresh token selection | 4 |
| train-only fit | 23 |
| raw/V18 wave | 319 |
| ICER wave | 332 |
| screen | 5 |
| **total** | **708 (~11.8 min)** |

Therefore the prior cache scan/cold-I/O bottleneck has largely been removed. V20 keeps pre-deserialization token filtering and independent arm replay. It also removes the 23-second fit stage because the V19 heads are copied exactly.

No additional invasive multi-arm evaluator optimization is justified before algorithmic validation.

---

## 10. Data discipline

The inspected V19 fresh 500 scenes are now permanent design data.

- previous design exclusion: 2700;
- V19 fresh: 500;
- V20 exclusion: **3200 unique validation tokens**.

V20 fresh selection uses token identity + fixed hash only and reads no teacher/regret/reliability label.

---

## 11. Engineering audit

Final code audit found and fixed one branch-specific instrumentation error: the new all-flagged delegated path skips dominance-head inference, but diagnostic serialization initially referenced dominance arrays initialized only inside the learned branch. V20 now initializes scalar/profile dominance diagnostic arrays to zero before branch selection.

Final validation:

- V64.3.6–V64.3.20 targeted: **95/95 PASS**;
- full repository: **383/383 PASS**;
- warnings: **36**, unchanged existing Transformer `nested_tensor/norm_first` warnings;
- V20 raw/scalar/dual contracts: PASS;
- launcher shell syntax: PASS;
- 5000 randomized safe-domain V19/V20 cases: **0 action differences, 0 admissible-mask differences, 0 support-logit differences, 0 dominance-logit differences**.

No teacher/future signal was added to runtime. V20 introduces no new learned parameter and no validation-derived threshold.

---

## 12. Decision tree after V20

1. **Structural-domain identity fails** -> engineering/deployment-path bug; do not alter reliability model.
2. **V19 safe-domain mechanism fails to reproduce** -> audit token/config/operator identity; do not tune thresholds.
3. **Signed dual not incremental** -> demote dual, retain scalar ICER-DC; do not tune 0.5 mixing weight.
4. **Recovery/preservation fail** -> audit extremal false positives; keep selector/B/M/guards frozen.
5. **Everything except regret passes** -> next allowed branch is TRAIN-only teacher-improvement magnitude / robust extremal ordering on the same frozen deployment-complete frontier.
6. **Full screen passes** -> one independent frozen full-validation reproduction only; test/closed-loop remain forbidden until reproduction passes.
