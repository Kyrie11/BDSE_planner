# V64.3.17 EAF-DALER Postmortem and V64.3.18 EAF-DACER Design

## Executive decision

V64.3.17 should **not** be promoted to full/test/closed-loop. The screen stop is correct, but the causal interpretation needs refinement.

The prior diagnosis remains partly correct: frozen EAF contains generalizable evidence-attributed reliability information, and the difficult step is converting that information into a reliable extremal intervention. V64.3.17 improves preservation/deployment alignment, but it does **not** establish all-challenger extremal recovery. The primary newly identified failure is earlier than representation capacity: V64.3.17 incorrectly promoted the upstream legacy utility-refinement candidate pool to a hard deployment-admissibility condition, collapsing almost every learned listwise scene to zero or one challenger.

Therefore V64.3.18 keeps acquisition, B/M, EAF value, DARM/DBR, evidence certificate, one-sided guard, and structural guard frozen. It corrects the learned candidate semantics, then causally tests incumbent-relative counterfactual ordering and exact signed selected-atom attribution.

Main V64.3.18 method: **EAF-DACER — Evidence-Attributed Deployment-Admissible Counterfactual Extremal Recovery**.

Recommended novelty wording:

> **evidence-attributed counterfactual dominance for deployment-admissible extremal recovery under a fixed planner-interface evidence budget**

Here “counterfactual” means the operational candidate-versus-frozen-incumbent contrast available inside one frozen planner frontier; it is not a causal-identification claim.

---

## 1. What V64.3.17 actually showed

### Fresh 500-scene endpoints

| Arm | Teacher match | Teacher regret | Flip | Beneficial | Harmful |
|---|---:|---:|---:|---:|---:|
| DARM anchor | 17.60% | 25196.11 | — | — | — |
| raw EAF | 17.80% | **12349.05** | 58.80% | 9.60% | 9.40% |
| scalar EAIR | 22.00% | 13805.09 | 32.80% | 5.60% | 1.20% |
| RAER | **22.60%** | 12840.36 | 39.20% | 6.60% | **1.60%** |
| DALER | 22.20% | 12573.01 | 40.20% | 7.00% | 2.40% |

Important causal facts:

- raw EAF regret is about **51.0% lower** than the DARM anchor on this fresh set. Frozen EAF value remains highly useful; there is no evidence to reopen acquisition/selector/B/M.
- DALER reduces harmful intervention by **7.0 absolute points** versus raw EAF while retaining **72.9%** of raw beneficial interventions.
- DALER regret is **2.08% lower than RAER**, while match is only 0.4 pp lower; thus the prior paired endpoint rule is satisfied.
- final one-sided/evidence guard blocks **0%** of DALER proposals, so V64.3.17 did improve pre-selection/final-guard alignment.
- these preservation/endpoint gains are real, but they do not prove runner-up recovery.

### Reliability signal still exists

- V64.3.17 train internal-holdout exact-executable AUC: **0.7812**.
- fresh exact-executable AUC: **0.8316**.
- frozen RAER fresh all-frontier AUC: **0.7409**.

The fresh DALER all-frontier AUC is only **0.4729**, but this is an out-of-support evaluation for a head trained almost entirely on the collapsed utility-equivalent subset. It should not be interpreted as “EAF attribution lost reliability information.”

### The recovery mechanism did not work

On 476 fresh raw-EAF proposal scenes:

- RAER alternative recovery: **1.68%** (8 scenes), precision **50%**, mean teacher margin **+0.156**.
- DALER alternative recovery: **0.21%** (1 scene), precision **0%**, mean teacher margin **-0.425**.
- DALER selected-nonanchor teacher-better rate is high (**84.58%**), but this is mostly a high-quality keep/filter behavior, not alternative recovery.

So V64.3.17 is best characterized as a stronger **preservation/abstention mechanism**, not a demonstrated all-challenger recovery operator.

---

## 2. Why the V64.3.17 listwise story was structurally starved

The screen’s `instrumentation_valid=false` is **not** a missing-prefix/instrumentation bug. Diagnostics are present. The old checker bundled `executable_edge_count >= 512` into the instrumentation gate; fresh V64.3.17 has only **285** exact executable edges, so that composite gate failed.

The actual candidate masks are decisive.

### Fresh 500 scenes

| Mask | Edge count | Mean/scene | Zero-candidate scenes | Singleton scenes | Multi-candidate scenes |
|---|---:|---:|---:|---:|---:|
| final-guard prerequisites (`daler_guard_executable`) | **5205** | 10.41 | 218 | 8 | **274** |
| legacy utility-equivalent pool | **481** | 0.962 | 24 | **473** | 3 |
| V64.3.17 intersection (`daler_executable`) | **285** | 0.57 | **219** | **279** | **2** |

Of the 481 fresh utility-equivalent edges, **476 are exactly the legacy raw-top action**; only 5 are alternatives.

### Train 3000 scenes

| Mask | Edge count | Mean/scene | Zero-candidate scenes | Singleton scenes | Multi-candidate scenes |
|---|---:|---:|---:|---:|---:|
| final-guard prerequisites | **30352** | 10.12 | 1322 | 43 | **1635** |
| legacy utility-equivalent pool | **2788** | 0.929 | 227 | **2763** | 10 |
| V64.3.17 intersection | **1676** | 0.559 | **1326** | **1673** | **1** |

Only **one train scene** contained multiple V64.3.17 executable challengers. Consequently, the nominal scene-level listwise objective was almost always just anchor-versus-one-edge classification. The model had essentially no training support for learning alternative challenger ordering.

### Root semantic error

`_certificate_utility_refinement_context` defines the candidate set used by an **upstream incumbent-construction heuristic** (score slack/top-k/pair certificate/utility cost). It is not the final execution guard. V64.3.17 mistakenly elevated membership in that pool to a hard learned-intervention admissibility condition.

This error did not leak teacher/future data, and it did improve guard alignment, but it removed the very alternatives required to test the paper’s recovery claim.

---

## 3. Is the previous bottleneck solved?

Only partially.

The previous bottleneck was:

> frozen EAF has useful evidence-attributed reliability information, but reliability does not yet enter the extremal selection operator correctly.

V64.3.17 alleviated the **preservation** half of this problem:

- harmful flips are strongly reduced;
- beneficial retention remains useful;
- final guard performs no hidden cleanup;
- endpoint regret is competitive with RAER.

But it did not solve the **extremal recovery** half. More importantly, the new evidence shows that V64.3.17 did not provide a fair test of listwise recovery because its hard candidate mask eliminated almost every alternative before the learned operator ran.

The bottleneck should therefore be split into two ordered questions:

1. **Candidate semantics:** is the learned operator competing over the actual final-guard-admissible complete frontier rather than an upstream incumbent heuristic?
2. **Relative extremal ordering:** given a healthy multi-challenger admissible frontier, can selected-evidence attribution identify an alternative that is better than both the anchor and the frozen incumbent?

V64.3.18 addresses (1) deterministically and tests (2) with pre-registered causal ablations.

---

## 4. What happens to the paper story and novelty

### Do not revert the mainline

The fixed-interface/evidence-attribution story remains the strongest coherent thread. The new mainline is:

**fixed planner-interface evidence cap B<=16**
→ auditable evidence atoms
→ terminally frozen acquisition, M=24
→ selected evidence
→ frozen EAF complete DARM-anchor frontier value
→ exact signed selected-evidence attribution
→ **complete final-guard-admissible challenger frontier**
→ **anchor support + incumbent-relative evidence-attributed dominance**
→ **counterfactual extremal recovery / explicit anchor abstention**
→ unchanged one-sided + evidence certificate
→ unchanged all-flagged structural-risk guard
→ final decision preservation.

This is an upgrade/refinement of the V64.3.17 mainline, not a change of research direction.

### Refine the novelty sentence

V64.3.17 wording:

> evidence-attributed, deployment-aligned listwise reliability for extremal decision selection under a fixed planner-interface evidence budget

is too generic and is weakened by the fact that V64.3.17 rarely contained a real list to rank.

V64.3.18 wording:

> **evidence-attributed counterfactual dominance for deployment-admissible extremal recovery under a fixed planner-interface evidence budget**

is more precise because the paper-level mechanism is not “we trained a listwise classifier.” It is:

- a fixed, auditable evidence bottleneck;
- exact decomposition of the frozen EAF value into selected-evidence contributions;
- a deployment-admissible complete frontier that does not reuse an incumbent heuristic as a gate;
- an incumbent-relative evidence representation;
- an extremal operator that can either recover a better alternative or explicitly abstain to the anchor;
- unchanged downstream certificates.

The paper should say **all final-guard-admissible challengers on the complete frozen EAF anchor frontier**, not “all valid actions.” Actions that fail frozen safety/margin/evidence prerequisites are intentionally outside the learned intervention set.

CCF-A-level novelty is plausible only if the fresh causal ablations, independent full-val reproduction, and later closed-loop evaluation support this full chain. The algorithm story alone cannot guarantee acceptance.

---

## 5. V64.3.18 EAF-DACER algorithm

### 5.1 Final-guard-admissible frontier

For DARM anchor `a`, challenger `b` is learned-admissible only if runtime-only frozen conditions hold:

1. valid action;
2. if an unflagged valid action exists, `b` is unflagged;
3. frozen EAF DARM-anchor margin `M_EAF(b,a) >= 0.015`;
4. frozen score gain `score(b)-score(a) >= 0`;
5. unchanged evidence certificate passes.

When all valid actions are safety-flagged, DACER abstains from learned re-ranking and leaves the scene to the existing continuous structural-risk guard.

The legacy utility-equivalence membership bit is **not** a hard mask and **not** a learned feature. It is retained only for diagnostics / deterministic exact-tie-breaking. Legacy utility cost/rank remain part of the previously audited scalar runtime representation; they do not define admissibility.

The contract additionally fails if frozen residual robust-margin corrections (`residual_beta_uncertainty`, `residual_epsilon`) become nonzero without updating the pre-selection admissibility model.

### 5.2 Exact signed selected-atom attribution profile

V64.3.18 exports the existing EAF residual’s exact additive selected-atom contribution matrix privately. For each challenger, the column sum is the unchanged frozen EAF residual; no new evidence is queried and EAF arithmetic is unchanged.

The 42-feature main representation contains the audited 25 V64.3.17 scalar features plus 17 structured attribution features:

- selected atom count normalized by B;
- candidate contribution L1, positive-mass fraction, top-1 absolute concentration, effective support;
- the same four statistics for candidate-minus-legacy contributions;
- top-4 signed normalized candidate contributions;
- top-4 signed normalized candidate-minus-legacy contributions.

The candidate-minus-incumbent profile is the key new representation: it asks whether the *same selected evidence* supports an alternative differently from the frozen incumbent, rather than collapsing attribution to one global RSS magnitude.

### 5.3 Anchor-augmented counterfactual objective

A standardized shared linear score is deliberately retained; architecture depth is not the claimed novelty.

Anchor logit is fixed:

`score(anchor) = 0`.

For admissible challenger `b`, `score(b)=w^T z_b + beta`.

The main DACER objective contains three fixed train-only terms:

1. **anchor-augmented listwise CE**: target the teacher-best positive-margin admissible challenger, else anchor;
2. **class-balanced support BCE**, fixed weight 1: candidate better than anchor;
3. **incumbent-dominance pair loss**, fixed weight 1: supervise `sign(teacher_margin(b)-teacher_margin(legacy))` with `score(b)-score(legacy)`.

There is no validation tuning of anchor logit, threshold, support weight, dominance weight, B/M, guard, or certificate.

Runtime chooses the maximum learned score over `{anchor} U admissible challengers`. The unchanged final guard remains after this operator and should perform essentially zero hidden cleanup.

---

## 6. Why structured per-atom representation is not introduced blindly

The V64.3.17 screen’s emitted `next_action` suggested structured per-atom/query-conditioned reliability. The new audit shows that jumping directly to a larger representation would confound representation capacity with a candidate-set semantic bug.

Therefore V64.3.18 uses three nested learned controls on the same fresh scenes:

- **G-DALER**: candidate semantic correction only; guard-admissible frontier + old scalar listwise/support objective;
- **DACER-scalar**: same frontier/features + incumbent-dominance objective;
- **DACER-profile**: same frontier/objective + exact signed selected-atom profile.

This makes the causal questions identifiable:

- V17 DALER → G-DALER: did fixing the hard utility mask restore a real listwise frontier?
- G-DALER → DACER-scalar: does incumbent-relative dominance supervision improve recovery?
- DACER-scalar → DACER-profile: does structured exact selected-evidence attribution add recovery signal?

If the profile arm adds no fresh causal gain, it must not be sold as a paper novelty.

---

## 7. Design-only headroom from the now-contaminated V64.3.17 screen

These numbers are **algorithm-design diagnostics only** and must never be promoted or placed in the paper.

After removing only the erroneous hard utility-equivalence condition:

- fresh V64.3.17 design scenes: 57.56% of raw-proposal scenes have >=2 guard-admissible challengers;
- 57.77% have at least one non-incumbent admissible alternative;
- 33.40% contain an alternative whose teacher margin is greater than `max(anchor, incumbent)`;
- train: 58.96% are multi-admissible and 38.33% contain such a counterfactual opportunity.

This establishes sufficient *design headroom* to justify one clean fresh DACER screen. It is not evidence of generalization.

---

## 8. V64.3.18 causal experiment

All heads are fit on TRAIN only. One new, untouched, hash-selected 500-scene validation screen runs six paired arms:

1. raw EAF;
2. frozen RAER;
3. frozen V64.3.17 DALER hard-utility-mask control;
4. G-DALER;
5. DACER-scalar;
6. DACER-profile (main).

All six arms must replay exactly the same 500 scenario tokens.

### Candidate-semantic support gates

- fresh admissible edges >= 2048;
- fresh multi-admissible raw-proposal scene rate >= 25%;
- admissible support >= 5x V64.3.17 exact-executable support, with absolute minimum 2048;
- train admissible edges >= 8192;
- train multi-admissible scenes >= 512.

### Capacity/generalization gates

- profile train holdout support AUC >= 0.65;
- profile train holdout dominance AUC >= 0.60 with >=512 dominance pairs;
- fresh support AUC >= 0.65;
- fresh dominance AUC >= 0.60.

### Recovery mechanism gates

- proposal changed >= 5%;
- alternative recovery >= 3%;
- alternative recovery precision >= 70%;
- alternative teacher-margin mean > 0;
- counterfactual recovery precision (`selected alt > max(anchor, incumbent)`) >= 60%;
- counterfactual opportunity capture >= 5%;
- selected non-anchor teacher-better rate >= 75%.

### Causal ablation gates

DACER-scalar must beat G-DALER in at least one recovery-specific mechanism metric:

- dominance AUC +2 pp; or
- counterfactual capture +2 pp; or
- alternative recovery +1 pp.

DACER-profile must beat DACER-scalar in at least one structured mechanism metric:

- dominance AUC +1 pp; or
- counterfactual precision +5 pp; or
- counterfactual capture +1 pp; or
- alternative recovery +1 pp,

while remaining endpoint-non-harmful.

### Preservation / endpoint gates

- harmful absolute reduction vs raw >=5 pp;
- beneficial retention >=35%; beneficial > harmful;
- deployed flip >=3% and lower than raw;
- final post-selection guard block <=0.1%;
- teacher match >= DARM anchor +0.5 pp;
- regret <=1.02 x raw EAF;
- paired improvement/non-harm vs frozen RAER under the previous strict rule.

Passing this screen authorizes **only** one independent frozen full-validation reproduction. Test and closed-loop remain forbidden until that reproduction succeeds.

---

## 9. Data discipline

The V64.3.17 fresh 500 scenes are now design data and cannot be used for promotion again.

V64.3.18 exclusion file contains **2200 unique validation tokens**:

- 1700 previous design exclusions;
- +500 V64.3.17 fresh-screen scenes;
- overlap between those two sets: 0.

Train contains 3000 unique tokens and overlaps the new 2200-token validation design exclusion set by 0.

Fresh V64.3.18 selection uses only scenario identity plus fixed SHA256 seed `v64.3.18-eaf-dacer-fresh-v1`; no teacher label, match, regret, reliability score, or oracle statistic participates.

Budget wording remains:

> planner-interface evidence cap `B<=16`; exact `B=16` whenever at least 16 eligible proposal atoms exist.

The V64.3.17 fresh set used B=16 in 456/500 scenes (91.2%), and every scene with >=16 eligible proposal atoms used B=16.

---

## 10. Historical no-repeat constraints

The complete `ALGORITHM_CHANGELOG.md` was reviewed before V64.3.18 design. Continue to forbid:

- reopening BTP/RET/CET/AF/HAP or selector family allocation;
- increasing B or M;
- relaxing evidence/one-sided/safety/structural certificates;
- OCFI alpha/radius sweeps;
- EAIR/RAER probability-threshold sweeps;
- DALER/DACER anchor-logit or objective-weight tuning on validation;
- broad EAF representation unfreezing before the corrected candidate frontier is tested;
- reusing the old utility-equivalence set as a hard deployment mask;
- using any of the 2200 excluded validation tokens for promotion;
- claiming structured per-atom novelty if DACER-profile does not beat DACER-scalar on fresh causal mechanism metrics.

If DACER recovery works but regret remains too high, the next allowed algorithmic direction is a train-only teacher-improvement-magnitude / robust listwise ordering term on the **same frozen admissible frontier**, not a selector/acquisition change.

---

## 11. Engineering audit and validation

Implemented safeguards:

- DACER/RAER/DALER are mutually exclusive runtime arms;
- final-guard-admissible mask is independent of legacy utility-pool membership;
- all-flagged scenes abstain from learned re-ranking;
- exact selected-atom contribution columns sum to the frozen EAF residual;
- utility-pool membership is diagnostic / deterministic tie-break only, not a learned feature or hard gate;
- contract fails if robust-margin correction parameters cease to match the exact pre-selection admissibility model;
- DACER diagnostic prefixes are propagated through evaluator/train/metrics;
- fresh token replay uses `--require-all-scenario-tokens`;
- checker separates instrumentation validity from candidate-support collapse;
- no teacher/future signal enters runtime DACER features.

Regression:

- V64.3.6–V64.3.18 targeted tests: **84/84 PASS**, 6 existing Transformer warnings;
- full repository: **372/372 PASS**, 36 existing Transformer warnings;
- 5000 randomized raw V64.3.17 versus raw V64.3.18 tournament cases: **0 action differences, 0 score differences, 0 frozen-public-diagnostic case differences**;
- design exclusion audit: **2200 unique**, exact union, **0 train overlap**.

Thus the V64.3.18 raw configuration is a frozen-planner no-op apart from instrumentation; learned changes activate only in the fitted G-DALER/DACER causal arms.

---

## 12. Next execution

Run only:

```bash
bash RUN_V64_3_18_EAF_DACER_SCREEN_2GPU.sh
```

Return the complete `outputs_v64_3_18_eaf_dacer_screen_2gpu_v1` directory/zip.

If the launcher prints `STOP SCREEN`, follow the emitted `next_action`; do not manually tune thresholds/objective weights/B/M/guard/certificate. Even if the screen passes, the only allowed next stage is an independent frozen full-validation reproduction.
