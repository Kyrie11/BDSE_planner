# V64.3.28 PTMC uploaded-result postmortem and V64.3.29 FCR design

## 0. Scope and causal discipline

This note treats the uploaded V64.3.28 run as the only new untouched evidence for the V28 mechanism. The V28 TRAIN diagnostic is retained only as a design audit because PTMC was designed after inspecting earlier TRAIN failures. Historical results in `ALGORITHM_CHANGELOG.md` are used to rule out repeated branches and to interpret path-level behavior, not to pool validation blocks.

The paper-level question is kept fixed:

> Under a fixed planner-interface evidence budget, can the planner retain enough action-decision evidence to support safe, monotone extremal recovery?

The literal current configuration remains `B<=16`, proposal pool `M=24`, but the scientific object is a **fixed bounded interface**, not the number 16 itself.

---

## 1. V28 result validity

The uploaded V28 screen is a valid algorithm result rather than an engineering false STOP:

- V28 TRAIN gate: PASS;
- fresh split A: FAIL;
- fresh split B: FAIL;
- both-independent-block promotion: FAIL;
- no pooled rescue;
- the V28 checker itself reports the next action as: tail-mode confirmation is not incremental over V25 DRC; do not tune threshold / proposal coverage / weights.

The no-fallback selected-set containment contract passes on both blocks, so the failure cannot be blamed on PTMC inventing a different fallback candidate.

---

## 2. What V28 proves, what it falsifies, and what remains unresolved

### 2.1 Mechanisms supported by the fresh result

1. **No-fallback monotonicity remains valid.** PTMC only removes an aggregate proposal and never creates/re-ranks a new alternative. This remains a useful structural invariant for future reliability mechanisms.

2. **V25 aggregate DRC can produce a safe selected direct-replacement path on these new blocks.**
   - A: 22 direct replacements, teacher-improvement sum `+1.799253`, worst `-0.056853`, NegRMS `0.012821`, path regret delta sum `-35,985.07`.
   - B: 26 direct replacements, teacher-improvement sum `+1.250624`, worst `-0.000471`, NegRMS `0.000129`, path regret delta sum `-25,012.47`.

3. **Asymmetric incumbent preservation should remain frozen.** V28 itself makes zero learned admissible-incumbent -> anchor changes. Historical V19/V20/V21 results show that reopening that path is split-unstable: a path that was strongly beneficial on one screen became strongly harmful on another. Current V28 V20 happens to benefit from incumbent->anchor on both blocks, but that does not erase the earlier sign reversals.

### 2.2 Mechanisms falsified by the fresh result

1. **V28 type-only global PTMC is not a useful high-retention fresh confirmation mechanism.** It vetoes three behavioral aggregate proposals across A/B and all three are teacher-positive. It removes no direct nonpositive proposal.

2. **The V28 TRAIN separation is not stable enough to promote the PTMC mechanism.** The TRAIN type-only Gaussian result was highly selective against the known catastrophic proposal, but the untouched blocks show false-positive vetoes and no tail incrementality.

3. **The V27-era statement “proposal generation is no longer the main bottleneck” can no longer be carried forward unconditionally.** On V28 fresh, aggregate DRC captures only a small fraction of direct positive opportunities, while a veto-only Stage 2 is structurally unable to create missing proposals.

### 2.3 What V28 does *not* prove

1. It does **not** prove that `B<=16` is too small. Literal capacity was not varied.
2. It does **not** prove that type semantics contain no catastrophic information. Neither fresh block contains a selected aggregate direct proposal with teacher improvement `<= -0.5`; therefore catastrophic false-negative recall is not observed here.
3. It does **not** prove that tail risk is solved globally. Earlier untouched V25 DRC blocks contained severe tails (worst about `-0.99`, NegRMS about `0.20-0.29`). The current blocks are simply not tail-stressing in the same way.
4. It does **not** justify reopening learned admissible-incumbent -> anchor recovery just because V20 is favorable on this particular A/B.

---

## 3. Scene-level PTMC attribution

PTMC changes only three direct aggregate proposals on the current untouched blocks.

| split | scene | incumbent | anchor | aggregate proposal | proposal improvement vs incumbent | aggregate DRC | PTMC confirmation | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A | `62eb30e1177857a5` | 20 | 23 | 0 | `+0.000346929` | `+0.062408` | `-2.505528` | false veto |
| B | `cc8d5f615b4758fc` | 23 | 0 | 1 | `+0.003732445` | `+0.027140` | `-2.934317` | false veto; proposal is teacher action |
| B | `1faefabae51b50c3` | 3 | 19 | 4 | `+0.000437265` | `+0.037803` | `-0.434931` | false veto; proposal is teacher action |

Consequences:

- A direct count `22 -> 21`, precision `0.7273 -> 0.7143`, NegRMS `0.012821 -> 0.013123` (worse), positive-regret RMS `256.43 -> 262.46` (worse).
- B direct count `26 -> 24`, precision `0.8077 -> 0.7917`, NegRMS `0.0001286 -> 0.0001338` (worse), positive-regret RMS `2.5718 -> 2.6768` (worse).

Thus the fresh failure is not “confirmation too weak.” It is the opposite operational error: **the confirmation deletes useful recovery without deleting a harmful direct proposal.** Threshold/coverage tuning would therefore be post-hoc rescue and is prohibited.

---

## 4. Why endpoint coverage is now the dominant bottleneck

### 4.1 V20 is high-coverage context, not a mechanism to restore

On the same fresh blocks:

| split | arm | teacher match | teacher regret | direct opportunity capture |
|---|---|---:|---:|---:|
| A | V20 | 0.220 | 14279.34 | **0.3826** |
| A | V25 DRC | 0.144 | 14755.03 | **0.1074** |
| B | V20 | 0.216 | 13013.99 | **0.3571** |
| B | V25 DRC | 0.158 | 13414.58 | **0.1364** |

V20 also obtains large current-block regret reduction from admissible-incumbent -> anchor:

- A: 100 events, regret delta sum `-94,855.85`;
- B: 119 events, regret delta sum `-150,977.55`.

But this operator role is historically unstable: V20/V21 also produced strongly harmful incumbent->anchor blocks. Therefore the V20 endpoint gap cannot be “fixed” by restoring that path.

The useful diagnostic from V20 is instead: **there is much more direct recovery opportunity in the current frontier than V25/DRC is transmitting.**

### 4.2 Stage-1 gate decomposition

For direct-admissible incumbent scenes with at least one teacher-positive alternative:

#### Split A — 149 positive-opportunity scenes

- at least one positive alternative with support `>0`: `100/149 = 67.1%`;
- scalar dominance `>0`: `67/149 = 45.0%`;
- support and scalar both positive: `62/149 = 41.6%`;
- support + scalar + DRC `>0`: `17/149 = 11.4%`;
- actual selected teacher-positive direct replacement: `16/149 = 10.7%`.

#### Split B — 154 positive-opportunity scenes

- support `>0`: `104/154 = 67.5%`;
- scalar dominance `>0`: `67/154 = 43.5%`;
- support and scalar: `58/154 = 37.7%`;
- support + scalar + DRC `>0`: `21/154 = 13.6%`;
- actual selected teacher-positive direct replacement: `21/154 = 13.6%`.

The largest incremental collapse is therefore **before/at the DRC proposal**, not after it. A perfect veto-only PTMC can never recover the missing `~25-30pp` opportunity mass between support/scalar-feasible alternatives and DRC-positive proposals.

### 4.3 Fixed-budget evidence compression is a credible next mediator

The untouched aggregate arm reports:

| metric | A | B |
|---|---:|---:|
| retained decision atoms | 15.372 | 15.420 |
| proposal candidate atoms | 22.036 | 22.200 |
| proposal decisive-atom recall | 0.7531 | 0.7621 |
| selected decisive-atom recall | 0.5561 | 0.5622 |
| proposal interaction-decisive recall | 0.7247 | 0.7325 |
| selected interaction-decisive recall | 0.5059 | 0.5024 |
| exact evidence certificate fraction | 0.920 | 0.930 |
| pair-full action match | 0.188 | 0.174 |
| local pair-full action match | 0.182 | 0.158 |

This does not prove capacity insufficiency. It does show a concrete **transmission loss from the already available Top-M evidence into the retained B-set**, especially for interaction-decisive evidence. Given the historical exhaustion of learned acquisition losses and multiple risk-estimator geometries, the next falsifiable hypothesis should test **same-budget evidence allocation**, not another tail classifier.

### 4.4 Revised dominant bottleneck

The dominant bottleneck should now be stated as:

> **safe recovery coverage under fixed-budget decision-evidence transmission**

or more paper-oriented:

> **decision-evidence sufficiency for extremal recovery under a fixed planner-interface budget.**

The current operational mediator is sparse Stage-1 proposal coverage; the next hypothesis is that the selected B-set compresses the complete action-contrast frontier too weakly for support/DRC to recover enough positive alternatives.

This bottleneck should be addressed now because changing only a veto head cannot improve it structurally.

---

## 5. B<=16 policy and paper storyline

### 5.1 Keep B<=16 frozen in V29

Yes. The next experiment should continue to freeze the current literal budget for causal isolation:

- changing B would mix **allocation quality** and **capacity**;
- V28 fresh contains no catastrophic proposal, so there is no valid evidence that “B=16 cannot observe catastrophe”;
- the next test can ask a cleaner question: **can the same budget transmit a more decision-sufficient subset?**

### 5.2 Do not make “16” the novelty

The user's framing is correct. The paper headline should be **under a fixed/bounded budget, preserve sufficient evidence to decide the action**, not “B=16 is special.”

`B=16` should be presented as the principal operating point for controlled experiments. Once the mechanism is frozen, a later budget-performance ablation can show robustness/sensitivity. That ablation is secondary evidence, not the main algorithmic idea.

If V29 cannot improve same-budget transmission despite being active and mathematically reducing complete-frontier compression error, then a small diagnostic budget-sensitivity study becomes scientifically justified to separate:

1. wrong allocation target at fixed capacity, versus
2. actual interface-capacity insufficiency.

That diagnostic should occur **after** V29, not inside it.

---

## 6. Historical no-repeat constraints retained

The following branches must remain closed unless a new experiment provides a genuinely different falsifiable hypothesis:

- V40--V43 DACC / deletion beam / swap / repair / budget-layer combinatorial coreset search;
- V64.3.8--V64.3.12 BDMU / AF / HAP / BTP / RET / CET proposal-acquisition loss variants; the changelog already records `exact_acquisition_exhausted=true` and a pivot away from another acquisition loss;
- V24 sorted/L1-normalized attribution-spectrum geometry;
- V26 family+aggregate flat KNN;
- V27 type-KNN threshold / K / type-weight / group-weight tuning;
- another V28 “classifier v2/v3”, catastrophic threshold tuning, proposal-coverage tuning, or naive aggregate+type concatenation;
- mean-SE + DRC mixing, downside multiplier sweeps, zero-boundary sweeps, support/dominance threshold rescue;
- action blacklist, transition-geometry mainline, signed-profile ranking, KNN radius/OOD guard, simple failed-view AND stacking;
- pooled A/B rescue;
- reopening learned admissible-incumbent -> anchor changes;
- changing B/M in the same V29 mechanism test.

---

## 7. V64.3.29 algorithm: EAF-ICER-FCR

Full name:

> **Frontier-Contrast Evidence Rebinding under a Fixed Planner-Interface Budget**

The intended single causal change is the **retained evidence binding**, after EAF outputs already exist.

### 7.1 Frozen components

V29 keeps frozen:

- evidence atoms and the current Top-M proposal pool (`M=24`);
- current literal interface budget (`B<=16`);
- EAF checkpoint and exact DARM+EAF frontier arithmetic;
- AOCC baseline selector as the fallback interface;
- ICER support head and scalar incumbent-dominance head;
- V25 DRC recipe: aggregate 18-D evidence, `K={32,64}`, downside RMS, multiplier 1, boundary 0, scalar extremal ranking;
- admissible-incumbent preservation;
- exact downstream evidence certificate and structural-risk guard.

PTMC is removed from the V29 main mechanism.

### 7.2 Complete full-M frontier-contrast target

For the already queried full Top-M set, compute the complete DARM+EAF anchor-star margin vector around the full-M selected-local anchor. This is the reference action-decision contrast that the retained interface should approximate.

No teacher label, future state, validation fit, or new evidence query enters this target.

### 7.3 Deterministic same-cardinality rebinding

Starting from the same frozen Top-M evidence pool, construct a candidate B-set with exactly the same retained cardinality as the baseline AOCC set. Greedy forward selection minimizes the lexicographic compression error:

1. `L_inf` error to the full-M complete anchor-star;
2. RMS error as the tie-break objective.

Budget feasibility is enforced at every step. Atom cost and atom index are only deterministic feasibility/tie-break terms, not learned weights.

This is deliberately not DACC:

- no deletion beam;
- no one/two-swap repair;
- no combinatorial winner-preservation search;
- no target candidate expansion.

It is also not a V64.3.8--.12 acquisition variant:

- no learned proposal loss;
- no acquisition adapter unfreezing;
- no teacher utility target;
- no change to Top-M admission;
- no additional model query.

### 7.4 Monotone acceptance contract

The candidate rebind is deployed only when all of the following hold:

1. same retained cardinality as baseline AOCC;
2. same fixed budget feasibility;
3. candidate selected-local anchor equals the full-M selected-local anchor;
4. candidate exact downstream target action equals the full-M exact target action;
5. complete frontier compression error is strictly lexicographically better than the baseline B-set.

Any failure returns the exact baseline AOCC selection.

This makes FCR a **monotone interface refinement**: it can improve the frontier representation only under hard decision-preservation constraints; it cannot force a different full-M target merely to create recovery.

### 7.5 Downstream DRC handling

Because the selected evidence distribution changes, the unchanged DRC recipe is re-fit on the exact same frozen 3000 TRAIN scenes. This is not a new estimator or a validation-tuned threshold: only the feature values induced by the new retained B-set change. The TRAIN gate verifies that the same `K={32,64}`, downside-RMS, multiplier-1, zero-boundary operator remains 5/5 scene-fold safe before any fresh scene can be selected.

---

## 8. V29 causal experiment

### 8.1 Frozen TRAIN gate before any fresh GPU

Replay the exact historical 3000 TRAIN identities on two arms in parallel:

- baseline V20 selector;
- FCR-V20 selector.

Hard-control the baseline V25 reproduction:

- 3000 scenes;
- 75,133 frontier rows;
- 1,455 replacement edges;
- 310 replacement scenes;
- aggregate DRC 5/5 fold-safe;
- 71 selected proposals;
- teacher-improvement sum `5.527642` within numerical tolerance.

FCR TRAIN must additionally satisfy:

- enabled on every scene;
- finite full-M/B frontier error on >=99%;
- final frontier error never increases;
- at least one accepted rebind (mechanism is not dead code);
- every accepted rebind preserves cardinality, budget, full-M local anchor and exact target;
- unchanged DRC recipe remains 5/5 fold-safe with >=64 selected and nonnegative total selected teacher improvement.

Any failure is a TRAIN STOP. Do not tune FCR objective weights (there are none), acceptance thresholds, DRC K, multiplier or boundary.

### 8.2 Fresh identity discipline

V28 consumed 1000 new untouched validation identities, so V29 exclusion is updated from 6700 to **7700 unique inspected validation tokens**.

Select a new untouched 1000 with a new SHA-based seed, split A/B = 500/500, and require zero overlap with:

- all 7700 prior inspected validation tokens;
- frozen 3000 TRAIN identities.

### 8.3 Fresh arms

Each block runs:

1. raw EAF;
2. V20 historical high-coverage context;
3. V25 aggregate DRC causal control;
4. FCR-V20 interface-mechanism control;
5. **FCR + unchanged aggregate DRC recipe** main arm.

V20 is not eligible to rescue the method; it is included to expose recoverable coverage and path decomposition.

### 8.4 Pre-registered V29 fresh gates

Both blocks independently require:

**FCR mechanism active and exact**
- FCR enabled rate 100%;
- >=5 accepted rebinds / 500;
- >=99% finite frontier-error instrumentation;
- no frontier-error increase;
- every accepted rebind passes all hard contracts.

**safe recovery coverage expansion over V25 DRC**
- direct positive-opportunity capture `>= V25 + 3pp`;
- at least `+5` teacher-positive direct replacements.

**tail preservation**
- no selected catastrophic replacement under the pre-existing definition `teacher improvement <= -0.5`;
- no increase versus V25 in positive-regret RMS;
- no increase in worst regret increase;
- no increase in teacher NegRMS;
- worst teacher improvement not worse than V25;
- selected direct path regret delta sum `<=0` and precision `>=60%`.

**historical invariants**
- zero learned admissible-incumbent -> anchor events;
- all-flagged structural-domain identity/delegation remains exact;
- harmful/flip preservation versus raw remains within fixed tolerances.

**endpoint**
- noninferior to V25 DRC on match/regret in each block;
- at least one block must show a strict endpoint signal before full-val promotion.

No pooled rescue.

### 8.5 Interpretation matrix

- **FCR inactive / hard contract violation** -> engineering or mechanism-definition failure; stop.
- **FCR active + frontier error improves, but safe coverage does not** -> full-M anchor-star compression is not the missing decision-evidence target; do not tune greedy weights. Reassess interface target/observability.
- **coverage improves but selected tail degrades / catastrophe appears** -> same-budget rebinding exposes unsafe recovery modes; stop FCR, do not relax guards.
- **coverage + tail pass, endpoint noninferior but no endpoint signal** -> safe mediator moved but downstream recovery ranking/value remains limiting; audit ranking, not B size.
- **both A/B pass + endpoint signal** -> freeze V29; allow exactly one independent full-validation reproduction. Test/closed-loop remain blocked until then.

---

## 9. Candidate paper contribution after V28

The current paper should not promote PTMC as validated novelty. The more durable mainline is:

> fixed bounded planner-interface evidence -> complete evidence-attributed action frontier -> decision-sufficient fixed-budget evidence binding -> deployment-admissible incumbent-contrastive recovery -> downside-aware extremal intervention -> monotone preservation -> unchanged structural guard -> endpoint.

If V29 succeeds, a stronger candidate mechanism claim is:

> **Evidence-attributed frontier-contrast compression for decision-sufficient evidence rebinding under a fixed planner-interface budget, coupled with monotone extremal recovery.**

The novelty is **not** “greedy subset selection” and not the literal number `B=16`. The potentially publishable mechanism is that the full already-queried EAF action-contrast frontier defines what the bounded interface must preserve, and a candidate evidence rebind is allowed only when it improves that complete contrast representation while preserving the exact full-interface decision.

This remains a **candidate novelty**, not a CCF-A-level claim, until untouched A/B, independent full-val, broader baselines/ablations, and a dedicated related-work novelty audit support it.

---

## 10. Implementation and engineering audit

Implemented in this package:

- `bdse/planner/frontier_contrast_rebinding.py`;
- FCR hook in `bdse/planner/nuplan_planner.py`;
- `bdse/configs/v64_3_29_eaf_icer_fcr_v20.yaml`;
- frozen 3000 TRAIN manifest and 7700-token design exclusion;
- V29 TRAIN fitter and hard contract checker;
- V29 A/B split checker and double-fresh checker;
- V29 unit tests;
- fail-closed 2-GPU launcher.

The FCR implementation recomputes the candidate frontier using the production DARM/EAF primitives before accepting it, so the final contract does not trust the incremental greedy arithmetic. It also fails closed to the exact baseline AOCC selection on shape, budget, anchor, evaluator, exact-target, or compression-error failure.

See `V64_3_29_ENGINEERING_VALIDATION.txt` for the final regression record.
