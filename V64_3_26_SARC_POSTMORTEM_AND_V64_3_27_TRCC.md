# V64.3.26 EAF-ICER-SARC Postmortem and V64.3.27 EAF-ICER-TRCC Design

## 1. Result-validity verdict

The uploaded V64.3.26 result is a **correct fail-closed TRAIN STOP**. It is not an engineering false negative and it never used fresh validation.

The observed chain is:

`prerequisite re-audit -> targeted regression -> frozen 3000-scene V20/EAF semantic instrumentation replay -> V26 5-fold SARC fit/gate -> STOP TRAIN`.

Audited facts:

- TRAIN scenes: exactly 3000 unique tokens;
- frontier rows: 75,133;
- semantic frontier size: 764,601,702 bytes;
- frontier SHA256: `0d1d2442f6268b06a2590723bb765e60c0ca5c376233d7321da53f48c99e4c0a`;
- TRAIN token SHA256: `b36a847e7a3d7caa3c785ac96b6789ddefed071fae050170482108d950447da4`;
- complete marker exists;
- server targeted regression: 88 passed, 2 existing Transformer warnings;
- no traceback/runtime exception;
- `fresh_validation_used=false`.

Therefore there is no scientific basis for a fresh V26 attribution: no V26 fresh block exists by design.

## 2. Why the TRAIN gate failed

The frozen V25 aggregate-downside control remains stable:

| arm | fold-safe | selected | teacher improvement sum | negative RMS | worst |
|---|---:|---:|---:|---:|---:|
| V25 aggregate DRC | **5/5** | 71 | **+5.527642** | **0.064782** | **-0.545757** |
| V26 semantic-family SARC | **4/5** | 64 | +3.608093 | **0.142243** | **-0.998528** |

V26 fails two independent preregistered requirements:

1. it is not 5/5 fold-safe: fold 3 selects only 7 replacements, below the frozen per-fold minimum of 8;
2. much more importantly, its selected negative tail is substantially worse than the aggregate control, so `semantic_tail_incremental_on_train=false`.

Relaxing the fold-3 count would **not** rescue V26. The tail contract still fails by a large margin.

## 3. Scene-level causal attribution

The decisive counterexample is TRAIN scene `faf93d61f8bd5238`, challenger action 21.

Its actual normalized teacher improvement relative to the incumbent is:

`-0.998528`.

The same edge is rejected by the V25 aggregate DRC:

`aggregate DRC = -0.003552`.

V26 changes only the local risk geometry and accepts it:

`semantic-family SARC = +0.023354`.

For the semantic K32 neighborhood:

- local mean = `+0.023887`;
- local downside RMS = `0.000533`;
- worst observed neighbor = only `-0.001835`.

So SARC is not merely slightly miscalibrated. The family representation moves the candidate into a locally benign neighborhood that is missing its real catastrophic outcome mode. In contrast, the aggregate view keeps enough adverse neighborhood mass to make its certificate negative.

## 4. Shared / exclusive selected populations

Exact token+action comparison of the two fixed-fold selectors:

| population | count | teacher sum | positive ratio | worst | negative RMS |
|---|---:|---:|---:|---:|---:|
| shared | 45 | +3.613395 | 75.6% | -0.545757 | 0.081359 |
| aggregate-only | 26 | **+1.914248** | 61.5% | **-0.009837** | **0.001940** |
| semantic-only | 19 | **-0.005302** | 52.6% | **-0.998528** | **0.229078** |

This is the key mechanism result: the family representation does not merely veto good aggregate candidates. It creates a new exclusive population whose net contribution is negative and whose tail contains the new approximately `-1` catastrophe.

## 5. What V26 does and does not prove

V26 **does falsify** the proposed main mechanism:

> coarse semantic-family alignment by flat concatenation into the local KNN risk metric.

It does **not** support finalizing the paper around `semantic-aligned tail-regret certification`.

However, V26 still does **not prove** that the fixed `B<=16` evidence interface lacks capacity. The V26 representation is much coarser than the interface:

- it sums all selected atoms into only five family coordinates for the candidate and five for candidate-minus-incumbent;
- `decision_boundary` is structurally zero under the current evidence generator in this population;
- the 10-D family representation has mean absolute inter-dimension correlation about 0.407;
- standardized PCA explains about 94.45% of variance with only the first three components and 98.86% with five.

The fixed interface still contains finer atom-type identity and exact candidate/incumbent contribution correspondence. V26 never tests these as an independent confirmation view.

Thus the capacity hypothesis becomes more plausible, but it is **not yet identified**.

## 6. Updated dominant bottleneck

V24 showed representation-induced neighborhood fragmentation from sorted/L1-normalized attribution spectra.

V25 showed that returning to the aggregate 18-D geometry does not solve fresh catastrophic semantic aliasing, even under dense TRAIN support.

V26 now shows that coarse family semantics, when allowed to replace the aggregate neighborhood geometry, can resurrect a catastrophe that the aggregate view itself rejects.

The bottleneck therefore tightens to:

> **selected-path tail certification under representation-conditional neighborhood instability and within-interface semantic outcome aliasing.**

The key design requirement is no longer “find a richer single KNN metric.” It is:

> **new evidence semantics must provide independent evidence about the already-proposed candidate without being allowed to change which candidate is proposed.**

## 7. Directions now ruled out / deprioritized

Do not repeat:

- V24 abs-sorted/L1-normalized full attribution spectra;
- V26 family features flat-concatenated into the aggregate KNN metric;
- adding still more semantic dimensions to the same single distance metric;
- tuning semantic/group/type weights;
- mean-SE + DRC interpolation;
- downside multiplier, K, zero-boundary, support or scalar-dominance threshold sweeps;
- standalone KNN-radius/OOD gates as the primary fix;
- transition geometry or signed-profile ranking as main mechanisms;
- action/maneuver blacklists;
- broad acquisition/selector/EAF/B/M unfreezing before the remaining within-interface semantic test;
- pooled fold or fresh-block rescue.

A post-hoc diagnostic `aggregate DRC AND V26 family DRC, no fallback` is also not a viable main mechanism: it retains only 47 replacements and is only 3/5 fold-safe (diagnostic only, not promotion evidence). Therefore V27 does **not** simply bolt the failed V26 family gate onto V25.

## 8. V64.3.27 EAF-ICER-TRCC

Full name:

**Evidence-Attributed Incumbent-Contrastive Extremal Recovery with Type-Resolved Tail-Regret Candidate Confirmation.**

V27 makes one causal change relative to the frozen V25 proposal operator.

### Stage 1: frozen aggregate proposal

Exactly V25:

- final-guard-admissible incumbent is preserved by default;
- alternative requires frozen support `>0`;
- frozen scalar dominance `>0`;
- aggregate 18-D downside certificate `>0` using K={32,64}, inverse-distance weighting, multiplier=1, zero boundary;
- one extremal alternative is proposed by the existing scalar-dominance ranking/tie rule.

### Stage 2: independent type-resolved confirmation

The current evidence generator has the fixed atom types:

1. `occupancy`;
2. `ttc`;
3. `gap`;
4. `drivable_area`;
5. `wrong_way`;
6. `speed_limit`;
7. `red_light`;
8. `route_connector`;
9. `local_comfort_accel`;
10. `local_comfort_jerk`;
11. `local_comfort_curvature`;
12. `local_comfort_brake`.

For each type `t`, over the same selected `B<=16` atoms:

`S_t(b) = sum a_e(b)` for selected atoms of type `t`,

and

`D_t(b,i) = sum [a_e(b)-a_e(i)]` on those same atoms.

This gives a fixed 24-D type view.

Constraints:

- no magnitude sorting;
- no L1 normalization;
- no learned embedding;
- no type/group weights;
- no concatenation with the 18-D aggregate view;
- same K={32,64}, same downside-RMS statistic, same zero boundary.

### Monotone no-fallback invariant

Only the **single V25 aggregate-proposed candidate** is evaluated by the type view.

If its type certificate is positive: accept that same candidate.

If its type certificate is non-positive: preserve the incumbent.

There is **no fallback/reselection to the second-best alternative**.

Therefore, by construction:

`TRCC selected replacements subseteq V25 aggregate-DRC selected replacements`.

This is a much stronger causal/engineering invariant than V24-V26: the new representation cannot create a new selected alternative.

## 9. V27 TRAIN causal experiment

One frozen 3000-scene instrumentation replay is necessary because V26 provenance does not contain the 12 type-resolved coordinates. EAF, selector, frontier, evidence budget, safety guards and learned support/dominance heads remain frozen.

TRAIN produces three readouts on the identical replacement population and fixed V23 scene folds:

1. **V25 aggregate DRC control** — causal control;
2. **type-only direct-selector diagnostic** — answers whether type semantics are outcome-sufficient when used alone; diagnostic only;
3. **TRCC main** — aggregate proposal plus type confirmation/no fallback.

TRCC must satisfy before any fresh GPU:

- 5/5 fixed folds selected-path safe;
- total selected replacements >=64 (same support floor; not relaxed after V26);
- total teacher-improvement >=0;
- exact subset/no-fallback invariant in every fold;
- selected negative RMS non-worse than V25 aggregate;
- selected worst outcome non-worse than V25 aggregate;
- at least one of the two tail metrics strictly better.

If this fails: TRAIN STOP, no fresh, no parameter sweep.

## 10. V27 fresh experiment if and only if TRAIN passes

V26 used no fresh scenes, so the frozen design exclusion remains exactly **6700** unique inspected validation tokens. V27 uses a new hash seed to select another untouched 1000, split A500/B500.

Each block runs only:

- raw EAF;
- frozen V20;
- V25 aggregate DRC control;
- V27 TRCC main.

Each independent block must verify:

- exact token identity and zero TRAIN/design overlap;
- frozen deployment/interface invariants;
- support/dominance signal;
- incumbent-default / zero learned incumbent->anchor intervention;
- **TRCC selected-replacement subset of aggregate DRC**;
- selected replacement path non-harm;
- recovery/capture;
- selected-tail incrementality over aggregate DRC;
- preservation non-degradation;
- endpoint non-inferiority/gain.

A and B must both pass independently. No pooling.

## 11. Paper mainline and novelty after V26

Do **not** finalize the mainline as `semantic-aligned tail-regret certification`: V26 directly fails its preregistered TRAIN causal gate.

The code-faithful mainline for V27 is:

`fixed B<=16 -> auditable selected evidence / exact EAF contribution -> frozen complete DARM-anchor frontier -> complete deployment-admissible frontier -> frozen support + scalar incumbent dominance -> aggregate downside proposal -> type-resolved independent confirmation of the same candidate -> no-fallback incumbent preservation -> unchanged evidence/one-sided certificate -> unchanged structural-risk guard -> final decision preservation -> preservation + endpoint`.

A candidate headline **only if V27 survives TRAIN + double-fresh + independent full-val** is:

> **evidence-attributed monotone cross-view incumbent-contrastive tail-regret confirmation for deployment-admissible extremal recovery under a fixed planner-interface evidence budget.**

This is intentionally narrower than the V26 semantic-aligned headline. The novelty is no longer “a richer semantic metric”; it is the **monotone cross-view decision structure** that separates proposal geometry from semantic tail confirmation and prevents a new representation from resurrecting alternatives.

If V27 also fails under this no-fallback structure, evidence for insufficient observability inside the current fixed interface becomes substantially stronger. At that point it is reasonable to reconsider interface/acquisition capacity or a fundamentally different nonlocal distributional model rather than continuing KNN representation variants.
