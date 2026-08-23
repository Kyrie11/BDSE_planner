# V64.3.29 FCR untouched result postmortem and V64.3.30 FBIC design

## Executive decision

V64.3.29 is a **valid algorithmic failure**, not an engineering false negative. FCR is highly active and reliably reduces the objective it was designed to reduce, but the untouched A/B results show that the objective itself is not the missing mediator for recovery. The strongest new conclusion is:

> **Global full-M frontier-compression fidelity is not equivalent to candidate-specific decision sufficiency at the direct incumbent-replacement boundary.**

The next highest-information step is therefore **not** another B=16 selector, FCR-v2, tail classifier, or DRC threshold. The pre-registered next branch is a single-point retained-interface capacity ceiling over the *same already queried M=24 evidence bank*. V64.3.30 implements exactly that test as FBIC.

---

## 1. Paper idea understood from the uploaded manuscript

The manuscript's stable research motivation is stronger than any specific PTMC implementation: a real-time planner cannot retain an unbounded predictive world model, so the scientific question is which bounded evidence is sufficient to support the downstream action. The paper explicitly distinguishes reconstruction fidelity from **decision sufficiency**, builds an Evidence-Attributed Frontier (EAF) over a fixed candidate bank, and uses deployment-admissible incumbent-contrastive recovery with structural containment.

The current `.tex` is still centered on BDSE-PTMC. Its key structural claims are:

1. bounded/auditable planner-interface evidence rather than generic reconstruction;
2. complete EAF anchor-to-challenger instrumentation with exact selected-evidence attribution;
3. direct incumbent-to-alternative recovery restricted by deployment admissibility, support and scalar dominance;
4. at most one extremal proposal;
5. no-fallback confirmation can only return the incumbent or the same proposal;
6. all-flagged structural scenes are delegated to the unchanged structural-risk guard;
7. TRAIN diagnostics are design evidence only and untouched A/B must independently validate the mechanism.

The last item is decisive: because V28 untouched A/B falsified PTMC and V29 falsified FCR as a decision-sufficiency proxy, the current manuscript should **not** be submitted with PTMC as the headline method. The structural problem formulation survives; the method section must eventually be rewritten after the V30 capacity/consumer ambiguity is resolved.

---

## 2. V29 experiment validity

### TRAIN

- 3000 frozen TRAIN scenes;
- 75,133 frontier rows;
- FCR accepted 2,339 scenes (77.97%);
- accepted contract pass rate 100%;
- mean accepted L-infinity reduction 0.04409;
- mean accepted RMS reduction 0.03525;
- refit aggregate DRC: 5/5 path-safe folds, 107 selected edges, teacher-improvement sum +11.763551.

This shows the code can execute the intended FCR intervention and that the mechanism is not trivially inactive. It is still development evidence only.

### Untouched A/B

The official double-fresh checker returns:

- TRAIN gate: PASS;
- split A: FAIL;
- split B: FAIL;
- both independent blocks pass: FALSE;
- safe coverage gain both: FALSE;
- tail noninferiority both: FALSE;
- catastrophe free both: FALSE;
- FCR monotone local contract both: TRUE;
- endpoint noninferiority both: FALSE.

The uploaded logs contain no traceback/OOM/action-provenance failure that would invalidate this causal interpretation. The earlier targeted repository stack was 107/107 PASS on the server.

---

## 3. What FCR actually accomplished

| quantity | Split A | Split B |
|---|---:|---:|
| accepted FCR scenes | 361/500 (72.2%) | 367/500 (73.4%) |
| accepted local contract pass | 100% | 100% |
| mean accepted L-inf reduction | 0.03075 | 0.03275 |
| mean accepted RMS reduction | 0.02206 | 0.02359 |

This is important positive evidence: FCR is not failing because its optimizer is too weak. It demonstrably changes the evidence set and improves its declared full-frontier compression objective on fresh scenes.

But the pure interface control almost does not move useful recovery:

| arm | A direct positive-opportunity capture | B capture |
|---|---:|---:|
| V20/AOCC | 34.97% | 31.68% |
| FCR-V20 | 36.20% | 31.88% |
| absolute change | **+1.23 pp** | **+0.20 pp** |

The pre-registered signal was +3 pp independently on both blocks. Therefore the hypothesis

> “preserving the complete full-M anchor frontier more faithfully through the same B=16 set will materially recover missing direct intervention opportunities”

is **not supported**.

This falsifies the target proxy more strongly than a no-op failure would: FCR reduces the proxy substantially, yet the downstream decision quantity barely changes.

---

## 4. Where the FCR+DRC pipeline loses useful opportunities

### Split A gate decomposition

`163 positive direct opportunities`

- V25/AOCC: `163 -> support 115 -> scalar 77 -> support+scalar 68 -> DRC+ 20 -> selected 19`
- FCR: `163 -> support 122 -> scalar 83 -> support+scalar 78 -> DRC+ 9 -> selected 8`

The FCR representation exposes *more*, not fewer, teacher-positive opportunities to the support/scalar heads. The dominant collapse occurs at the DRC boundary: 20 DRC-positive opportunities become 9.

The identity shift is even stronger: the 19 positive scenes captured by V25 and the 8 captured by FCR have **zero overlap**. FCR loses all 19 V25 captures and substitutes 8 different ones.

### Split B gate decomposition

- V25/AOCC: `161 -> 117 -> 66 -> 57 -> 15 -> 15`
- FCR: `160 -> 120 -> 71 -> 66 -> 10 -> 8`

Capture identity:

- common: 6;
- V25-only: 9;
- FCR-only: 2.

### Mechanism implication

FCR changes the aggregate representation sufficiently that the recovery reliability neighborhood/score semantics move. Re-fitting the **same** DRC recipe on the same 3000 TRAIN population is not enough to make the new representation stable on untouched scenes.

Therefore the relevant failure is not “DRC threshold slightly wrong.” It is:

> **representation-conditioned replacement reliability / semantic aliasing at the candidate-specific intervention boundary.**

Do not continue by tuning K, downside multiplier, boundary, support/scalar thresholds, or FCR objective weights.

---

## 5. Selected-tail behavior: non-reproducible sign reversal

### Split A

| arm | direct repl. | positive | sum DeltaT | worst | NegRMS |
|---|---:|---:|---:|---:|---:|
| V25 DRC | 28 | 19 | -0.914731 | -0.929157 | 0.175594 |
| FCR+DRC | 18 | 8 | +2.341384 | -0.022551 | 0.005319 |

FCR looks substantially safer on A and removes the V25 catastrophe `eca13b6114895ee4` (-0.929157).

### Split B

| arm | direct repl. | positive | sum DeltaT | worst | NegRMS |
|---|---:|---:|---:|---:|---:|
| V25 DRC | 25 | 15 | +0.383170 | -0.605138 | 0.121033 |
| FCR+DRC | 21 | 9 | -2.581144 | -0.989800 | 0.332721 |

FCR creates two new severe catastrophes:

- `444554532a375e54`: incumbent 13 -> 3, DeltaT=-0.989379; FCR replaces all 16 retained atoms, DRC=+0.100162;
- `66c3346eaa795dd1`: incumbent 24 -> 23, DeltaT=-0.989800; FCR changes 12/16 atoms, DRC=+0.018597.

It also retains the baseline B catastrophe `dec739cd22e05639` (7 -> 0, DeltaT=-0.605138).

This A-good/B-bad reversal is precisely the kind of split instability the project has repeatedly used as a stop condition. It is not a reason to add another tail veto.

---

## 6. A V29 logic-contract issue found in addition to the algorithm failure

Split A has 20 all-flagged structural scenes. ICER's structural-delegation diagnostic is 100%, but final identity versus raw is only 90%.

Two exact violations are:

- `8fc79d869dcb594b`: raw 0 -> FCR final 9;
- `1b1a8a5fd2205cfd`: raw 3 -> FCR final 10.

Both FCR rebindings were accepted and changed 12 evidence atoms.

The FCR acceptance contract audits an exact intermediate full-M tournament target, but that is not equivalent to preserving the **complete final structural-domain operator**. Rebinding can change tournament context/scores before the downstream structural path even when the later ICER flag says “delegated.”

This is a logic-contract defect. It does not explain away the main FCR failure: even the pure safe-domain FCR-V20 recovery signal is too small. V30 therefore does **not** repair FCR; instead it makes the new capacity intervention a strict no-op on the all-flagged domain.

---

## 7. Which mechanisms are now supported, rejected, or need escalation

### Supported / keep frozen

- **bounded auditable planner interface** as the research setting;
- **EAF complete selected-evidence attribution** as planner instrumentation;
- deployment-admissible action frontier;
- incumbent-contrastive direct replacement framing;
- asymmetric admissible-incumbent preservation;
- structural-domain separation;
- no-fallback/monotone intervention containment;
- strict independent fresh protocol with no pooled rescue;
- aggregate downside DRC as the historical control mechanism, not as a universally reliable final solution.

### Rejected as next main direction

- PTMC classifier/threshold variants;
- local type-KNN / semantic concat / another risk head;
- learned incumbent->anchor recovery;
- full attribution-spectrum KNN geometry;
- FCR-v2 using the same global frontier-compression target;
- tuning FCR L-inf/RMS weights, greedy order or acceptance tolerance;
- broad B/M sweep;
- V40-V43-style coreset beam/swap repair;
- V64.3.8-.12 acquisition-loss family;
- support/scalar/DRC threshold rescue.

### Needs escalation

The remaining question is no longer “which B=16 subset best approximates the full frontier?” It is:

> **What evidence is needed for one concrete candidate-specific incumbent replacement, and can the current bounded interface transmit/use it without catastrophic aliasing?**

Before designing a candidate-conditioned completion mechanism, one causal ambiguity must be resolved: literal B16 retained capacity versus downstream consumer semantics.

---

## 8. Dominant bottleneck after V29

The previous V28 diagnosis should be narrowed from generic “safe recovery coverage” to:

> **candidate-/intervention-conditioned decision sufficiency at the direct incumbent->alternative boundary, under a bounded auditable evidence interface and a hard catastrophic-tail constraint.**

The missing mediator is **not global frontier reconstruction fidelity**. V29 directly tested that hypothesis and obtained a negative causal result.

The unresolved sub-question is:

- **capacity:** B16 drops useful information that is already in queried M24; or
- **consumer semantics:** even when richer evidence is available, current support/scalar/DRC recovery semantics do not use it reliably.

V30 is designed to resolve this split with the minimum new assumption.

---

## 9. Paper mainline after V29

The current `.tex` remains useful as a record of the problem formulation and structural invariants, but PTMC must no longer be the final headline. FCR should also not replace PTMC in the headline after this result.

A tighter CCF-A-oriented mainline is currently:

> **Bounded planner evidence should be judged by intervention-level decision sufficiency, not reconstruction fidelity: the interface must expose evidence sufficient to support or reject a concrete deployment-admissible incumbent replacement, while monotone structural contracts prevent the learned recovery layer from creating new unsafe action paths.**

Stable mechanism chain:

`bounded auditable interface -> EAF action/evidence attribution -> deployment-admissible candidate-specific intervention -> incumbent-default monotone recovery -> structural/tail constraints -> independent fresh validation`

What is intentionally *not* fixed yet is the concrete “candidate-specific decision-sufficient evidence” operator. V30 determines the correct branch before that part is written as a final method.

---

## 10. V64.3.30 EAF-ICER-FBIC

FBIC = **Full-Bank Interface Capacity Ceiling**.

This is a single-point causal diagnostic, not a final novelty claim.

### Core intervention

The **entire upstream V20 configuration remains B=16**:

- global `evidence.budget=16`;
- fallback budget stages `[16]`;
- Top-M `M=24`;
- historical AOCC constructed at B16;
- same acquisition/model/checkpoint;
- same EAF/ICER support/scalar semantics;
- FCR disabled.

After B16 selection, only in non-structural scenes, FBIC exposes every already queried active decision Top-M atom to the downstream planner using a separate `full_bank_capacity_probe.interface_budget=24` ceiling.

There are **zero additional model/evidence queries**. This measures retained transmission capacity only.

### Capacity-only hard contract

FBIC is applied only if:

1. baseline selection is valid/unique;
2. reference bank is valid and no larger than M=24;
3. baseline B16 is a subset of the reference bank;
4. full reference is affordable under the diagnostic interface ceiling;
5. cardinality actually expands;
6. scene is not all-flagged structural.

It may add already queried atoms but may **never remove/reallocate any B16 atom**. Any failure returns the exact B16 selection.

### Why the global evidence budget stays 16

An initial implementation audit caught an important potential confound: setting global `evidence.budget=24` would also modify upstream selector/fallback behavior. That would no longer isolate retained capacity. The implementation was corrected before final delivery so that only the post-selector diagnostic interface ceiling is 24; every upstream budget setting remains bit-semantically frozen against V20.

---

## 11. V30 experiment and pre-registered interpretation

Each untouched block runs five paired arms:

1. raw;
2. B16 V20;
3. B16 V25 DRC;
4. FBIC full-bank V20 — **pure capacity control**;
5. FBIC full-bank + unchanged V25 DRC recipe — main capacity+recovery arm.

The DRC family is not changed: same 18-D aggregate feature schema, K={32,64}, downside multiplier 1, zero boundary, scalar extremal ranking. It is re-estimated on the same frozen 3000 TRAIN scenes only because the retained representation changes.

The screen checks exact per-scene query-count parity, zero baseline-atom removal, structural no-op, incumbent->anchor zero, gate decomposition, direct positive-opportunity capture, selected tail and endpoint.

### Branch A: capacity mediator supported

If pure FBIC-V20 gains at least +3 pp on **both** A/B and FBIC+DRC converts that gain safely with no selected catastrophe, B16 retained capacity is a reproducible mediator. The next algorithm should be an **adaptive candidate-conditioned bounded completion** that recovers the full-bank signal without making fixed B=24 the paper thesis.

### Branch B: capacity exposes signal, DRC cannot consume it

If pure FBIC-V20 gains on both A/B but FBIC+DRC remains unsafe/low-coverage, stop selector work. The bottleneck becomes **candidate-conditioned recovery reliability on the capacity-complete view**.

### Branch C: capacity-only expansion does not help

If pure FBIC-V20 does not reproducibly gain, stop B sweeps and same-bank allocation. Focus directly on candidate-conditioned counterfactual recovery semantics/operator mismatch.

This branch does not prove M24 contains no latent information. It proves that merely transmitting more of the already queried bank through the frozen downstream semantics is not sufficient.

---

## 12. Fresh-data discipline

V29 consumed another 1000 untouched scenes. V30 uses:

- permanent design exclusion: 8,700 unique tokens = old 7,700 + exact V29 1,000;
- exclusion SHA256: `d868dce298250a78ac74cc0b443eed8b6364acfbdd6daef50bacacf37ac6ff53`;
- new untouched 1,000 scenes split A/B=500/500;
- same frozen 3,000 TRAIN manifest;
- strict baseline V25 replay gate before fresh: `3000 / 75,133 / 1,455 / 310 / 5-of-5 / 71 / +5.527642`.

No pooled A/B rescue is permitted.

---

## 13. Engineering audit of the delivered V30 code

Final local validation after the capacity-isolation fixes:

- `python -m compileall -q bdse`: PASS;
- launcher `bash -n`: PASS;
- V29+V30 focused tests: 13/13 PASS;
- V13-V30 targeted mechanism stack: 115/115 PASS after final hardening;
- full repository: 445/445 PASS after final hardening;
- warnings: 36, all existing PyTorch Transformer `nested_tensor/norm_first`; no new warning class.

The exact final counts are regenerated in `V64_3_30_ENGINEERING_VALIDATION.txt` at package time; if those counts differ because an additional packaging-only test is added, that file is authoritative.

Important logic hardening performed in V30:

- structural-domain capacity probe is a forced no-op;
- upstream B16 budget is frozen; B24 exists only as a separate post-selector interface ceiling;
- budget accounting is split explicitly into upstream selector B16 and retained-interface B24, fixing a generic evaluator logging bug that would otherwise mark the capacity arm as over-budget;
- capacity expansion must be a set superset of B16 — no baseline removal/reallocation;
- duplicate/invalid baseline/reference indices fail closed;
- full-bank cost feasibility is checked;
- no new query is encoded and screened by paired per-scene query counts;
- V29 FCR is disabled in V30;
- all historical no-repeat constraints remain in `ALGORITHM_CHANGELOG.md`.

---

## 14. Recommended next command

```bash
cd bdse_v64_3_30_eaf_icer_fbic
bash RUN_V64_3_30_EAF_ICER_FBIC_SCREEN_2GPU.sh
```

Do not run a different B, change thresholds, or add a classifier before interpreting the two independent V30 blocks.
