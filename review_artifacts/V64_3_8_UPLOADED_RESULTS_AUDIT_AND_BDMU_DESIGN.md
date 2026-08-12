# V64.3.8 — uploaded-result audit and BDMU design

## 1. Executive conclusion

The newly uploaded V64.3.7 matrix changes the decision point relative to the previous review because **both BROAD and LITERAL are now complete and protocol-valid**. The next step no longer needs to wait for the LITERAL arm.

The current evidence supports three conclusions:

1. **DARM+DBR is a real positive downstream value mechanism.** Both arms improve the fixed-B teacher action while acquisition is numerically unchanged, so the gain is attributable to pair-value/final aggregation rather than proposal selection.
2. **LITERAL is the better provisional main value branch for the paper endpoint, but BROAD exposes a severity/regret signal that must not be ignored.** LITERAL has the best exact action preservation; BROAD has substantially better teacher/pair-full regret.
3. **The next bottleneck is now acquisition, not another pair architecture.** Critical Top-M stays at 0.3601533 and selected-critical at 0.2605364 in both arms. The most defensible next algorithm is therefore a theorem-aligned continuous acquisition target, not another binary critical classifier.

I implement **V64.3.8 BDMU: Budgeted Decisive-Margin Marginal Utility**, trained only through the existing zero-init CCBR representation primitive while freezing the promoted DARM+DBR value path.

The tightened mechanism chain is:

`fixed planner-interface budget -> auditable evidence atoms -> budget-feasible decisive-margin marginal utility -> budgeted acquisition -> DARM one-sided margin preservation -> final decision preservation`.

No new numerical BDMU claim is made in this delivery because the nuPlan caches/GPU runtime are not available in the delivery environment. The code is designed so BDMU is hard-gated behind a full-pipeline DARM+DBR reproduction before any new acquisition GPU spend.

---

## 2. Uploaded V64.3.7 BROAD vs LITERAL audit

Immutable epoch -1 anchor for both arms:

| metric | anchor |
|---|---:|
| final teacher action match | 0.264 |
| pair-full teacher action match | 0.264 |
| selected-local pair-full | 0.264 |
| teacher regret | 14484.4613 |
| pair-full regret | 14079.9774 |
| exact critical Top-M micro recall | 0.3601533 |
| selected exact-critical micro recall | 0.2605364 |
| proposal decisive recall | 0.7915091 |

Selected protocol-valid rows:

| metric | BROAD e3 | delta | LITERAL e2 | delta |
|---|---:|---:|---:|---:|
| final teacher match | 0.282 | **+1.8pp** | **0.286** | **+2.2pp** |
| pair-full match | 0.274 | +1.0pp | **0.280** | **+1.6pp** |
| local pair-full | 0.264 | 0 | 0.264 | 0 |
| teacher regret | **13367.1395** | **-1117.32** | 14027.5786 | -456.88 |
| pair-full regret | **13673.7794** | **-406.20** | 14031.3813 | -48.60 |
| beneficial residual intervention | 0.022 | +0.022 | **0.028** | +0.028 |
| harmful residual intervention | 0.012 | +0.012 | 0.012 | +0.012 |
| residual net | +0.010 | — | **+0.016** | — |
| beneficial compression | **0.016** | +0.012 | 0.012 | +0.008 |
| harmful compression | 0.008 | +0.004 | **0.006** | +0.002 |
| compression net | **+0.008** | — | +0.006 | — |
| exact critical Top-M | 0.3601533 | 0 | 0.3601533 | 0 |
| selected exact-critical | 0.2605364 | 0 | 0.2605364 | 0 |
| proposal decisive recall | 0.7915091 | 0 | 0.7915091 | 0 |

### Interpretation

**LITERAL should be the provisional full-pipeline value candidate.** It is not a separate replacement for BROAD: it retains BROAD winner/hard/near support and adds a bounded quota of exact teacher-winner -> leave-one-atom-out flip pairs. Its +2.2pp final teacher gain and +1.6pp pair-full gain are the strongest direct evidence for the paper's decision-preservation endpoint.

However, **do not throw away BROAD as “worse.”** BROAD reduces teacher regret by 1117.32 versus only 456.88 for the selected LITERAL epoch, and pair-full regret by 406.20 versus 48.60. This indicates that exact literal boundary supervision improves decision identity more aggressively than decision severity. That action/regret tension is useful evidence for the next target: acquisition should preserve a *continuous decisive margin object* over several nearest rivals rather than learn only a sparse winner-flip event.

The raw LITERAL epochs also show a checkpoint Pareto tension: e2 is action-best (0.286 teacher match), while e3 is lower at 0.280 but has much better teacher regret 13479.08. Full runs therefore save every epoch and use an explicit validation-only selection/audit instead of silently trusting `best.pt`.

### Ablation component worth carrying into the main value branch

The only current ablation component with enough positive evidence to promote is the **bounded literal winner-flip pair quota inside DBR training**. It should enter the *provisional* DARM+DBR main value branch for the full-pipeline confirmation because it improves both final teacher match and pair-full-over-local value correction. BROAD remains the necessary ablation to demonstrate that exact-boundary emphasis changes the decision-preservation endpoint.

No old acquisition ablation should be merged now. AP-WCCA/AP-WRCCA, FPCCA/CCBR objectives, BCHA, larger B/M, full proposal unfreeze and binary literal-critical BCE have already provided adequate negative/ceiling evidence in the algorithm log.

---

## 3. Main bottleneck after DARM+DBR

The causal result is unusually clean: for BROAD and LITERAL, critical Top-M, selected-critical and proposal-decisive recall are unchanged from the immutable anchor, while final decisions improve. Therefore:

- downstream decisive value/aggregation was a real bottleneck and DARM+DBR partially fixes it;
- acquisition is now the dominant unresolved bottleneck;
- critical Top-M ~=36% is too low for a paper whose central claim is budgeted decision-sufficient evidence;
- the next acquisition objective should use the same theorem object as DARM: teacher winner-vs-decisive-rival one-sided margins.

A CCF-A-level story becomes tighter if every learned component can be attached to one preservation object rather than introducing another heuristic selector loss.

---

## 4. V64.3.8 BDMU algorithm

### 4.1 Fixed reference interface

Let `w` be the scalar full-information teacher winner and let `S_B` be an **immutable, budget-feasible B=16 reference set derived from the frozen promoted V64.3.7 DARM+DBR foundation**. For training throughput, the implementation constructs `S_B` with the existing all-GPU one-shot MARS/HAB budget surrogate (`_fast_pair_margin_surrogate_masks`), rather than putting the exact NumPy backward-elimination selector on every training step. This is deliberately a reference-set approximation, not a claim that the training target reruns the exact deployed selector. During V64.3.8 training, DARM, DBR, the foundation proposal/family stack, deployed selector semantics, B=16 and proposal Top-M are frozen; exact end-to-end selector behavior is still measured by validation/open-loop/closed-loop evaluation.

The current trainable proposal is

`q(i) = q_foundation(i) + r_BDMU(i)`

where the new CCBR proposal residual is zero initialized. The implementation reconstructs `q_foundation` exactly by subtracting the trainable residual before building `S_B`, so the BDMU target cannot chase its own changing ranking.

### 4.2 One-sided decisive margin deficit

For the nearest `R` valid teacher rivals `b`, define normalized full teacher margin

`m_T(w,b) = [J_T(b)-J_T(w)] / s`,

with scene scale `s`. The margin that BDMU asks the budgeted interface to preserve is

`gamma_b = min(m_T(w,b), max(rho*m_T(w,b), gamma_floor), gamma_cap)`.

For the frozen B-set,

`delta_b(S_B) = [gamma_b - m_{S_B}(w,b)]_+`.

Rivals are softly weighted toward the nearest decision boundaries. The default is `R=4`, deliberately broader than literal R=1 so the target can retain BROAD's severity signal while remaining decisive-boundary focused.

### 4.3 Strict budget-feasible marginal utility

The final implementation does **not** use an infeasible B+1 counterfactual.

For a selected atom `i in S_B`, utility is the increase in one-sided margin deficit when removing it:

`U_remove(i) = sum_b pi_b [ delta_b(S_B - {i}) - delta_b(S_B) ]_+`.

For a missed atom `i not in S_B`:

- if genuine budget slack allows adding `i`, use the corresponding deficit reduction;
- otherwise compute the best **budget-feasible single exchange** `S_B - {j} + {i}` over selected atoms `j` whose removal makes the query feasible:

`U_swap(i) = max_j sum_b pi_b [ delta_b(S_B) - delta_b(S_B - {j} + {i}) ]_+`.

The acquisition target is

`U_BDMU(i) = U(i) / c_i^p`,

with default query-cost power `p=1`. Thus every positive target corresponds to a valid fixed-budget local intervention. Exact literal winner-flip atoms are a high-value limiting case rather than the only positive label.

### 4.4 Acquisition loss

BDMU normalizes positive utility within each scene and uses listwise cross entropy against the current proposal logits. A tiny residual L2 prior preserves the zero-init foundation ranking where the teacher utility supplies no reason to move.

Only `critical_proposal_adapter` is trainable. CCBR is used **only as a complete-candidate factorized representation primitive**; old CCBR/LEA/HCBE/ACRA/literal BCE objectives are all zero/disabled. DBR must be present in the warm-start checkpoint and is not allowed as a missing prefix.

### 4.5 Theorem bridge to add to the paper

Let the decisive-rival set `R(w)` contain every challenger used by the downstream preservation certificate, and define the non-negative aggregate deficit

`D(S)=sum_{b in R(w)} pi_b [gamma_b-m_S(w,b)]_+`, with every `pi_b>0` and every `gamma_b>0`.

Then `D(S)=0` implies `m_S(w,b)>=gamma_b>0` for every decisive rival. Under the same validity/safety assumptions already used by the DARM preservation theorem, this is a direct sufficient condition for the budgeted interface to keep `w` as the final action. BDMU's swap target is exactly the local decrease `D(S_B)-D(S_B-{j}+{i})` restricted to budget-feasible exchanges. Therefore BDMU does not need a new independent correctness theorem: it supplies an **acquisition-side descent object for the existing decisive-margin certificate**.

The practical `R=4` target is an approximation to the full decisive-rival set, so the paper should not claim that training loss alone certifies preservation. The exact downstream certificate remains the final verifier. The `R=1` vs `R=4` ablation tests how much multi-rival coverage is needed empirically, while the theorem statement should be written for the complete decisive-rival set.

---

## 5. Why this is a better novelty direction

The novelty claim should not be “decision-focused selection” in general. That space already contains strong recent theory and task-aware sensing work. The defensible paper-specific novelty is the combination of:

1. a **fixed planner-interface evidence budget** rather than unconstrained feature attention;
2. **auditable queryable evidence atoms** with explicit query costs;
3. a theorem-coupled **one-sided decisive action-margin** preservation object;
4. **budget-feasible local marginal utility** defined around an immutable deployment-derived reference set;
5. a frozen DARM+DBR decision interface so acquisition gains have clean causal attribution;
6. final action preservation/regret and closed-loop evaluation under the exact same B=16 interface.

This aligns the algorithmic object, ablations and theorem rather than accumulating selector modules.

---

## 6. Experiment plan and falsifiable gates

### Mandatory causal sequence

**A. V64.3.7 full confirmation.** Run 50k train / 1k val DARM+DBR-LITERAL from the same V62 foundation. BDMU is blocked unless the existing teacher-oriented full audit still returns `full_promotion=true`.

**B. BDMU screen.** 12k train / 500 val, 4 epochs, both GPUs. The screen must show:

- adapter actually moved;
- non-zero BDMU utility support;
- Top-M BDMU utility capture improves by at least +2pp;
- selected utility capture or exact-critical Top-M provides corroborating mechanism gain;
- final teacher match improves >=+0.5pp **or** teacher regret improves >=2%;
- teacher match and regret stay inside no-harm tolerances.

Utility improvement alone cannot promote the model.

**C. BDMU full.** 50k / 1k, 8 epochs, every epoch saved. Re-run the same checker on the full log; only a full-promotion row is eligible for final evaluation.

### Theory ablations — run only after main full BDMU succeeds

1. **R1 vs R4:** isolate whether multi-rival decisive utility is responsible for balancing exact action preservation and regret severity.
2. **No-cost normalization:** set `cost_power=0` while all other semantics are fixed. This tests whether “value per planner query cost” is essential rather than decorative.
3. **Existing BROAD vs LITERAL:** already completed and should be retained as the value-side ablation; do not rerun unless a publication-sized replication is required.

### Analysis experiment with no new training

Bucket validation scenes by **foundation BDMU missed-utility fraction / reference margin deficit**, then report BDMU-vs-foundation teacher-action gain and regret gain in each bucket. A strong mechanism result should concentrate improvements in scenes where the frozen B=16 interface leaves measurable decisive utility outside Top-M/selected evidence. This is much stronger causal evidence than another aggregate selector recall table.

### Closed-loop protocol

After checkpoint and hyperparameters are frozen from validation:

- CL20 non-reactive is integration/safety debug only;
- CL100 non-reactive and CL100 reactive use the exact same frozen checkpoint;
- use the same scenario tokens for V64.3.7 DARM+DBR-LITERAL and V64.3.8 BDMU and report paired deltas / bootstrap confidence intervals;
- do not tune on the held-out test open-loop or the final reactive closed-loop result.

---

## 7. Engineering and efficiency changes

### BDMU-only loss fast path

The existing training framework constructs many legacy objectives even when their weights are zero. In uploaded V64.3.7 logs, the loss stage alone is material (for example LITERAL e3: 73.63 s of a 217.81 s epoch; pair sampling ~=1.92 s). V64.3.8 therefore detects the strict case where BDMU is the **only positive loss weight** and skips construction of all zero-weight legacy losses.

A configuration trap was also fixed: `load_config` injects a non-zero default `family` loss when omitted. Every V64.3.8 configuration now sets `family: 0.0` explicitly, so the fast path is truly active. This is a semantics-preserving compute optimization, not a model simplification.

No measured V64.3.8 GPU speedup is claimed here because the dataset/GPU runtime is unavailable. The next logs export `bdmu_fast_path_active` and the same stage-wall instrumentation so the actual loss/epoch speedup can be measured directly.

### Legacy-eval compatibility

BDMU teacher utility diagnostics are computed only when the V64.3.8 utility block is enabled. V64.3.7-and-earlier evaluation does not pay the new diagnostic overhead.

### Checkpoint contract

V64.3.8 allows the new zero-init proposal adapter to be absent from the warm start, but **does not allow `decisive_boundary_pair_adapter.*` to be missing**. Accidentally pointing BDMU at V62 instead of a promoted V64.3.7 DARM+DBR checkpoint fails the config/checkpoint contract instead of silently training the wrong algorithm.

### Full-run selection

Both V64.3.7 and V64.3.8 full launchers save every epoch and default to training only. Evaluation is run only after a validation audit selects the epoch, avoiding wasted evaluation of an opaque `best.pt` and preventing test-set checkpoint selection.

---

## 8. What is intentionally not changed

- evidence budget remains B=16;
- proposal Top-M remains unchanged;
- DARM final aggregation semantics remain unchanged;
- DBR architecture/weights are frozen during BDMU;
- evidence atom definitions and teacher construction are unchanged;
- DA-EPC / safety / fallback semantics are unchanged;
- no full proposal/foundation unfreeze;
- no AP-WCCA/AP-WRCCA, FPCCA, BCHA, binary literal-critical BCE, global tournament, global/evidence/set potential, beam/swap/bruteforce selector retry.

This is important for paper attribution: V64.3.8 asks one narrow question — **can the fixed B=16 acquisition interface recover more of the already validated DARM decisive-margin value by ranking atoms according to budget-feasible marginal utility?**

---

## 9. Expected paper positioning if the experiment succeeds

A successful result should be presented as a progression rather than a module stack:

1. full information defines the teacher decision and decisive margins;
2. evidence is compressed into auditable query atoms under a hard interface budget;
3. DARM+DBR shows that correcting decisive margins can improve the final decision without changing acquisition;
4. BDMU then learns which atoms are worth querying **because of their marginal contribution to those same decisive margins**;
5. the final theorem/empirical story connects margin preservation to teacher-action preservation, bounded regret and closed-loop behavior.

That is substantially tighter than describing CCBR/DBR/DARM/selector losses as independent tricks.
