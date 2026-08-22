# V64.3.30 corrected TRAIN postmortem and V64.3.31 OMCER design

## Executive decision

The uploaded **gate-fixed V64.3.30** result is technically valid and is suitable for causal attribution **at the frozen TRAIN gate only**. It is not a fresh-validation result: the launcher stopped before `fresh_selection`, so the V30 run consumed no new A/B validation scenes and the inspected-validation exclusion remains exactly 8,700 tokens.

No new engineering error was found. The previous zero-selection plumbing defect is fixed: the authoritative selector diagnostics contain 327 proposal locks, the V30 fitter consumes those locks, and the corrected cross-fit selects 38 proposals rather than zero. The raw PCWER row/edge provenance hashes match the fitter report; both arms replay the same frozen 3,000-scene TRAIN manifest; and the historical V25 reproduction is exact (75,133 frontier rows, 1,455 eligible replacement edges, 310 replacement scenes, 71 selected, teacher-improvement sum +5.5276423258).

The corrected V30 therefore represents a **real mechanism failure**, not an engineering failure.

---

## 1. What V30 establishes

### 1.1 PCWER is active, contract-correct, and nevertheless fails the TRAIN safety gate

PCWER attempts a risk-free proposal on 327/3,000 scenes and accepts 126 fixed-budget evidence rebindings. Accepted rebindings satisfy the intended budget/cardinality/proposal/incumbent/anchor contracts. Mean proposal-margin L-infinity reconstruction error improves by about 0.03443. Thus the mechanism is not inert.

Yet the proposal-locked DRC cross-fit has only 2/5 path-safe folds, selects 38 proposals, has mean positive-opportunity capture 12.697%, and creates two selected catastrophes:

- fold 0 worst teacher improvement: -0.9906337634;
- fold 2 worst teacher improvement: -0.9893587528.

The TRAIN gate correctly stops before fresh validation.

### 1.2 Same-proposal/no-fallback is not the failure

Replaying the **same 327 risk-free proposal identities on the original AOCC evidence**, with the frozen V25 DRC allowed only to accept/reject the same proposal, gives:

- 35 selected proposals;
- teacher-improvement sum +1.994451;
- mean capture 12.329%;
- worst selected outcome only about -5.36e-4;
- **0 selected catastrophes**.

Therefore the monotone proposal-lock/no-second-best operator is a useful safety invariant. V30's catastrophic regression is caused by changing the evidence representation, not by the same-proposal operator.

### 1.3 Both V30 catastrophes are accepted PCWER rebindings and are sign flips of the risk certificate

For the two catastrophic selected proposals, the original AOCC representation rejects the same proposal, while PCWER makes the DRC score positive:

- `67a57ae417045162`: same q=2, teacher improvement -0.9906338; baseline DRC -0.070323 -> PCWER DRC +0.030633.
- `bb77e9686029538d`: same q=20, teacher improvement -0.9893588; baseline DRC -0.223642 -> PCWER DRC +0.175645.

Both rebindings pass the PCWER evidence contract. Across shared accepted proposal identities the score shift is biased upward. Hence PCWER can improve its proposal/anchor/incumbent reconstruction objective while systematically changing the local risk geometry in the wrong direction.

### 1.4 V29's central conclusion can now be strengthened

V29 showed:

> Better reconstruction of the **global complete action frontier** is not equivalent to more decision-sufficient evidence for downstream extremal recovery.

V30 adds:

> Better reconstruction of the **proposal-conditioned proposal↔anchor / proposal↔incumbent witness geometry** is still not equivalent to more **outcome-risk-sufficient** evidence for the selected extremal proposal.

This rules out a broader class of "reconstruct the decision object more faithfully, therefore risk becomes safer" assumptions. Frontier/witness fidelity and selected-outcome risk sufficiency are distinct objects.

Accordingly, **PCWER should be removed from the main algorithm rather than tuned**. No PCWER-v2, witness weighting, lexicographic-order, acceptance-threshold, or B/M rescue is justified.

---

## 2. Updated gate decomposition and dominant bottleneck

The corrected V30 proposal generator produces 327 unique risk-free proposal scenes. When their exact teacher improvements are recovered, the proposal population contains:

- 327 proposals total;
- 219 teacher-positive proposals (66.97%);
- 26 catastrophic proposals with teacher improvement <= -0.5;
- a heavy negative tail down to -9.839625.

Thus proposal generation is neither empty nor mostly wrong. It contains substantial recoverable positive mass, but a sparse heavy tail makes naive extremization unsafe.

The critical mismatch is now narrower than the V28/V29 wording "safe recovery coverage under fixed-budget decision-evidence transmission":

> **safe extremal proposal coverage under operator-induced post-selection / winner's-curse risk mismatch**.

The historical DRC memory is fit on all support-positive/scalar-dominance-positive eligible edges, whereas deployment chooses an extremal proposal using those same support/dominance signals. An evidence-only risk certificate therefore evaluates a *selected* candidate while being blind to the selector state that made it a winner.

This is an operator/risk-alignment problem, not yet evidence-capacity evidence. B=16 must stay frozen.

A paper-level phrasing is:

> **operator-aligned catastrophic-tail admissibility for extremal recovery under a fixed planner-interface evidence budget.**

---

## 3. Which mechanisms remain, which stop

### Retain / strengthen

1. Fixed bounded interface (`B=16` as the controlled operating point, not the novelty claim).
2. EAF attributed deployment-admissible complete frontier.
3. Frozen support + scalar incumbent-dominance semantics.
4. Asymmetric admissible-incumbent preservation.
5. Monotone intervention: once a proposal is formed, no independent view may rerank into a second-best fallback.
6. Structural all-flagged delegation / incumbent-default deployment guard.
7. Catastrophic tail `teacher improvement <= -0.5` as a hard experimental safety contract.

### Stop as main mechanisms

1. PTMC / type-KNN confirmation: already fresh-falsified.
2. FCR global complete-frontier rebinding: fresh-falsified.
3. PCWER proposal-conditioned witness rebinding: corrected V30 TRAIN-falsified on tail safety.
4. Any claim that reconstruction fidelity itself is the sufficient risk statistic.

### New rejected diagnostics from corrected V30 TRAIN

These were tested only to locate the next mechanism and must **not** become new tuning loops:

- hard proposal-only KNN memory (327 proposal-path population): too sparse; only 18 selected, ~61% precision, and catastrophic failures recur;
- one-neighbor-edge-per-source-scene deduplicated memory: collapses to zero selection;
- generic residual/gap/count feature concatenations: do not recover the required safe coverage;
- transition/signed-profile/full-attribution rescue: historically falsified branches remain forbidden.

---

## 4. Critical structural refinement: risk must precede extremization

V30 used the chain:

`risk-free unique q -> DRC confirms/vetoes q -> no fallback`.

This is safe on original evidence but structurally conservative: if the scalar-best q is unsafe, the operator immediately preserves the incumbent even when a different alternative could have been both safe and beneficial.

The more precise decomposition is:

1. compute deployment-admissible support/scalar-positive alternatives;
2. evaluate **risk admissibility for every such alternative** using a certificate aligned with the same operator state;
3. extremize **once** over the risk-admissible set;
4. after that one proposal is formed, preserve the no-rerank/no-second-best invariant for any later guard.

This is not post-veto fallback. Risk is part of proposal **admissibility before proposal formation**. There is only one extremization event.

A controlled TRAIN placement diagnostic using the same new 19-D certificate gives:

- post-extremal risk-free-q then confirm: 43 selected, teacher sum +4.1616, mean capture 14.78%, 0 catastrophes, but only 4/5 count/sum folds;
- **pre-extremal risk admissibility then single extremization**: 80 selected, +8.7537, mean capture 26.46%, 0 catastrophes, 5/5 folds.

This is the main structural reason V31 moves risk before extremization.

---

## 5. V31: EAF-ICER-OMCER

**OMCER = Operator-Margin Catastrophic-Excess Regret Certification.**

V31 deliberately keeps the original AOCC evidence and removes PCWER/FCR from the main path.

### 5.1 Minimal operator-conditioning statistic

For every deployment-admissible candidate already satisfying the frozen support/scalar semantics, append exactly one runtime scalar to the historical 18-D aggregate evidence vector:

\[
m_{op}=\min(\ell_{support},\ell_{scalar-dom}).
\]

This is the weakest-link signed distance to the existing joint eligibility boundary `support>0 AND scalar_dominance>0`. It is already computed by the frozen selector, adds no evidence query, no learned head, no validation-tuned weight, and no new threshold.

### 5.2 Tail-aligned local certificate

The V25 DRC uses local mean minus RMS of **all negative** outcomes. That treats a tiny negative such as -1e-4 as the same type of downside event as a catastrophic replacement, which suppresses recall while not explicitly encoding the paper's catastrophic-tail contract.

V31 keeps the local mean but defines catastrophic excess relative to the already frozen threshold `tau_cat=-0.5`:

\[
e_i=\min(y_i-\tau_{cat},0), \qquad \tau_{cat}=-0.5,
\]

and for each fixed neighborhood scale K in {32,64}:

\[
C_K = \mu_K - \sqrt{\sum_i w_i e_i^2}.
\]

The runtime score is `min(C_32,C_64)` and the existing semantic zero boundary is unchanged: risk-admissible iff score > 0.

The TRAIN memory remains the full 1,455 eligible edge population. K, distance weighting, multiplier=1, decision boundary=0, B=16 and M=24 are all frozen.

### 5.3 Why both factors are necessary: frozen TRAIN 2x2

| Representation / certificate | selected | teacher sum | mean capture | catastrophes | full 5-fold gate |
|---|---:|---:|---:|---:|---:|
| 18-D evidence + V25 all-negative downside | 71 | +5.5276 | 21.10% | 1 (`-0.5458`) | tail FAIL |
| 18-D evidence + catastrophic-excess | 94 | +4.3720 | 28.41% | 1 (`-1.7026`) | FAIL |
| 18-D + operator margin + all-negative downside | 56 | +6.1351 | 17.64% | 0 | count FAIL (4/5) |
| **18-D + operator margin + catastrophic-excess (OMCER)** | **80** | **+8.7537** | **26.46%** | **0** | **5/5 PASS** |

Interpretation:

- catastrophic-excess alone restores coverage but is unsafe;
- operator margin alone stabilizes the tail but is too conservative;
- their combination is complementary on frozen TRAIN.

This is a design-screen result, **not fresh evidence**. It licenses exactly one double-fresh test of the fixed mechanism, not threshold/feature search.

---

## 6. Mechanism-chain update

The V29/V30 candidate chain was:

`bounded evidence interface -> attributed complete frontier -> unique incumbent-contrastive proposal -> proposal-conditioned witness compression -> same-proposal downside confirmation -> incumbent preservation -> structural deployment guard`.

After corrected V30, it should be tightened to:

> **bounded evidence interface -> attributed deployment-admissible complete frontier -> frozen support/scalar eligibility -> operator-conditioned catastrophic-tail risk admissibility -> single incumbent-contrastive extremization -> incumbent-default monotone intervention -> structural deployment guard**.

The two major deletions are intentional:

- remove proposal-conditioned evidence compression;
- remove post-extremal downside confirmation as the primary risk stage.

The no-fallback principle remains, but it is now applied **after one risk-admissible proposal has been formed**, rather than forcing an unsafe scalar winner to consume the only proposal slot before risk is known.

This chain is conceptually tighter and closer to a CCF-A-level mechanism statement because each stage has a distinct semantic role and a falsifiable ablation.

---

## 7. V31 experiment protocol

### TRAIN

- exact frozen 3,000-scene manifest and fold seed retained;
- exact historical V25 reproduction required;
- 2x2 is reported on TRAIN only;
- OMCER main must be 5/5 path-safe and catastrophe-free;
- at least 64 selections, >=5 more than V25, >=+3 percentage points mean capture over V25, nonnegative teacher sum, and mean negative RMS noninferior;
- failure stops before fresh; no tuning of B/M/K/-0.5/multiplier/zero boundary/operator-margin definition.

### Fresh

Corrected V30 consumed no fresh validation, so exclusion remains exactly 8,700 inspected tokens. A new hash seed selects a new untouched 1,000 scenes, split A/B=500/500.

Five arms:

1. raw;
2. V20 high-coverage historical reference;
3. V25 evidence-only aggregate-downside DRC;
4. V30 proposal-lock-only DRC (post-extremal placement control);
5. **V31 OMCER main**.

Each A/B split must independently satisfy:

- exact paired identity and instrumentation;
- learned incumbent->anchor = 0;
- all-flagged structural identity/delegation contracts;
- selected direct replacement count >=8 and endpoint regret-delta sum <=0;
- **zero selected teacher improvement <= -0.5**;
- OMCER capture >= V25 +3pp **and** >=5 additional positive direct recoveries;
- OMCER capture >= proposal-lock control +3pp **and** >=5 additional positives;
- selected-tail negative RMS/worst/positive-regret RMS noninferior to both controls;
- endpoint noninferior to V25 and raw under the existing tolerances;
- at least one split must show a strict endpoint signal for full-validation promotion;
- no pooled A+B rescue.

Passing both blocks authorizes one independent full-validation reproduction only. Test/closed-loop remain forbidden.

---

## 8. Stop rules after V31

If V31 fails tail safety, stop OMCER. Do not tune K, `-0.5`, multiplier, decision boundary, operator-margin weight/transform, B or M to rescue it.

If V31 is catastrophe-free but fails the required coverage gain, do not add arbitrary operator feature combinations. The conclusion is that the minimal selector-state conditioning is insufficient; next work should audit proposal-generation semantics or perform a controlled capacity/observability study.

If mechanism gates pass but endpoint fails, audit mediation through final structural guards/endpoint metric composition before changing the certificate.

Do not reopen PTMC, type-KNN, FCR, PCWER, signed profile, transition geometry, action blacklists, OOD radius, DACC/beam/swap, learned incumbent->anchor, failed acquisition branches, or pooled rescue.

---

## 9. Engineering validation at delivery

- V31 fitter reproduces the precomputed frozen TRAIN 2x2 exactly.
- V31 hard config/memory contract: PASS.
- `python -m py_compile` / repository compile checks: PASS.
- V13--V31 targeted regression: **122/122 PASS**.
- full repository: **451 PASS / 1 inherited packaging FAIL / 36 warnings**. The only failure remains the historical missing root file `V64_SAQA_BCC_NEXT_COMMANDS.sh`; V31 does not depend on it and no fabricated replacement was added.
- `bash -n RUN_V64_3_31_EAF_ICER_OMCER_SCREEN_2GPU.sh`: PASS.
- no V31 fresh effectiveness result is fabricated locally; server cache/checkpoint execution is required.
