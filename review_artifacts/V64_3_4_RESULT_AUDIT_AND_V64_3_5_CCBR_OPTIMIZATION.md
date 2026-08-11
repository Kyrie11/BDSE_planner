# V64.3.4 result audit and V64.3.5 CCBR+LEA optimization

## 1. Bottom line

The uploaded V64.3.4 acquisition ablations do **not** contain a positive algorithm that should be merged into the main method. AP-WRCCA+LCV, FPCCA-noLBA, FPCCA+LBA (F6), and FPCCA-F8+LBA all leave the same-subset literal teacher winner-flip critical Top-M micro recall exactly unchanged at `0.3601532567`. Several downstream acquisition diagnostics regress slightly. The adapters demonstrably train and produce non-zero residuals, so this is not another “module was not wired” result.

The most important new evidence is that FPCCA's representation support is structurally too small. F6 LBA has maximum literal-boundary representability `0.12408`; F8 only reaches `0.20250`. The independent validation diagnostic says only `0.20479` of literal critical boundaries have both endpoints in base top-6, and the teacher winner itself lies in base top-6 only `0.36626` of eligible scenes. Therefore increasing F is an inefficient way to repair the semantic support problem.

The next main algorithm is V64.3.5 **Complete-Candidate Boundary Routing (CCBR)** with optional **Literal Endpoint Attribution (LEA)**. It exposes every valid candidate action as a possible boundary endpoint using two factorized O(EK) atom-to-action routers, rather than enumerating O(EK^2) pair tokens. It reuses action embeddings and J0 that already exist at deployment, adds no new expensive atom-action evidence query, and leaves fixed `B=16` and proposal `M` unchanged. LEA supervises only exact literal winner-flip critical atoms with their teacher winner and leave-one-out flip endpoint; non-critical atoms receive no endpoint target.

## 2. V64.3.4 same-subset screen results

Anchor values on the common 500-scene validation subset:

- literal critical Top-M micro recall: `0.3601532567`
- literal critical selected micro recall: `0.2605363985`
- proposal decisive recall: `0.7915091362`
- teacher action match: `0.264`
- mean literal critical count on eligible scenes: `0.537037`
- literal critical atom fraction: about `0.01181`
- teacher winner in base top-6: `0.366255`
- literal boundary in base top-6: `0.204787`

Screen deltas relative to the same anchor:

| Variant | Δ critical Top-M | Δ critical selected | Δ proposal decisive | Δ teacher match | boundary support max | decision |
|---|---:|---:|---:|---:|---:|---|
| AP-WRCCA+LCV | 0.0000 | -0.00383 | -0.01852 | -0.006 | 0 | negative |
| FPCCA-noLBA | 0.0000 | -0.00766 | -0.02829 | -0.006 | n/a | negative |
| FPCCA+LBA F6 | 0.0000 | -0.00766 | -0.01456 | -0.004 | 0.12408 | negative |
| FPCCA+LBA F8 | 0.0000 | -0.00766 | -0.01573 | -0.004 | 0.20250 | negative |

`FPCCA-noLBA` was marked invalid by the old checker only because the string test `"LBA" in variant.upper()` also matches `noLBA`. This is a checker bug, not an algorithmic invalidity. It is fixed in V64.3.5. Even after correcting that bookkeeping, FPCCA-noLBA remains a negative result because critical Top-M does not move and proposal decisive recall regresses by more than 2 percentage points.

Adapter activation is not the explanation. AP-WRCCA+LCV reaches residual RMS about `1.027`; FPCCA variants reach roughly `0.406--0.488`, with non-zero parameter deltas and non-zero ACRA/LBA losses. The learned correction can move logits but does not admit additional literal-critical evidence into Top-M.

## 3. Why FPCCA failed

FPCCA addressed the V64.3.3 failure mode by replacing one mostly-wrong base winner with a top-F pair frontier. The current results show that the remaining support assumption is still too strong: the teacher-relevant literal boundary often involves an endpoint outside that top-F set. F8 approximately doubles learned LBA support relative to F6 (`0.124 -> 0.203`) but produces **zero** Top-M improvement. This is exactly the pattern expected when capacity is spent on a larger but still severely truncated boundary set.

The correct response is therefore not F=10/F=12, a larger M, a larger B, more ACRA weight, or a relaxed certificate. Those would either retain the same semantic support problem or confound the fixed-budget contribution.

## 4. Current bottleneck ordering

### Bottleneck A: complete literal-boundary acquisition support

This is the first bottleneck because all four clean screen variants keep literal critical Top-M at the anchor value. The target is extremely sparse: roughly 1.18% of active atoms are literal-critical in validation and eligible scenes contain only about 0.54 such atoms on average. A representation that excludes 80--88% of their true boundaries cannot make efficient use of direct attribution supervision.

### Bottleneck B: possible frozen family-slot admission ceiling

The current atom residual changes within-family atom ranking, while HAB family allocation remains frozen. Existing results do not directly show whether that creates a second hard ceiling. V64.3.5 therefore adds a diagnostic-only `teacher_exact_winner_flip_frozen_family_slot_oracle_topm_recall`: it gives exact critical atoms oracle-dominant scores while keeping the runtime family logits, family ids, B and M unchanged. This metric does not train the model or alter deployment.

Interpretation:

- oracle >= 0.90 and learned critical Top-M remains low: family slots are not the main blocker; the atom representation/ranking is failing.
- oracle materially below 0.90: an atom-only residual cannot solve acquisition. Only then is a small zero-initialized boundary-aware **family residual** justified as a conditional experiment. Do not unfreeze the full historical family/proposal stack.

### Bottleneck C: atom-to-action / pair-boundary value representation

Prior full-support V64.3.3 evidence showed pair-full/candidate teacher match around `0.236/0.224`. That remains a likely later bottleneck, but the V64.3.4 screens do not reach the condition required to switch focus: critical Top-M never improves. If CCBR materially raises Top-M/selected recall and pair-full/candidate teacher match stays flat, stop acquisition work immediately and move to a decision-boundary value model. Do not repeat V55 global action-potential projection or V59 generic set-conditioned interaction potential; the next value model should be explicitly tied to the literal/decisive action boundary identified by the acquisition mechanism.

## 5. V64.3.5 algorithm: CCBR + LEA

### Complete-Candidate Boundary Routing (CCBR)

For every atom i and every valid candidate action a, CCBR uses the existing action embedding together with normalized J0 cost and J0 rank. Two atom-conditioned routers produce distributions over all K valid candidate actions:

- winner endpoint router alpha_i(a)
- flip endpoint router beta_i(a)

The two routed action contexts are composed with difference and multiplicative interaction features, then passed through a zero-initialized residual head added to the immutable legacy HAB atom proposal. Complexity is O(EK), not O(EK^2), and K is the candidate bank already encoded by the planner.

Key invariants:

- fixed retained evidence budget B=16 unchanged;
- proposal M unchanged;
- no extra atom-action evidence query;
- no teacher input at deployment;
- legacy proposal/family heads remain frozen;
- step-zero proposal is exactly the previous HAB proposal because the final CCBR residual head is zero initialized;
- invalid/non-finite candidate actions are masked from endpoint routing.

### Literal Endpoint Attribution (LEA)

For an exact teacher-interface literal critical atom, the full teacher interface supplies winner w and leave-one-out winner f_i. LEA applies CE to alpha_i -> w and beta_i -> f_i. It is gated by the exact critical mask and teacher-scalar alignment. Non-critical atoms get no endpoint label.

Unlike FPCCA/LBA, representability is the complete valid candidate bank; on aligned critical examples it should be near 1.0. The screen checker requires >0.95 endpoint representability for a LEA arm to be considered properly wired.

## 6. CCF-A novelty path

The strongest paper story is not “another selector network”. It is a fixed-budget decision-sufficiency chain:

1. decision-sufficient evidence is defined by preservation of decisive action margins under a fixed planner-interface budget;
2. literal winner-flip supervision identifies which evidence actually changes the decision;
3. CCBR provides **complete candidate-boundary support with factorized linear-time routing**, avoiding both a single-winner semantic anchor and truncated top-F pair support;
4. LEA ties the acquisition representation to exact boundary identity without leaking teacher information at deployment;
5. DA-EPC and the existing exact downstream selector certify preservation under the same B=16 interface.

For a strong paper, the mechanism ablation should show the causal sequence `top-F support ceiling -> full candidate support -> critical Top-M -> selected critical recall -> action/value bridge -> closed-loop`. If the chain breaks later, the next module should repair that exact bridge rather than adding generic capacity.

## 7. Extra DA-EPC fast runs are not V64.3.4 algorithm evidence

The three uploaded fast full-style runs (FPCCA-noLBA, FPCCA+LBA F6, FPCCA+LBA F8) found zero safe historical foundation checkpoints (`candidate_count=0`, `safe_candidate_count=0`) and fell back to a freshly rebuilt fast foundation. All three then failed the immutable-anchor gate:

- noLBA: base winner-rival sign accuracy `0.34598 < 0.62`, full-interface match `0.391`
- F6 LBA: `0.35370 < 0.62`, full-interface match `0.395`
- F8 LBA: `0.35056 < 0.62`, full-interface match `0.400`

Therefore these runs must not be used to claim FPCCA success or failure. Their useful contribution is engineering: they show that automatic foundation rebuilding can waste large runs and confound the experiment. V64.3.5 full wrappers now hard-require an explicit existing `FOUNDATION_CKPT` and a promotion report from the CCBR screen.

## 8. Efficiency audit

The V64.3.4 vectorized pair sampler is a successful optimization and is retained. In the acquisition matrix, pair-sampling time is about `1.35--1.76 s/epoch`, versus roughly `250 s/epoch` in the historical V64.3.3 analysis. No supervision/quota reduction was used for this speedup.

Current screen-time bottlenecks are instead:

- loss construction: roughly `40--113 s/epoch` in the matrix runs;
- data wait: highly variable, roughly `10--136 s/epoch`;
- model forward: roughly `8--13 s/epoch`.

V64.3.5 does not remove active objectives merely to benchmark faster. The recommended two-GPU settings remain `GPUS=0,1`, batch 16/GPU, 12 train workers/GPU, 4 validation workers/GPU, prefetch factor 2. CCBR itself is O(EK) and uses only already-computed action embeddings/J0; it avoids the O(K^2) boundary materialization that a complete-pair FPCCA would require.

The invalid fresh-foundation runs are much slower (loss stage around 0.9 ks/epoch) and are eliminated from the normal V64.3.5 path by the explicit-foundation gate.

## 9. Next experiments and stop rules

### Experiment A: CCBR-noLEA screen

Purpose: isolate the representation change. Same 12k/4-epoch same-subset protocol, frozen legacy HAB anchor, ACRA retained, LEA weight 0.

### Experiment B: CCBR+LEA screen

Purpose: test whether exact endpoint identity helps once representation support is complete. Same settings as A except LEA weight 0.25.

Promotion rule remains strict: critical Top-M must improve by at least +1.0 percentage point absolute; selected critical may not fall more than 0.5 pp; proposal decisive may not fall more than 2 pp; teacher match may not fall more than 0.5 pp. LEA must also show endpoint support >0.95.

### Diagnostic C: frozen-family-slot oracle

Read the new validation metric before adding any family-level algorithm. If it is low, run one conditional family-residual experiment. If high, do not touch family allocation.

### Full run

Only the promoted arm can enter the 50k/full DA-EPC pipeline. The launcher requires both the explicit matched foundation checkpoint and the promotion JSON.

### Stop rules after full run

- Top-M does not improve, family oracle high: stop acquisition architecture escalation; audit feature separability/label predictability for the rare literal-critical atoms.
- Top-M improves but selected recall does not: inspect exact selector/family cap/exchange behavior, not more acquisition loss.
- Top-M and selected improve but pair-full/candidate teacher match is flat: switch to explicit pair-boundary value modeling.
- open-loop improves but CL20 does not: move to interactive prediction/candidate dynamics/replanning. Do not keep stacking evidence-selection losses.

## 10. Engineering validation

After the V64.3.5 changes:

- targeted CCBR/FPCCA tests pass;
- full repository regression: `285 passed, 0 failed, 30 warnings`;
- new train/eval configuration contracts pass for CCBR-noLEA and CCBR+LEA;
- new launchers pass `bash -n`;
- old FPCCA screen metadata typo is corrected in the delivered tree;
- noLBA checker substring bug is corrected;
- full wrappers block automatic foundation rebuild and block unpromoted variants.

No GPU nuPlan training was executed in this delivery environment, so V64.3.5 performance is a hypothesis to be tested, not a claimed result.
