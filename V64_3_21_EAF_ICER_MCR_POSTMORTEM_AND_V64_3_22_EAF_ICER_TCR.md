# V64.3.21 EAF-ICER-MCR postmortem and V64.3.22 EAF-ICER-TCR design

## Executive conclusion

V64.3.21 is **not** a single-mechanism failure. The two pre-registered independent fresh blocks fail for different causal reasons:

- **Split A:** selection-conditioned incumbent retention succeeds, but direct incumbent→alternative replacements have a large positive regret tail. Profile-mean direct replacements contribute about **+143.7k** regret versus raw; consensus contributes about **+143.8k**.
- **Split B:** profile-mean direct replacement is nearly regret-neutral (**+4.65k**), but incumbent→anchor retention veto contributes **+99.34k** regret. The consensus operator raises binary direct precision from 56.8% to 60.0% but makes direct replacement regret much worse (**+108.07k**), proving that binary precision is not the correct extremal objective.

The complete frontier, support/dominance signal, deployment-complete all-flagged delegation, and exact selected-evidence attribution remain useful. The remaining failure is **regret-tail reliability under extremal action changes**.

## V64.3.21 double-fresh attribution

### Split A

- instrumentation: PASS
- deployment-complete domain alignment: PASS
- candidate support: PASS
- support/dominance signal: PASS
- selected-incumbent magnitude retention: PASS
- recovery: FAIL
- preservation: FAIL
- endpoint: FAIL

Profile retention on 307 admissible incumbents:

- retention AUC: 0.8066
- sign accuracy: 0.7166
- predicted fallback teacher-margin sum: **-0.9908** (beneficial direction)

Path decomposition versus raw:

- incumbent→anchor: 30 scenes, **-19,846.5** regret
- direct incumbent→alternative, profile mean: 91 scenes, **+143,723.9** regret
- direct incumbent→alternative, consensus: 83 scenes, **+143,812.5** regret

The largest profile/consensus loss is a repeated raw action transition `1→4`: one scene contributes +75.7k regret and another +24.6k.

### Split B

- instrumentation: PASS
- deployment-complete domain alignment: PASS
- candidate support: PASS
- support/dominance signal: PASS
- selected-incumbent magnitude retention: FAIL
- recovery: FAIL
- preservation: FAIL
- endpoint: FAIL

Profile retention on 296 admissible incumbents:

- retention AUC: 0.6921
- sign accuracy: 0.6723
- predicted fallback teacher-margin sum: **+4.9670** (harmful direction)

Path decomposition versus raw:

- incumbent→anchor: 23 scenes, **+99,340.2** regret
- direct incumbent→alternative, profile mean: 88 scenes, **+4,647.9** regret
- direct incumbent→alternative, consensus: 75 scenes, **+108,074.2** regret

Again, two large `1→4` replacements contribute approximately +59.0k regret.

## What the V21 ablations prove

### 1. Selection-conditioned incumbent retention magnitude is not stable enough to be a hard veto

It is helpful in A and harmful in B. More importantly, the deterministic TRAIN internal holdout already contained a warning that the previous fitter did not gate:

- profile predicted-fallback teacher-margin sum: **+6.3153**
- scalar predicted-fallback teacher-margin sum: **+4.1743**

So AUC/sign accuracy were insufficient diagnostics. A selected-incumbent→anchor mechanism must be promoted only if its **TRAIN-only path teacher-improvement sum is non-positive** on an independent internal holdout.

### 2. Corroborated dominance should be retained as an ablation, not the main operator

On B, both-positive consensus raises direct replacement precision from 56.8% to 60.0%, but worsens direct path regret from +4.65k to +108.07k. The deleted replacements include high-benefit true positives. Therefore a binary two-view consensus is not a regret-aware extremal operator.

The signed attribution view still contains information, but should be used as a **ranking view**, not a hard binary gate. V22 therefore fixes replacement eligibility with the frozen scalar-dominance semantic zero boundary, then lets the exact signed selected-evidence profile modify only the extremal ranking score through an equal scalar/profile mean. This keeps the scalar→profile comparison causally clean and avoids the V21 consensus failure mode.

### 3. The remaining replacement failures are transition-structured, not merely random calibration errors

A repeated raw action transition `1→4` appears as a high-regret false-positive tail in TRAIN and both fresh blocks. In the frozen TRAIN complete frontier:

- final-guard-admissible `1→4` examples: **78**
- teacher-positive fraction: **28.2%**
- total candidate-minus-incumbent teacher improvement: **-66.82**

The error therefore already exists in TRAIN. The current evidence-only reliability representation lacks explicit semantics for *how the planner action changes* from incumbent to challenger. V64.3.22 must encode planner transition semantics without memorizing raw candidate-slot IDs.

## Paper mainline and novelty

The headline novelty is retained:

> **evidence-attributed incumbent-contrastive reliability for deployment-admissible extremal recovery under a fixed planner-interface evidence budget**.

V64.3.22 strengthens the mechanism with a subordinate component:

> **planner-transition-conditioned regret reliability**.

The mainline remains:

**fixed planner-interface evidence cap B<=16 → auditable evidence atoms → frozen M=24 acquisition → selected B<=16 evidence → frozen EAF complete DARM-anchor frontier → exact selected-evidence attribution → complete final-guard-admissible frontier → frozen anchor-support + incumbent-dominance reliability → transition-conditioned regret-risk reliability → conservative extremal replacement/retention → unchanged evidence/one-sided certificate → unchanged structural-risk guard → final decision preservation**.

No evidence query, B/M, selector, acquisition, EAF value checkpoint, certificate, safety rule, or structural guard is changed.

## V64.3.22 EAF-ICER-TCR

Full name: **Evidence-Attributed Incumbent-Contrastive Extremal Recovery with Transition-Conditioned Regret Reliability**.

### Runtime planner-transition representation

For every candidate relative to a frozen reference action, TCR computes an auditable runtime-only transition vector from the already-generated candidate bank:

- candidate/reference maneuver-family one-hot descriptors;
- same-maneuver / safe-like / progressive family relations;
- terminal progress, lateral displacement and speed differences;
- path-length and maximum lateral-excursion differences;
- endpoint, mean-path and maximum-path separation;
- terminal yaw difference as sine/cosine.

The representation contains no teacher/future signal and no raw action-slot identity. It adds no planner-interface evidence query.

Two references are used:

- incumbent-relative transition for candidate replacement risk;
- anchor-relative transition for selected-incumbent retention risk.

### Magnitude-weighted expected-improvement risk objective

For selected incumbent retention:

`delta_ret = teacher_margin(incumbent vs anchor)`.

For alternative replacement:

`delta_rep = teacher_margin(candidate vs anchor) - teacher_margin(incumbent vs anchor)`.

Each head uses fixed magnitude-weighted logistic loss:

`weight = abs(delta)`, `label = 1[delta > 0]`, fixed L2=1e-3.

The sample weights are normalized only by a positive TRAIN-only scalar. Therefore logit zero remains the fixed expected-improvement boundary; there is no validation threshold sweep.

Replacement-risk training is **selection-conditioned**: only TRAIN alternatives that are final-guard-admissible and already satisfy frozen support-positive + **scalar-dominance-positive** ICER eligibility are used. The scalar and signed-profile arms therefore share the exact same TRAIN population and regret-risk head. Signed selected-evidence attribution is never a hidden training-sample gate; it can affect only final extremal ranking. This targets the actual replacement tail while keeping the scalar→profile ablation causally identifiable.

### Fail-close TRAIN contract

V21 allowed a retention head to proceed despite a positive TRAIN-holdout fallback teacher-margin sum. V22 prevents this.

Before fresh validation, the transition-conditioned main head must satisfy on a deterministic TRAIN scene holdout:

- predicted incumbent→anchor fallback teacher-margin sum <= 0;
- at least 8 direct replacements are selected by the frozen scalar-eligibility operator plus risk veto;
- selected direct-replacement candidate-minus-incumbent teacher-improvement sum >= 0;
- transition feature coverage >=95%.

If this fails, the launcher stops **before any fresh validation replay**.

### Runtime operator

All-flagged structural-domain delegation is unchanged.

If the raw incumbent is final-guard-admissible:

1. retention regret-risk logit <0 can demote it to the anchor baseline; otherwise incumbent is preserved;
2. an alternative must pass frozen anchor-support >0;
3. it must pass the frozen **scalar incumbent-dominance** semantic zero boundary;
4. it must additionally pass replacement regret-risk >0;
5. scalar TCR ranks survivors by scalar dominance; signed-profile TCR uses the same eligibility and risk head but ranks survivors by the equal mean of scalar and signed-profile dominance. Regret risk is a veto/tie view, not a second arbitrary extremal score.

If the raw incumbent is not final-guard-admissible, the existing anchor-relative support recovery is unchanged.

## V64.3.22 causal experiment

Use 1000 untouched validation tokens selected by token identity + fixed SHA256 after permanently excluding **4700** inspected validation tokens (3700 prior + the V21 A/B 1000).

Split into independent A500 and B500; no pooled rescue.

Five arms per block:

1. raw EAF;
2. frozen V21 profile-retention + dual-mean control;
3. evidence-only magnitude-weighted regret-risk + scalar eligibility + signed-profile ranking;
4. transition-conditioned regret-risk + frozen scalar eligibility/ranking;
5. transition-conditioned regret-risk + the **same scalar eligibility/risk head** + signed-profile equal-mean ranking (main).

Causal questions:

- V21 control → evidence-risk: does regret-sensitive training itself fix the tail?
- evidence-risk → transition-risk dual: do planner transition semantics add value under identical scalar eligibility?
- transition scalar → transition dual: does exact signed selected-evidence attribution add value **only through extremal ordering**, with identical TRAIN population, risk head and eligibility?
- raw → transition dual: do mechanism, preservation and endpoint close simultaneously?

Each 500-scene block independently requires:

- frozen-interface and transition instrumentation validity;
- deployment-complete structural identity;
- support AUC >=0.65, dominance AUC >=0.70, regret-risk AUC >=0.60;
- incumbent→anchor regret delta sum <=0;
- direct incumbent→alternative regret delta sum <=0;
- direct replacement precision >=60%, capture >=8%;
- transition-conditioned direct path no worse than evidence-only risk and endpoint no worse by >1%;
- harmful absolute reduction >=5pp, beneficial retention >=35%;
- match >= DARM anchor +0.5pp;
- regret <=1.02x raw.

If dual main fails but transition-scalar independently passes both blocks, signed-profile attribution remains an ablation and scalar TCR becomes the candidate for full-val; no view-weight tuning is allowed.

Passing both blocks authorizes only one frozen independent full-validation reproduction. Test/closed-loop remain forbidden until that reproduction passes.

## No-repeat constraints carried forward

Do not:

- reopen BTP/RET/CET/AF/HAP, selector or acquisition;
- increase B or M;
- relax evidence/one-sided/safety/structural guards;
- restore V17 utility-equivalence hard masking;
- sweep OCFI/EAIR/RAER/DACER/ICER thresholds or TCR zero boundaries;
- tune scalar/profile weights or resurrect both-positive consensus as the main operator;
- retry the V21 linear-MSE retention veto without a TRAIN path-regret fail-close gate;
- blacklist action slots such as `1→4`; the new representation must use planner-semantic trajectory/maneuver transitions;
- broad-unfreeze EAF before transition-conditioned risk is causally tested;
- pool A/B to rescue a failed replication;
- use any of the 4700 inspected validation tokens for promotion.

## Engineering validation

The implemented V64.3.22 code adds transition features, risk heads, TRAIN fail-close fit, contract checker, double-fresh checker and launcher. It also reduces transition frontier I/O by writing incumbent-relative geometry only on admissible edges and anchor-relative geometry only on the raw incumbent row.

Final verification is recorded in `V64_3_22_ENGINEERING_VALIDATION.txt`.
