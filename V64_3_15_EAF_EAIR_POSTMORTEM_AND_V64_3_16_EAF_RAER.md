# V64.3.15 EAF-EAIR Postmortem and V64.3.16 EAF-RAER Design

## Executive conclusion

The uploaded V64.3.15 package is **not a completed EAIR experiment**. It contains only raw EAF train and validation-discovery replay plus regression/contract/re-audit artifacts. The pipeline stops before EAIR fitting because the evaluator/trainer failed to propagate the `decisive_frontier_eair_*` diagnostics that `tournament.py` already produced. A second latent bug would also have made fresh-token replay incomplete because the cache was truncated before `--scenario-token-file` filtering.

Therefore this round cannot be used to claim that EAIR passed or failed. What it does establish is that raw EAF still has the same characteristic bottleneck: it substantially reduces teacher regret relative to the DARM anchor, but over-intervenes and reduces exact teacher-action preservation.

## 1. Uploaded result attribution

### Raw EAF — train, n=3000

| Metric | Result |
|---|---:|
| Teacher action match | 15.23% |
| Teacher regret | 14946.01 |
| DARM/selected-local anchor match | 17.10% |
| DARM/selected-local anchor regret | 25276.41 |
| Pair-full match | 18.20% |
| Pair-full regret | 18906.62 |
| Deployed flip | 50.40% |
| Beneficial intervention | 7.37% |
| Harmful intervention | 9.23% |
| Complete EAF frontier coverage | 100% |
| Evidence certificate fraction | 87.53% |
| Mean retained decision atoms | 15.954 |
| Mean proposal atoms | 23.777 |

### Raw EAF — validation discovery, n=1200

| Metric | Result |
|---|---:|
| Teacher action match | 14.17% |
| Teacher regret | 12572.56 |
| DARM/selected-local anchor match | 18.00% |
| DARM/selected-local anchor regret | 19296.85 |
| Pair-full match | 20.17% |
| Pair-full regret | 15848.99 |
| Local pair-full match | 18.25% |
| Raw EAF deployed flip | 60.67% |
| Beneficial intervention | 9.08% |
| Harmful intervention | 12.92% |
| Complete EAF frontier coverage | 100% |
| Evidence certificate fraction | 93.67% |
| Retained decision atoms | exactly 16 on all 1200 scenes |
| Mean proposal atoms | 23.998 |

### Interpretation

The key contradiction is real and stable:

- raw EAF improves teacher regret by roughly 35% versus the DARM anchor on validation discovery;
- at the same time exact teacher match drops from 18.00% to 14.17%;
- harmful interventions (12.92%) exceed beneficial interventions (9.08%);
- pair-full reaches 20.17% exact match, so the value pathway still has a meaningful ceiling;
- complete-star coverage is already 100%, so missing frontier edges are not the immediate explanation.

This is evidence for a **value-to-extremal-decision interface bottleneck**, not a reason to reopen acquisition.

On validation scenes where raw EAF proposes a non-anchor action (1136/1200 = 94.67%):

- 64.35% of proposed challengers are actually better than the anchor by teacher cost;
- raw EAF proposed-margin AUC for this event is only 0.620;
- proposed attribution-scale AUC is 0.655;
- teacher-better proposals have higher mean attribution scale (0.05365) than teacher-worse proposals (0.03854).

Attribution therefore remains useful as a support/reliability cue, but raw margin argmax is not a reliable selector.

## 2. Why V64.3.15 has no formal EAIR conclusion

The upload has no:

- fitted EAIR YAML;
- EAIR fit report;
- frozen fresh-val token list;
- paired fresh raw replay;
- paired fresh EAIR replay;
- V64.3.15 final screen JSON.

The direct engineering cause is diagnostic propagation. `tournament.py` emitted the EAIR feature fields, but `evaluate_open_loop.py` and `train.py` did not whitelist the `decisive_frontier_eair_` prefix. On the 2839 train proposal edges, the launcher's required explicit EAIR feature coverage is therefore 0%, and the launcher stops before fitting.

A **design-only** replay using the fitter's backward-compatible fallback aliases gives:

- train scene-group internal-holdout AUC: 0.779;
- all already-inspected 1136 validation proposal edges: AUC 0.649;
- `p>=0.5` keeps about 31.95% of validation proposals and increases their teacher-better fraction to 76.86%.

These are not promotion results. They are used only to decide what mechanism to test next, so the full 1200-scene validation-discovery set is now design-contaminated.

## 3. Engineering/protocol faults fixed

### Fault A — EAIR diagnostic loss

Fixed in:

- `bdse/experiments/evaluate_open_loop.py`
- `bdse/experiments/train.py`

Both now propagate `decisive_frontier_eair_*` and `decisive_frontier_raer_*` scalar diagnostics.

### Fault B — token replay silently drops frozen scenes

Old behavior:

1. construct dataset with `max_scenarios=500`;
2. keep only the first 500 cache paths;
3. apply a 500-token scenario filter.

Any frozen token outside the cache prefix is never evaluated.

New behavior:

1. if a token filter is present, do not prefix-cap the dataset;
2. filter by requested token first;
3. apply `--max-scenarios` to matched tokens;
4. optional `--require-all-scenario-tokens` makes missing tokens fatal.

### Fault C — cache-order-biased fresh split

V64.3.15 selected the first eligible fresh scenes in discovery order. V64.3.16 instead hash-ranks unseen eligible scenario tokens with a fixed preregistered SHA256 seed. No label or metric participates in scene selection.

### Fault D — contamination boundary too narrow

The previous exclusion covered 500 V64.3.14 design scenes. This postmortem explicitly examines all 1200 V64.3.15 discovery scenes, so all 1200 are now excluded from V64.3.16 promotion. The old 500 are verified to be a subset of the new 1200-token exclusion.

### Interface-accounting correction — exact B=16 is not universal on train

33/3000 uploaded train scenes have only 10--14 eligible proposal atoms, so no selector can retain 16 distinct decision atoms there. Validation discovery has 16/16 on all 1200 scenes. The faithful claim is a **B<=16 interface budget, with exact B=16 whenever at least 16 eligible proposal atoms exist**, plus an exact-B scene-rate diagnostic. No acquisition change or filler atom is introduced.

## 4. What should be kept, what should stop

### Keep

1. fixed planner-interface budget and auditable evidence atom representation;
2. terminally frozen acquisition/proposal stack after the V64.3.8--V64.3.12 negative branches;
3. DARM anchor and DBR baseline;
4. V64.3.13 complete DARM-anchor EAF frontier value;
5. exact additive selected-evidence attribution;
6. teacher-improvement supervision `J_T(challenger) < J_T(anchor)` rather than exact teacher-winner classification;
7. unchanged evidence certificate / structural safety guard;
8. independent full-val reproduction before test/closed-loop.

### Stop / do not iterate

1. OCFI radius/alpha/constant-radius variants;
2. scalar EAIR threshold sweeps;
3. BTP/RET/CET or other acquisition reopening;
4. increasing B or M;
5. relaxing the evidence certificate;
6. optimizing only average complete-frontier pair-sign while ignoring extremal selection;
7. broad representation unfreezing before testing the all-frontier reliability structure;
8. reusing any of the 1200 observed validation-discovery scenes for promotion.

## 5. Main bottleneck

The current chain is better described as:

`fixed selected evidence`
`-> informative complete-frontier value`
`-> noisy/extremal top challenger selection`
`-> one-sided intervention`
`-> preservation/regret tradeoff`.

V64.3.15 scalar EAIR, even if repaired, is structurally limited because it acts **after** the raw EAF top challenger has been selected. If that challenger is rejected, it can only return the anchor. It cannot recover a second challenger that is supported by the same B evidence and genuinely improves teacher cost.

This is the remaining mismatch between learning and deployment: complete-frontier value is trained mostly edgewise, while deployment is an extremal selection problem.

## 6. V64.3.16 main algorithm — EAF-RAER

**Evidence-Attributed Reliability-Aware Extremal Re-ranking**.

Mainline:

`fixed planner-interface evidence budget`
`-> auditable evidence atoms`
`-> terminally frozen budgeted acquisition`
`-> B<=16 selected evidence (exact B=16 when available)`
`-> frozen EAF complete DARM-anchor frontier value`
`-> exact selected-evidence attribution`
`-> all-challenger evidence-attributed one-sided reliability`
`-> reliability-aware extremal re-ranking`
`-> unchanged evidence certificate`
`-> final decision preservation`.

For every valid challenger b of the DARM anchor a, the train-only shared readout estimates

`p_b = P[J_T(b) < J_T(a) | runtime frozen-EAF evidence statistics]`.

The fixed selection rule is

`u_b = p_b * max(M_EAF(b,a), 0)`

with `p_b >= 0.5`, positive raw EAF margin, validity and the frozen safety mask. The challenger with maximum `u_b` is passed to the unchanged one-sided/evidence certificate. If none qualifies, the anchor is retained.

No additional evidence is queried. Acquisition, DARM, DBR, the EAF checkpoint/value, pair-full/local-pair-full semantics and certificate remain frozen.

### Why this is a stronger paper story than scalar EAIR

The novelty is not “a logistic confidence gate.” The method makes exact selected-evidence attribution part of the **decision operator over the complete frontier**. It explicitly addresses selection-induced extremal unreliability: evidence is first budgeted, then attributed to value, then attributed to **which frontier intervention can be trusted enough to compete at the extremum**.

That gives a cleaner progression:

`decision-sufficient evidence selection -> evidence-attributed value -> evidence-attributed extremal decision reliability`.

## 7. Next screen — three paired arms

Same fresh 500 scenes, same frozen checkpoint:

1. raw EAF;
2. repaired scalar EAIR post-top1 control;
3. RAER all-frontier re-ranking.

### Primary endpoints

- teacher action match;
- teacher regret;
- harmful / beneficial intervention;
- deployed flip rate;
- DARM anchor preservation;
- pair-full/local-pair-full invariance;
- complete frontier coverage;
- B/M and evidence-certificate invariance.

### Mechanism endpoints

- all-frontier fresh-val teacher-better AUC;
- raw top-1 teacher-better rate;
- RAER-selected teacher-better rate;
- mean teacher margin of raw top vs RAER selected;
- proposal-changed rate;
- anchor-fallback rate;
- alternative-recovery rate for non-anchor runner-ups.

### Promotion criteria

RAER must simultaneously establish reliability generalization, a real change to extremal selection, preservation improvement and an endpoint improvement. It is not promoted if it merely abstains more often or improves AUC without reducing final regret.

## 8. Pre-registered next branches

- Low all-frontier AUC: move to a small structured query-conditioned/per-atom reliability representation, still without touching acquisition.
- Good AUC but no extremal-selection improvement: change the reliability feature/objective structure, not the probability threshold.
- Good re-ranking and preservation but regret remains poor: add a train-only teacher-improvement magnitude / listwise extremal-ordering objective over the same frozen frontier.
- Full screen pass: independent full-val reproduction first; only after reproduction passes may test and closed-loop be scheduled.

## 9. Engineering validation

- V64.3.6--V64.3.16 targeted regression: 68/68 PASS.
- Full repository: 356/356 PASS.
- Warnings: 36, all the historical PyTorch Transformer nested-tensor / `norm_first` warning family; no new warning class.
- Raw V64.3.16 config contract: PASS.
- Shell syntax: PASS.
- V64.3.15 old 500 exclusion subset of V64.3.16 1200 exclusion: 500/500 verified.
- Uploaded train/val scenario-token overlap: 0.
- RAER unit safety fix: a safety-flagged challenger cannot be resurrected when the DARM anchor is the only unflagged action.
