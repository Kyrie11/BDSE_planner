# V64.3.23 EAF-ICER-RCR Postmortem and V64.3.24 EAF-ICER-ARC Design

## Executive conclusion

The uploaded V64.3.23 screen is a valid complete double-fresh experiment, not an interrupted or misconfigured run. TRAIN passed; both independent A/B 500-scene blocks and all five intended arms completed with paired token identity and zero TRAIN/fresh overlap. The official double-fresh promotion failed.

The failure is not broad loss of incumbent-contrastive reliability. Split A shows strong non-harmful direct replacement and endpoint improvement; Split B reverses the direct replacement path. Per-scene attribution shows that B is dominated by four catastrophic negative replacements around -0.93 to -0.99 normalized candidate-vs-incumbent teacher improvement.

V23's central limitation is therefore not “local evidence contains no signal”. Its local certificate controls uncertainty of the *neighborhood mean* (`mean - SE`), while the extremal decision is harmed by the downside of one selected outcome in a heavy-tailed/multimodal conditional population. The new question is whether exact within-budget attribution structure can resolve the hidden outcome mode well enough to certify individual replacement downside.

V64.3.24 implements that hypothesis as EAF-ICER-ARC: **Attribution-Resolved Regret Certification**.

## 1. Execution-integrity audit

The result package was split by the user into configs/logs and two provenance archives because the JSONL files are large. This split did not interrupt the official experiment.

Verified properties:

- V23 TRAIN RCR fit completed with `train_gate_pass=true`.
- Fresh selection produced 1000 unique untouched validation identities, then A500/B500.
- A/B each contain all five arms: raw, frozen V20, evidence scalar, evidence RCR, transition RCR.
- Every arm contains exactly 500 unique rows and exactly the same ordered token list within its block.
- A/B overlap = 0.
- TRAIN memory manifest has 3000 unique tokens; TRAIN vs A/B overlap = 0.
- No traceback, missing-arm, incomplete-token, or config-mix failure exists in official logs.
- Both split checkers and the double-fresh checker completed.

The local-memory NPZ files themselves were omitted from the user's split upload. This does not invalidate the run: the runtime loader SHA-checks the memory, so missing/mismatched memory would abort the fresh arm before producing complete rows/edges. Re-fitting from the uploaded TRAIN frontier reproduces the reported replacement population and cross-fit statistics, although the original omitted NPZ bytes cannot be independently SHA-verified from the split package alone.

Conclusion: **V23 is reliable for algorithm attribution.**

## 2. Official STOP reason

TRAIN did **not** stop. It passed:

- evidence-local selected improvement sum: `+9.8463`, 5/5 folds path-safe;
- transition-local selected improvement sum diagnostic: `+14.4132`, but only 3/5 folds path-safe.

The double-fresh screen stopped because neither independent split satisfied the full promotion contract:

- A next action: signed-profile ranking is not incremental; keep scalar RCR candidate and do not tune view weights.
- B next action: evidence-local selected replacement path is harmful; audit local-tail structure and do not tune K/zero boundary.

The scientifically load-bearing failure is B's selected replacement path. A also exposes that the historical preservation checker still encoded an abstention-style `harmful -5pp / flip<raw` requirement incompatible with the new incumbent-default asymmetric operator; V24 corrects that protocol *before new fresh identities are selected*.

## 3. Split A

Evidence scalar direct replacement:

- count = 48;
- path regret delta sum = `-119410.75`.

Evidence signed RCR:

- count = 45;
- path regret delta sum = `-86669.17`;
- match = 18.0%;
- regret = 13904.29 vs raw 14077.62;
- endpoint gate passes.

Transition RCR diagnostic:

- count = 47;
- path regret delta sum = `-174440.42`.

But signed-profile ranking is not incremental over scalar. Transition being excellent on A is not enough to promote it because it fails cross-split stability.

## 4. Split B

Evidence scalar:

- count = 31;
- path regret delta sum = `+57819.90`.

Evidence RCR:

- count = 29;
- path regret delta sum = `+57895.19`;
- direct replacement precision is below the target;
- regret = 14392.97 vs raw 14277.18;
- endpoint fails.

Transition RCR:

- count = 33;
- path regret delta sum = `+115708.44`.

Thus transition geometry is even less stable than evidence-local risk across fresh blocks.

## 5. Per-scene heavy-tail attribution

The B evidence-RCR selected path is dominated by four scenes:

| scenario | legacy→selected | normalized improvement |
|---|---:|---:|
| `0e612278ebd05d5e` | 5→19 | -0.989897 |
| `5ef81ac81e9d54ed` | 4→2 | -0.988400 |
| `e384cbf203735c60` | 0→21 | -0.929033 |
| `6ab7225c12445343` | 26→10 | -0.928512 |

Their combined normalized loss is about `-3.84`, exceeding the full B selected-path normalized sum in magnitude; many smaller positive replacements partially offset them.

Neighborhood inspection shows why V23 can confidently make the wrong decision. For example:

- `5ef81...`: K64 local mean 0.02965, SE 0.02941, lower bound +0.000245, runtime outcome -0.9884;
- `e384...`: K64 local mean 0.10493, SE 0.05091, lower bound +0.05402, runtime outcome -0.9290;
- `6ab7...`: K32 lower bound +0.02224, runtime outcome -0.9285.

The local neighborhood can have a positive estimated mean while containing or aliasing a rare catastrophic mode.

The TRAIN selection-conditioned replacement population is itself heavy-tailed: 1455 alternatives, positive sign rate around 62%, yet overall mean candidate-vs-incumbent improvement is negative. This is exactly the regime where binary reliability or confidence in a conditional mean can be insufficient for an extremal decision.

## 6. Mechanism conclusions

### What is retained

- fixed planner-interface cap `B<=16`;
- auditable selected evidence;
- frozen EAF complete frontier;
- exact selected-evidence attribution;
- complete final-guard-admissible frontier;
- frozen support and scalar incumbent-dominance;
- structural-domain delegation;
- admissible-incumbent default preservation;
- path-level direct replacement audits;
- double-fresh no-pooling protocol.

### What is removed from the main

**Signed-profile equal-mean ranking.** It is not incrementally useful on either A or B. Do not tune its weight.

**Transition-local risk.** It is excellent on A and harmful on B; its TRAIN fold stability was already weaker. Do not make trajectory geometry the headline mechanism and do not tune transition group weights.

## 7. Bottleneck refinement

Previous bottleneck:

> selection-conditioned local regret coherence under extremal replacement.

V23 refines it to:

> **outcome/downside regret certification under evidence-neighborhood aliasing.**

The next key question is:

> **After frozen support/dominance has isolated the true extremal replacement population, can the exact B<=16 selected-evidence attribution structure disambiguate local hidden outcome modes sufficiently to certify the downside of the individual candidate that will actually replace the incumbent?**

This is more precise than asking for higher edge AUC or a larger network.

## 8. V64.3.24 EAF-ICER-ARC

ARC = **Attribution-Resolved Regret Certification**.

Proposed novelty refinement:

> **evidence-attributed incumbent-contrastive regret certification for deployment-admissible extremal recovery under a fixed planner-interface evidence budget.**

The mainline remains:

`fixed B<=16 -> auditable selected evidence -> frozen EAF complete frontier -> exact selected-evidence attribution -> complete deployment-admissible frontier -> frozen support/scalar incumbent dominance -> attribution-resolved local downside regret certificate -> incumbent-default extremal recovery -> unchanged certificates/structural guard -> decision preservation`.

### Full attribution-resolved representation

The existing top-4 attribution summary is insufficiently resolved for the new tail question. V24 instruments all selected evidence contributions without increasing the evidence budget:

- full 16-entry candidate signed spectrum;
- full 16-entry candidate-minus-incumbent signed spectrum;
- absolute-contribution sorting for permutation invariance;
- L1 normalization;
- zero padding when fewer than 16 evidence atoms exist.

The pre-existing 18 evidence features retain attribution magnitude and frontier scale. The local metric balances three groups equally: aggregate evidence, candidate spectrum, delta spectrum.

### Downside certificate

V23 mean certificate:

`mean - SE`.

V24 downside certificate:

`local mean improvement - weighted RMS of negative local improvements`,

minimum over fixed K32/K64.

The downside multiplier is fixed at 1.0 and the decision boundary stays zero. There is no validation sweep.

### Orthogonal four-way ablation

TRAIN and fresh configs:

1. aggregate + mean-SE;
2. aggregate + downside;
3. attribution-resolved + mean-SE;
4. attribution-resolved + downside (**main**).

All four share the exact same TRAIN replacement population and frozen scalar ranking. This cleanly separates the effect of risk statistic from the effect of full evidence attribution.

## 9. New validation protocol

V23 A/B are now design data, so V24 permanently excludes 5700 validation tokens (`4700 + 1000`, zero overlap).

A new untouched 1000-token set is split into A/B 500 each. Six arms are run per block:

- raw EAF;
- frozen V20;
- aggregate mean-SE;
- aggregate downside;
- attribution mean-SE;
- attribution downside main.

Both blocks must independently pass. No pooled rescue.

The main must show both:

- downside certificate incremental over aggregate mean-SE;
- full attribution spectrum incremental over aggregate downside.

Otherwise the corresponding mechanism is not allowed into the paper main.

## 10. Preservation protocol correction

Because the V24 main never learned-vetoes an already final-guard-admissible incumbent to anchor, the old `harmful reduction >=5pp` and `flip<raw` requirement is structurally incompatible with the operator.

V24 pre-registers asymmetric preservation as:

- zero learned admissible-incumbent→anchor action changes;
- direct incumbent→alternative path regret delta sum <=0;
- harmful rate must not materially exceed raw;
- flip rate must not materially exceed raw;
- endpoint match/regret still must pass;
- frozen V20 remains the separate abstention/preservation control.

This is not a post-hoc change made after seeing V24; it is fixed before fresh V24 identities are selected.

## 11. Engineering validation

A real implementation bug was found during development: the full attribution diagnostic was initially referenced in the ICER serializer before being initialized on the common ICER path. The historical regression suite caught it (`NameError`) and it was fixed before delivery.

The user-uploaded V23 source archive also omitted historical root file `V64_SAQA_BCC_NEXT_COMMANDS.sh`; this caused an unrelated V64.2 repository test failure. V24 restores the exact 32,559-byte verified file from the previous V23 delivery archive.

Final validation:

- V64.3.6–V64.3.24 targeted: **119/119 PASS**;
- complete repository, three independent batches: **407/407 PASS**;
- warnings: **36**, all existing Transformer `nested_tensor/norm_first` warnings;
- Python compile: PASS;
- V24 launcher `bash -n`: PASS;
- synthetic TRAIN frontier -> four V24 memories/configs -> four contract checks: PASS.

Runtime current-scene inputs contain no teacher/future label. TRAIN teacher improvement exists only inside offline local memory. New fresh identities are hard checked against the TRAIN memory token manifest.

## 12. Do not repeat

Do not reopen acquisition/selector/B/M, OCFI/EAIR/RAER thresholds, BTP/RET/CET/AF/HAP, utility-equivalence hard masking, guard relaxation, learned admissible-incumbent→anchor veto, transition/action blacklist, signed-profile weight tuning, transition group-weight tuning, or broad EAF unfreezing before the ARC hypothesis is causally tested.

Do not tune K32/K64, downside multiplier=1, zero boundary, or attribution group weights on V24 fresh results.

If aggregate-downside succeeds but attribution-downside does not, keep aggregate-downside and conclude the proposed full attribution spectrum is not useful enough. If attribution-downside still shows a catastrophic tail, the next audit must ask whether the fixed evidence interface lacks necessary latent state before considering any representation expansion.
