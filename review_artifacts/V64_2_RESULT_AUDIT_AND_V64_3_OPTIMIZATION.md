# V64.2 Result Audit and V64.3 Optimization

## Executive conclusion

The uploaded V64.2 run is not primarily failing Minimum because B=16 is too small. The historical checker reported Protocol PASS, Minimum FAIL, Competitive FAIL, but a code audit found a missing protocol invariant: evidence calibration used `beta=0`, while deployed AOCC used `beta=1` plus `prior_radius=0.02`. The calibration application copied epsilon only. This creates an uncalibrated deployment-only pessimism of roughly `M * radius = 24 * 0.02 = 0.48`, matching the observed AOCC initial deficit and explaining `evidence_certified_fraction=0.048` / `fallback_rate=0.952`.

The corrected V64.3 gate checker flags the old run as Protocol-invalid with exactly one new failure:

```text
evidence calibration/deployment beta mismatch: calibration=0.0, candidate=1.0
```

This must be repaired before interpreting the Minimum certificate failure as an algorithm-capacity failure.

## Historical V64.2 gate status

- Protocol (old checker): PASS.
- Minimum: FAIL.
  - evidence certified fraction `0.048 < 0.40`;
  - fallback rate `0.952 > 0.60`.
- Competitive: FAIL.
  - teacher action-match gain over foundation `+0.007 < +0.015`;
  - residual teacher-match gain over same-checkpoint local `0 < +0.005`;
  - pair-full residual gain `0 < +0.005`;
  - beneficial/harmful residual deployed flips `0/0`;
  - evidence certified fraction `0.048 < 0.55`;
  - proposal decisive recall `0.755817 < 0.80`;
  - fallback `0.952 > 0.40`;
  - teacher literal winner-flip critical Top-M recall `0.348168 < 0.80`;
  - teacher literal winner-flip critical selected recall `0.332125 < 0.50`;
  - paired teacher-regret comparison does not beat foundation.

## Algorithm vs engineering classification

### Engineering/protocol: primary cause of Minimum failure

Calibration JSON:

- evidence beta: `0.0`;
- evidence epsilon: `3.6137e-05`;
- evidence raw-error MAE: `0.04883`.

Deployed calibrated candidate config:

- `adverse_certificate_beta=1.0`;
- `adverse_certificate_prior_radius=0.02`;
- calibrated epsilon `3.6137e-05`.

The AOCC selector forms an omitted-evidence adverse radius as `beta * sigma + epsilon`; when no learned uncertainty is used, sigma is the prior radius. V64.2 therefore added about `0.02` adverse deficit per acquisition atom after calibrating a score with no such term. With M=24 this is about 0.48, matching the measured initial deficit. B=16 recovers about 0.32 and cannot erase the artificially added deficit, so fallback becomes almost deterministic.

V64.3 copies calibration beta/prior-radius/epsilon as one atomic contract and makes mismatch a Protocol failure.

### Algorithm: critical acquisition did not improve

V64.2 candidate vs matched foundation:

- proposal decisive recall: `0.7558` vs about `0.8036` (worse);
- selected decisive recall: `0.5817` vs about `0.6141` (worse);
- teacher critical Top-M recall: `0.3482` vs about `0.3548` (worse/slightly flat);
- teacher critical selected recall: `0.3321` vs about `0.3314` (essentially unchanged).

Top-M to B16 teacher-critical loss is only about 1.6 percentage points, so the dominant bottleneck is HAB acquisition, not the exact B=16 selector. HCBE changed the loss but fine-tuning the entire legacy proposal/family stack caused broad recall drift without increasing literal critical recall.

### Algorithm: query extension is harmful in nominal

- candidate full-interface teacher action match: `0.182`;
- foundation full-interface teacher action match: `0.359`.

The support-aware query extension is the major semantic difference in the dense learned interface and has no evidence of compensating benefit. V64.3 keeps the module for checkpoint compatibility but sets nominal scale to 0 and freezes it. This restores the checkpoint-supported 12-D anchor in nominal; the 6-D extension remains an ablation, not a core contribution.

### Algorithm/training protocol: residual is not actually ready

V64.2:

- residual proposal rate: `0.516`;
- deployed residual flip rate: `0`;
- residual calibration epsilon: `0.9891`;
- residual raw-error MAE: `1.6905`;
- residual sigma mean: `0.0567`.

The residual is not merely blocked by an arbitrary threshold; its calibration error is very large relative to useful decision-boundary corrections. In addition, the primary checkpoint is epoch 1 while residual curriculum scale is only 0.05 for epochs 0–2. Thus the paper checkpoint was promoted before the residual path received full-strength training.

V64.3 shortens the curriculum and forbids best-checkpoint promotion before zero-based epoch 3.

## What V64.2 learned

1. Planner-interface contracts and fixed B=16 accounting are stable.
2. HAB retains its own learned dense decision well: dense-to-HAB Top-M action preservation is about `0.981`.
3. Effective selected decisive recall remains about `0.747`, and selected interaction decisive recall about `0.597`.
4. Candidate teacher action match `0.238` slightly exceeds foundation `0.231`.
5. Pair-full teacher action match `0.249` exceeds foundation `0.235`, indicating that sparse evidence/pair modeling contains some useful correction before final gating/compression.

## What V64.2 did not learn

1. It did not learn teacher literal winner-flip critical acquisition.
2. It did not preserve the broad proposal recall of the foundation.
3. It did not learn a trustworthy residual whose conformal error is small enough to flip deployed winners.
4. It did not preserve the dense foundation interface once the 6-D query extension was trained.
5. It did not produce enough total action-level gain to beat the competitive threshold or paired-regret control.

## V64.3 algorithm: CC-AOCC + AP-WCCA

### Calibration-Consistent AOCC

Evidence beta, prior radius and epsilon are calibrated and deployed as one auditable contract. No threshold is relaxed. With the current V64.2 calibration beta=0, the correct deployed evidence beta is also 0. The prior radius can remain recorded, but has no mathematical contribution when beta=0.

### Anchor-Preserving Winner-Conditioned Critical Acquisition

The strong V62 legacy HAB proposal is frozen. A small zero-initialized residual adapter is added:

```text
r_crit = MLP([
  evidence embedding,
  proposal-feature embedding,
  scene embedding,
  candidate-set summary,
  family embedding,
  frozen base-winner action embedding
])

proposal_score = frozen_legacy_HAB_score + r_crit
```

Why this targets the observed failure:

- literal criticality is winner-relative, but the old proposal atom head had no explicit winner action embedding;
- zero initialization preserves legacy proposal ranking at step 0;
- freezing legacy proposal/family modules prevents the broad-recall regression observed in V64.2;
- teacher is used only to construct training critical labels; deployment uses the base winner available to the planner;
- hard HAB Top-M, M=24, exact B=16 selector, evidence atoms and literal winner-flip definition remain unchanged.

Objective rebalance:

- exact critical proposal weight: `8 -> 12`;
- BCC coverage: `2 -> 4`;
- HCBE exchange: `1 -> 2`;
- global/dense winner proposal pressure: `20 -> 6`;
- hardest-negative critical ranking: `0.25 -> 0.10`.

This is intentionally a residual correction around the proven proposal anchor, rather than another wholesale proposal retraining attempt.

## Should diagnostic closed-loop run when gates fail?

Yes, with one strict condition: **the corrected Protocol gate must pass**. If Protocol is valid but Minimum/Competitive fails, run a paired CL20 on validation/tune scenarios as a diagnostic. Do not treat it as an official result and do not use it to waive gates.

Why it is useful here:

- certificate coverage can be conservative even when selected actions are behaviorally acceptable;
- open-loop teacher match cannot expose reactive interaction, candidate dynamics, repeated replanning or cache/state accumulation failures;
- candidate/local/foundation paired CL20 can show whether the small open-loop +0.7 pp gain produces any real trajectory improvement or whether action differences are behaviorally irrelevant/harmful.

For the existing V64.2 run, first perform the calibration-consistent open-loop replay. Only after corrected Protocol PASS should diagnostic CL20 be run.

## Fixed-budget SOTA roadmap

Near-term priority order after V64.3:

1. Repair certificate contract and verify Minimum on the existing checkpoint without retraining.
2. Fresh V64.3 training: recover foundation proposal recall while increasing literal teacher-critical Top-M recall.
3. Require a fully trained residual checkpoint; inspect calibrated residual MAE/epsilon before any attempt to tune flip margins.
4. Run paired diagnostic CL20 after corrected Protocol PASS even if Competitive remains below threshold.
5. If critical Top-M rises but selected critical does not, then modify selector tie/exchange policy without increasing B.
6. If selected critical rises but action/CL does not, the next bottleneck is atom-action value quality or candidate bank/dynamics, not acquisition.
7. If open-loop gains are real but CL remains flat, focus on candidate dynamics, reactive interactions, replan state/cache and candidate availability. Do not keep increasing proposal-loss complexity.

## Runtime profile and speed plan

Measured V64.2 stage wall time:

- anchor audit: 15.1 min (4.3%);
- training: 299.2 min (84.8%);
- calibration: 24.4 min (6.9%);
- three-way open-loop: 14.3 min (4.0%).

Training epoch profiling shows data wait (roughly 156–387 ms/step) and loss construction (118–207 ms/step) dominate model forward (21–37 ms/step). Therefore V64.3 defaults to:

- 2 GPUs;
- batch 16 per GPU, global batch 32;
- 8 DataLoader workers/GPU;
- prefetch 3; persistent workers already enabled;
- base LR 1.7e-5;
- frozen legacy proposal/query modules; train only critical adapter and residual heads;
- verified-cache speed path uses audited 12-D cached query features and zero-pads the disabled extension, avoiding useless 6-D runtime recomputation;
- open-loop remains 2 workers/GPU because it is already only 4% of total wall time.

The exact selector is not approximated and B is unchanged.

## Validation performed

- full pytest: 258 passed, 0 failed;
- V64.3 config contract: PASS;
- Python compile: PASS;
- modified shell syntax: PASS;
- corrected old-gate replay detects exactly the evidence beta mismatch.

No fresh GPU experiment is claimed from this environment.
