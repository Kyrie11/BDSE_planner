# V64.3.30.2 uploaded-result engineering postmortem and V64.3.30.3 audit fix

## Executive decision

Do **not** promote the uploaded V64.3.30.2 A/B to a paper-level capacity conclusion in this iteration.

The six fresh planner arms did execute, paired query accounting is exact, the FBIC mechanism itself did not crash, and all-flagged structural preservation passes. The official screen nevertheless returns `engineering_valid=false` because the frozen V30.2 split checker encodes `safe_applied_rate >= 0.90` as an engineering contract. Both fresh blocks fall below that threshold (A 0.862869, B 0.849687).

That threshold is semantically wrong for FBIC: `safe_applied_rate` is the prevalence of safe scenes for which the already-queried reference bank is strictly larger than the frozen B16 baseline. Reason code 6 is an intentional no-op when `reference_count <= baseline_count`; it is not a failed capacity intervention. Requiring a fixed application prevalence would force a valid experiment either to fabricate unqueried atoms or to alter upstream acquisition, violating the capacity-only causal isolation.

However, the `0.90` test existed in the frozen checker and is serialized under `preregistered_thresholds` in the V30.2 reports. Correcting it after observing A/B and then treating the same 1000 scenes as an independent paper result would be a post-hoc protocol repair. Therefore V30.3 fixes the audit semantics **before** selecting a new untouched population and permanently adds the spent V30.2 1000 tokens to the design exclusion.

Per the project discipline, no new planning/recovery algorithm is introduced in V30.3 and no V30.2 scientific/mechanism conclusion is promoted here.

## 1. What did execute correctly in the uploaded V30.2 run

- TRAIN B16/B24 replay and the historical B16 V25 fit completed; the expected B24 DRC 4/5 fold-safety failure reproduced.
- The fresh selector generated 1000 label-free/hash-selected validation tokens with zero overlap with the then-frozen 8700-token design set and the frozen TRAIN set.
- Fresh A and B each contain raw / B16-V20 / B24-V20 arms with 500 paired scenes.
- No traceback, CUDA/OOM, runtime exception, or truncated evaluation appears in the uploaded logs.
- Query parity is exact between B16 and B24 on all three logged query counts in both A and B.
- The B24 retained-interface ceiling is 24, upstream configured budget remains 16, retained-budget pass is 100%, no-new-query is 100%, and removed-B16-atom mean is zero.
- Structural preservation passes: A has 26 all-flagged scenes and B has 21; FBIC applied rate in that domain is zero and B24 final action identity versus raw is 1.0.

The uploaded result package is missing `A_b24_v20_rows.jsonl`, although the original run must have contained it because the A split checker completed and serialized its report. This is a review-bundle packaging omission, not evidence of a runtime failure. V30.3 writes a provenance SHA256 manifest to make future multipart uploads auditable for completeness.

## 2. Exact V30.2 checker defect

The frozen V30.2 checker defines:

```python
fbic_contract = (
    enabled_rate == 1.0
    and safe_applied_rate >= 0.90
    ...
)
```

and maps any failure to `engineering_capacity_contract_failure`.

But FBIC's reason codes are explicitly:

- `0`: applied;
- `1`: structural-domain no-op;
- `6`: no capacity expansion over baseline, no-op;
- `2/3/4/5/7/8`: genuine fail-closed invalid/reference/budget/nesting cases.

The current fresh reports contain only reasons `0/1/6`:

- A: 409 applied, 26 structural no-op, 65 reason-6 no-expansion;
- B: 407 applied, 21 structural no-op, 72 reason-6 no-expansion.

For the uploaded B rows, the audit-fixed pointwise checker verifies:

- 407/407 expandable safe scenes applied correctly;
- 72/72 safe no-expansion scenes are exact reason-6 no-ops;
- 21/21 structural scenes are exact reason-1 no-ops;
- zero global budget/query/accounting violations.

Thus the population-level 0.90 application prevalence is not an engineering-integrity property.

## 3. V30.3 corrected pointwise contract

V30.3 freezes the following contract before selecting its new fresh scenes.

For every non-structural safe scene:

1. if `reference_count > baseline_count`, the probe must have reason 0, must be applied, and final retained set cardinality must equal the reference cardinality;
2. if `reference_count == baseline_count`, the probe must have reason 6, must not be applied, and final count must exactly equal baseline/reference;
3. any safe reason 2/3/4/5/7/8 is an engineering failure.

For every all-flagged structural scene:

- reason must be 1;
- probe must not apply;
- final retained count must equal baseline;
- the existing final-action identity-versus-raw structural check remains required.

For every scene:

- upstream budget = 16;
- retained-interface ceiling = 24;
- retained-interface budget pass = 1;
- no-new-query = 1;
- removed baseline atoms = 0;
- B16/B24 query counts are exact per scene.

The old `safe_applied_rate >= 0.90` is removed from engineering validity and remains descriptive as `safe_expandable_rate`. The already-frozen mean retained-atom increase >=4 criterion is retained as a separate capacity-exposure adequacy gate.

## 4. Independence repair

The V30.2 1000 fresh tokens are now spent. V30.3 creates:

`bdse/configs/v64_3_30_3_design_exclude_v64_3_30_2_screen_tokens.txt`

with exactly 9700 unique tokens = historical 8700 + V30.2 1000. SHA256:

`cc2f7228ed802f8f605f8d1c7a48f3fe889130daa89307a4b0118c373ee33253`

The new hash seed is:

`v64.3.30.3-eaf-icer-fbic-pure-auditfix-double-fresh-v1`

No V30.2 fresh scene is reused for paper-level promotion.

## 5. Compute-preserving rerun

V30.3 does not rerun the already-valid 3000-scene B16/B24 TRAIN experiment. The launcher re-audits the prior V30.2 TRAIN provenance and requires all of the following to reproduce:

- `engineering_contract_valid=true`;
- `historical_B16_V25_reproduced=true`;
- `B24_DRC_fail_is_selected_path_fold_safety_failure_not_runtime_error=true`.

Only the new untouched fresh A/B raw/B16-V20/B24-V20 arms are evaluated.

## 6. Scientific interpretation policy

Because this round identified and repaired a frozen screening-contract bug, V30.2 is retained as **spent diagnostic evidence only**. Its numerical A/B outcomes must not be used as the formal independent answer to the capacity-mediator question.

The next algorithm/mechanism decision is intentionally deferred until V30.3 passes the corrected engineering/preservation contract on the new untouched population. This prevents a post-hoc audit change from contaminating the mechanism chain.
