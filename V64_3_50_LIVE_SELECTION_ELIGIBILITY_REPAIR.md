# V64.3.50 live-selection eligibility / cohort-semantics repair

## Trigger

The first two native closed-loop pairs produced synchronized live RSMR proposal
events.  The third frozen V49 scene (`04e294754b4d56a6`) completed both CONTROL
and TREATMENT simulations but produced `0/0` `first_proposal_now` markers.  The
probe diagnostics themselves were present on every planner tick, so this is not
a missing-log failure: neither arm observed a live direct RSMR proposal during
the native closed-loop scenario.

No paired score or hard-safety outcome from this third scene was used to choose a
threshold, feature, loss, or model.  The repair changes cohort/event semantics,
not the SIOR estimator.

## Root cause

Earlier V50 code conflated two populations:

1. the **offline V49 discovery cohort**: 502 TRAIN scenes in which the offline
   V49 instrumentation selected an RSMR winner; and
2. the **live intervention cohort**: native closed-loop states in which the
   frozen RSMR policy actually emits a direct proposal and therefore defines a
   one-shot selected-action treatment.

The first does not imply the second.  This is the same offline-to-live boundary
that earlier repairs exposed through absolute iteration and state-local action
slot assumptions.  A symmetric `0/0` live-proposal result is therefore a valid,
label-free selection-transport observation, not an outcome-class label and not
an engineering failure.

## Corrected protocol

The 502 V49 scenes remain frozen as the discovery cohort.  Every token is still
run in paired CONTROL/TREATMENT native nuPlan simulations.

### Live-eligible event

If both arms reach the same first live RSMR proposal event, V50 keeps all prior
hard invariants:

- identical pre-intervention iteration/time trace;
- identical pre-intervention deployed candidate trajectory fingerprints;
- identical live proposal slot/fingerprint/maneuver semantics across arms;
- identical live Q/P/E across arms;
- CONTROL preserves incumbent;
- TREATMENT executes that live winner exactly once and returns to incumbent at
  every later proposal;
- no rerank/second-best/fallback.

Only this stratum receives the paired `safe_benefit` label and enters SIOR
fit/calibration/test.

### Symmetric no-live-proposal event

If neither arm ever emits a proposal, the row is recorded as
`pair_status=no_live_proposal`, `live_intervention_eligible=0`.  It is **not**
converted into a negative label.  The collector requires the entire
CONTROL/TREATMENT action trace and quantized candidate-trajectory fingerprints
to be identical, zero proposal/intervention counters, and equivalent official
aggregate/hard metrics.  The scene remains in the 502-row cohort audit but is
excluded from SIOR outcome fitting.

If exactly one arm emits a proposal, or the no-treatment traces diverge, the run
still stops as an engineering violation.

## Population-support gate

No post-hoc eligibility-rate threshold is introduced.  Instead V50 uses only the
minimum support already required by the frozen V48/V49 estimator:

- every nested fit split: >=32 safe-benefit and >=32 nonbenefit live events;
- every calibration fold: >=16 safe-benefit live events;
- every held-out fold: both outcome classes present;
- final fit/calibration satisfy the same inherited minima.

If these conditions fail, V50 records the offline-to-live cohort transport and
scientifically stops before risk fitting/fresh evaluation.  It does not rescue
the run by labeling no-proposal scenes as bad outcomes or by lowering the
frozen estimator requirements.

## Comparator correction

The historical V49 OBS comparator is reconstructed on the full frozen offline
selected training population for each fold, not on the live-eligible subset.
It is then evaluated on the same live Q/P/E test states and paired closed-loop
labels as SIOR.  This prevents live eligibility from leaking into the baseline
training distribution.

## Interpretation

V50 now tests selected-outcome identification conditional on a **real live
selection event**.  The audit separately reports the mapping

`V49 offline selected scene -> native closed-loop live proposal / no proposal`.

A large no-live-proposal stratum is itself scientific evidence about
selection-regime transport and must be reported; it is not silently discarded.
The candidate paper claim must therefore distinguish offline discovery coverage
from on-policy selected-event outcome identification.
