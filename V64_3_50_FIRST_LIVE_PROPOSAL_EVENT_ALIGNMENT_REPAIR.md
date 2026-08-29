# V64.3.50 first-live-proposal event-state alignment repair

## Trigger

The first V50 paired collection attempt stopped on `03dac455f9ec5792` with:

`first RSMR proposal must occur at planner iteration 0, got 7/7`.

The important fact is that CONTROL and TREATMENT agreed (`7/7`).  The old code
had encoded an implementation assumption—"a selected scene must expose its
first runtime proposal at absolute nuPlan iteration zero"—as if it were a causal
invariant.  It is not.

The exception occurs before `paired_selected_outcomes.csv` is updated, so no
paired causal row was committed.  This repair was made without consuming the
pair's score/hard-metric outcome as algorithm evidence.

## Why deleting the check is unsafe

The original fit consumed V49 offline Q/P/E values.  If a closed-loop treatment
actually occurs at iteration 7, merely accepting the row would pair a later
intervention outcome with covariates from a different state.  That creates a
state/label mismatch and invalidates the selected-outcome interpretation.

## Repaired invariant

The intervention event is now the first synchronized **live pre-post-selection
RSMR proposal**.  It may occur at any non-negative planner iteration.  The pair
is accepted only if:

1. CONTROL/TREATMENT have identical probe time/action traces before intervention;
2. first proposal iteration and simulation time match exactly;
3. the live proposal action equals the byte-locked V49 `full_selected_action`;
4. historical post-selection can only have kept that proposal or vetoed it to
   incumbent—any third action is a no-rerank violation;
5. frozen Q/P/E coordinate definitions are evaluated at this exact live event
   and match across the two arms before treatment;
6. CONTROL preserves incumbent and TREATMENT executes the exact proposal once;
7. control executes zero interventions and treatment exactly one;
8. all previously frozen hard-metric and no-fallback checks remain active.

The SIOR fit consumes these event-aligned live Q/P/E values.  V49 offline Q/P/E
and persisted risk are retained only as provenance/audit information.  The
historical observational sign model is reconstructed from its frozen training
population and evaluated on the same live Q/P/E test state, so the comparator is
not advantaged or disadvantaged by a state mismatch.

## Scientific status

This is a pre-result engineering/protocol repair, not a promotion.  It does not
change RSMR, the candidate bank, Q/P/E coordinate definitions, SIOR model class,
`lambda=1`, retention budget, or the same-winner/incumbent no-fallback operator.
It replaces an invalid absolute-iteration assumption with the actual causal
requirement: treatment, proposal identity, and covariates must describe the same
pre-intervention state.

A future STOP because the live proposal does not equal the frozen V49 winner is
not eligible to be bypassed.  Such a failure would mean the offline selected
population does not transport to the closed-loop runtime selector and would
need a separate scientific diagnosis.
