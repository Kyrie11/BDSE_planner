# BDSE v47 Result Diagnosis and v48 DBCE Design

## 1. Did closed loop run?

No.  The strict paired open-loop gate returned FAIL under `set -e`, so the
pipeline terminated before invoking nuPlan simulation.  The uploaded
`closed_loop` directory is empty.  This is a deliberate gate stop, not a
simulator launch failure.

There is also a command-flow defect in the old helper: its CL20 block was only a
commented example.  Thus even a future PASS would have required a separate
manual command.  The v48 pipeline executes CL20 automatically after PASS when
`RUN_CLOSED_LOOP_AFTER_GATE=1`.

## 2. Gate result

| Gate item | v47 candidate | control / threshold | Decision |
|---|---:|---:|---|
| Teacher action match | 0.249 | 0.238 + 0.020 | FAIL; gain +0.011 |
| Winner/rival sign | 0.692164 | 0.638368 | PASS |
| Near-tie sign | 0.468943 | 0.583448 | severe regression |
| Evidence sufficiency | 0.073575 | 0.070400 + 0.010 | FAIL; gain +0.003175 |
| Pair-full action match | 0.248 | >=0.300 | FAIL |
| Certified pair fraction | 0.135116 | >=0.500 | FAIL |
| Fully certified scenes | 0.028 | diagnostic | too low |
| Planner latency p95 | 1365.15 ms | <=500 ms | FAIL |
| Median teacher regret | 40.93 | control 30.62 | FAIL |
| p90 teacher regret | 38570.69 | control 30794.33 | FAIL |
| Exact selector fraction | 1.0 | >=0.99 | PASS |
| Calibration provenance | active | required | PASS |

## 3. Experiment integrity warning

The output contains two launcher logs beginning roughly two minutes apart.  The
training JSONL contains 24 rows but only 12 unique epochs: every epoch appears
twice.  Two independent jobs therefore wrote checkpoints and logs to one
`OUT_ROOT`.  This can create race-dependent checkpoints.  The metrics are useful
for failure analysis but should not be reported as a clean paper run.

V48 adds PID-bearing output and pipeline locks and makes duplicate epochs a gate
failure.

## 4. Did the model learn critical evidence?

Only partially, and not in the intended causal sense.

Positive evidence:

- selected decisive recall increased from 0.403 to 0.513;
- selected interaction-decisive recall increased from 0.316 to 0.579;
- dense full-interface match increased from 0.265 to 0.319;
- broad winner/rival sign accuracy increased from 0.638 to 0.692;
- total sparse queries fell from about 11252 to 2285 per scene;
- pair-full to budget preservation reached 0.895.

Contradictory evidence:

- decisive and interaction recall are slightly negatively correlated with
  teacher correctness;
- near-tie sign accuracy fell by 0.115;
- the critical-pair loss stayed nearly constant (`3.612 -> 3.610`);
- proposal loss decreased, yet proposal decisive recall was exactly unchanged;
- 96.4% of selected decision atoms came from interaction families;
- pair-full correctness was only 0.248 and teacher regret worsened.

The model learned “select many interaction atoms with positive teacher support,”
not “select the lowest-cost atoms whose removal would destroy the current
teacher/rival boundary.”

## 5. Component analysis

### Exact selector supervision

**Effective.** Exact coverage reached 1.0.  This fixes the old 6.25% alignment
problem, but exact supervision cannot compensate for a wrong target or wrong
pair interface.

### Exact-tournament AOCC target

**Partially effective.** Harmful pair-full-to-budget compression is only 1.8%,
so AOCC usually preserves its own target.  However, dense-to-pair conversion
breaks 15.4% of scenarios and pair-full teacher match is 0.248.  AOCC protects a
weak upstream decision.

### Integrable local margin plus pair residual

**Promising but defective in v47.** It improved broad winner/rival signs and
reduced queries.  In residual mode, however, v47 bypassed the existing
local/pair calibration.  Full-strength residual addition damaged near-tie
signs—the exact region that determines action flips.

### Counterfactual critical-evidence loss

**Not actually counterfactual in v47.** The target was positive oriented atom
support divided by cost.  Large support received credit even when removing the
atom left the decision safely above the boundary.  The pair loss remained flat.

### Proposal supervision

**Optimization worked; deployment path did not.** `proposal_top_m=64` exceeded
the average decision evidence pool of about 30 atoms, so all relevant atoms were
materialized regardless of ranking.  A decreasing proposal loss could not alter
runtime selection.

### Interaction reservation

**Harmful saturation.** Reserving 24 interaction Top-M slots with a B=16 budget
caused almost the entire budget to be interaction evidence.  Recall rose
mechanically while cross-family decision competition disappeared.

### Calibration and certificate

**Provenance is correct; geometry is too conservative.** Calibration was
log-disjoint and active.  Raw atom-pair error was about 0.02, while a 0.10 prior
radius was accumulated over omitted atoms.  The full order could certify most
targets, but B=16 certified only 13.5% of pairs, triggering fallback in 88.7% of
scenes.

### Latency

**Query reduction is effective; gate is not met.** Sparse query count fell by
about 79.7%, but prediction remains the dominant stage and p95 is 1365 ms.
Further selector micro-optimization is not the main latency lever.

## 6. v48 DBCE changes

### 6.1 Leave-one-out deployment-boundary target

For teacher action `w`, rival `r`, full teacher margin `m_wr`, atom contribution
`d_i`, and target boundary `gamma`, v48 defines:

```text
c_i(w,r) = relu(gamma - (m_wr - d_i)) - relu(gamma - m_wr)
```

The quantity is positive only if removing atom `i` increases the teacher/rival
boundary deficit.  It is cost normalized and becomes the shared listwise target
for pair-residual utility and proposal ranking.

### 6.2 Dual-source rival mining

The training frontier is the union of:

- teacher-nearest rivals, which represent the true decision boundary;
- model-most-confused rivals, which represent current errors.

This avoids self-confirming mining based only on the model's current margins.

### 6.3 Confidence-shrunk sparse residual

The pair margin is local integrable margin plus sparse antisymmetric residual.
The combined result is shrunk toward the local margin when residual variance is
large or local/residual signs disagree.  This preserves interaction-specific
capacity without allowing the residual head to arbitrarily overturn near ties.

### 6.4 Tournament-active certificate frontier

Instead of certifying every pair incident to the target, v48 protects the
near-boundary target rivals plus safety-crossing rivals.  The cap is deterministic
and diagnostics expose original frontier size and retained pair-weight mass.

### 6.5 Real fixed-budget competition

- proposal Top-M: 64 -> 24;
- reserved interaction slots: 24 -> 8;
- decision-family boost: 0.75 -> 0.25.

The model must now trade interaction, route, progress, regularity and other
families under the actual budget.

### 6.6 Bound and runtime changes

- prior radius: 0.10 -> 0.02;
- target rivals: 16 -> 6;
- selector pairs: 128 -> 96;
- runtime pair-query cap: 320 -> 192;
- residual pairs: 48 -> 32;
- tournament rivals: 16 -> 12;
- utility Top-K: 12 -> 8.

### 6.7 Boundary-aligned checkpoint and gate

Checkpoint selection now prioritizes teacher action, full interface,
pair-full/sparse-full, near-tie sign, budget preservation and regret.  Raw
critical recall has only a small diagnostic weight.

The strict gate additionally rejects duplicate epochs, near-tie regression,
interaction saturation, low frontier retained weight and excessive fallback.

## 7. Novelty assessment

The parts with the strongest research potential are the joint formulation:

1. leave-one-out decision-boundary criticality under query cost;
2. integrable local pair margins with uncertainty-shrunk sparse interaction
   residuals;
3. a deployment-identical nested certificate frontier that preserves the final
   fixed-budget action;
4. explicit decomposition of candidate, interface and compression errors.

Each ingredient has nearby literature—active feature acquisition, pairwise
ranking, conformal risk control and planning uncertainty—but the combination
must be presented as a specific fixed-budget planning method, not as a claim
that no related concept exists.  Novelty must be supported by ablations showing
that each component improves teacher/pair-full correctness, certificate
coverage, budget curves and closed-loop outcomes.

## 8. Required experiment order

1. Use a fresh `OUT_ROOT`; do not reuse the contaminated v47 directory.
2. Build/reuse log-disjoint `val_tune` and `val_calib` manifests.
3. Train v48 from the frozen v30 checkpoint on `val_tune` only.
4. Calibrate on `val_calib` with prior radius 0.02.
5. Replay v48 and frozen control on identical 1000 val-tune rows.
6. Run the strict gate.
7. Let the pipeline automatically start CL20 only after PASS.
8. Run CL100 only after CL20 safety non-inferiority.
9. Keep the official test untouched until all design choices are frozen.

## 9. Validation of the delivered code

- full test suite: **158 passed, 5 warnings**;
- Python compile: pass;
- `run_v48_dbce.sh` syntax: pass;
- `V48_DBCE_NEXT_COMMANDS.sh` syntax: pass;
- v48 gate replay on v47 correctly detects all original failures and duplicate
  concurrent training.

The new code is designed to improve the causal alignment of training and
runtime.  It cannot honestly be claimed to pass before the new GPU experiment
is executed.
