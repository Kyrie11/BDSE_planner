# v36 SCIDE runtime-gate analysis and v37 SAGE design

## 1. Result diagnosis

The v36 result is not a selector-recall failure. Structural hard coverage is 1.0, effective hard recall is 1.0, selected soft-interaction recall is approximately 0.61–0.64, effective interaction recall is approximately 0.68–0.70, fallback is zero, and query counts are far below budget. The failures are concentrated in final-action semantics and margin fidelity:

- raw selected safety-flag rate: 0.035;
- teacher action match: 0.209–0.211;
- interaction pair-sign accuracy below the v35 non-regression floor;
- teacher regret approximately 13.1k–13.5k, above the v35 floor.

The identical 0.035 safety-flag rate across all four selector configurations is a strong indication that it is not controlled by selector allocation.

## 2. Full-chain root causes

### 2.1 Raw safety-flag rate conflated two events

The tournament hard guard selects a safe valid action whenever one exists. Therefore a flagged final action can remain primarily when all valid candidates are flagged. The old gate treated this candidate-bank failure as if the selector had ignored an available safe action.

v37 separates:

- `avoidable_selected_action_safety_flag`: a safe candidate existed, but the final action was flagged;
- `all_actions_safety_flagged`: every valid candidate was flagged;
- raw selected flag rate, retained as a diagnostic;
- all-flagged hard-risk regret after the structural risk guard.

This is a semantic correction, not a relaxation: avoidable unsafe choices are still capped at 0.5%, and all-flagged cases must be handled by a continuous risk-minimizing guard.

### 2.2 Structural-mask overreach removed soft feasibility evidence

v36 used:

```text
hard_atom OR family_id == feasibility
```

as the structural mask. The feasibility family contains both hard constraints and soft decision evidence. Consequently route-connector and speed-limit evidence were removed from Top-M and B=16 along with true hard atoms.

v37 sets `structural_safety_include_feasibility=false`. Only genuinely hard atoms bypass the budget; soft feasibility evidence returns to the decision certificate.

### 2.3 Binary safety masks destroyed graded teacher margins

A hard occupancy or drivable-area atom is not only a binary violation indicator. Its continuous cost also represents clearance, boundary proximity, and near-critical TTC. Removing all hard evidence from the margin decomposition preserved a yes/no feasible set but lost the ranking signal among feasible actions.

v37 computes a deployment-only structural residual from:

- hard agent overlap risk;
- TTC risk;
- hard off-route risk;
- soft agent risk;
- soft off-route risk;
- red-light risk.

Each component is robustly rank-normalized, combined, and added to `J0` at normalized pair-margin scale. It is budget-exempt because it is a compressed safety-interface prior, not a selected evidence atom. It uses no teacher future and no logged-future label.

### 2.4 Hard pair deletion caused checkpoint support shift

v36 restricted the pair graph to the hard viability frontier. That is appropriate only with a perfect safety oracle and a model trained on exactly that graph. Runtime flags are conservative, and the unchanged v30 checkpoint was trained on a broader comparison support.

v37 retains the full logical graph but changes acquisition weights:

- safe–safe: 1.0;
- safe–flagged: 0.35;
- flagged–flagged: 0.10;
- all-flagged scenes: minimum-hard-risk frontier receives full weight, outside pairs retain a small weight.

This preserves calibration comparisons and checkpoint compatibility without increasing pair count.

### 2.5 All-flagged scenes need a graded action guard

A binary filter cannot choose among an all-flagged candidate bank. v37 first applies red-light hierarchy, then forms a near-minimum continuous hard-risk pool, and finally uses certificate score, soft risk, and TTC as tie-breakers. The resulting hard-risk regret is logged.

## 3. v37 SAGE algorithm

SAGE means **Safety-Always-on Graded Evidence**.

```text
complete structural constraints
          │
          ├── binary feasibility / final hard guard
          └── compressed graded safety residual → J0

soft feasibility + interaction + precedence + route + comfort evidence
          │
          └── fixed B=16 selector and signed tournament
```

The architecture preserves the paper's key claim: a fixed number of decision-sufficient evidence atoms is selected after a complete safety interface. It avoids claiming that basic collision/map checks themselves are optional evidence.

## 4. Gate semantics

The v37 gate retains the meaningful non-regression requirements:

- teacher action match ≥ 0.215 and no more than 0.003 below v35 baseline;
- interaction/winner-rival/hard pair-sign accuracy no more than 0.005 below baseline;
- teacher regret no more than 3% above baseline;
- fixed query limits;
- structural and effective hard coverage ≥ 0.98;
- selected soft-interaction recall ≥ 0.32;
- effective interaction recall ≥ 0.35.

Safety is checked as:

- avoidable flagged-action rate ≤ 0.005;
- raw flagged-action rate must not exceed all-flagged rate by more than 0.001;
- the all-flagged continuous risk guard must cover all all-flagged cases.

## 5. Checkpoint recommendation

Use the unchanged v30 checkpoint for the next runtime-only gate. v37 adds no learned parameters and does not alter checkpoint tensor shapes. Training before runtime passes would confound selector/channel changes with weight adaptation.

Only after a v37 runtime configuration passes and CL20 is non-degrading should controlled v30-initialized finetuning be run.

## 6. Validation performed in this environment

- full regression: 108 tests passed;
- Python compilation passed;
- shell syntax check passed;
- all v37 configurations loaded successfully;
- no evidence-budget or proposal-size increase;
- no new neural parameters.

Numerical nuPlan validation cannot be executed here because the nuPlan cache and v30 checkpoint are not available in this environment.
