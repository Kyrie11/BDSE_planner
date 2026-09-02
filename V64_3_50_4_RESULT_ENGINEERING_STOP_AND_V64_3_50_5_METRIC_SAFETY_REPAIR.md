# V64.3.50.4 result audit → V64.3.50.5 metric-safety repair

## Formal verdict

**ENGINEERING/DATA STOP. No V50 algorithm attribution is permitted from the uploaded result.**

The uploaded V50.4 resume result is not a completed 502-pair experiment:

- control certified: **452 / 502**;
- treatment certified: **324 / 502**;
- paired certified: **324 / 502**;
- no complete paired-outcome table;
- no nested PIOR identification result;
- no causal-retention gate result.

This alone is sufficient to block GO/STOP attribution. In addition, the resume exposed a new metric-pipeline failure, so the remaining work is not merely “continue the last batch.”

## New failure

Two independent arm/batch failures end inside nuPlan metric computation:

- treatment `batch_0006`: **63 success / 1 failed**, failed token `a2326fd4694d5191`;
- control `batch_0008`: **49 success / 1 failed**, failed token `fbd37ced14a15418`.

Both fail in `drivable_area_compliance.py` when constructing a `TimeSeries`: the history timestamps and metric values have different lengths, triggering the `TimeSeries` length assertion and causing `Metric Engine failed`.

The probe event itself has already fired for every scenario in both incomplete batches (64/64 treatment, 50/50 control), which localizes the new failure after the PIOR one-shot intervention logic, in the evaluation callback rather than the proposal execution path.

## Root-cause diagnosis

nuPlan constructs metric engines keyed by scenario type and reuses a metric engine for all simulations of that type. `DrivableAreaComplianceStatistics` consumes `EgoLaneChangeStatistics.corners_route` and `ego_driven_route`, which are mutable state written for the currently computed history. V50.4 executes scenarios through `single_machine_thread_pool` with four workers per arm. Consequently, two scenario threads of the same type may call the same stateful metric engine concurrently.

This explains the observed cross-history length mismatch and, more importantly, means a successful batch is not automatically proof that every metric value was race-free: an interleaving can change state without necessarily triggering the length assertion.

Therefore the already certified V50.4 metric outputs are **not promoted as paper-grade causal labels**.

## V64.3.50.5 repair contract

V50.5 is engineering-only. The scientific V50 mechanism is frozen byte-for-byte:

- exact 502 frozen full-set RSMR proposal population;
- same exact-anchor cached physical proposal identity;
- same one-shot treatment and incumbent control;
- same Q / (P-Q) / (E-P) risk state;
- same zero-bias pairwise sign-risk and fixed lambda;
- same PIOR outcome label and hard-safety coordinates;
- same nested identification gate and causal-retention gate;
- same veto-only/no-rerank/no-second-best/no-fallback deployment contract.

The repair wraps nuPlan's official simulation entrypoint and serializes the entire `run_metric_engine` callback inside each arm process with a process-wide re-entrant lock. **Simulation and planner execution remain threaded with four workers.** Only metric evaluation is serialized.

The frozen V50 paired collector is not edited. A small spawn shim checks its exact SHA256 and rewrites only the child nuPlan module to the metric-safe wrapper.

## Why full 502/502 rerun is required

Do **not** resume from V50.4's 324 paired certificates. Because the identified failure is shared mutable metric state, absence of an assertion is not sufficient to prove that drivable/TTC/dependent metrics were not silently computed from an interleaved state. Those metrics participate directly in PIOR hard-safety labels and therefore can change the learned outcome law.

V50.5 uses a new output root and recomputes all 502 treatment/control outcomes under one homogeneous, metric-safe evaluator.

## Runtime/speed decision

Safe acceleration is retained:

- two GPUs remain concurrent, one causal arm per GPU;
- `WORKERS_PER_ARM=4` remains unchanged;
- batch size and exact population remain unchanged;
- planner caching/replan cadence remains unchanged;
- the new lock covers only the short metric callback at scenario end.

The repair therefore avoids the approximately 4x wall-time penalty of making all simulation sequential. A four-scenario-per-arm engineering sentinel runs first using the two V50.4 failure tokens plus two timestamp-compatible fixed TRAIN tokens. It is never consumed by PIOR fitting. If the sentinel fails, the 502-scene job does not start.

No result-dependent retry, metric truncation, metric dropping, safety-label substitution, threshold sweep, or scenario removal is introduced.

## Scientific status after this repair

V50 remains **scientifically unevaluated** until V50.5 reaches exact 502/502 paired completion and the already preregistered PIOR identification + causal-retention gates execute.

Accordingly this round makes **no new algorithm-family closure, no mechanism promotion, no V51 design, and no change to the dominant bottleneck claim**. The V49→V50 preregistered branch remains the only legitimate next scientific decision rule.

## Command

```bash
cd bdse_v64_3_50_5_eaf_icer_pior_metricsafe
bash RUN_V64_3_50_5_EAF_ICER_PIOR_TRAIN_2GPU.sh
```

The launcher first verifies the original V50 source manifest and the V50.5 engineering manifest, then runs the metric-race sentinel. Only after the sentinel passes does it launch the full new 502×2 paired collection, then a metric-safety provenance audit, then the unchanged nested PIOR fit/gates.
