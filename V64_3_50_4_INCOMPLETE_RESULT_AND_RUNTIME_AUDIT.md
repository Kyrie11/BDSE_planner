# V64.3.50.4 incomplete-result reliability and runtime audit

## Formal verdict

**ENGINEERING/DATA WAIT / INCOMPLETE COLLECTION.** The current uploaded package cannot support V50 PIOR algorithm attribution because the preregistered paired data gate has not completed.

Completed certificates:

- control: 260/502;
- treatment: 260/502;
- paired certified prefix: 260/502.

`batch_0005` was still running at archive time and has no `.pior_batch_complete.json`; its partial probe/metric files are diagnostic only.

All completed V50.4 certificates are healthy (`failed=0`, exact expected probe count). This is positive engineering evidence that the V50.4 physical-identity repair is working, but it is not outcome-law evidence.

## Runtime diagnosis

Four completed 64-scene production batches per arm show approximately 22--23 scenarios/hour/arm. With two arms on two GPUs, a full run from scratch is therefore on the order of ~22 wall-hours. From 260 certified scenarios per arm, a clean certificate-based resume has 242 scenarios/arm remaining, approximately 10.5--11 wall-hours at the observed throughput.

The closed-loop planner dominates cost. Profiled planner calls show ~80% cached-plan reuse; non-cached replans contain the expensive certificate/value path. GPUs are already commonly saturated, so increasing worker concurrency is not a defensible free speedup. Preflight/source/regression/config overhead is tiny compared with simulation.

## Safe acceleration decision

For the current frozen V50 experiment, the only high-confidence acceleration is **resume, not algorithm approximation**: preserve the exact certified 260+260 prefix and run only the remaining deterministic batches. A helper audit/launcher is included. The wrapper reduces parent monitoring frequency but intentionally leaves planner profiling and all scientific configuration unchanged.

No V51 mechanism is designed from this incomplete package. Once 502/502 is complete, apply the already frozen V50 identification and causal-retention GO conditions before any algorithm branch is selected.
