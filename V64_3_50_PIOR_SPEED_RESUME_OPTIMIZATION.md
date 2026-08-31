# V64.3.50 PIOR engineering speed/resume optimization

This patch is engineering-only. It does **not** change the V64.3.50 PIOR scientific mechanism, the exact 502-scene frozen V49 full-set RSMR proposal population, treatment/control action semantics, Q/P/E state, fitting loss, capture-derived retention budget, closed-loop challenge, or final paired-outcome label.

## Why the original run looks frozen after `PASS ... probe configs`

The next stage launches two 502-scenario nuPlan subprocesses and redirects each child's stdout/stderr to `closed_loop_train/{control,treatment}/run.log`. The parent used blocking `subprocess.run()` and printed nothing until both arms finished. Therefore the console can remain unchanged for hours even while simulations are progressing.

## Concrete bottlenecks found

1. **All TRAIN DB directories were handed to nuPlan.** The exact 502 tokens already carry log-name information, but the original runner passed four whole city split directories. nuPlan therefore discovers/opens many irrelevant TRAIN DBs during scenario construction.
2. **GPU inference was explicitly serialized.** Each arm uses a 4-worker thread pool but set `BDSE_SERIALIZE_GPU_INFERENCE=1`, turning the shared read-only CUDA model into a process-wide inference critical section. The repository's optimized fixed-budget closed-loop runner already uses the same shared-model mode with serialization disabled.
3. **Huge per-tick diagnostics I/O.** `BDSE_CLOSED_LOOP_DIAG` serialized the complete selector/tournament diagnostic tree on every simulator tick. PIOR only needs a durable certificate that the one-shot probe fired once per scenario.
4. **No heartbeat.** Child logs were hidden from the parent, so there was no indication whether time was spent in nuPlan DB/scenario construction, planner/model initialization, or simulation.
5. **No scientifically auditable partial checkpoint.** A 502-scenario arm had to finish completely before it could be reused.

## Optimizations

### Safe raw DB restriction

The scientific identity is the exact frozen `scenario_token`, not a guessed filename. The manifest therefore resolves each token to the narrowest **safe DB candidate set** it can prove: exact DB stem first; then the repository-consistent stable nuPlan log identity after stripping a numeric crop suffix such as `_00718_00912`; then an optional read-only SQLite token lookup for multi-chunk log families. If a local DB schema cannot prove one file, the whole stable-log family is retained; if naming cannot establish even that family, only that token's known city split is used. nuPlan still receives the exact frozen `scenario_filter.scenario_tokens`, so widening the DB search set cannot silently add a scientific sample. Each batch passes only the union of its tokens' safe candidate sets.

Important bug fix: nuPlan log identities contain dots. The previous speed package used `Path(log_name).stem`, which incorrectly converted `2021.08.23.18.41.38_veh-28` to `2021.08.23.18.41`. The hotfix never uses generic path-stem parsing on a log identity and strips only a literal `.db` suffix.

### Probe-only diagnostics

The PIOR runner sets `BDSE_CLOSED_LOOP_DIAG_MODE=pior_probe_events`. The planner writes only the unique fired PIOR event and a minimal contract payload; non-probe ticks write nothing. Historical/default behavior remains the original full diagnostic mode.

A process-wide append lock prevents thread interleaving of the lightweight JSONL event certificates.

### Concurrent shared-model inference

The optimized runner defaults to `BDSE_SERIALIZE_GPU_INFERENCE=0` while keeping `BDSE_SHARE_MODEL_PER_PROCESS=1` and one physical GPU per arm. This changes only execution scheduling of independent read-only eval forwards. Set `PIOR_SERIALIZE_GPU_INFERENCE=1` to restore the conservative old lock if the server exhibits CUDA/thread instability.

### Heartbeat and timing profile

Every `PIOR_HEARTBEAT_SECONDS` (default 30 s), the parent prints:

- arm and batch index;
- elapsed wall time;
- PIOR probe-event count;
- physical GPU utilization and memory;
- detected phase (`process-starting`, `nuplan-init/scenario-build`, `planner-loaded/simulating`, `metrics/finalizing`);
- latest child-log line.

Each completed batch also emits `bdse_closed_loop_profile.json`, aggregating planner-stage timing such as runtime feature construction, candidate generation, evidence enumeration, certificate stages, model timing, and trajectory conversion.

### Validated batch resume

The 502 tokens are divided deterministically into batches (`PIOR_BATCH_SIZE`, default 64). A batch is reusable only after all of the following are complete and hashed:

- exact batch token list/hash;
- exact treatment/control config hash;
- checkpoint hash;
- challenge identity;
- exact raw-DB path-list hash for that batch;
- `N/N successful, 0 failed`;
- exactly `N` PIOR probe events;
- exact `N` token-addressed per-scenario metric rows;
- normalized metric file hash;
- probe-event file hash.

On rerun with `PIOR_RESUME=1`, only such batches are skipped. An interrupted/incomplete batch is deleted and rerun from scratch. This does not change sample membership, fold assignment, action semantics, or labels.

## Can the already-running old job be resumed?

**Partially completed old arms are intentionally not reused.** The old format did not bind a token to a completed simulation, its unique probe event, and its final metric in one completion certificate. Reconstructing a partial arm from scattered old files would weaken the paired-intervention audit.

A **fully complete** old arm is safe to reuse. The optimized runner validates all of: 502 successful, zero failed, 502 probe fires, and exact 502 token-addressed aggregator metrics. If an old control or treatment arm satisfies this, it is converted and reused automatically with `--allow-legacy-full-arm-resume` (enabled by the launcher when `PIOR_RESUME=1`).

## Knobs

```bash
# Default optimized run
bash RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh

# More CPU simulation concurrency if the server has headroom
WORKERS_PER_ARM=6 bash RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh

# Finer resume granularity
PIOR_BATCH_SIZE=32 bash RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh

# Fewer process/model reloads (coarser resume granularity)
PIOR_BATCH_SIZE=128 bash RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh

# Restore old serialized GPU inference for debugging
PIOR_SERIALIZE_GPU_INFERENCE=1 bash RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh

# Disable resume deliberately
PIOR_RESUME=0 bash RUN_V64_3_50_EAF_ICER_PIOR_TRAIN_2GPU.sh
```

Do **not** change `planner.replan_interval_ticks`, the simulation horizon, candidate count, RSMR, treatment/control action, challenge, or metrics for speed: those change the scientific intervention/outcome semantics rather than only the engineering execution path.
