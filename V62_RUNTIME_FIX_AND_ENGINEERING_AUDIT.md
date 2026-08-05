# V62 DCAB-EWFC Runtime Fix and Engineering Audit

## 1. Scope and algorithmic invariants

This patch was made after aligning the uploaded paper, the V62 revised manuscript,
`NEXT_COMMANDS_V62_DCAB_EWFC.txt`, and `ALGORITHM_UPDATE_LOG.md`.

The paper-facing novelty remains unchanged:

- a fixed planner-interface **evidence-atom budget** rather than a dense-world reconstruction objective;
- local, auditable, action-queryable decision evidence atoms;
- HAB proposal / deterministic fixed-budget selection;
- a risk-aware action rule and calibrated one-sided margin protection;
- V62's deployment-complete all-valid action bridge and literal leave-one-atom-out winner-flip criticality.

No loss weight, model architecture, atom budget, selector objective, tournament rule,
calibration threshold, or fallback threshold was changed.  The changes below fix
shape semantics, metric accounting, robustness, and avoid redundant evaluation work.

## 2. Root cause of the reported crash

Reported operation:

```text
active_dense_g = dense_g * active_atoms[:, None]
dense_g:       (128, 32)
active_atoms:  (48,) -> (48, 1)
```

`predict_dense_numpy` returns the configured padded interface `[E_max, K] = [128, 32]`.
The cached scene stores only its actually constructed evidence atoms (48 in the
failing scene).  The old evaluator used:

```python
active_atoms = active_atoms[: dense_g.shape[0]]
```

Slicing cannot extend a 48-element mask to 128 entries, so NumPy correctly rejected
the multiplication.  Cropping `dense_g` to 48 would hide the crash but would make the
diagnostic depend on storage length and could silently break padded-interface
semantics.

### Correct fix

- explicitly align masks to the dense tensor axes by zero-padding or truncating;
- padded/nonexistent atoms are inactive and padded/nonexistent actions are invalid;
- validate `[K]`, `[E,K]`, mask shapes, and finite values on active/valid entries;
- use `np.where(active_mask, g, 0)` instead of `0 * g`, because `0 * NaN` remains NaN;
- preserve the full `[128,32]` dense diagnostic and literal LOO winner-flip logic.

## 3. Additional confirmed engineering fixes

### 3.1 V62 query-budget accounting

The all-valid action bridge evaluates each selected evidence atom on every queried
valid action.  The former diagnostics preferred `B * |rival_pairs|` whenever a rival
graph existed, even when the deployed bridge was action-conditioned.  That
under-reported the paper-facing selected-stage interface.

For an actually executed action bridge, `selected_certificate_query_count` and
`effective_query_count` now report:

```text
B * number_of_queried_valid_actions
```

Legacy/direct pair-only paths retain pair-based accounting.

### 3.2 Dense diagnostic hot path

The former open-loop path encoded the same scene twice and called the complete model
`forward`, including proposal/set/residual/pair plumbing not consumed by the dense
local-interface diagnostic.  The optimized path:

- shares the certificate-stage encoded context through the existing thread-local
  prediction-cache scope;
- evaluates only the dense local head needed by `J0/g/g_var` diagnostics;
- uses `torch.inference_mode()`;
- verifies numerical equivalence to the former full-forward path.

The training-time open-loop validation path uses the same cache scope.

### 3.3 Avoided unnecessary GPU serialization

The model adapters return NumPy outputs, so the CUDA-to-CPU copies needed for the
planner result already synchronize the relevant work.  Extra global
`torch.cuda.synchronize()` calls around each scene serialized concurrent workers and
were removed from the certificate-stage latency interval.  This does not change any
model output.

### 3.4 Memory-bounded open-loop aggregation

The evaluator no longer retains every `BDSEMetricResult` and every JSONL row in RAM.
It now:

- computes finite-value metric means online in `O(number_of_metric_keys)` memory;
- writes per-sample JSONL incrementally with a buffered writer;
- retains the previous finite-value mean semantics exactly.

This matters for a completed test set with tens of thousands of scenes.

### 3.5 Deterministic iterable handling

`deterministic_order()` previously converted its input iterable to a list more than
once.  A generator was consumed by the first conversion and could produce an empty or
inconsistent second result.  It now materializes the iterable exactly once.

## 4. Validation performed

- Python bytecode compilation: pass.
- YAML parsing: **315 files** parsed successfully.
- Shell syntax: **41 executable shell/command scripts** passed `bash -n`.
- Full unit suite: **240 passed**, 0 failed.
- Regression coverage added for:
  - short active mask -> padded dense atom axis;
  - V62 all-valid `B*K` selected query accounting;
  - optimized dense `J0/g/g_var` equivalence to full forward;
  - one-shot iterable ordering;
  - online metric aggregation equivalence.
- V62-config semantic audit (reduced hidden width only for speed): dense output
  equivalence passed at `E=128`, `K=32`.

The Transformer nested-tensor message caused by `norm_first=True` is a PyTorch
optimization warning, not the reported failure.  Changing normalization order would
alter checkpoint/model semantics and was intentionally not done.

The `156/180 compatible tensors` message is also not the crash.  The listed missing
parameters are V62 residual-action/variance heads absent from the older foundation
checkpoint; the V62 warm-start policy explicitly initializes new heads instead of
requiring a strict load.

## 5. Runtime microbenchmark

A reduced CPU synthetic scene measured the combined certificate + dense diagnostic
path over three independent process runs:

| Version | Mean seconds/scene |
|---|---:|
| Uploaded implementation | 0.133892 |
| Optimized implementation | 0.116890 |
| Relative reduction | 12.70% |

This benchmark validates removal of redundant work, not GPU paper-grade throughput.
Actual speedup depends on GPU, worker count, I/O, atom count, and candidate validity.
No model output or decision rule was relaxed to obtain it.

## 6. Dataset/readiness interpretation

The uploaded diagnostics contain 58,418 validation samples and 67,042 currently
available test samples, with no duplicate split identities in either diagnostic.
However, the current test diagnostics fail more paper-scale checks than validation:

- validation E0 failures: safe-candidate coverage and candidate/log ADE p50/p90;
- current test E0 failures: full-interface action match, B16 oracle decision
  sufficiency, runtime decision sufficiency, safe-candidate coverage, candidate/log
  ADE p50/p90, and logged-ego route-distance p90;
- test full-interface action match is 0.93428 versus the 0.95 gate;
- test B16 oracle decision sufficiency is 0.83992 versus the 0.85 gate;
- test runtime decision sufficiency is 0.64073 versus the 0.72 gate;
- the diagnostic reports a negative full-interface teacher regret, which should be
  audited before treating the split as a publication-grade final benchmark.

Therefore the incomplete test cache is usable as a **single frozen-checkpoint stress
test** for algorithm comparison, provided no checkpoint, threshold, or version is
selected from it.  It should not be used as the final headline test result until test
construction and parity/readiness audits are complete.  This distinction preserves a
valid held-out protocol without preventing useful engineering analysis.

## 7. Recommended rerun

After replacing the repository with this package, rerun the same V62 wrapper.  The
failed foundation-quality output JSON was not completed, and the evaluator opens its
JSONL in overwrite mode, so the stage will be regenerated cleanly.

To resume and reuse any already valid split/provenance/checkpoint artifacts, use:

```bash
export PIPELINE_FORCE=0
bash V62_DCAB_EWFC_NEXT_COMMANDS.sh |& tee "$OUT_ROOT.pipeline.console.log"
```

Keep `PIPELINE_FORCE=1` only when a deliberately fresh split/training/calibration run
is required.  No claim of gate pass, closed-loop improvement, or CCF-A-level result is
made until the paper-grade GPU pipeline and paired nuPlan closed-loop evaluation are
actually rerun.
