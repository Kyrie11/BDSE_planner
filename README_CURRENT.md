# BDSE current algorithm — V64.3.56 EAF-ICER-RCPR

**Realized Constraint-Process Retention**

V55 is reliable for TRAIN attribution. Its REALIZED-DOMINANCE arm identifies an
unweighted paired-outcome Pareto functional (`concordance=0.567756`, 4/5 folds
above random and 5/5 above the V52 static Pareto control), but the unchanged
deployment gate still fails beneficial retention and fold-wise nonharm. The
PREDICTED-DOMINANCE branch was correctly not evaluated.

This closes the combination of **static one-replan realized ego motion + tested
static sign/Pareto functionals** as deployment-sufficient. V56 executes the
last preregistered internal state family: realized operator-relative
interaction/safety constraint consequence.

## Frozen components

- bounded EAF interface, support/admissibility;
- frozen full-set RSMR winner;
- same-winner/incumbent, veto-only, no fallback/rerank/second-best;
- exact V50.5 metric-safe 502 paired outcomes;
- V52 QPE+D effect-support hurdle;
- V54 realized one-replan ego endpoint mediator;
- **V55 unweighted Pareto functional and exact deployment gate**;
- V53 fixed planned endpoint + DCT-II k=1,2 profile;
- five folds, `lambda=1`, `alpha=0.07791855203619909`.

## Branch A — REALIZED-CONSTRAINT-PROCESS

Replay the exact V50.5 treatment/control one-shot intervention only through the
first scheduled replan (5 ticks), with final metrics disabled. At each current
simulated state record three lower-is-safer runtime-semantic channels:

1. agent occupancy interaction risk;
2. agent TTC risk;
3. hard route-corridor excess.

The paired process is `control risk - treatment risk` on post-intervention ticks
1..5. Larger is better on every channel. No safety weight, scalarization,
attention, DCT, horizon sweep, or new outcome label is introduced.

The outcome state is:

`QPE+D + exact V54 realized endpoint + 15-D realized constraint process`.

The functional is exactly V55. It must beat random and the exact V55
REALIZED-DOMINANCE concordance in aggregate and >=4/5 folds, then pass the
unchanged deployment gate.

This is a diagnostic/oracle branch, not t0-deployable.

## Branch B — PREDICTED-CONSTRAINT-PROCESS

**Not fit/scored unless Branch A fully passes.**

A zero-bias, zero-preserving `lambda=1` ridge predicts the realized constraint
process from the fixed V53 planned profile plus `D * t0 current constraint
risk`; the V54 endpoint predictor is the exact V55 model family. Nested inner
OOF predictions are used to train the outcome ranker.

Both nuisance predictions must beat zero-response baselines aggregate and >=4/5
folds, and the same outcome/deployment gates must pass. A full pass freezes the
algorithm immediately for runtime integration and untouched validation.

## Final internal convergence rule

V56 is the final internal state-family test.

- Oracle FAIL -> stop internal algorithm search by falsification; no V57/V58
  feature/state variants.
- Oracle PASS but the single preregistered t0 bridge FAIL -> stop state-family
  search; do not increase nuisance capacity.
- Predicted branch full PASS -> freeze; only runtime integration, untouched
  validation, then external baselines/official benchmarking.

## Run

```bash
cd bdse_v64_3_56_eaf_icer_rcpr
bash RUN_V64_3_56_EAF_ICER_RCPR_TRAIN.sh
```

Two GPUs are used for treatment/control short-horizon replay by default. Set
`GPU_TREAT=0 GPU_CONTROL=0` for one-GPU sequential execution. Final paired
outcomes are reused from V50.5; only the new 5-tick constraint mediator is
recollected.
