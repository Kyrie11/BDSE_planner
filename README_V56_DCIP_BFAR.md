# V56 DCIP-BFAR-DBAP

**Dual-Certificate Integrable-Potential Boundary-Focused Anchor-Residual Decision-Budget Action Preservation**

V56 keeps the fixed-budget causal chain introduced in V52–V55:

```text
complete decision boundary
→ winner / hard / near-tie pairs
→ decisive evidence
→ exact B=16 AOCC
→ selected-local action anchor
→ certified residual correction
→ global action winner
→ paired closed loop
```

It fixes two failures exposed by V55:

1. AOCC/evidence certification was contaminated by residual uncertainty even though the residual changed no deployed action.
2. A Hodge projection of an arbitrary pair field plus a scene-level potential target did not identify which evidence should change which action.

## Main changes

- **Dual certificates**: the exact evidence selector is certified only against the selected-local action interface; residual action flips use a separate global uncertainty certificate.
- **Direct integrable evidence potential**: each selected atom predicts one action-cost correction `h_i(a)`; summation is globally integrable by construction.
- **Atomwise causal distillation**: `h_i(a)` is supervised by the exact teacher-minus-local per-atom action cost, with extra weight on the teacher winner, the wrong anchor action, and interaction evidence.
- **Exact selected-local no-op**: zero residual potential gives exactly the B=16 selected-local argmin.
- **Pure controls**: local/foundation controls disable both residual mean and residual uncertainty.
- **Closed-loop acceleration**: all planners in one nuPlan worker process reuse a single CUDA model and one device inference lock; the default worker count is four and summary-PDF rendering is disabled.
- **Closed-loop protocol audit**: combined summaries report the real scenario count and three-way CL20 token identity is verified.
- **Frozen preliminary test protocol**: the partial test cache may be used once after model/config/gates are frozen, with checkpoint/config hashes recorded.

## Primary command

Use `NEXT_COMMANDS_V56_DCIP_BFAR.txt` or `V56_DCIP_BFAR_DBAP_NEXT_COMMANDS.sh`.

## Claim boundary

V56 has been compiled and unit-tested locally, but no fresh nuPlan training or closed-loop simulation was run in this environment. Gate pass, closed-loop gain, real-time speedup, and fixed-budget SOTA remain experimental questions.
