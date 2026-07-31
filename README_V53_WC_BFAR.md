# V53 WC-BFAR-DBAP

**Winner-Consistent Boundary-Focused Anchor-Residual Decision-Budget Action Preservation**

V53 corrects two blockers found in the uploaded V52 experiment:

1. the immutable foundation gate was using final budgeted-action regret and therefore rejected the same unchanged anchor under a different runtime configuration;
2. the nominal action-aware losses were computed on frozen base+local outputs and supplied no gradient to the residual/selector modules.

V53 keeps the paper's causal line:

`full decision boundary -> flip-critical pairs -> decisive evidence under B=16 -> certified residual correction -> action/closed-loop effect`.

New components include an anchor-only provenance gate, explicit interface-specific regret metrics, safe zero-residual initialization, residual-tournament teacher-winner losses, B=16 winner-preservation loss, sparse cycle-consistency, and separate minimum-completeness versus competitive gates.

Run the full pipeline with:

```bash
bash V53_WC_BFAR_DBAP_NEXT_COMMANDS.sh
```

See:

- `V53_WC_BFAR_ANALYSIS_AND_NEXT_STEPS.md`
- `NEXT_COMMANDS_V53_WC_BFAR.txt`
- `ALGORITHM_UPDATE_LOG.md`
