# V55 PC-BFAR-DBAP

V55 introduces **Potential-Consistent Boundary-Focused Anchor-Residual Decision-Budget Action Preservation**.

The V54 selector already retained a strong fraction of decisive evidence, but its pair tournament did not make the selected-local action cost the actual deployed anchor. V55 fixes that interface:

1. build the fixed-budget selected-local cost `J_B^L` directly from the selected B=16 evidence;
2. interpret the learned pair head only as a residual edge field;
3. project that edge field onto a globally integrable action potential with a weighted Hodge projection;
4. add the potential to `J_B^L` and select the action from the resulting global cost;
5. allow a residual action flip only when its uncertainty-shrunk global margin is certified;
6. make the same-checkpoint local control remove both residual mean and residual uncertainty.

At zero residual, V55 is guaranteed by construction to return the direct selected-local action. The deployed B=16 selector remains exact AOCC.

Main entry point:

```bash
bash V55_PC_BFAR_DBAP_NEXT_COMMANDS.sh
```

See:

- `V55_PC_BFAR_ANALYSIS_AND_NEXT_STEPS.md`
- `NEXT_COMMANDS_V55_PC_BFAR.txt`
- `ALGORITHM_UPDATE_LOG.md`
