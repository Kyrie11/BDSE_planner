# BDSE v35 DICE

**DICE-BDSE: Direction-Invariant Complementary Evidence selection**

v35 targets the only failed v34 runtime gate metric: selected interaction-decisive recall. It keeps the evidence budget `B=16`, proposal size `M=64`, runtime pair cap `480`, signed tournament, and the v30 checkpoint architecture unchanged.

## Root fixes

1. **Complementary soft-interaction floor**: hard occupancy atoms no longer satisfy the TTC/gap interaction reservation.
2. **Agent-diverse reservation**: Top-M and final B=16 selection prefer different interacting agents before redundant evidence for one agent.
3. **Direction-invariant selector influence**: after reciprocal query canonicalization, both positive and negative predicted pair contributions can rank interaction evidence; the final tournament remains signed.
4. **Non-regression runtime gate**: optionally compares each v35 candidate against the v34 action-rank JSON.

## Runtime-only gate

```bash
SKIP_TRAIN=1 \
V35_CKPT=outputs_v30/train/bdse_v30_pmvrbsr.best.pt \
OUT_ROOT=outputs_v35_runtime_v30ckpt \
V34_BASELINE_JSON=outputs_v34_runtime_v30ckpt/open_loop/open_loop_v34_abiq_action_rank.json \
RUN_MODE=open_loop \
OPEN_LOOP_MAX_SCENARIOS=1000 \
ENFORCE_RUNTIME_GATE=1 \
bash run_v35_dice.sh
```

Do not run CL20 or training unless `open_loop/runtime_gate.txt` contains at least one PASS.
