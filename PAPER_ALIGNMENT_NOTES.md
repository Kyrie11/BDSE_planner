# BDSE Paper Alignment Notes

This update aligns the implementation with the BDSE paper's deployment interface:

- **Positioning**: BDSE is a budgeted planner-interface module for autonomous driving. It selects a small set of auditable, local evidence atoms that preserve the full-information teacher's trajectory choice under a finite candidate bank.
- **Cost decomposition**: the model learns `J0(a)` and atom-local costs `g_i(a)`, with pair evidence labels `d_i(a,b)=g_i(b)-g_i(a)` and aggregate residual supervision.
- **HAB**: the runtime path now uses a Hierarchical Atom Builder with learned family logits, family-aware Top-M proposal allocation, strict family budget caps, and a free reserve.
- **Uncertainty**: each sparse atom-action query predicts both mean and variance. The selector and tournament use lower-confidence margins `mean - beta * sigma - epsilon_cal`.
- **Sparse runtime discipline**: action loss, open-loop evaluation, calibration, and planner runtime no longer leak dense offline evidence; they use HAB Top-M atoms and only queried action evidence for the runtime rival/tournament graph.
- **Training additions**: pair atom-margin loss, heteroscedastic uncertainty loss, family listwise loss, proposal loss, residual loss, rank loss, calibration surrogate, and deployment-consistent action loss are all exposed through `compute_bdse_losses`.
- **Validation**: repository tests pass with `29 passed, 1 warning`.

Main touched files:

- `bdse/model/bdse_model.py`
- `bdse/model/losses.py`
- `bdse/planner/hab.py`
- `bdse/planner/selector.py`
- `bdse/planner/tournament.py`
- `bdse/planner/nuplan_planner.py`
- `bdse/data/tensorizer.py`
- `bdse/experiments/evaluate_open_loop.py`
- `bdse/experiments/calibrate.py`
- `bdse/configs/default.yaml`
- `bdse/tests/test_hab_uncertainty_alignment.py`
