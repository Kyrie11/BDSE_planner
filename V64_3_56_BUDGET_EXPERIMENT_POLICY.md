# V64.3.56 fixed-budget experiment policy

1. **Primary result:** B=16, M=24, identical ordered test tokens, metric-safe full closed loop.
2. **External trainable adapters:** train and validate independent checkpoints at B=8,16,24 using the same TRAIN/VAL split and checkpoint-selection rule.
3. **BDSE:** do not refit after internal-search convergence. The frozen B16 EAF checkpoint and frozen RSMR model are the headline runtime method. Evaluate them at B=8/16/24 only as a cross-budget interface-robustness curve.
4. **Why not silently retrain BDSE B8/B24?** V13 EAF training was explicitly B16-only and RSMR is fit on the resulting B16 frontier distribution. A true matched-training B8/B24 BDSE curve therefore requires retraining EAF and refitting RSMR per budget. That is a valid separate experiment only if preregistered before test inspection; it is not equivalent to changing one YAML field.
5. **Reporting:** keep the B16 matched-interface table separate from the cross-budget robustness plot/table. Label controlled architecture adapters as `*-inspired` unless official repositories are reproduced independently.
