# v49 DBAP 使用说明

## 目标

v49 将 v48 的主瓶颈从“证书可达性”转向“部署边界动作保留”：

- 纯 leave-one-out boundary-critical target；
- integrable local margin + bounded uncertainty-gated residual；
- exact-B nested AOCC；
- cross-family prefix capacity；
- validation/checkpoint/gate 与最终 pair-conditioned tournament 完全对齐；
- cache provenance 防止 resume 混入旧配置。

## 主要文件

- `bdse/configs/v49_bdse_dbap_train_2gpu.yaml`
- `bdse/configs/v49_bdse_dbap_cl.yaml`
- `run_v49_dbap.sh`
- `V49_DBAP_NEXT_COMMANDS.sh`
- `BUILD_MATCHED_TEST_SET.sh`
- `BDSE_V48_RESULT_V49_DBAP_REPORT.md`
- `ALGORITHM_UPDATE_LOG.md`

## 两张 A30 执行

```bash
export NUPLAN_ROOT=/path/to/nuplan
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2
export BDSE_VAL_CACHE_ORIGINAL=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2
export V30_CKPT_IN=outputs_v30/train/bdse_v30_pmvrbsr.best.pt
export OUT_ROOT=outputs_v49_dbap_exact_2gpu_v1

PIPELINE_DETACH=1 \
RUN_CLOSED_LOOP_AFTER_GATE=1 \
RUN_CL100_AFTER_CL20=0 \
bash V49_DBAP_NEXT_COMMANDS.sh
```

监控：

```bash
tail -f "$OUT_ROOT"/logs/pipeline_*.log
```

流水线：

1. 按 nuPlan log group 构建 `val_tune` / `val_calib`；
2. 从冻结 v30 checkpoint 在两张 GPU 上干净训练；
3. 仅在 `val_calib` 校准 adverse bounds；
4. 两卡并行 replay 相同 1000 个 `val_tune` 场景；
5. 重建同 split 的 frozen control；
6. 严格 paired gate；
7. PASS 后自动运行两卡 CL20；
8. 可选 CL100。

## 构建匹配的 test cache

使用全新输出目录：

```bash
TEST_OUT=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_v49_matched \
VAL_DIAGNOSTICS=/path/to/diagnostics_val.json \
bash BUILD_MATCHED_TEST_SET.sh
```

该脚本恢复 `include_drivable_polygons=true`，移除旧命令的每 log 512 上限，并启用严格 cache provenance。重新构建只能消除配置/缓存不一致，不能保证 official test 与 train/val 的场景分布相同。

## 调试模式

仅检查 open-loop，不自动运行 closed-loop：

```bash
PIPELINE_DETACH=0 \
RUN_CLOSED_LOOP_AFTER_GATE=0 \
bash V49_DBAP_NEXT_COMMANDS.sh
```

不要通过降低 gate 门槛制造 PASS。失败项应继续归因到 candidate、interface、compression、certificate 或 runtime 中的对应层。
