# BDSE V61 训练报错、数值修复与 test-set 使用建议

## 1. 结论

1. 本次 smoke training 失败不是 `torchrun`/DDP 通信故障，也没有证据指向缓存读取或 checkpoint 损坏。两个 rank 都在 epoch 0、batch 0、global step 0 的同一组量上失败：`loss`、`L_proposal_logit_stability`、`proposal_logit_rms_mean`。
2. 根因是 proposal 无效 atom 的极小掩码哨兵与 V61 新增 RMS 稳定项的运算顺序组合：先平方极小 FP32 值得到 `inf`，再乘零掩码得到 `NaN`。
3. 修复原则不是用 `nan_to_num` 吞掉异常，而是让 inactive atom 在任何平方/非线性运算之前通过 `torch.where` 置零；active atom 上的真实 `Inf/NaN` 仍会被训练全局 finite check 捕获。
4. 上传的 test diagnostics 可以作为当前的 **development/shadow test**，但不应作为最终论文 test：缓存仍在构建，且 test 相对 val 存在明显难度/质量/分布偏移。任何根据 test 结果进行的算法修改都会使该 test 变成开发数据。
5. 已增加 `RUN_V61_TEST_OPEN_LOOP.sh`：
   - `development_test`：允许不完整缓存，默认 1000 条，结果强制标记为开发诊断；
   - `final_test`：要求完成性、manifest、train/val 泄漏审计，并自动评估 diagnostics 中的全部 test 样本；不会在 test 上校准、选 checkpoint 或调阈值。

## 2. 对论文与 V61 代码算法的理解

论文定义的 BDSE 主线是：有限候选轨迹集、可查询 evidence atom、teacher cost 的 base/evidence 分解、对 winner-versus-rival 决策边界的预算证据选择，以及带校准下置信 margin 的 pairwise tournament。论文 TeX 中对应的核心位置包括候选集与 evidence budget（约第 80 行）、HAB（约第 178 行）、训练与校准（约第 233–282 行）、nuPlan 协议（约第 291–293 行）和误差/遗憾分解（约第 721–733 行）。

当前 V61 代码是在这条主线上做的 deployment-aligned 扩展，而不是论文逐行复刻：

- 从 immutable foundation anchor warm start，只训练 proposal/family 与 set-conditioned residual 相关模块；
- proposal 训练不再以 global Top-M 作为主目标，而是使用与部署 HAB 一致的 family allocation、family-conditioned atom acquisition、interaction reserve、structural-evidence exclusion/refill；
- fast GPU HAB 覆盖全部训练场景，抽样场景使用 exact runtime HAB hard mask；
- evidence budget 为 16，proposal pool 为 24；
- 以 dense/full-interface winner preservation、boundary-focused residual correction、dual certificate、group-disjoint calibration 和 paired candidate/local/foundation evaluation 为主；
- V61 新增 translation-invariant proposal surrogate 和 proposal logit center/RMS regularization，本次错误正发生在这一新增稳定项。

## 3. 报错根因

### 3.1 直接数据流

原代码的关键路径是：

```text
family_logits
  -> softmax -> family_pi / atom_pi
  -> proposal_head logits + log(atom_pi)
  -> invalid atoms masked with finfo(dtype).min / 2
  -> proposal_logit_rms = sqrt(sum(proposal_logits^2 * active_mask) / count)
```

在 AMP 路径下，`proposal_head` 输出与 `torch.log(atom_pi)` 相加后可成为 FP32。无效 atom 随后被写成大约 `-1.7014e38`。原 RMS 计算顺序为：

```python
proposal_logits.pow(2) * active_mask
```

因此无效位置发生：

```text
(-1.7014e38)^2 -> inf
inf * 0        -> NaN
```

这也解释了报错字段的精确组合：

- mean 分支没有平方，极小哨兵乘零仍为有限的 0，所以 `proposal_logit_abs_mean` 没有被列为 non-finite；
- RMS 分支先平方，因此 `proposal_logit_rms_mean` 为 NaN；
- RMS 进入 `L_proposal_logit_stability`，再污染总 `loss`。

两个 DDP rank 在第一个 batch 同时复现，因此 `ChildFailedError` 只是 torch elastic 对子进程失败的汇总，不是根因。

### 3.2 为什么不应采用 `nan_to_num` 或跳过 batch

这类处理会掩盖 active atom 上的真实数值发散。修复必须区分：

- inactive atom 的掩码哨兵不应进入矩计算；
- active atom 若出现 `Inf/NaN`，仍应触发当前训练代码的 finite check。

## 4. 已完成的代码修改

### 4.1 `bdse/model/losses.py`

- 新增 `_masked_logit_mean_rms()`：先将 logits 转 FP32，再用 `torch.where(active, logits, 0)` 去除 inactive atom，最后计算 mean/RMS。
- `_masked_center()` 同样改为先 `torch.where`，避免极端哨兵参与 masked reduction。
- `_neg_mask_value()` 从 `finfo.min / 2` 改为足够用于 top-k/softmax 的有限值：FP16/BF16 为 `-1e4`，FP32 为 `-1e9`。
- V61 stability loss 改用安全 helper。

### 4.2 `bdse/model/bdse_model.py`

- family/proposal invalid logit 改用有限哨兵；
- family softmax 概率保持 FP32，避免先转 FP16 后小的有效概率下溢为 0；
- `log(atom_pi)` 前加下界，避免有效 family gate 出现主动 `-inf`。

### 4.3 `bdse/tests/test_v61_dehab_bfar.py`

新增回归测试：

- FP32 dtype-min 哨兵位于 inactive atom 时，mean/RMS 必须有限，inactive gradient 必须为 0；
- active atom 为 `inf` 时不得被 helper 隐藏。

### 4.4 `RUN_V61_TRAINING_SMOKE.sh`

原 wrapper 强制要求 `BDSE_VAL_CACHE_ORIGINAL`，但真正委托的 runner 使用 `BDSE_VAL_CACHE`。已兼容两个变量并统一导出。用户当前给出的启动命令不再依赖 shell 中是否残留旧变量。

### 4.5 test evaluation

- 新增 `RUN_V61_TEST_OPEN_LOOP.sh`；
- 增强 `bdse/tools/check_test_set_readiness.py`：检查 expected count、completion marker、manifest 与 diagnostics 数量一致性、manifest 身份重复、failed preprocess 比例、train/val overlap、配置一致性；
- final mode 要求完整 manifest 与 train/val manifest，且 candidate/local/foundation 必须在完全相同的 scenario/timestamp 集合上成对评估；
- provenance 写入 checkpoint/config SHA256，并显式记录：
  - `checkpoint_selected_on_test=false`
  - `calibration_fit_on_test=false`
  - `thresholds_tuned_on_test=false`
  - `full_test_evaluated`
  - `ready_for_final_claim`

## 5. 验证结果

已执行：

```text
python -m py_compile ...                         PASS
bash -n RUN_V61_TRAINING_SMOKE.sh               PASS
bash -n RUN_V61_TEST_OPEN_LOOP.sh               PASS
pytest targeted V61/AMP/training stability      13 passed
readiness synthetic complete-cache fixture      INTEGRITY_PASS_COMPLETE
uploaded aggregate diagnostics only             PRELIMINARY_PASS
```

当前环境没有用户服务器上的 `/data0/...` cache、checkpoint 和双 GPU，因此不能在这里声称完整 smoke training 已跑通；需要在原环境执行下面的命令做端到端确认。

## 6. val/test diagnostics 对比

| 指标 | val | 当前 test | 解释 |
|---|---:|---:|---|
| 样本数 | 58,418 | 67,042 | test 数量更大，但仍未证明构建完成 |
| E0 gate 失败项 | 3 / 13 | 7 / 13 | test 明显更难/更异常 |
| B16 oracle decision sufficiency | 0.9120 | 0.8399 | test 低于 0.85 gate |
| full-interface action match | 0.9657 | 0.9343 | test 低于 0.95 gate |
| runtime decision sufficiency | 0.7490 | 0.6407 | 显著下降 |
| safe candidate exists | 0.7173 | 0.5779 | candidate coverage 明显恶化 |
| teacher candidate ADE p90 | 12.915 | 17.141 | 尾部候选质量更差 |
| logged ego route distance p95 的 p90 | 3.263 m | 17.550 m | test 存在强 route/coverage 异常信号 |
| far-from-route reject rate | 0.0513 | 0.2963 | test 约 29.6% 场景受该质量项影响 |
| candidate hard violation rate | 0.5668 | 0.6700 | test 更高 |
| teacher hard violation rate | 0.2827 | 0.4221 | test 更高 |
| valid candidate count | 27.902 | 27.958 | 数量近似，问题主要不是候选数量不足 |
| quality keep rate | 0.9175 | 0.6791 | test 的有效质量覆盖明显更低 |

上传的 diagnostics 有利信号是：test 中 67,042 个 identity 全部唯一、missing-label skip 为 0、与 val 的 `config_summary` 一致。但上传的 ZIP 不含 cache manifests、预期最终样本索引、log/city 覆盖与 preprocessing completion marker，因此目前只能判定为 `PRELIMINARY_PASS`，不能证明无 train/val 泄漏或完整性。

## 7. 是否现在使用 test

### 建议

可以使用，但角色必须是 **development/shadow test**，而不是最终 test。

适合当前用途：

- 在固定 checkpoint、固定 calibration、固定阈值下做 paired candidate/local/foundation 比较；
- 判断算法改善是否仅局限于 val 分布；
- 分析失败类型、route/candidate coverage 敏感性和鲁棒性；
- 所有结论标记为 preliminary/development-only。

不允许：

- 根据 test 指标选择 epoch/checkpoint；
- 在 test 上拟合 calibration epsilon；
- 根据 test 修改 gate/阈值后仍声称它是 untouched final test；
- 只汇报当前不完整 test 的绝对分数作为论文最终性能。

### 关于“构建完成对最终性能影响应该不大”

现有证据不支持这个假设。未完成构建通常不是随机缺失；慢场景、特定 log/city、route graph 复杂场景或预处理失败场景可能在后期集中加入。当前 test 已显示 route distance、safe candidate coverage 和 hard violation 有大幅偏移，因此后续样本可能明显改变绝对值，甚至改变不同算法的相对排序。

### 最终协议

1. 继续用 `val_tune` 选 checkpoint/超参；
2. 只用 group-disjoint `val_calib` 拟合 calibration；
3. 当前 test 作为 development test 使用后，应视为已被消费；
4. 若论文需要真正 untouched final test，最好另保留一个从未看过的 final split，或在算法完全冻结后重新指定独立 holdout；
5. 若仍决定把当前 test 用作最终结果，则从某个明确时间点开始冻结方法，不再根据其结果修改算法，并在完整性检查后做一次全量 one-shot evaluation。

## 8. 运行命令

### 8.1 重新运行 smoke

```bash
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2
export BDSE_VAL_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2
export BDSE_SPLIT_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v53_split
export BDSE_TEST_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_test_v2
export FOUNDATION_CKPT=/home/senzeyu2/code/BDSE_planner/outputs_v51_far_dbap_2gpu_v1/foundation_anchor/train/bdse_v51_foundation_anchor.best.pt
export CONTROL_CKPT="$FOUNDATION_CKPT"
export GPUS=0,1

SMOKE_OUT_ROOT=outputs_v61_dehab_smoke \
TRAIN_BATCH_SIZE_PER_GPU=4 \
TRAIN_NUM_WORKERS_PER_GPU=2 \
GPUS=0,1 \
bash RUN_V61_TRAINING_SMOKE.sh
```

不再需要额外设置 `BDSE_VAL_CACHE_ORIGINAL`。

### 8.2 当前不完整 test：开发诊断

```bash
export OUT_ROOT=outputs_v61_dehab_bfar_dbap_2gpu
export TEST_DIAGNOSTICS=/path/to/dataset/diagnostics_test.json
export VAL_DIAGNOSTICS=/path/to/dataset/diagnostics_val.json

TEST_ROLE=development_test \
TEST_MAX_SCENARIOS=1000 \
TEST_WORKERS_PER_GPU=2 \
GPUS=0,1 \
bash RUN_V61_TEST_OPEN_LOOP.sh
```

输出：

```text
$OUT_ROOT/test_open_loop/development_test/test_readiness.json
$OUT_ROOT/test_open_loop/development_test/suite/parallel_open_loop_suite_report.json
$OUT_ROOT/test_open_loop/development_test/test_evaluation_provenance.json
```

### 8.3 完成构建后的冻结全量 test

先从独立的官方索引/构建计划得到预期成功样本数，而不是把当前 diagnostics 数字直接当作预期值：

```bash
export TEST_EXPECTED_SAMPLES=<完整构建的预期成功样本数>

TEST_ROLE=final_test \
GPUS=0,1 \
bash RUN_V61_TEST_OPEN_LOOP.sh
```

final mode 默认把 `TEST_MAX_SCENARIOS` 设置为 diagnostics 的完整样本数，并在 provenance 中核对实际评估数量。任何完成性、manifest、泄漏、paired-protocol 或全量覆盖检查失败都会阻止 `ready_for_final_claim=true`。
