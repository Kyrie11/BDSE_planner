# v44 RADS 双 A30 训练加速报告

## 结论

训练慢的首要原因不是 GPU 算力，而是 action supervision 中的部署 selector：原实现对每个 batch 的 8、16、24 三个预算分别执行一次 GPU→CPU 同步和逐场景 NumPy/Python margin-coreset 搜索。双卡 DDP 中每个 rank 都执行同样的 CPU 搜索，较慢的 rank 会让另一个 rank 在梯度同步处等待。

其次，每个 epoch 都运行 1000 场景 open-loop，并启用 dense full-interface diagnostic；这部分属于验证而非训练，却会显著拉长总运行时间。第三，配置中 `beta_uncertainty=0`，部署不使用 pair uncertainty，但训练仍为权重仅 0.02 的辅助项计算完整 pair variance head。

## 已实施的代码优化

### 1. margin-coreset 等价向量化

`bdse/planner/selector.py` 中原先逐候选调用目标函数：

- 删除阶段：每删一个原子，都逐个尝试当前所有候选；
- swap 阶段：逐个遍历 selected × removed；
- 每个尝试又执行 Python pair 循环和 action inference。

新实现把同一轮的所有删除候选或 swap 候选组成矩阵，一次完成 Huber、符号保持、winner certificate 和 action-preservation 计算。算法目标、tie-break 和最终选择不变。

随机 40 组测试中，新旧实现选择结果完全一致。64 atoms、192 pairs、2 swap passes 的 CPU 微基准：

- 原实现：约 0.938 s/次；
- 新实现：约 0.014 s/次；
- selector 核心约 65× 加速。

### 2. 跨预算监督改为双卡分层轮转

原目标是权重为 0.75、1.5、0.75 的 B=8、16、24 加权平均。新默认配置使用 `weighted_round_robin`：

- step 0：GPU0 训练 B=8，GPU1 训练 B=16；
- step 1：GPU0 训练 B=24，GPU1 训练 B=16；
- 两步循环。

因此两张卡合计的预算频率仍严格为 1:2:1，与原权重比例一致，但每个 rank 每个 batch 只运行一个 selector，而不是三个。DDP 平均梯度后，仍是原多预算目标的分层估计。

如需逐 batch 完全计算三个预算，可设置：

```yaml
training:
  deployment_budget_strategy: all
```

### 3. 训练 selector 跳过局部 swap

快速配置中 `deployment_selector_swap_passes=0`。训练仍使用同一个 signed-margin coreset 删除目标，只跳过末尾局部交换。验证与部署继续使用原配置的 2 次 swap，因此最终评估没有降级为 fast selector。

如需训练阶段也完全一致：

```yaml
training:
  deployment_selector_swap_passes: 2
```

由于 selector 已向量化，开启 2 次 swap 也比原代码快很多。

### 4. 关闭部署不用的方差分支

快速配置设置：

```yaml
training:
  compute_pair_uncertainty: false
  freeze_unused_pair_variance_head: true
  loss_weights:
    uncertainty: 0.0
```

当前 `tournament.beta_uncertainty=0`、`selector.lambda_info=0`，所以 pair variance 不参与部署 selector 或 tournament。该分支关闭后可减少 pair-head 前向和反向计算。原配置仍保留，可用于 uncertainty ablation。

### 5. 增大 A30 上的计算 chunk

默认快速配置使用：

```yaml
local_forward_atom_chunk: 128
query_projection_atom_chunk: 128
pair_forward_atom_chunk: 32
pair_forward_pair_chunk: 128
```

相比原 16×64 的 pair chunk，减少 Python 循环和 CUDA kernel launch 次数。若显存不足，可退回：

```bash
TRAIN_CONFIG=bdse/configs/v44_bdse_rads_train.yaml
```

或把 pair chunks 调回 16/64。

### 6. 训练期验证轻量化

脚本默认：

- 每 2 个 epoch 验证一次；
- 每次 256 场景；
- 不做 dense diagnostic；
- 训练结束后仍自动执行 1000 场景 open-loop。

这不会减少最终评估规模，只减少中间重复验证。需要原验证设置时：

```bash
VAL_SCENARIOS=1000 \
VAL_EVERY_N_EPOCHS=1 \
VAL_DENSE_DIAGNOSTIC=1 \
bash run_v44_rads.sh
```

### 7. 训练吞吐日志

每个 epoch 日志新增：

- `train_epoch_wall_time_s`；
- `train_samples_per_second`。

便于比较优化前后的真实吞吐，而不是只观察 GPU utilization。

## 推荐运行

```bash
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/
export BDSE_VAL_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/

GPUS=0,1 \
V30_CKPT_IN=outputs_v30/train/bdse_v30_pmvrbsr.best.pt \
OUT_ROOT=outputs_v44_rads_fast_2gpu \
RUN_MODE=train_open_loop \
BATCH_SIZE_PER_GPU=4 \
NUM_WORKERS_PER_GPU=6 \
PREFETCH_FACTOR=2 \
bash run_v44_rads.sh
```

## 精确训练模式

使用向量化 selector，但恢复全部三个预算、pair uncertainty、2 次 swap 和完整逐 epoch 验证：

```bash
GPUS=0,1 \
TRAIN_CONFIG=bdse/configs/v44_bdse_rads_train.yaml \
VAL_SCENARIOS=1000 \
VAL_EVERY_N_EPOCHS=1 \
VAL_DENSE_DIAGNOSTIC=1 \
SAVE_EVERY_N_EPOCHS=1 \
RUN_MODE=train_open_loop \
OUT_ROOT=outputs_v44_rads_exact_2gpu \
bash run_v44_rads.sh
```

即使使用精确模式，向量化 coreset 仍会显著减少 CPU selector 时间。

## 已完成测试

```text
133 passed, 5 warnings
```

警告均来自 PyTorch Transformer nested tensor 提示，与本次修改无关。
