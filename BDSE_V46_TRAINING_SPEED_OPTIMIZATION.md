# BDSE v46 AOCC 双卡训练性能分析与等价优化

## 1. 当前命令实际执行的训练路径

`run_v46_aocc.sh` 使用 `torchrun --nproc_per_node=2` 启动两张 GPU 的 DDP 训练，并启用 AMP。用户给定参数下：

- 每卡 batch size：4；全局 batch size：8。
- `deployment_selector_backend=exact_cpu`。
- `SELECTOR_SCENES_PER_RANK=0`：每个 rank 的 4 个场景都运行 exact selector。
- `SELECTOR_EVERY_N_STEPS=1`：每一步都运行。
- `deployment_budget_strategy=primary_plus_aux`：每步运行主预算和一个辅助预算，共 2 个预算。
- 因此每个 optimizer step，每个 rank 要处理 `4 × 2 = 8` 个“场景-预算”选择任务，两张卡共 16 个；DDP 会等待较慢的 rank。
- 每 500 step 保存一次完整训练 checkpoint。
- 每 2 epoch 对 1000 个验证场景做 open-loop validation，并额外启用 dense diagnostic。
- `RUN_MODE=train_open_loop` 在训练结束后还会用两张 GPU 分片执行一次 open-loop 测试。

## 2. 已定位的主要瓶颈

### 2.1 AOCC 选择完成后重复计算未使用的 action-rank utility

v46 的 selector 配置为：

- `force_fill_budget=false`
- `min_selected_atoms=0`
- `mandatory_hard_quota=0`，且结构安全证据不占决策预算
- `decision_family_quota=0`
- `interaction_family_quota=0`
- `soft_interaction_quota=0`
- `proposal_fill_weight=0`
- `direction_invariant_interaction_weight=0`

在这些条件下，AOCC 输出的选择集合不会经过 post-fill、配额补齐或 utility 排序修改。但原代码仍在每个场景、每个预算后无条件调用 `_action_rank_atom_utility`，重新扫描 atom-pair/action 关系；该结果随后没有参与选择。

这是当前 exact selector CPU 路径中最大的重复计算。

### 2.2 多预算重复执行相同 GPU→CPU 搬运和 AOCC 预算无关部分

主预算和辅助预算使用完全相同的模型输出、pair graph、mask、成本和 metadata，只有 budget 不同。原代码为每个预算分别：

1. 切片 batch；
2. 将相同 tensor 拷贝到 CPU；
3. 重建相同的 AOCC 候选状态和嵌套贪心顺序；
4. 再按预算截断。

AOCC 的 nested greedy order 与预算无关，预算只决定从同一顺序中能物化多少 atom。因此这些工作可以安全复用。

### 2.3 cycle consistency 中的细粒度 CUDA 同步

原 `_pair_cycle_consistency_loss` 在 Python 三角形枚举循环中频繁执行：

- `bool(cuda_tensor)`
- `tensor.item()`
- 对每条边 margin 再执行 `.item()`

每次都会迫使 CPU 等待 CUDA stream。每场景最多选择 64 个三角形，候选构建过程会触发大量同步，GPU 利用率容易呈现“计算一小段—等待 CPU—再计算”的锯齿。

### 2.4 每个 loss scalar 单独做 finite check

v46 返回二十多个标量 loss/metric。原训练循环对每个标量分别执行 `torch.isfinite(...).item()`，常见路径每步产生二十多次 GPU→CPU 同步，然后才执行 DDP finite 状态同步。

### 2.5 训练外的固定墙钟开销

以下不是模型前向/反向本身，但会显著拉长总运行时间：

- `SAVE_EVERY_N_STEPS=500`：序列化 model、optimizer、GradScaler 等，并伴随 DDP 同步和磁盘写入。
- `VAL_SCENARIOS=1000`、`VAL_EVERY_N_EPOCHS=2`、`VAL_DENSE_DIAGNOSTIC=1`：验证时除了预算化 open-loop，还执行 dense full-interface scoring。
- 训练完成后的 1000 场景 open-loop 测试。

这些参数在优化代码中全部保持原样，以保证用户命令的训练/测试流程不变。

## 3. 代码优化内容

### 3.1 按需计算 post-fill utility

文件：`bdse/planner/selector.py`

新增严格条件判断。仅当下列任一机制确实可能改变选择集合时，才计算 action-rank / flip utility：

- 存在 active mandatory atom；
- 任一 family/interaction quota 大于 0；
- `force_fill_budget=true`；
- 当前选择数量不足 `min_selected_atoms`。

v46 当前配置会跳过该无效计算。其他实验配置只要启用了上述任一机制，就走原来的 utility 计算路径，保持通用行为。

同时，在无需 fill 时不再对所有 active atom 构造和排序 `filler_order`。

### 3.2 多预算只做一次 CPU 数据快照

文件：`bdse/model/losses.py`

新增：

- `_build_predicted_pair_numpy_cache`
- `_predicted_pair_certificate_masks_multi_budget`

同一步内的多个 budget 共用一次 CPU 输入缓存，不再重复搬运相同输出和 batch 数据。

### 3.3 复用 AOCC nested greedy state

文件：`bdse/planner/selector.py`

将 AOCC 拆分为：

- 构建预算无关状态与 nested greedy order；
- 对具体 budget 物化选择结果。

每个场景在主预算与辅助预算之间共享状态。缓存带有完整数组和标量参数一致性检查；只有输入逐元素完全一致时才复用，否则自动重建，避免错误命中。

### 3.4 批量构建 cycle consistency 离散拓扑

文件：`bdse/model/losses.py`

将 pair topology、mask、valid action、target 和 detached margin 作为一次批量 CPU 快照阶段处理，在 CPU 上完成离散三角形选择。随后从原始 GPU `predicted_margin` 中批量 gather 被选中的边并计算 Huber loss。

保持：

- later duplicate edge overwrite 语义；
- 反向边取负；
- triangle 排序规则；
- `max_triangles_per_scene`；
- loss 数值；
- 对原始 margin 的梯度。

### 3.5 聚合 finite check

文件：`bdse/experiments/train.py`

所有 scalar loss 的 `isfinite` predicate 在设备端 stack 后一次 `all()`。正常训练路径只需一次最终同步；只有真的出现非有限值时，才逐项生成错误名称。

DDP 的跨 rank `MIN all_reduce` 和“任一 rank 非有限即全体退出”的安全语义保持不变。

## 4. 等价性验证

新增测试：`bdse/tests/test_v46_training_speed_exactness.py`

覆盖：

1. 多预算复用输出 mask 与原逐预算调用完全相等；
2. AOCC state cache 与无 cache 的 selected set、objective 和关键 diagnostics 相等；
3. 新 cycle loss 与 legacy 实现的 loss tensor 及梯度逐元素相等；
4. 聚合 finite check 正确识别非有限 loss；
5. v46 无 post-fill 配置下，确认不会调用未使用的 action-rank utility。

完整测试结果：

```text
152 passed, 5 warnings
```

warnings 均为已有 PyTorch Transformer nested-tensor 提示，与本次优化无关。

## 5. 本地热点基准

当前执行环境没有 CUDA，也没有用户的 nuPlan cache，因此无法提供真实双卡端到端吞吐数字。对与 v46 尺寸接近的 synthetic exact-selector CPU 热点进行相同输入比较：

- 64 actions
- 64 atoms
- 192 pairs
- 同一步 2 个 budgets
- 原实现：124.419 ms / 两预算
- 优化实现：2.385 ms / 两预算
- 选择集合完全相同
- 该热点约 52.2×，耗时下降约 98.1%

该数字只表示 selector CPU 热点，不等于整体训练 52×。整体收益取决于模型 GPU 前后向、数据读取、validation 和 checkpoint 在总墙钟中的占比。真实环境应重点观察训练日志中的：

- `selector_exact_wall_time_s`
- epoch `loss` stage wall time
- samples/s
- GPU utilization
- checkpoint 和 validation 前后的墙钟跳变

## 6. 使用方法

无需更改原命令。解压优化包并在代码根目录执行原来的 `bash run_v46_aocc.sh` 即可。训练仍使用两张 GPU，预算、损失权重、exact selector、AMP、验证、保存和 open-loop 测试参数均保持不变。

若后续只希望进一步缩短墙钟、并能接受改变“中间评估/容灾频率”而不改变梯度目标，可单独考虑增大 `SAVE_EVERY_N_STEPS` 或关闭 `VAL_DENSE_DIAGNOSTIC`；本次交付没有替用户修改这些行为。
