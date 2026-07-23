# V44 双 A30 训练稳定性与速度修复说明

## 1. 上传日志中的真正中断原因

`run.log` 中只有一个致命堆栈：`torchrun` 收到外部 `SIGHUP`，随后主动向两个 worker 发送 `SIGHUP` 并退出。日志没有出现 CUDA OOM、NCCL collective failure、DataLoader worker crash 或模型 RuntimeError。

普通 `nohup bash run_v44_rads.sh &` 不能完全解决这个问题，因为 `torchrun` 会安装自己的 SIGHUP 处理器。新脚本提供 `DETACH=1`，使用 `setsid + nohup` 创建独立 session，从 SSH/父 shell 的进程组中脱离。

此外，新训练器每 `SAVE_EVERY_N_STEPS` 步原子更新 latest checkpoint，默认500步。中断后 `AUTO_RESUME=1` 会自动从 latest 恢复；新 checkpoint 能恢复到 epoch 内的 batch index，旧 checkpoint 仍可按 epoch 恢复。

## 2. 脚本配置错误/歧义

旧脚本默认 `GLOBAL_BATCH_SIZE=24`，当调用层没有正确传入 `BATCH_SIZE_PER_GPU` 时，会静默得到每卡12。上传日志实际记录的就是每卡12、全局24，而不是用户命令中的每卡4。

新脚本直接定义：

```bash
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-4}
GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NPROC_PER_NODE))
```

启动日志会再次打印最终生效值。脚本还将废弃的 `NCCL_ASYNC_ERROR_HANDLING` 改为 `TORCH_NCCL_ASYNC_ERROR_HANDLING`，并不再默认设置当前平台不支持的 `expandable_segments`。

## 3. 训练速度的主瓶颈

上传日志显示：

- epoch 0：约774.9秒，64.55 samples/s；
- epoch 1：约5604.4秒，8.92 samples/s；
- epoch 2：约5856.2秒，8.54 samples/s。

配置在 epoch 1 开始启用 predicted deployment selector。该 selector 是 stop-gradient 的 CPU/NumPy 过程，旧实现会把整个本地 batch 的多组张量同步拷回 CPU，然后逐场景运行 HAB、pair filtering 和 greedy selector。DDP 的两个 rank 都会阻塞，并在梯度同步处等待较慢 rank。

## 4. loss/selector 优化

### 4.1 轮换子批精确 selector 监督

快速默认值：

```yaml
deployment_selector_scenes_per_rank: 2
deployment_selector_every_n_steps: 2
deployment_selector_full_last_n_steps: 128
```

每两步，每个 rank 仅对轮换的2个场景执行精确 CPU selector；其他场景沿用已有 oracle curriculum mask。训练最后128步对本地全 batch 使用精确 predicted selector，使优化末尾重新对齐部署路径。

- 每卡 batch=4 时，常规阶段精确 selector 场景调用约减少4倍；
- 上传日志中的每卡 batch=12 时，约减少12倍；
- 最后128步仍为全精确。

这是一项有意识的训练目标近似，不是逐样本逐步完全等价。需要完全复现旧目标时使用：

```bash
SELECTOR_SCENES_PER_RANK=0
SELECTOR_EVERY_N_STEPS=1
SELECTOR_FULL_LAST_N_STEPS=0
```

其中 scenes=0 表示本地全 batch。

### 4.2 先在 GPU 上切片，再拷贝 CPU

精确 selector 只接收选中的场景。J0、pair delta、pair index、proposal logits、evidence metadata 等张量先在 GPU 上 `index_select`，只有小子批跨 PCIe 到 CPU。

### 4.3 移除 loss 中的逐步 CUDA 同步

旧 loss 多次使用 `if mask.sum() > 0`。Python 判断 CUDA tensor 会触发 device synchronization。新实现使用 branchless masked mean / weighted mean：

- residual Huber；
- pair ranking classification/regression；
- proposal BCE/rank；
- full-interface margin；
- hard feasibility；
- calibration；
- generic robust loss。

无有效元素时通过分母 clamp 自然返回0，不再每项同步 GPU。

### 4.4 loss meter 留在 GPU

旧训练器每个 step 都把全部 loss scalar stack 后 `.cpu().tolist()`，导致一次强制同步。新训练器在 GPU 累加，epoch 结束时只传输一次。

### 4.5 DDP/CUDA 小优化

- A30 默认启用 TF32 matmul/cudnn；
- `DDP(broadcast_buffers=False, gradient_as_bucket_view=True)`；
- 保留 AMP；
- pair uncertainty 分支继续关闭，因为部署配置的 `beta_uncertainty=0`；
- 保留较大的 forward chunk，减少 kernel launch。

## 5. 推荐运行

### 5.1 新建一套可比实验：每卡4、全局8

```bash
export BDSE_TRAIN_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2/
export BDSE_VAL_CACHE=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2/

DETACH=1 \
GPUS=0,1 \
V30_CKPT_IN=outputs_v30/train/bdse_v30_pmvrbsr.best.pt \
OUT_ROOT=outputs_v44_rads_fast_2gpu_v2 \
RUN_MODE=train_open_loop \
AUTO_RESUME=0 \
BATCH_SIZE_PER_GPU=4 \
NUM_WORKERS_PER_GPU=6 \
PREFETCH_FACTOR=2 \
SAVE_EVERY_N_STEPS=500 \
bash run_v44_rads.sh
```

### 5.2 从上传日志对应的旧输出继续

日志中的旧根目录是 `outputs_v44_rads_2gpu`，且实际每卡 batch=12。保持该 batch 可避免恢复后突然改变全局 batch：

```bash
DETACH=1 \
GPUS=0,1 \
OUT_ROOT=outputs_v44_rads_2gpu \
RUN_MODE=train_open_loop \
AUTO_RESUME=1 \
BATCH_SIZE_PER_GPU=12 \
NUM_WORKERS_PER_GPU=6 \
SAVE_EVERY_N_STEPS=500 \
bash run_v44_rads.sh
```

旧 latest checkpoint 没有 epoch 内 batch 字段时，会从最近完成的 epoch 开始；此后保存的新 checkpoint 支持 epoch 内恢复。

### 5.3 查看后台状态

```bash
cat outputs_v44_rads_fast_2gpu_v2/logs/train.pid
tail -f outputs_v44_rads_fast_2gpu_v2/logs/train_2gpu.out
```

不要再在外层额外套普通 `nohup`；使用脚本的 `DETACH=1`。

## 6. 测试

- `bash -n run_v44_rads.sh`：通过；
- 4段脚本内嵌 Python：全部编译通过；
- `python -m compileall -q bdse`：通过；
- pytest：138 passed，5个既有 Transformer nested-tensor warning。

由于本地环境没有用户的两张 A30 和50k缓存，无法给出真实端到端新吞吐。按日志分解，batch=12 的 predicted-selector 额外耗时占绝大部分，常规阶段将该场景调用减少约12倍后，理论上可从约90多分钟/epoch降到约20分钟量级；这是估算，实际值需以新日志的 `train_samples_per_second` 为准。
