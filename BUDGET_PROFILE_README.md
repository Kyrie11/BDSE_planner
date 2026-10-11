# V64.3.56 在线预算测量扩展

本扩展保留上传版本的原始源文件、权重逻辑、训练脚本和科学 manifest。新增 opt-in profiler 与独立 nuPlan subclass；只有调用新入口才测量。原 `RUN_V64_3_56_EAF_ICER_RCPR_TRAIN.sh` 是历史诊断/分支拟合，不能用它的训练或 probe 时长代表在线规划压力。

## 测量范围

1. `RUN_BDSE_BUDGET_PROFILE.sh`：从 NPZ 读取过去/当前 runtime fields；重新生成候选和 evidence，执行 frozen deployment core。batch=1、单进程、单 GPU。NPZ I/O 单独报告，不计为 planning core service。
2. `RUN_BDSE_RUNTIME_BUDGET_PROFILE.sh`：实际 nuPlan call，含 runtime feature 构造、core、输出轨迹转换；不含外部传感器/感知、arrival queue、后续发布和 controller。
3. `MODE=native`：仅完整调用开始/结束同步 GPU，用于服务延迟统计。
4. `MODE=attribution`：另在 model tensorize/encoder/sparse action/sparse pair、certificate prediction/stage 边界同步。可解释归因，但屏障改变 scheduling，不能据此声称原生实时尾延迟。

deadline 仅用于统计，没有实际超时丢弃、延迟注入或 fallback。FIFO replay 仅为单服务器周期到达假设下的诊断，没有推进仿真世界。

## 环境与输入

在原代码的 PyTorch/nuPlan 环境运行。core profile 不需要 nuPlan simulation，但原 core 的 import 依赖 PyTorch，即使 `--rule-only` 也如此。`--list-only`、trace summary 和多数 unittest 不需要 PyTorch。

指定 frozen deployment RSMR config 和 EAF checkpoint；不要用 treatment/control 或 `selected_outcome_probe.enabled=true` 配置。示例路径中的 outputs 根目录换成自己的实际位置：

```bash
export OWN_CONFIG=/实际outputs根目录/outputs_v64_3_47_eaf_icer_fsfr_screen_2gpu_v1/provenance/v64_3_47_rsmr.yaml
export EAF_CKPT=/实际outputs根目录/outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1/train/checkpoints/bdse_v64_saqa_bcc.epoch_0004.pt
```

此处 epoch_0004 来自既有冻结链路，不是假定 V56 训练出了新 checkpoint。如果自己的冻结 artifact 位置不同，以部署使用的实际 config/checkpoint 为准。RSMR config 内可能还有 runtime fit 的绝对路径，需确保这些依赖仍存在。

## 第一步：目标环境验收

```bash
python -m unittest bdse.tests.test_planning_budget_unittest -v
```

应看到 13 项全部通过；若 PyTorch 缺失，原 core trajectory invariance 检查会跳过。它使用真实原 core 的 CPU rule path 验证包裹前后结果一致，不验证 learned checkpoint 或 nuPlan integration。

先做 learned core smoke：

```bash
CACHE_ROOT=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2 \
LIMIT=10 WARMUP=2 BUDGETS="16" PROPOSALS="24" \
PROFILE_GPU=0 MODE=native \
bash RUN_BDSE_BUDGET_PROFILE.sh
```

核对 `provenance.json` 的 GPU、checkpoint、config、rule_only=false；trace 的 core_calls=1、actual query counts 与 frozen deployment 相符。再用相同 workload 跑 attribution，检查 action/trajectory 是否与未包裹 frozen planner 相符。source 没改不代表可以免除实际设备验收。

## 第二步：固定 M 扫 B

```bash
CACHE_ROOT=/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2 \
LIMIT=1000 WARMUP=20 MODE=native PROFILE_GPU=0 \
bash RUN_BDSE_BUDGET_PROFILE.sh
```

默认 B=8/16/24，M=24/24/24，共用相同抽样 manifest；原冻结权重不重训。因此为接口/成本和跨预算鲁棒性实验，不是每预算最佳质量实验。

输出目录默认 timestamp，必须是新目录。可以显式给 `PROFILE_ROOT=/新输出路径`，禁止混入历史 trace。trace 每进程一个文件，多线程行写入有锁。

首轮 1000 帧用于 screening；正式 p99/rare miss 需要更多独立 logs、完整压力子集和重复硬件运行。选择策略按采集时间父目录作为 log，随机打乱后 round-robin；不是自然全体分布的无偏抽样。正式总体结论另跑全体或带明确分层权重。

## 第三步：固定 B 扫 M；另测接口扩张

```bash
BUDGETS="16 16 16" PROPOSALS="24 48 128" \
MODE=native LIMIT=1000 PROFILE_GPU=0 \
bash RUN_BDSE_BUDGET_PROFILE.sh

BUDGETS="16 32 64 128" PROPOSALS="24 64 128 128" \
MODE=native LIMIT=1000 PROFILE_GPU=0 \
bash RUN_BDSE_BUDGET_PROFILE.sh
```

M 是 proposal acquisition 范围，B 是保留接口。若实际计算量由 M 主导，仅扫 B 不一定省时间。config 的其他 selection/admissibility cap 仍生效，不能凭 B=128 宣称已计算全部物理 evidence；查 trace 中执行计数。

改变 M/B 也可能改变 ranking/support 的训练分布。用于质量比较时注明 frozen cross-interface，并给 matched training/calibration 或单独成本表。

## 第四步：GPU 同步归因

```bash
MODE=attribution LIMIT=200 PROFILE_GPU=0 \
bash RUN_BDSE_BUDGET_PROFILE.sh
```

`spans` 有 inclusive/exclusive，避免把 encoder 包含在 prediction 中又重复求和。已有 `legacy_host_timings_not_cuda_attributed` 仅作 CPU/host 路径参考；其 GPU 阶段没有新增逐段同步，不能用来准确划分 GPU 花销。候选、evidence enumeration 等旧计时是补充诊断，并非全新的精确 profiler。

默认 shell 保持单 GPU 单 job，不能同时在同卡跑 train、paired control/treatment 或多个场景以模拟部署；多 job 用于研究仿真 throughput 的实验另表报告。

## 可直接用的 Python CLI

只导出 workload，不加载模型：

```bash
python -m bdse.tools.profile_planning_budget \
  --cache-root /data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2 \
  --subdirs train_boston train_pittsburgh train_singapore train_vegas_2 \
  --config "$OWN_CONFIG" --limit 2000 --list-only --output outputs/profile_workload
```

复用同一 workload：

```bash
python -m bdse.tools.profile_planning_budget \
  --cache-root /data0/senzeyu2/dataset/nuplan/data/cache/bdse_train_v2 \
  --manifest outputs/profile_workload/workload_manifest.json \
  --config "$OWN_CONFIG" --checkpoint "$EAF_CKPT" \
  --device cuda:0 --budget 16 --proposal-top-m 24 \
  --mode native --deadline-ms 100 --warmup 20 \
  --output outputs/profile_B16_M24_newrun
```

--deadline-ms 是描述性比较阈值，不能从它自动声称系统有 100 ms 必须满足的真实 deadline。--repeats 用于 warm repeat，重复同场景不增加独立样本。--limit 0 全量扫描。--rule-only --device cpu 仅用于工程 smoke，禁止用于 learned planner 预算主张。

## 实际 nuPlan call

NPZ 无法替代 raw DB/maps。使用既有闭环链路能够匹配 raw DB 的场景 token manifest；若用 cache 中 token，需要审计 extraction 起点、DB split 和匹配成功率，不能只凭名称假定都可用。

```bash
export NUPLAN_DB_ROOT=/真实raw_DB根目录
export NUPLAN_MAP_ROOT=/真实maps目录
export SCENARIO_TOKENS_FILE=/匹配raw_DB的场景tokens.txt

PROFILE_GPU=0 BUDGET=16 PROPOSAL_TOP_M=24 \
MODE=native CL_LIMIT=100 REPLAN_INTERVAL_TICKS=1 \
bash RUN_BDSE_RUNTIME_BUDGET_PROFILE.sh
```

另跑不指定 REPLAN_INTERVAL_TICKS 的 frozen 原设置。强制 interval=1 是主动行为/频率消融，不等价于复现既有闭环质量。trace 将 cached 与真正 core replan 分开。当前 wrapper 中 cached local trajectory 的时间/坐标复用语义也需单独审核，不能因缓存很快就称 safe high-frequency planning。

`DRY_RUN=1` 只打印命令，尚未检验实际 Hydra composition、DB 可用性或完整 simulation。launcher 使用 metric-safe simulation wrapper，单 worker 防止 scenario 并发干扰计时。

若需要手工集成原 evaluate 入口，raw Hydra override 为：

```text
planner.bdse_planner._target_=bdse.planner.profiled_nuplan_planner.ProfiledBDSEnuPlanPlanner
```

然后设置 `BDSE_BUDGET_TRACE_PREFIX`、`BDSE_BUDGET_PROFILE_MODE=native`。没有 prefix 时 subclass 直接调用原 planner。

## 结果解读

```bash
python -m bdse.tools.summarize_planning_budget \
  --traces /本次trace目录 \
  --deadlines-ms 50 100 200 --periods-ms 100 200 \
  --output outputs/budget_summary_newrun.json
```

只给本 profiler 的目录或文件，不要把混有其他工具 JSONL 的全 outputs 树输入 summary。未知 schema、坏行与 failed call 均报告，不静默伪装为成功；异常停止的 simulation 仍需核对整体完成率。

关键字段：

- `wall_ms`: 相应 scope 的一次同步完成服务时间。
- `trajectory_sha256`、`trajectory_shape`：core 输出轨迹身份，哈希在计时结束后计算；便于匹配同输入/预算的 native 与 attribution 结果。缓存复用帧没有 core 轨迹哈希。
- `pre_boundary_sync_ms`: 开始前清空该 device 的等待，排除于 wall_ms；不是已测真实 arrival queue。
- `counts.total_sparse_query_count`: 实际查询计数，和 effective_query_count/retained budget 分开看。
- `counts.proposal_atom_count`: proposal 实际范围。
- `runtime_shape`: slots、active agents、history 形状、route 点数；固定 padding 可能使自然密度与 encoder 时长脱钩。
- `warmup_wall_ms`: warmup 条目（core 初始包含冷调用）；nuPlan 多场景每实例初始调用不等于完全冷模型启动。
- `cuda_peak_allocated_bytes`: 进程级分配峰值，不是每帧峰值。
- `deadline_sensitivity_service_only`: 描述性超时比例，不含 arrival queue/control/publish。
- `periodic_FIFO_diagnostics_not_real_latency`: 理想周期、单 FIFO、不 drop 的假设；序列可能不是完整真实帧流。

分别比较 native 与 attribution，单卡/并发、cache core/完整 planner、cached/replan、cold/warm。不要把它们混为一张平均 latency 表。profile trace 写出耗时不计入服务时间；原可选 diagnostics 写出若在 planner 内开启则会计入，应一致配置并单独说明。

单次 timeout 百分比不证明 closed-loop safety harm。下一阶段 latency-aware simulator 必须让世界在等待时继续推进并运行合法旧控制，对齐发布时间的绝对坐标 trajectory；`sleep` 同步 simulator 只会减慢实验。

## 本次可验证范围

本开发环境没有用户 cache/checkpoint、PyTorch/nuPlan/GPU。13 项 unittest 中 12 passed、1 skipped（真实原 core），编译/两份 shell 语法、manifest/summary smoke、nuPlan dry-run passed。未执行 learned inference、完整 simulation 或 GPU stage profile。该状态不能替代目标环境验收。

原文件 SHA 比对保持一致；新增测量扩展不会回写历史结果。研究方案与文献判断见另附 `BDSE_story_and_budget_implementation.md`。
