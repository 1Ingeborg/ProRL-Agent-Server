# ProRL-Agent-Server

用在线 GRPO 训练 SWE-agent（Qwen3.5-4B，SWE-smith 任务）。训练和 agent rollout 分别跑在两台机器上：

- **训练端（GPU）**：slime + Megatron + SGLang，colocate 模式，同一批 GPU 交替做推理和训练。
- **rollout 端（CPU + Docker）**：Polar 的 rollout/gateway 服务，用 SWE-agent 在任务容器里修 bug，再由 verifier 跑测试给出奖励。

两边通过 SSH 隧道互通：训练端访问 rollout 服务（`127.0.0.1:18080` 和 `18100`），rollout 端的 agent 调用训练端的 SGLang router。

本仓库基于 [NVIDIA-NeMo/ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)（Polar，commit `45feeff`），许可证为 Apache-2.0。

## 目录

```
src/polar/          Polar：rollout 服务、gateway、runtime、trajectory 构建、evaluator
src/slime_bridge/   slime ↔ Polar：提交 rollout 任务、轨迹转成训练样本、奖励
adapters/           SWE-agent 接入：harness / runtime / runner / gateway bridge / verifier，
                    remote_sync_rollout（slime 的 rollout 函数），qwen35_safe_attention（模型 spec）
examples/swegym_slime_grpo/   Qwen3.5-4B 的 Megatron 参数与 HF → torch_dist 权重转换
scripts/            启动脚本、服务管理、隧道、slime/SGLang 补丁、checkpoint 拉取
config/             拓扑模板与训练配置示例
```

## 依赖版本

| 组件 | 版本 |
|---|---|
| slime | `bf14dc2`（v0.3.0），再应用 `docker/patch/latest/megatron.patch` 到 Megatron |
| Megatron-LM | `1dcf0da` |
| SGLang | 0.5.13（sglang-kernel 0.4.3+cu129） |
| torch | 2.11.0+cu129 |
| transformers / TE / FLA / mbridge | 5.8.1 / 2.10.0 / 0.5.0 / 0.15.1 |
| ray / torch_memory_saver | 2.59.0 / 0.0.10 |

rollout 端另外需要 [SWE-agent](https://github.com/SWE-agent/SWE-agent) 1.1.0 源码（`repos/SWE-agent`）以及 SWE-smith 任务的 Docker 镜像。

slime 和 Megatron 放在 `repos/slime`、`repos/Megatron-LM`，安装后执行一次：

```bash
bash scripts/patch/patch_sglang_0513_token_metadata.sh
SLIME_DIR=repos/slime bash scripts/patch/patch_slime_router_tokens.sh
SLIME_DIR=repos/slime bash scripts/patch/patch_slime_training_fixes.sh
```

## 数据

`scripts/prepare_remote_training.py` 根据 SWE-smith 任务生成 `datasets/slime/{train,val}.jsonl` 和 `manifest.json`。每一行是一个 Polar 任务 spec，指定 runtime、agent、evaluator。

## 运行

**rollout 端**

```bash
cp config/model.env.example config/model.env   # 填 MODEL_BASE_URL 等
scripts/services.sh start                       # serve_rollout + serve_gateway
python scripts/managed_job.py rl-bridge start   # SWE-agent ↔ gateway 桥
cp config/remote-tunnel.example.json config/remote-tunnel.json
python scripts/remote_tunnel.py start           # 到训练端的 SSH 隧道（不会自动重连）
```

**训练端：300 条数据 demo**

`config/full300-8a800.env.example` 用 `train.jsonl` 里的全部 300 题跑一个 epoch：每轮 12 题、每题采样 8 次、共 25 轮。训练前先做一次基线验证，之后在第 7、14、21、25 轮后各验证一次并保存模型。

```bash
bash scripts/autodl_train.sh convert                                  # 只需一次：HF 权重 → torch_dist
cp config/full300-8a800.env.example config/full300-8a800.env
bash scripts/autodl_train_full.sh plan                                # 打印完整的 train.py 命令
RUN_ID=$(date +%Y%m%dT%H%M%S)-full300 bash scripts/autodl_train_full.sh start
bash scripts/autodl_train.sh status                                   # stop 用来停止
```

启动前会做预检：GPU 数量、数据的 sha256 与 `manifest.json` 是否一致、full300 是否正好覆盖 300 题、reference 权重是否已转换、磁盘空间是否足够、rollout 服务（`127.0.0.1:18080/health`）是否可达。

每次启动都要换一个新的 `RUN_ID`，因为任务 ID 由它生成，复用会在 rollout 服务上冲突。日志在 `runs/training/<RUN_ID>/job.readable.log`，TensorBoard 日志在 `runs/tensorboard/<RUN_ID>`。验证集的 pass@1 是日志中 `eval <N>:` 那行里的 `polar/eval/resolved_rate`。

在 8×A800 上，每轮 rollout 约 15 分钟，加上训练每轮约 37 分钟，加上 5 次验证，跑完一个 epoch 大约需要 16–17 小时。

**checkpoint**：在 rollout 端运行 `scripts/pull_checkpoints.sh <RUN_ID> 24`。它会在每次验证结束后，把对应的 `iter_N` 拉过来并做校验；校验通过后删除训练端的副本，最后一份（iter 24）两边都保留。

## 主要配置

- 8 卡 colocate，TP2 CP4；每轮 12 题 × 8 采样，GRPO，lr 1e-6，KL 0.001。
- `MAX_TOKENS_PER_GPU × CP` 是单条轨迹的长度上限，超过的轨迹会被丢弃；`POLAR_MIN_COMPLETE_ACCEPT_FRACTION` 规定每组至少要保留的比例。
- 验证集每题采样 1 次算 pass@1（`polar/eval/resolved_rate`），每隔 `EVAL_INTERVAL` 轮验证并保存一次；epoch 结束时一定会验证。
- agent（由 `prepare_remote_training.py` 写入每条数据）：最多 80 次调用，上下文最多 73k token、单次输出 8k token（两者相加不超过 81920 的单条样本上限），单次工具输出截断到 2 万字符。
- rollout 端（`config/topology.template.yaml`）：最多 96 个 agent 同时运行，正好是一轮的 12 × 8 条采样。
