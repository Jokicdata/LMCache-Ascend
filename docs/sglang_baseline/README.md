# SGLang 基线部署与 TTFT 压测手把手教程（Qwen3.8-27B × 8×910B4）

> 目标：在接入 LMCache 之前，先跑通一个**纯净的 SGLang 基线**——安装、起容器、启动推理服务、用 benchmark 量出 TTFT/TPOT/吞吐。后续所有优化（LMCache MP mode 适配等）都以这套基线数据作为对照组。
>
> 环境约定：宿主机 8×Ascend 910B4（A2），模型 `/workspace/models/Qwen3.8-27B`，容器内同路径挂载。

---

## 0. 前置认知（1 分钟）

| 组件 | 角色 |
|---|---|
| 宿主机 NPU 驱动 + 固件 | 硬件层，`npu-smi info` 能看到 8 张卡即可 |
| CANN | 昇腾算子库（镜像内已内置，宿主机不用装） |
| SGLang 官方镜像 | 已集成 torch-npu、sgl-kernel-npu、ascend attention backend，**拉下来就能用** |
| Qwen3.8-27B | 混合 Mamba（GDN）+ 全注意力架构，27B 级别，TP8 部署 |

为什么不源码安装：SGLang NPU 版依赖 torch-npu、triton-ascend、sgl-kernel-npu、CANN custom_ops 多个组件的版本对齐，镜像全部打包好了，源码安装只在你需要改 SGLang 代码时再走。

---

## 1. 宿主机检查

```bash
# 1) 确认 8 张卡都在、驱动正常
npu-smi info

# 2) 确认设备节点存在
ls /dev/davinci*     # 应有 davinci0~7、davinci_manager、devmm_svm、hisi_hdc

# 3) 确认模型目录
ls /workspace/models/Qwen3.8-27B/   # 应有 *.safetensors、config.json、tokenizer 等
```

---

## 2. 拉取镜像并启动容器

### 2.1 选镜像

910B（A2）与 910C（A3）镜像不通用，**务必选 `910b` 后缀**：

```bash
# A2 (910B) 开发镜像（滚动更新）
export SGLANG_IMAGE=quay.io/ascend/sglang:main-cann9.0.0-910b
# 稳定版可在 https://quay.io/repository/ascend/sglang 找 v0.5.x-cann9.x.0-910b 形式的 tag，
# 例如（以 quay 上实际存在的 tag 为准）：
# export SGLANG_IMAGE=quay.io/ascend/sglang:v0.5.16-cann9.0.0-910b

docker pull "$SGLANG_IMAGE"
```

> 官方 SGLang 文档以 v0.5.16 为基准验证 Qwen3 系列部署；若追求稳定优先选稳定版 tag。

### 2.2 启动容器

```bash
docker run -itd \
  --name sglang-npu \
  --shm-size=100g \
  --net=host \
  --ipc=host \
  --privileged=true \
  -v /workspace:/workspace \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/queue_schedule:/var/queue_schedule \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --entrypoint=bash \
  "$SGLANG_IMAGE"

docker exec -it sglang-npu bash   # 进入容器
```

关键点：
- `--shm-size=100g`：SGLang 多进程（TokenizerManager / Scheduler / Detokenizer）走共享内存 IPC，太小会 OOM。**TP8 必须给够**。
- `--net=host`：直接用宿主机端口，省去 -p 映射。
- 8 个 `davinci` 节点 + `davinci_manager/devmm_svm/hisi_hdc` 是昇腾容器标准挂法。

### 2.3 容器内验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 部分镜像 entrypoint 已 source，重复执行无害
python -c "import torch, torch_npu; print(torch_npu.npu.device_count())"  # 期望输出: 8
python -c "import sglang; print(sglang.__version__)"
```

---

## 3. 启动推理服务

### 3.1 环境变量 + 启动命令

在容器内执行：

```bash
# 昇腾侧常规调优变量（与 vLLM 用法一致）
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export OMP_NUM_THREADS=8
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

python -m sglang.launch_server \
  --model-path /workspace/models/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --tp-size 8 \
  --host 0.0.0.0 \
  --port 8998 \
  --dtype bfloat16 \
  --context-length 8192 \
  --chunked-prefill-size 4096 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  2>&1 | tee sglang_server.log
```

参数对照（与 vLLM 的迁移映射）：

| SGLang 参数 | vLLM 等价物 | 说明 |
|---|---|---|
| `--tp-size 8` | `--tensor-parallel-size 8` | 27B 权重切 8 份 |
| `--context-length 8192` | `--max-model-len 8192` | 先短上下文跑通基线 |
| `--chunked-prefill-size 4096` | `--max-num-batched-tokens` | Sarathi 式分片 prefill |
| `--mem-fraction-static 0.85` | `--gpu-memory-utilization` | 静态显存占比（权重+KV池） |
| `--disable-cuda-graph` | `--enforce-eager` | **默认不开**，NPU Graph 默认启用 |
| （默认开启） | `--enable-prefix-caching` | RadixAttention 默认生效 |

### 3.2 混合 Mamba 模型注意项

Qwen3.8-27B 属于 Qwen3.5/3.6/3.8 混合架构（GDN 层与全注意力层交错）。如果上面的标准命令启动报错，按顺序排查：

1. **架构未被当前 SGLang 版本收录** → 升级镜像 tag 或到 SGLang ascend 支持矩阵确认 Qwen3.8 支持状态；
2. **KV 布局类报错** → 尝试显式 `--disable-cuda-graph`（排除 NPU Graph 捕获对混合层的兼容性）；
3. **启动日志中出现 attention block size 相关行** → 记下该值，后续接 LMCache MP mode 时它决定 `--chunk-size`（Qwen3.8-27B 在 vllm-ascend v0.30 上的经验值是 896，见 `docs/mp_mode/qwen3_5.md`）。

### 3.3 启动成功的标志

日志末尾出现：

```
The server is fired up and ready to roll!
```

另开一个终端看进程拓扑（MP mode 的实际形态，8 个 Scheduler 进程各绑一张卡）：

```bash
docker exec sglang-npu ps -ef | grep sglang | head
```

---

## 4. 功能验证

```bash
curl -s http://127.0.0.1:8998/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B",
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

正常返回 JSON，`choices[0].message.content` 有内容。去掉 `chat_template_kwargs` 即开启思考模式。

服务自带 Prometheus 指标端点（观测 batch/队列的 Level-2 工具）：

```bash
curl -s http://127.0.0.1:8998/metrics | grep -E "num_running_req|num_queue_req|token_usage"
```

---

## 5. TTFT Benchmark（vLLM 式压测，SGLang 自带）

### 5.1 随机长度数据集（控制变量实验首选）

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 8998 \
  --model Qwen3.8-27B \
  --dataset-name random \
  --random-input 1024 --random-output 256 \
  --random-range-ratio 1 \
  --num-prompts 32 \
  --max-concurrency 4 \
  --seed 42 \
  2>&1 | tee bench_random_c4.log
```

- `--random-range-ratio 1`：输入长度在 [1,1024] 均匀采样 → **长短混合负载**，最能暴露调度差异（chunked prefill 的受益场景）。
- `--seed 42`：可复现，A/B 对比的前提。

### 5.2 ShareGPT 真实数据集（贴近生产）

```bash
python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 8998 \
  --model Qwen3.8-27B \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 64 \
  --max-concurrency 8 \
  --seed 42
```

### 5.3 读输出：TTFT 在哪一行

```
============ Serving Benchmark Result ============
Benchmark duration:            ...
Total input tokens:            ...
Successful requests:           ...
TTFT (s):   mean=0.xxx  median=0.xxx  p99=1.xxx   ← 首token延迟（本教程主角）
TPOT (s):   mean=0.xxx  median=0.xxx  p99=0.xxx   ← 每输出token延迟
ITL (s):    mean=0.xxx  ...
Output token throughput:  xxx tok/s
==================================================
```

### 5.4 基线实验矩阵（建议照跑）

| 实验 | 改动量 | 看什么 |
|---|---|---|
| baseline：上面原样命令 | - | TTFT/TPOT/吞吐基准 |
| 关闭 NPU Graph | 加 `--disable-cuda-graph` | 图捕获对 TPOT 的收益 |
| 并发扫描 | `--max-concurrency` 4/8/16 | 服务容量拐点（TTFT p99 何时起飞） |
| 关闭 RadixAttention | 加 `--disable-radix-cache` | 前缀复用对 TTFT 的贡献 |

所有结果 log 存到固定目录（如 `bench_results/`），文件名带参数含义，之后接 LMCache 时直接对照。

---

## 6. 本仓库 `benchmark/` 目录使用指南

仓库自带两套压测工具，**注意它们是 vLLM/内嵌 connector 时代的产物，本教程的 SGLang 基线用不到**，但后续 CacheBlend 实验会用到，先学会用法。

### 6.1 `benchmark/v1/rag/` —— RAG 场景 LMCache/CacheBlend 压测

**定位**：模拟 RAG 负载（多文档 + 问题），对比 **LMCache 前缀缓存** vs **CacheBlend** 的吞吐、平均 TTFT、生成质量。基于 vllm-ascend v0.18.0 开发验证。

| 文件 | 作用 |
|---|---|
| `rag.py` | 主脚本。两种模式：默认**离线**（KV 预计算+请求在同一实例内跑完）；`--online` 则作为客户端向已启动的服务发请求 |
| `online.sh` | 一键：自动起 vLLM 服务（挂 `LMCacheAscendConnectorV1Dynamic` 内嵌 connector）→ 跑 rag.py → 关服务 → 出 CSV → 自动调 analysis.py 画图。支持 `server/client/both` 三种参数 |
| `offline.sh` | 一键离线跑两个场景（`config/lmcache.yaml` 与 `config/lmcache_blend.yaml`），产出两份 CSV |
| `analysis.py` | 读取 CSV 汇总对比 + 出图 |
| `config/lmcache.yaml` | LMCache 配置（`max_local_cpu_size`、`chunk_size` 等） |
| `config/lmcache_blend.yaml` | 开启 blending 的配置（与 prefix cache 互斥） |

**使用步骤**：

```bash
cd benchmark/v1/rag
pip install -r requirements.txt
# 1. 下载 musique_s.json（CacheBlend 官方数据集，链接见下）
#    https://github.com/YaoJiayi/CacheBlend/blob/main/inputs/musique_s.json
# 2. 编辑 online.sh / offline.sh 顶部：
#    MODEL_NAME、DATASET_PATH、END（样本数）
# 3. 运行
./online.sh          # 或 ./offline.sh
# 4. 结果在 *_benchmark_results/ 下的 CSV 与图
```

**核心参数速查**（`rag.py`）：

| 参数 | 说明 |
|---|---|
| `--dataset` | JSON 数据集路径，每条含 `ctxs`(文档列表)+`question`+`answers` |
| `--start-index/--end-index` | 数据集切片范围 |
| `--qps` / `--time` / `--step-interval` | 请求发送速率 / 时长上限 / 步进间隔 |
| `--max-tokens` / `--temperature` | 输出长度 / 温度 |
| `--online` + `--openai-api-base/--openai-api-key` | 在线模式：打到 `http://host:port/v1` |
| `--kv-storage-size` / `--kv-chunk-size` | 预计算 KV 的存储上限 / chunk 大小（须与 LMCache yaml 一致） |
| `--output` | 结果 CSV 路径 |

**重要注意**：
1. 该工具用的是**旧版内嵌 connector** `LMCacheAscendConnectorV1Dynamic`（LMCache 0.4.5 时代），与 MP mode（`LMCacheMPConnector` + 独立 server）**不能共存于同一 Python 环境**；
2. CacheBlend 与 vLLM prefix caching **互斥**（脚本里已自动 `--no-enable-prefix-caching`）；
3. 官方实测（musique_s）：CacheBlend 吞吐接近翻倍，平均质量分数有轻微损失。

### 6.2 `benchmark/cacheblend/` —— Hole-mode CacheBlend 压测（研究型）

**定位**：基于 hydra 配置的研究向压测，验证 CacheBlend **hole 模式**（非连续复用：只重算文档中被修改的"洞"，其余 KV 全复用）。当前配置默认 NPU 设备、Qwen3 模型。

```yaml
# config/config-hole.yaml 关键字段
model_path: /data/weights/qwen3-8B/
data_path: .../musique_s.json
recompute_ratio: 0.5      # 重算比例
hole_mode: 'hole'         # hole 模式开关
hole_id / hole_pos: -1    # 单洞回退：选中间文档；多洞用 hole_ids[]/hole_positions[]
chunk_size: 256
tp_size: 1
inflate: 1                # 文档块在 prompt 中的重复倍数
phase2_repeats: 2         # Phase-2 重复次数（pass1 填洞，pass2+ 纯复用）
batch_size: 1
npu: 0                    # NPU 标志位
```

**使用**：改 `config-hole.yaml` 里的 `model_path`/`data_path` 后执行：

```bash
python benchmark/cacheblend/benchmark-hole.py --config-path ... # 或按 hydra 约定
# 输出在 ./outputs/<时间戳>/ 下
```

Phase-1（预计算各文档 KV）/ Phase-2（混合请求命中测试）两阶段设计，`phase2_repeats` 用于区分"填洞开销"与"纯复用收益"。

---

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| 容器内 `npu.device_count()` 为 0 | 检查 `--device` 挂载与 `/dev/davinci_manager` 等三个特殊节点 |
| 启动时 HCCL 报通信初始化失败 | `export HCCL_SOCKET_IFNAME=lo`；确认 `--net=host` |
| 多进程 IPC 卡死/报共享内存错 | 调大 `--shm-size`；确认 `--ipc=host` |
| 显存不够 | 降 `--mem-fraction-static`（如 0.8）或缩短 `--context-length` |
| bench 报模型名 404 | `--model` 要与 `--served-model-name` 一致 |
| Qwen3.8 启动即报架构错误 | 见 3.2 节三步排查；确认镜像内 SGLang ≥ 官方支持该混合架构的版本 |

---

## 8. 与后续任务的衔接

本教程完成后的状态 = **"SGLang 裸基线"**：有一套可复现的 TTFT/TPOT/吞吐数据和压测负载。下一步：

1. 用 `benchmark/v1/rag` 的 musique_s.json 负载思想构造你自己的 RAG jsonl；
2. 挂 LMCache MP mode（参考 `docs/mp_mode/`），对同一负载重跑 benchmark；
3. `3.3 节` 记下的 attention block size + `docs/mp_mode/qwen3_5.md` 的经验表 → 决定 LMCache 侧 `--chunk-size` 与 `--max-num-batched-tokens`。

—— 这就是"SGLang 适配 MP Mode"任务的完整前半程。
