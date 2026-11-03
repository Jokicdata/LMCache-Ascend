# SGLang + LMCache MP Mode on Ascend 910B4：手把手适配教程

> 适用环境：8×Ascend 910B4，已经能用 SGLang 启动 Qwen3.8-27B。
>
> 目标：在现有 SGLang 服务上接入独立 `lmcache server`，验证 KV 的注册、写入、命中和回灌，再按实际报错做最小代码修改。
>
> 重要结论：**不要从重写 SGLang connector 开始。** 上游 LMCache 已经提供 SGLang MP 集成；当前优先做版本和接口审计。版本齐全时，第一阶段可能不需要修改产品代码。

## 0. 面向初学者的背景：vLLM、SGLang 与本次适配的关系

### 0.1 vLLM 和 SGLang 分别是什么

vLLM 和 SGLang 都是大模型推理引擎，负责接收请求、调度 prefill/decode、管理显存中的 KV Cache，并调用模型完成计算。它们不是模型本身，模型目录只提供权重、配置和 tokenizer。

可以把 LMCache 理解为挂在推理引擎旁边的“外部 KV 缓存服务”，把 LMCache-Ascend 理解为让这个缓存服务能在昇腾 NPU 上高效搬运数据的插件：

```text
请求 → vLLM 或 SGLang → 模型计算
                │
                └── LMCache → LMCache-Ascend → CANN / NPU
```

### 0.2 当前已经完成的 vLLM 工作是什么

当前主线已经形成一条 vLLM MP 路径：

```text
vllm serve
  → LMCacheMPConnector
  → 独立 lmcache server
  → NPU IPC / 共享内存
  → LMCache-Ascend 原生块传输 kernel
```

其中包括 vLLM connector 生命周期、NPU KV 注册、块大小处理、MP 数据面、异步事件，以及相应部署文档和测试。这里的“已适配”指 **vLLM + LMCache MP + Ascend** 这条组合路径，不代表任意引擎都能自动复用同一套上层接口。

### 0.3 SGLang 适配与 vLLM 有什么不同

| 对比项 | vLLM 现有路径 | SGLang 要验证/适配的路径 |
|---|---|---|
| 接入方式 | `--kv-transfer-config` 指定 `LMCacheMPConnector` | `--enable-lmcache` + `--lmcache-config-file` |
| 上层 connector | vLLM `LMCacheMPConnector` | 上游 `UnifiedLMCacheMPConnector` |
| KV 组织 | vLLM paged KV buffer / layer dict | SGLang K/V pool、unified cache 的 FULL/SWA/MAMBA component |
| 进程协作 | vLLM scheduler role 与 worker role | SGLang scheduler、TP worker 和本地 RadixCache/unified cache 协作 |
| 本地前缀缓存 | vLLM prefix caching | SGLang RadixAttention / unified radix cache |
| 配置形式 | JSON connector 配置 | SGLang 参数 + LMCache YAML |
| 当前工作 | 已有 Ascend MP 主线和文档 | 审计上游接口，验证 NPU 注册和端到端闭环，按真实差异补小 patch |

两种引擎可以共享 `lmcache server`、缓存索引、存储后端和 Ascend 原生 kernel，**不能直接共享所有 connector 和 KV 生命周期代码**。因此本次工作的重点是“把上游已有的 SGLang MP 控制面接到现有 Ascend 数据面”，而不是把 vLLM connector 逐行翻译成 SGLang。

### 0.4 为什么还要做接口审计

“上游支持 SGLang”“本仓支持 Ascend”“当前模型能启动”是三个独立条件。只有以下条件同时成立，才可能零代码跑通：

1. 当前 LMCache 包含 `UnifiedLMCacheMPConnector`；
2. 当前 SGLang 能调用这套 unified MP 接口；
3. 当前模型的 KV/MAMBA layout 能被上游 adapter 识别；
4. NPU IPC、块传输和异步事件在 TP8 下正常；
5. SGLang 的 block/chunk 语义与 LMCache 对齐。

接口审计就是先把这五件事分清，避免一开始改错仓库、改错层。

## 1. 先回答三个基础问题

### 1.1 当前仓库包含完整 LMCache 源码吗

不包含。`LMCache-Ascend` 是昇腾插件，负责：

- NPU 原生扩展 `lmcache_ascend.c_ops`；
- NPU KV 布局和 `PageBufferShapeDesc` 补丁；
- MP 块传输 kernel、异步事件和完成通知；
- 一部分旧内嵌式 vLLM/SGLang 适配代码。

以下主体在上游 [LMCache](https://github.com/LMCache/LMCache)：

- `lmcache server`；
- ZMQ/gRPC 控制面；
- SGLang MP connector；
- MP transfer context 和 cache context；
- 缓存查找、存储和淘汰。

### 1.2 是否要先安装 LMCache，再安装 LMCache-Ascend

是。推荐两个仓库都以可编辑方式安装在**同一个 SGLang Python 环境**：

```text
SGLang 环境
  ├── sglang / torch / torch-npu / CANN
  ├── LMCache          上游缓存服务与 SGLang MP connector
  └── LMCache-Ascend   NPU 原生实现和补丁
```

不要把 MP 环境和旧 `LMCacheAscendConnectorV1Dynamic` 内嵌式环境混用。

### 1.3 LMCache 是否已经适配过 SGLang

**已经适配，包括 MP 路径。** 需要重点检查这些上游符号：

```text
lmcache.integration.sglang.unified_lmcache_mp_connector
  └── UnifiedLMCacheMPConnector

lmcache.integration.sglang.unified_kv_adapter
  └── SGLangUnifiedKVAdapter
```

`UnifiedLMCacheMPConnector` 负责 SGLang 与独立缓存服务通信；`SGLangUnifiedKVAdapter` 负责把 SGLang 的 FULL/SWA/MAMBA 等缓存组件映射到 LMCache。

LMCache v0.5.6 的发布说明同时包含：

- SGLang unified MP radix cache；
- SGLang engine-driven transfer；
- 移除 unified MP connector 的硬编码 CUDA 检查；
- Ascend NPU LMCache-driven backend。

本仓 `lmcache_ascend/__init__.py` 中的 `LMCACHE_UPSTREAM_TAG` 也指向 `v0.5.6`。因此教程优先使用 **v0.5.6 固定版本**，而不是一开始追逐滚动 `dev`。

## 2. 最终要搭出的运行结构

```text
终端 A：lmcache server
  ├── 监听 127.0.0.1:5556
  ├── L1 CPU 缓存
  └── NPU cache context / 数据搬运

终端 B：SGLang
  ├── --enable-lmcache
  ├── --lmcache-config-file /workspace/scripts/lmcache_mp.yaml
  ├── UnifiedLMCacheMPConnector
  └── 8 个 TP worker 注册各自的 NPU KV pool

终端 C：验证客户端
  └── 两次相同长前缀请求，观察 Stored / Retrieved 和 TTFT
```

当前已经运行的 SGLang 裸服务先用于保存基线数据。**不要在同一批 NPU 上同时启动第二个 27B TP8 服务**；开始 MP 验证前，先保存 benchmark，再停止裸服务。

## 3. 阶段一：冻结环境，确认源码位置

在 SGLang 容器内执行：

```bash
mkdir -p /workspace/src /workspace/scripts /workspace/results

python - <<'PY'
import sys
import torch
import torch_npu
import sglang

print("python:", sys.version)
print("sglang:", sglang.__version__, sglang.__file__)
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu_count:", torch.npu.device_count())
PY

python -m sglang.launch_server --help 2>&1 | grep -E "enable-lmcache|lmcache-config-file"
npu-smi info
ls /usr/local/Ascend
```

记录：

- SGLang 版本和源码路径；
- CANN 版本；
- 当前启动命令；
- 当前 TTFT/TPOT/吞吐基线；
- Qwen3.8 启动时的 attention block size 日志。

如果 `--enable-lmcache` 已存在，说明 SGLang 至少有 LMCache 接入开关；它不代表 NPU MP 已验证通过。

## 4. 阶段二：安装上游 LMCache 和本仓插件

### 4.1 获取上游 LMCache

```bash
cd /workspace/src
git clone https://github.com/LMCache/LMCache.git
cd LMCache
git checkout v0.5.6
git submodule update --init --recursive
pip install -e . --no-build-isolation
```

如果服务器无法访问 GitHub，请在有网机器下载对应源码包后拷贝到 `/workspace/src/LMCache`，不要直接安装一个不知版本的 PyPI 包替代。

### 4.2 获取 LMCache-Ascend

```bash
cd /workspace/src
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
git submodule update --init --recursive
```

如果你要基于自己的分支开发，也可以添加远程后 checkout：

```bash
git remote add jokic git@github.com:Jokicdata/LMCache-Ascend.git
git fetch jokic
git checkout -b xby jokic/xby
```

### 4.3 对齐 hcomm 与 CANN

先看容器实际 CANN 版本：

```bash
ls /usr/local/Ascend
cd /workspace/src/LMCache-Ascend/third_party/hcomm
git tag | tail -30
```

选择与容器 CANN 对应的 hcomm tag。例如**仅当** CANN 是 8.5.0 且远端存在对应 tag 时：

```bash
git checkout v8.5.0
```

你的 SGLang 镜像可能是 CANN 9.0.0，不能照抄 8.5.0。若没有明确匹配版本，先停下记录版本，不要随意选择一个相近 tag 编译。

### 4.4 安装昇腾插件

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONHASHSEED=0
cd /workspace/src/LMCache-Ascend
pip install -v --no-build-isolation -e .
```

该命令会编译并放置：

- `lmcache_ascend/c_ops*.so`；
- `libcache_kernels.so`；
- 适用时的 HCCL/HiXL 扩展。

### 4.5 验证实际加载路径

```bash
python - <<'PY'
import lmcache
import lmcache_ascend
import lmcache_ascend.c_ops as ops

print("lmcache:", lmcache.__file__)
print("lmcache_ascend:", lmcache_ascend.__file__)
print("c_ops:", ops.__file__)
print("has_group_transfer:", hasattr(ops, "execute_object_group_transfer"))
print("has_block_transfer:", hasattr(ops, "multi_layer_block_kv_transfer"))
PY
```

期望两个 Python 包都指向 `/workspace/src/...`，而不是另一份 `site-packages` 源码。

## 5. 阶段三：接口审计——正式改代码前必须完成

### 5.1 一键检查上游接口

```bash
python - <<'PY'
import importlib.util
import lmcache
import sglang

print("LMCache:", lmcache.__file__)
print("SGLang:", sglang.__version__, sglang.__file__)

modules = [
    "lmcache.integration.sglang.unified_lmcache_mp_connector",
    "lmcache.integration.sglang.unified_kv_adapter",
    "lmcache.v1.multiprocess.transfer_context.worker_transfer",
    "lmcache.v1.platform.devices.npu",
    "lmcache.v1.platform.devices.npu.ipc_wrapper",
    "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache",
]
for name in modules:
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        found = f"ERROR: {exc}"
    print(f"{name}: {found}")

from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    UnifiedLMCacheMPConnector,
)
from lmcache.integration.sglang.unified_kv_adapter import SGLangUnifiedKVAdapter
from lmcache.v1.platform.devices.npu import NpuDeviceSpec
from lmcache.v1.platform.devices.npu.ipc_wrapper import NpuIPCWrapper

print("UnifiedLMCacheMPConnector: OK")
print("SGLangUnifiedKVAdapter: OK")
print("NpuDeviceSpec / NpuIPCWrapper: OK")
PY
```

任何一步报错都先保存完整输出，不要马上重写 connector。

### 5.2 检查 SGLang 的实际挂接位置

```bash
grep -Rns "UnifiedLMCacheMPConnector\|enable_lmcache" \
  /sgl-workspace/sglang/python/sglang/srt/mem_cache \
  /sgl-workspace/sglang/python/sglang/srt/model_executor 2>/dev/null | head -80

grep -Rns "ComponentType.MAMBA\|MAMBA" \
  /workspace/src/LMCache/lmcache/integration/sglang | head -50
```

要回答四个问题：

1. `--enable-lmcache` 最终创建哪个 connector？
2. connector 在 scheduler 进程还是 TP worker 进程创建？
3. KV pool 注册发生在哪个函数？
4. 当前 Qwen3.8 是否经过 unified FULL/MAMBA component 路径？

### 5.3 检查原生 MP 数据面

```bash
cd /workspace/src/LMCache-Ascend
pytest -q tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py -s
```

该测试需要 NPU，验证：

```text
NPU IPC 注册
  → NpuDeviceSpec.create_cache_context
  → LMCacheDrivenTransferModule.store
  → retrieve
  → KV 内容一致
```

它当前使用 `EngineType.VLLM`，所以**通过它只证明 NPU MP 数据面可用，不证明 SGLang 控制面已适配**。

## 6. 根据审计结果决定是否需要改代码

| 审计结果 | 正确行动 |
|---|---|
| Unified connector、NPU IPC、c_ops 都存在 | 直接进入启动验证，暂不改产品代码 |
| LMCache 没有 unified SGLang MP | 升级到 v0.5.6+，不要基于旧 `multi_process_adapter.py` 另写一套 |
| SGLang 有开关但版本过旧、没有 unified 调用路径 | 优先升级/匹配 SGLang；不要强行走本仓旧内嵌式 adapter |
| 仍有硬编码 CUDA 检查 | 使用包含移除该检查的 LMCache 版本；旧版本才考虑 backport |
| NPU native plugin 没有加载 | 先定位上游 `NpuDeviceOps.ensure_native` 和 `wrap_kv_caches`，不要先改 C++ |
| Qwen3.8 的 MAMBA component 不支持 | 先用普通 GQA/MHA 模型打通，再单独处理混合状态缓存 |

### 6.1 哪些文件第一轮不应该改

以下属于旧内嵌式路径，不是当前 SGLang MP 的起点：

- `lmcache_ascend/integration/sglang/sglang_adapter.py`；
- `lmcache_ascend/v1/cache_engine.py`；
- `lmcache_ascend/v1/npu_connector/__init__.py::CreateNPUConnector`；
- `SGLangLayerwiseNPUConnector`。

尤其不要把旧 adapter 中的 `.cuda()` 全局替换成 `.npu()` 后认为 MP 适配完成，那只是另一条 in-process 路径。

### 6.2 如果确实需要修改，代码应该放在哪里

按错误归属修改，而不是所有问题都改 LMCache-Ascend：

| 问题 | 修改仓库/位置 |
|---|---|
| SGLang 没有启用 unified connector | SGLang 的 LMCache/radix-cache 注册路径 |
| Unified connector 的 KV 注册、请求生命周期不匹配 | 上游 LMCache `lmcache/integration/sglang/` |
| NPU IPC/device context 不匹配 | 上游 LMCache `lmcache/v1/platform/devices/npu/` 或对应 device-agnostic 调用点 |
| `PageBufferShapeDesc`、plane 几何或 NPU layout 不匹配 | 本仓 `shape_desc.py`、`kv_layer_groups.py` |
| 原生块传输不支持当前 KV 格式 | 本仓 `csrc/mp_mem_kernels.cpp` 及对应 kernel |
| 异步事件/完成通知异常 | 本仓 `event_recorder.cpp`、`completion_recorder.cpp` |

修改上游两个仓库时同样使用 `pip install -e .`，保证下次启动加载你的源码。

### 6.3 最小调试补丁应该怎么写

第一轮只加“观测探针”，不要先改传输逻辑。建议在实际 KV 注册函数打印：

```python
print(
    "[SGLANG-MP-PROBE]",
    "engine=sglang",
    "rank=", rank,
    "device=", first_tensor.device,
    "shape=", tuple(first_tensor.shape),
    "dtype=", first_tensor.dtype,
    "contiguous=", first_tensor.is_contiguous(),
)
```

并确认：

- 每个 TP rank 都执行了注册；
- device 是 `npu:rank`，不是 `cuda` 或 CPU；
- 每个 worker 的本地 device index 正确；
- KV layout 与上游注册格式一致；
- `import lmcache_ascend.c_ops` 已发生。

确认问题后删除探针，再写最小修复和回归测试。

## 7. 预计代码量

代码量取决于接口审计，不能预先承诺一个固定数字：

| 情况 | 产品代码预估 | 测试/配置 |
|---|---|---|
| 版本齐全，直接跑通 | 0 行 | 启动脚本、配置、验证脚本 |
| 只需补 NPU 加载或小兼容 patch | 约 20～100 行 Python | 约 100～300 行测试 |
| 需要补 SGLang KV 注册/生命周期适配 | 约 100～400 行 Python | 需要单元测试和 TP e2e |
| Qwen3.8 MAMBA 状态路径也不兼容 | 审计后单独评估，可能超过上述范围 | 必须增加多轮和状态恢复测试 |

**不建议以“重写整个 connector”作为工作量起点。** 上游已有实现时，重写通常意味着重复维护上千行协议和生命周期代码。

## 8. 修改后的安装和重启规则

只改 Python：

```text
修改源码 → 停止 SGLang/lmcache server → 重新启动 → 生效
```

改了 C++/AscendC/pybind：

```bash
cd /workspace/src/LMCache-Ascend
pip install -v --no-build-isolation -e .
```

然后重启两个进程。安装后再次检查：

```bash
python -c "import lmcache_ascend; print(lmcache_ascend.__file__)"
python -c "import lmcache_ascend.c_ops as o; print(o.__file__)"
```

不要只重启进程却怀疑为什么新 C++ 没生效。

## 9. 第一次端到端启动

### 9.1 普通 GQA/MHA 模型的起始配置

推荐先用普通稠密模型打通；如果当前只有 Qwen3.8-27B，则跳到 9.3，但把它视为混合模型实验路径。

创建 `/workspace/scripts/lmcache_mp.yaml`：

```yaml
mp_host: tcp://127.0.0.1
mp_port: 5556
chunk_size: 256
```

启动缓存服务（server 和 SGLang 两个进程都应设置相同的 `PYTHONHASHSEED=0`）：

```bash
export PYTHONHASHSEED=0
lmcache server \
  --host 127.0.0.1 --port 5556 \
  --http-port 8080 \
  --chunk-size 256 \
  --l1-size-gb 20 \
  --eviction-policy LRU \
  --supported-transfer-mode lmcache_driven \
  2>&1 | tee /workspace/results/lmcache_server.log
```

另一个终端检查：

```bash
curl -s http://127.0.0.1:8080/healthcheck
```

如果该版本健康端点路径不同，以 `lmcache server --help` 和启动日志为准。

### 9.2 启动 SGLang

保留你原来已经跑通的全部参数，只追加两个参数。以下命令演示追加方式；**Qwen3.8 必须先按 9.3 核对 chunk/block 对齐后再执行**：

```bash
export PYTHONHASHSEED=0
python -m sglang.launch_server \
  --model-path /data/models/Qwen \
  --served-model-name Qwen3.8-27B \
  --host 0.0.0.0 --port 8998 \
  --tp-size 8 \
  --dtype bfloat16 \
  --context-length 8192 \
  --chunked-prefill-size 4096 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --mamba-ssm-dtype bfloat16 \
  --enable-lmcache \
  --lmcache-config-file /workspace/scripts/lmcache_mp.yaml \
  2>&1 | tee /workspace/results/sglang_mp.log
```

`--mamba-ssm-dtype bfloat16` 是当前 910B 跑 Qwen3.8 的已有要求，继续保留。

### 9.3 Qwen3.8 的 chunk/block 注意事项

**不要直接把 vLLM 文档里的 896/1536 等数值照搬到 SGLang。** 那些表属于 vLLM-Ascend 版本组合。

正确方法：

1. 从当前 SGLang 启动日志找到 attention block size N；
2. 确认 unified KV adapter 对 FULL/MAMBA component 的分组；
3. 让 LMCache `chunk_size` 与 N 对齐；
4. 同步修改 server 的 `--chunk-size` 和客户端 YAML 的 `chunk_size`；
5. 若不明确，先保持默认做注册/短请求验证，不直接开展长前缀性能结论。

Qwen3.8 单卡放不下时不要强行 TP1；可以使用 TP8，但先验证 8 个 worker 都完成注册。

## 10. 如何证明 LMCache 真的在工作

仅看到服务启动不算成功。必须完成下面闭环：

### 第一步：发送一个长前缀请求

```bash
python - <<'PY' > /tmp/lmcache_req_a.json
import json
prefix = "KV cache reuse requires a sufficiently long shared prefix. " * 300
json.dump({
    "model": "Qwen3.8-27B",
    "prompt": prefix + "\nSummarize the topic in one sentence.",
    "max_tokens": 32,
    "temperature": 0,
}, open("/tmp/lmcache_req_a.json", "w"))
PY

curl -s http://127.0.0.1:8998/v1/completions \
  -H 'Content-Type: application/json' \
  -d @/tmp/lmcache_req_a.json
```

观察 `lmcache_server.log`：应出现与 `Stored` 等价的写入日志。

### 第二步：清掉 SGLang 本地 RadixCache

```bash
curl -s -X POST http://127.0.0.1:8998/flush_cache
```

目的：防止第二次只是命中 SGLang 本地缓存，无法证明数据来自 LMCache。

### 第三步：发送相同前缀、不同尾部的请求

```bash
python - <<'PY' > /tmp/lmcache_req_b.json
import json
prefix = "KV cache reuse requires a sufficiently long shared prefix. " * 300
json.dump({
    "model": "Qwen3.8-27B",
    "prompt": prefix + "\nGive one short title for this text.",
    "max_tokens": 32,
    "temperature": 0,
}, open("/tmp/lmcache_req_b.json", "w"))
PY

curl -s http://127.0.0.1:8998/v1/completions \
  -H 'Content-Type: application/json' \
  -d @/tmp/lmcache_req_b.json
```

成功标准：

- server 出现 `Retrieved` 等价的读取日志；
- 输出正常；
- 没有 IPC、block size、dtype、shape 错误；
- 重复测试时 TTFT 有可重复下降。

如果第二次只有 SGLang 本地命中日志而没有 LMCache retrieve，不能算 MP 验证通过。

## 11. 必做测试分层

按顺序执行，不要直接从 TP8 长上下文开始排错：

1. **API 审计**：第 5 节所有 import 通过；
2. **NPU 数据面**：`test_npu_lmcache_driven_e2e.py` 通过；
3. **上游 SGLang connector 单测**：

   ```bash
   cd /workspace/src/LMCache
   pytest -q tests/v1/test_unified_sglang_mp_connector.py
   ```

   文件名以实际 checkout 为准；

4. **SGLang 单 worker/最小可用 TP**：注册、store、retrieve；
5. **当前模型的 TP8**：8 个 rank 注册和请求闭环；
6. **性能对照**：相同 prompt、相同并发、相同 seed，比较无 LMCache 与 MP；
7. **混合 Mamba 专项**：多轮对话、flush 后恢复、长上下文、不同 rank 一致性。

如果修改了本仓，建议新增：

```text
tests/integration/sglang/
  ├── test_mp_api_surface.py
  ├── test_sglang_kv_registration.py
  └── test_sglang_mp_e2e.py
```

当前仓库没有完整 SGLang MP 测试，这些应成为适配交付的一部分。

## 12. 常见报错的归属

| 报错/现象 | 优先检查 |
|---|---|
| `ModuleNotFoundError: unified_lmcache_mp_connector` | LMCache 版本或安装路径 |
| 仍创建旧 `LMCacheLayerwiseConnector` | SGLang 模式选择/版本，当前走成 IP 而非 MP |
| `CudaIPCWrapper` / CUDA context 错误 | 上游版本过旧或仍走 CUDA 专用路径 |
| `NpuDeviceOps` 找不到 native | LMCache-Ascend 未安装、c_ops 路径错误、插件未触发 |
| shape/block/chunk 不匹配 | SGLang KV layout、block size、chunk size |
| 只有一个 rank 注册成功 | TP worker 初始化和 local rank/device 映射 |
| 第一次 Stored，第二次无 Retrieved | 未 flush 本地缓存、前缀未按 chunk 对齐、容量不足或 store 未完成 |
| 能启动但无性能收益 | 命中率低、prompt 太短、重复前缀不足、测量未隔离本地 RadixCache |
| Qwen3.8 仅在多轮时异常 | MAMBA recurrent state 生命周期，不是普通 attention KV 问题 |

## 13. 推荐的实际开发顺序

```text
第 1 步：保存裸 SGLang 基线
第 2 步：安装 LMCache v0.5.6 + LMCache-Ascend
第 3 步：跑接口审计和 NPU MP 原生测试
第 4 步：不修改代码，直接做一次 SGLang MP 启动
第 5 步：根据第一个真实错误决定改哪个仓库
第 6 步：先完成 Stored/Retrieved 正确性闭环
第 7 步：再处理 Qwen3.8 混合 Mamba 和 TP8 边界
第 8 步：最后做 TTFT 对照和整理 PR
```

核心原则：**先证明现有上游实现能不能跑，再写适配代码；先修控制面，再修数据面；先保证正确，再谈性能。**

## 14. 依赖安装完成后的立即执行清单

当 LMCache、依赖和 LMCache-Ascend 都已安装后，先不要改代码，按下面顺序执行。

### 第一步：确认安装和 NPU 后端

```bash
python - <<'PY'
import torch
import torch_npu
import numba
import aiofile
import lmcache
import lmcache_ascend
import lmcache_ascend.c_ops as ops

print("lmcache:", lmcache.__file__)
print("lmcache_ascend:", lmcache_ascend.__file__)
print("c_ops:", ops.__file__)
print("npu_available:", torch.npu.is_available())
print("npu_count:", torch.npu.device_count())
PY

lmcache server --help
```

如果仍出现 `CLI-only mode`，先回到依赖安装文档，不要继续启动 MP 服务。

### 第二步：执行并保存接口审计

执行本文第 5.1 节的完整 Python 审计脚本，并将输出保存到：

```text
/workspace/results/sglang_mp_api_audit.log
```

再执行第 5.2 节的两条 `grep`，确认 SGLang 实际挂接点和 MAMBA component 支持情况。

### 第三步：验证 NPU 原生 MP 数据面

```bash
cd /workspace/src/LMCache-Ascend
pytest -q tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py -s \
  2>&1 | tee /workspace/results/npu_mp_native_e2e.log
```

### 第四步：先分析结果，再决定代码修改

审计全部通过后，第一次启动仍然先使用现有上游实现，不主动添加 patch。只有出现真实、可复现的错误后，才按第 6 节判断错误属于 SGLang、上游 LMCache 还是 LMCache-Ascend。

需要保留并用于定位问题的材料：

- 三个仓库的版本和 commit；
- `sglang_mp_api_audit.log`；
- `npu_mp_native_e2e.log`；
- 第一次启动时最早的完整异常栈；
- 当前 Qwen3.8 的 attention block size 日志。

## 参考资料

- [LMCache Quickstart（含 SGLang MP 启动与验证）](https://docs.lmcache.ai/getting_started/quickstart.html)
- [LMCache v0.5.6 Release Notes](https://github.com/LMCache/LMCache/releases/tag/v0.5.6)
- [LMCache MP Configuration Reference](https://docs.lmcache.ai/mp/configuration.html)
- 本仓 `docs/mp_mode/README.md`
- 本仓 `tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py`
