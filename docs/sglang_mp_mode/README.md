# SGLang 适配 LMCache MP Mode：小白向方案说明

> 目标：把当前仓库已经跑通的 **vLLM + LMCache MP Mode + Ascend NPU** 能力，迁移到 **SGLang + Ascend NPU**。
>
> 阅读方式：先理解 LMCache 解决什么问题，再理解 MP Mode 改变了什么，最后看适配清单。本文只做方案和代码导读，不代表这些适配已经全部实现。

## 1. 当前代码仓的 LMCache-Ascend 在做什么

### 1.1 一句话

**LMCache-Ascend 是 LMCache 的昇腾适配插件**：让 vLLM/SGLang 在 NPU 上把请求产生的 KV Cache 保存到独立缓存系统，后续具有相同前缀的请求可以取回 KV，减少重复 prefill。

它不是：

- 模型权重或模型目录；
- 一个新的推理引擎；
- 只负责单卡算子的库。

它是“推理引擎的 KV 缓存旁路 + NPU 数据搬运实现”。

### 1.2 为什么需要它

没有外部 KV 缓存时：

```text
请求 A：长系统提示词 + 问题 A
  -> 计算长前缀 KV
  -> 请求结束或显存紧张后，KV 被淘汰

请求 B：同一个长系统提示词 + 问题 B
  -> 再次计算长前缀 KV
```

接入后：

```text
请求 A -> 计算 KV -> 交给 LMCache 保存
请求 B -> 查找相同前缀 -> 命中则取回 KV -> 只计算未命中部分
```

收益主要出现在 **TTFT（首 token 延迟）**，尤其是多轮对话、共享系统提示词、RAG 重复文档等场景。它不会直接改变 decode 每个 token 的计算公式，因此 TPOT 不一定改善。

### 1.3 仓库中各部分的职责

| 位置 | 作用 |
|---|---|
| `lmcache_ascend/__init__.py` | 插件入口：检查 LMCache 版本、平台、推理引擎，并安装必要 patch |
| `lmcache_ascend/v1/npu_connector/` | NPU KV 内存布局和搬运，如 `batched_to_gpu` / `batched_from_gpu` |
| `lmcache_ascend/v1/token_database.py` | 将 token 序列切块并计算 hash，形成缓存 key |
| `lmcache_ascend/v1/cache_engine.py` | 缓存查找、存储、淘汰等引擎逻辑 |
| `lmcache_ascend/v1/transfer_context.py` | 创建 NPU 传输上下文，选择设备/内存/传输模式 |
| `lmcache_ascend/v1/transfer_channel/` | HCCL/HiXL 等传输通道适配 |
| `csrc/mp_mem_kernels.cpp` | MP 模式需要的底层内存/块传输 kernel |
| `lmcache_ascend/integration/vllm/` | vLLM 专属集成和 patch |
| `lmcache_ascend/integration/sglang/` | SGLang 现有集成，主要是旧的内嵌式路径 |
| `lmcache_ascend/integration/patch/sglang/` | 对 SGLang scheduler/manager 的现有 patch |

“判断两个请求是否能复用”不是比较完整自然语言问题，而是比较 **token 前缀块的 hash**。例如：

```text
请求 A = [相同系统提示词][相同文档][问题 A]
请求 B = [相同系统提示词][相同文档][问题 B]
         <------ 可命中的前缀 ------>  <-> 只需重新计算尾部
```

相关代码入口：

- `lmcache_ascend/v1/token_database.py`：token 分块、hash；
- `lmcache_ascend/v1/cache_engine.py`：`lookup` / `store`；
- `lmcache_ascend/v1/npu_connector/npu_connectors.py`：KV 在缓存和 NPU 显存之间搬运。

## 2. MP Mode 是什么，有无 MP Mode 有什么区别

### 2.1 旧模式：LMCache 内嵌在推理引擎进程中

旧路径使用：

- `LMCacheAscendConnectorV1Dynamic`；
- `LMCacheAscendEngine`；
- `LMCacheAscendScheduler` / SGLang scheduler patch。

可以理解为：

```text
vLLM/SGLang 进程
  ├── 请求调度
  ├── 模型 forward
  └── LMCache 查找/存储/淘汰
```

缓存逻辑和引擎在同一个进程里，接入相对直接，但缓存工作会占用引擎进程资源，也不便于多个引擎实例共享同一个缓存服务。

### 2.2 MP Mode：缓存变成独立服务

MP = Multi-Process。核心变化是：

```text
独立进程：lmcache server
  ├── 缓存索引和查找
  ├── L1/L2 存储
  ├── 淘汰
  └── 部分模式下的数据搬运
          ▲
          │ ZMQ 控制面 + 共享内存/设备 IPC 数据面
          │
vLLM worker 进程（每个 TP rank 一个）
  └── LMCacheMPConnector
```

启动顺序也因此改变：先启动 `lmcache server`，再启动推理引擎，并通过 `--kv-transfer-config` 指定 `LMCacheMPConnector`。

文档：

- `docs/mp_mode/README.md`
- `docs/mp_mode/lmcache_driven.md`
- `docs/mp_mode/engine_driven.md`
- `docs/mp_mode/qwen3_5.md`

### 2.3 两种 MP 数据搬运模式

| 模式 | 谁搬 KV | 特点 |
|---|---|---|
| `lmcache_driven` | LMCache server 通过设备 IPC 句柄搬运 | 支持更复杂 KV 布局；Qwen3.5/3.8 混合 Mamba 走这条 |
| `engine_driven` | vLLM worker 从命名共享内存池自行 gather/scatter | 零拷贝路径，但只支持标准 GQA/MHA，不支持 Mamba |

NPU 上 `auto` 当前解析为 `lmcache_driven`。相关代码：

- `lmcache_ascend/v1/transfer_mode.py`：传输模式解析；
- `lmcache_ascend/v1/transfer_context.py`：创建传输上下文；
- `csrc/mp_mem_kernels.cpp`：底层块传输；
- `lmcache_ascend/v1/shape_desc.py`：KV page/layout 描述。

### 2.4 有无 MP Mode 的区别

| 对比项 | 旧内嵌式 | MP Mode |
|---|---|---|
| LMCache 所在位置 | 推理引擎进程内 | 独立 `lmcache server` 进程 |
| 连接器 | `LMCacheAscendConnectorV1Dynamic` | `LMCacheMPConnector` |
| 控制通信 | 进程内调用 | ZMQ |
| 数据通信 | 进程内内存操作 | 设备 IPC / 命名共享内存 |
| 多引擎共享 | 不自然 | 设计目标之一 |
| 启动依赖 | 只需引擎 | 先 server，再引擎 |
| 生命周期 | 随引擎 | server 与引擎需配套管理 |
| 适配重点 | NPU connector + scheduler | 还要处理跨进程初始化、块大小、传输模式、异步完成 |

**重要兼容性约束**：MP Mode 和旧内嵌式 connector 不能装在同一个 Python 环境。MP 主线要求 LMCache `dev` 分支；旧 SGLang 路径文档则绑定 LMCache `0.4.5`。

## 3. 当前 MP Mode 是否只适配了 vLLM

### 3.1 结论

**是的。当前仓库的 MP Mode 文档、激活逻辑和端到端测试都是围绕 vLLM 的。**

现有 SGLang 支持是另一条“旧内嵌式”路线，不等于 SGLang MP Mode 已支持。

证据：

1. `docs/mp_mode/lmcache_driven.md`、`engine_driven.md` 的启动方式都是 `vllm serve` + `LMCacheMPConnector`。
2. `lmcache_ascend/__init__.py` 的激活逻辑会检查 `import vllm`，没有 vLLM 时直接返回。
3. `tests/integration/test_npu_activation.py` 的引擎枚举逻辑目前只把 `vllm` 识别为可激活引擎。
4. `lmcache_ascend/integration/sglang/lmcache_connector.py` 和 `integration/patch/sglang/` 中，类型和注释明确指向 `LMCacheAscendConnectorV1Dynamic`、`LMCacheAscendEngine`、`LMCacheAscendScheduler`，没有 `LMCacheMPConnector` 路径。
5. `tests/integration/test_sglang_vllm_connector_e2e.py` 的 SGLang case 创建的是旧 connector，并显式跳过 MP 不兼容情况。

### 3.2 已经完成的 vLLM/Ascend 适配

| 类别 | 已有实现 |
|---|---|
| 插件激活 | `init_lmcache_ascend()`，检查平台/版本并安装 patch |
| KV 数据搬运 | `VLLMPagedMemNPUConnectorV2`、layerwise connector、MLA/DSA 相关 connector |
| MP 块传输 | `csrc/mp_mem_kernels.cpp`、event/completion recorder |
| 传输模式 | `transfer_mode.py`、`transfer_context.py` |
| vLLM 块大小 | `integration/vllm/logical_block_size.py` |
| vLLM 存储/MLA 特殊路径 | `integration/vllm/patch_dsv4_kvcache_ops.py` |
| 传输通道 | `transfer_channel/hccl_channel.py`、`hixl_channel.py` |
| 验证 | `tests/integration/test_npu_activation.py`、`tests/v1/multiprocess/`、MP e2e 测试 |

因此，SGLang 适配不是从零写 NPU 算子，而是把 **SGLang 的连接器/调度/进程生命周期** 接到已有 MP/NPU 能力上，同时补齐上游 SGLang 侧需要的接口。

## 4. SGLang 适配 MP Mode 需要修改什么

### 4.1 先确认上游接口是否具备

MP 连接器不是简单把字符串从 `LMCacheAscendConnectorV1Dynamic` 改成 `LMCacheMPConnector`。

需要先检查当前 LMCache `dev` 和 SGLang 版本：

- SGLang 是否有可挂接外部 KV connector 的统一接口；
- 是否已有 `LMCacheMPConnector` 的 SGLang 实现；
- connector 是否提供 lookup、retrieve、store、异步完成、请求释放等生命周期；
- TP worker 是否都能拿到各自的 NPU 设备上下文和传输上下文。

如果上游只有 vLLM 版 `LMCacheMPConnector`，则第一项工作就是实现 **SGLang MP connector 或适配层**，而不是只改 Ascend 仓库。

### 4.2 修改插件激活逻辑

当前 `lmcache_ascend/__init__.py` 只检查 vLLM。需要让“只安装 SGLang”的环境也能进入 Ascend 初始化：

```text
现在：import vllm 失败 -> return
目标：检测 vllm / sglang -> 按引擎安装对应 patch
```

注意不要把 vLLM 专属 patch 无条件套到 SGLang。例如：

- `integration/vllm/logical_block_size.py`；
- vLLM DSV4 MLA/DSA patch；
- vLLM `LMCacheMPConnector` 导入路径。

应拆成公共初始化和引擎专属初始化。

### 4.3 新增 SGLang MP connector/manager 适配

可复用现有文件的设计，但不能直接复用其旧 connector 假设：

- `integration/sglang/lmcache_connector.py`：当前工厂只返回 `LMCacheAscendConnectorV1Dynamic`；
- `integration/patch/sglang/sglang_manager_adapter.py`：当前适配的是 `LMCacheAscendScheduler`；
- `integration/patch/sglang/sglang_scheduler_adapter.py`：当前处理的是旧 `lookup` / `load_kv` / `store_kv` 流程。

需要新增 MP 路径，至少覆盖：

```text
请求进入 -> 向 lmcache server 查询前缀
       -> 命中时异步取回 KV
       -> 等待/继续调度
       -> prefill 后保存新增 KV
       -> 请求结束时释放/收尾
```

SGLang 的 tokenizer、scheduler、detokenizer 是多进程结构，必须明确：**哪些调用在 scheduler 进程，哪些 KV 搬运在 TP worker 进程**，不能只在主进程初始化一次就认为所有 worker 可用。

### 4.4 复用并校验 NPU connector

以下能力是主要复用对象：

- `VLLMPagedMemNPUConnectorV2` 的 NPU KV 读写思路；
- layerwise connector；
- `transfer_context.py`；
- `csrc/mp_mem_kernels.cpp`；
- `transfer_channel/`。

但类名中有 `VLLM` 不代表能直接在 SGLang 使用。需要检查：

- KV buffer 形状和页布局是否一致；
- slot mapping / block table 的含义是否一致；
- layerwise 与 batched 的调用时机是否一致；
- SGLang RadixAttention 的内部 KV 与 LMCache 外部 KV 是否重复保存或冲突；
- TP rank 到 NPU device 的映射是否正确。

必要时抽出不依赖 vLLM 的 NPU 传输层，再分别提供 vLLM/SGLang wrapper。

### 4.5 处理块大小和混合 Mamba 约束

Qwen3.8-27B 是混合 GDN/Mamba + 全注意力模型，不能只验证普通 Qwen3。

MP 文档要求：

- Mamba 混合模型使用 `lmcache_driven`；
- `--mamba-cache-mode align`；
- `--chunk-size` 与 attention block size 对齐；
- 不支持 `engine_driven` 的 Mamba 路径；
- `--l1-size-gb` 要装得下目标工作集。

SGLang 侧还要确认是否存在与 vLLM `--mamba-cache-mode align` 等价的配置；如果没有，需要在适配层明确保证状态对齐，不能假设参数名相同。

此前 SGLang 在 910B 上启动 Qwen3.8 还需要：

```text
--mamba-ssm-dtype bfloat16
```

这是因为 NPU 上部分 GDN 算子不支持 FP32 initial state。这个设置解决的是 SGLang 本身启动问题，与 LMCache MP 接入是两个独立约束，都要保留验证。

### 4.6 增加测试和文档

至少需要：

1. 只安装 SGLang 时的插件激活测试；
2. connector 工厂能选择 SGLang MP connector 的测试；
3. 单请求写入 + 同前缀第二请求命中的 e2e；
4. TP=8 多 worker 初始化测试；
5. 长短混合输入、重复前缀、多轮对话测试；
6. Qwen3.8-27B 混合 Mamba 测试；
7. server 未启动、server 异常退出、shm 不足等失败路径测试；
8. `docs/mp_mode/` 新增 SGLang 启动示例和限制说明。

## 5. 推荐实施顺序

```text
1. 固定 SGLang 裸服务基线
   Qwen3.8-27B + 910B4 + --mamba-ssm-dtype bfloat16
   跑通 random-ids benchmark，记录 TTFT/TPOT/吞吐

2. 确认 LMCache/SGLang 上游 MP 接口
   明确 connector API 和进程调用位置

3. 先做最小标准 GQA/MHA 模型适配
   不先挑战 Mamba；验证 lookup/store/retrieve 全链路

4. 接入 NPU MP 传输上下文
   验证 TP=1，再扩展 TP=8

5. 做重复前缀 TTFT A/B
   无 LMCache / 旧内嵌式 / SGLang MP 三组对比

6. 最后适配 Qwen3.8 混合 Mamba
   处理 chunk/block 对齐和 recurrent state 生命周期
```

最小验收标准不是“服务能启动”，而是：

- 第一次请求日志出现缓存写入；
- 第二次相同前缀请求出现命中和取回；
- TTFT 有可重复的下降；
- 输出语义正常；
- 多轮/多 TP 下无死锁、无 device 上下文错误；
- 关闭 LMCache 后，SGLang 裸服务行为不变。

## 6. 本次适配的核心判断

这次工作本质不是“把 vLLM 参数翻译成 SGLang 参数”，而是：

> **把 LMCache MP 的跨进程缓存协议，接入 SGLang 的调度器和 TP worker 生命周期，并复用现有 Ascend NPU KV 搬运能力。**

其中最容易低估的三点是：

1. 上游 SGLang 是否真的已有 MP connector 接口；
2. 多进程下每个 worker 的 NPU/共享内存上下文初始化；
3. Qwen3.8 混合 Mamba 的状态缓存与 KV 块对齐。

先把普通 GQA/MHA 模型的 MP 链路打通，再处理 Qwen3.8，会比直接在混合模型上联调更容易定位问题。
