# LMCache Connector 详解：它是什么、属于谁、在本仓库哪里

> 面向第一次接触 LMCache 集成的开发者。本文回答四个问题：
> 1. connector 是什么；
> 2. 它是属于 LMCache 插件，还是属于 vLLM/SGLang 引擎；
> 3. 它具体做什么；
> 4. 代码在本仓库的哪里。
>
> 行号为撰写本文时的版本；如果代码有变动，请以实际文件为准。

## 1. 一句话定义

**Connector 是连接"推理引擎"和"LMCache 缓存系统"的粘合层。**

- LMCache 只负责三件事：KV 存到哪（L1/L2/后端）、怎么查（token 前缀 hash + lookup）、怎么淘汰。
  它**不知道** vLLM/SGLang 的显存长什么样，不知道什么时机该存该取。
- vLLM/SGLang 只负责调度和模型 forward，它**不知道**外部缓存的协议、分块、存储后端。
- connector 坐在中间：朝引擎一侧实现引擎规定的接口（生命周期钩子），朝 LMCache 一侧调用
  `retrieve/store/lookup` 和内存对象接口。

所以答案不是二选一：**connector 的"壳"是引擎概念**（vLLM 的 `KVConnectorBase_V1`、
SGLang 的 `--enable-lmcache` 开关），**connector 的能力来自 LMCache**，而本仓库提供的是
**Ascend NPU 版本的粘合实现**。

```text
vLLM / SGLang 引擎
   │  引擎侧接口：KVConnectorBase_V1 / LMCacheConnector
   ▼
Connector（粘合层）  ← 本仓实现 NPU 版本；MP 主线的壳在上游
   │  LMCache 侧接口：lmcache_engine.retrieve/store/lookup
   ▼
LMCache 缓存引擎 → 存储后端
```

## 2. 两种 connector：指挥部与工程队

把 connector 拆成两层，是理解全部代码的关键：

| 层 | 俗称 | 职责 | 主要在哪 |
| --- | --- | --- | --- |
| 引擎级 connector | 指挥部 | 生命周期编排：前向开始前加载、前向结束后保存、请求完成通知、块释放延迟、抢占处理 | 壳在上游 LMCache；本仓只做 Ascend 定制薄层 |
| 设备级 connector | 工程队 | 真正的 KV 搬运：paged KV ↔ LMCache 内存对象，调 NPU 原生 kernel | 本仓完整实现 |

- 指挥部决定"**什么时候**搬、哪些请求、哪些 token"；
- 工程队决定"**怎么**搬"（哪层、哪个 plane、哪个 block、什么 stride）。

数据方向固定两条：

- `batched_to_gpu()`：**retrieve**。LMCache 内存对象 → 引擎分页 KV（把缓存回灌显存）。
- `batched_from_gpu()`：**store**。引擎分页 KV → LMCache 内存对象（把显存 KV 卸载进缓存）。

## 3. 本仓库里的 connector 清单

### 3.1 引擎级 connector（本仓 = 继承上游 + Ascend 定制）

| 类 / 函数 | 位置 | 上游基类 | 作用 |
| --- | --- | --- | --- |
| `LMCacheAscendConnectorV1Dynamic` | [lmcache_ascend_connector_v1.py](../lmcache_ascend/integration/vllm/lmcache_ascend_connector_v1.py#L14-L20) | `LMCacheConnectorV1Dynamic` | vLLM 旧内嵌式动态连接器，本体几乎全部沿用上游 |
| `LMCacheAscendConnectorV1Impl` | [vllm_v1_adapter.py](../lmcache_ascend/integration/vllm/vllm_v1_adapter.py#L31-L48) | `LMCacheConnectorV1Impl` | 挂在 worker 上的实现，覆盖 `start_load_kv`/`wait_for_save`/`get_finished`/`request_finished` |
| `LMCacheAscendHoleConnectorV1Dynamic` | [lmcache_ascend_hole_connector_v1.py](../lmcache_ascend/integration/vllm/lmcache_ascend_hole_connector_v1.py#L21-L24) | `LMCacheConnectorV1Dynamic` | Hole/断层特性专用连接器，Impl 在 `vllm_v1_adapter_hole.py` |
| `LMCacheConnector__init__`（SGLang） | [sglang_adapter.py](../lmcache_ascend/integration/sglang/sglang_adapter.py#L16-L72) | 上游 SGLang connector `__init__` | 建 `lmcache_engine`、保存 K/V 池并 `post_init` |
| `LMCacheLayerwiseConnector_start_load_kv`（SGLang） | [sglang_adapter.py](../lmcache_ascend/integration/sglang/sglang_adapter.py#L95-L164) | 上游逐层加载入口 | `lookup(pin=True)` → TP 全服最小 token → `retrieve_layer` 逐层装入 |

### 3.2 设备级 connector（本仓完整实现，真正搬 KV）

全部在 [npu_connectors.py](../lmcache_ascend/v1/npu_connector/npu_connectors.py)：

| 类 | 行号 | 面对谁 |
| --- | --- | --- |
| `VLLMBufferLayerwiseNPUConnector` | [45](../lmcache_ascend/v1/npu_connector/npu_connectors.py#L45) | vLLM 逐层 + blend |
| `VLLMPagedMemNPUConnectorV2` | [458](../lmcache_ascend/v1/npu_connector/npu_connectors.py#L458) | vLLM 标准分页路径 |
| `VLLMPagedMemLayerwiseNPUConnector` | [1238](../lmcache_ascend/v1/npu_connector/npu_connectors.py#L1238) | vLLM 逐层分页 |
| `SGLangNPUConnector` | [1613](../lmcache_ascend/v1/npu_connector/npu_connectors.py#L1613) | SGLang 标准路径 |
| `SGLangLayerwiseNPUConnector` | [1623](../lmcache_ascend/v1/npu_connector/npu_connectors.py#L1623) | SGLang 逐层路径 |

工厂函数 [CreateNPUConnector](../lmcache_ascend/v1/npu_connector/__init__.py#L24-L99)
按 `EngineType` + `use_layerwise` + blending 选择上面的实现，替换上游的 `CreateGPUConnector`。

## 4. vLLM 旧内嵌式：connector 怎么被调用

以 `VLLMPagedMemNPUConnectorV2` 所在的标准链路为例：

```text
vllm serve 启动
→ 引擎按 --kv-transfer-config 的 kv_connector 名创建
  LMCacheAscendConnectorV1Dynamic                    [本仓壳，上游逻辑]
→ worker 上绑定 LMCacheAscendConnectorV1Impl         [本仓]
→ 注册 KV 池 → CreateNPUConnector 创建设备 connector [本仓]
→ 每个请求：
    前向开始前：Impl.start_load_kv
                  → 上游 → lmcache_engine.retrieve
                  → VLLMPagedMemNPUConnectorV2.batched_to_gpu / to_gpu
                  → NPU kernel 把缓存回灌分页 KV
    模型 forward（读分页 KV）
    前向结束后：Impl.wait_for_save
                  → 上游 → lmcache_engine.store
                  → batched_from_gpu / from_gpu
                  → NPU kernel 把分页 KV 卸载到内存对象 → 存储后端
    调度收尾：Impl.get_finished / request_finished
                  → 保证异步保存完成前不提前释放请求的 KV 块
```

### 4.1 本仓在继承基础上加了哪些 Ascend 定制

[LMCacheAscendConnectorV1Impl](../lmcache_ascend/integration/vllm/vllm_v1_adapter.py#L31-L48)：

- [start_load_kv](../lmcache_ascend/integration/vllm/vllm_v1_adapter.py#L73-L82)：
  上游加载之后调用 `_mark_failed_p2p_loads_for_recompute()`，把 P2P 拉取失败的块标为
  无效，让 vLLM 本地重算，而不是用坏数据继续。
- [wait_for_save](../lmcache_ascend/integration/vllm/vllm_v1_adapter.py#L133-L141)：
  三处 Ascend 差异——MLA/DSA 默认 `save_only_first_rank` 时被动 rank 直跳保存；
  记录一个 NPU `ordering_event` 保证保存顺序；`slot_mapping` 先 `pin_memory()` 再搬 NPU。
- [get_finished](../lmcache_ascend/integration/vllm/vllm_v1_adapter.py#L384-L422) 与
  [request_finished](../lmcache_ascend/integration/vllm/vllm_v1_adapter.py#L448-L488)：
  异步 store 时用 `delay_free` 延后 vLLM 释放 KV 块，直到 `get_finished_stores` 确认保存
  完成；中止请求主动 `lookup_unpin` 并等待未完成保存。
- [handle_preemptions](../lmcache_ascend/integration/vllm/vllm_v1_adapter.py#L424-L446)：
  被抢占请求释放 lookup pin，必要时等待其异步 store 排空。

### 4.2 数据面在 NPU connector 里怎么走

[to_gpu](../lmcache_ascend/v1/npu_connector/npu_connectors.py#L784-L846) 的链路：

```text
to_gpu(memory_obj, start, end, kvcaches, slot_mapping)
→ initialize_kvcaches_ptr / _initialize_pointers   把各层 KV 张量地址收集成指针表
→ c_ops.multi_layer_kv_transfer(..., False, ...)   False = to_gpu 方向
   → C++/AscendC kernel 按 slot_mapping 把对象内容 scatter 到分页 KV
```

[from_gpu](../lmcache_ascend/v1/npu_connector/npu_connectors.py#L848-L950)：

```text
from_gpu(memory_obj, start, end, kvcaches, slot_mapping)
→ 指针表准备；slot_mapping 必要时先搬到 NPU
→ 直接路径：c_ops.multi_layer_kv_transfer(..., True, ...)   True = from_gpu 方向
→ 或 fused 路径：c_ops.fused_multi_layer_kv_transfer（带 staging 缓冲）
→ 目标不是 NPU 且未设 no_sync：store_stream.synchronize()  确保卸载完成
```

## 5. SGLang：patch 怎么把 connector 接进来

1. **安装期补丁**。[sglang_patch.py](../lmcache_ascend/integration/patch/sglang/sglang_patch.py#L78-L136)
   的 `_patch_init_memory_pool` 会直接改写已安装 SGLang 源码中 `init_memory_pool` 的
   Ascend 分支，注入：

   ```python
   if self.server_args.attention_backend == "ascend":
       if self.server_args.enable_lmcache:
           import lmcache_ascend
   ```

   作用与 vLLM 的 `--kv-transfer-config` 对应：把本仓插件接进 SGLang 启动路径。

2. **构造连接器**。[LMCacheConnector__init__](../lmcache_ascend/integration/sglang/sglang_adapter.py#L16-L72)
   拿到 K/V 池后调 `init_lmcache_engine` 创建引擎，`self.kvcaches = [k_pool, v_pool]` 并按
   NPU 期望的 `[layer, blocks, bs, heads, dim]` 布局传给 `post_init`。

3. **逐层加载**。[LMCacheLayerwiseConnector_start_load_kv](../lmcache_ascend/integration/sglang/sglang_adapter.py#L95-L164)
   `lookup(pin=True)` 命中后，经 `global_min_tokens` 做 TP 全服同步（NPU 高负载下避免
   compute kernel 与 HCCL all_reduce 竞争而死锁），再 `retrieve_layer` 生成逐层生成器，
   由引擎逐层推进。

4. **数据面**。与 vLLM 同一套 NPU 家族：`SGLangNPUConnector` / `SGLangLayerwiseNPUConnector`
   的 `batched_to_gpu` / `batched_from_gpu`，最终同样落到 `c_ops`。

## 6. MP 主线：connector 不在本仓，本仓提供"手脚"

当前 main 的推荐架构是独立 `lmcache server` + `LMCacheMPConnector`：

| 组件 | 位置 |
| --- | --- |
| `LMCacheMPConnector`（vLLM 挂载的 connector） | **上游 LMCache，不在本仓** |
| `UnifiedLMCacheMPConnector`（SGLang） | **上游 LMCache，不在本仓** |
| `NpuDeviceSpec` / `NpuIPCWrapper`（KV pool IPC 注册） | **上游 LMCache，不在本仓** |
| plane 几何补丁 `shape_desc.py` | 本仓 |
| 物理/逻辑 block 补丁 `logical_block_size.py` | 本仓 |
| 块级搬运 `csrc/mp_mem_kernels.cpp` + `pybind.cpp` + AscendC kernel | 本仓 |
| 插件激活 `lmcache_ascend/__init__.py` | 本仓 |

所以 MP 模式下"指挥部"（connector）在上游 LMCache，本仓的角色是把 NPU 的手臂接给指挥
部：把引擎 KV 布局翻译成字节几何、把 LMCache 对象地址解析成设备地址、用 AscendC kernel
完成跨进程块搬运。更完整的函数级调用图见
[code_call_graph_zh.md](mp_mode/code_call_graph_zh.md)。

## 7. 一张图总结

```text
引擎侧视角                        LMCache 侧视角
------------------------          -----------------------
KVConnectorBase_V1/--enable-lmcache    lmcache_engine.retrieve/store/lookup
        │                                      ▲
        ▼                                      │
  引擎级 connector（指挥部）           设备级 connector（工程队）
  本仓：Dynamic/Impl/Hole/SGLang init   本仓：NPUConnector 五件套
        └────────→ 编排调用 ────────────────→ batched_to_gpu / batched_from_gpu
                                              │
                                              ▼
                                        c_ops NPU kernel（AscendC）
```

- **指挥部**（引擎级 connector）：什么时候存、什么时候取、哪些请求完成才能释放显存块。
- **工程队**（设备级 connector）：NPU paged KV 与 LMCache 内存对象之间按层/plane/block 搬运字节。
- **MP 主线**：指挥部在上游 `LMCacheMPConnector`；本仓只提供 NPU 工程能力与布局补丁。