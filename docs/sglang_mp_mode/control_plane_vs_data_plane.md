# 控制面 vs 数据面：用一次真实的 Stored/Retrieved 讲清楚

> 本文目标：把 MP mode 里最常说的两个词——**控制面（control plane）**与**数据面（data plane）**——落到具体代码调用链上。
>
> 方法：以你已经跑通的 `test_npu_lmcache_driven_e2e.py`（3 passed）为锚点，把你的测试日志每一行对应到代码的哪一步；控制面在上游 LMCache，数据面在本仓 LMCache-Ascend。引用行号均来自你的真实测试日志 `/workspace/results/npu_mp_native_e2e.log`。

---

## 1. 一句话定义

| 面 | 一句话 | 传什么 | 走什么通道 | 代码在哪 |
|---|---|---|---|---|
| **控制面** | 让"谁、在哪、存/取什么"达成一致 | 小消息：lookup 结果、store/retrieve 请求、完成通知 | ZMQ/gRPC 套接字 | **上游 LMCache** |
| **数据面** | 真正把 KV 大块数据搬过去/搬回来 | 大块 KV 字节（几十 KB~几百 MB） | NPU IPC 句柄 + AscendC kernel 搬运 | **本仓 LMCache-Ascend** |

关键直觉：**控制面是一张"快递单"，数据面是"搬货"本身。** 控制面只传递"哪辆车、哪个地址、装多少件"，货物本体从不经过 ZMQ。

---

## 2. 完整链路图

### 2.1 冷请求（store）全流程

```text
┌────────────────────── SGLang / vLLM 引擎进程 ──────────────────────┐
│ scheduler                                                    │
│   │ 1. 请求到来，做 lookup 查询                                │
│   ▼                                                          │
│ LMCacheMPConnector / UnifiedLMCacheMPConnector   ← 上游      │
│   │ 2. ZMQ: {lookup: token_hash}                             │
│   ▼                                                          │
│ TP worker（每 rank 一个）                                     │
│   │ 3. 注册 KV pool：NpuIPCWrapper.wrap(...)  → 再走 ZMQ 发给 server
│   │    （只传"句柄"，不传 KV 内容）                            │
└───│──────────────────────────────────────────────────────────┘
    │ 控制面：ZMQ(/gRPC)  ── 小消息
    ▼
┌────────────────────── lmcache server 进程（上游） ─────────────────┐
│ server.py ZMQ handler                                        │
│   │ 4. 收到 store 请求（含 IPC 句柄、block_ids、方向）           │
│   ▼                                                          │
│ LMCacheDrivenTransferModule                       ← 上游     │
│   │   日志：Stored 256 tokens in 0.005 seconds                │
│   │   (lmcache_driven_transfer.py:834)                       │
│   ▼                                                          │
│ device_ops.multi_layer_block_kv_transfer(...)  ← 上游门面     │
└───│──────────────────────────────────────────────────────────┘
    │ 数据面：NPU IPC 句柄直读 + AscendC kernel 搬运
    ▼
┌────────────────────── 本仓原生层 ────────────────────────────────┐
│ lmcache_ascend.c_ops（pybind.cpp 暴露）                         │
│   │ 5. multi_layer_block_kv_transfer /                        │
│   │    execute_object_group_transfer   ← pybind 绑定          │
│   ▼                                                          │
│ csrc/mp_mem_kernels.cpp + csrc/kernels/multi_layer/*          │
│   │ 6. AscendC kernel：按 shape_desc 的 plane 几何逐块搬运      │
│   ▼                                                          │
│ 引擎 NPU 显存 KV plane ←→ 主机内存 L1 对象                      │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 热请求（retrieve）只是方向反过来

```text
server 收到 retrieve 请求
  → device_ops.multi_layer_block_kv_transfer(方向=H2D)
  → AscendC kernel 把主机 L1 对象 scatter 回引擎的相同 block
  → 日志：Retrieved 256 tokens in 0.004 seconds
       (lmcache_driven_transfer.py:1100)
```

注意控制面还负责**完成通知**：native 入队只代表"已提交"，不代表"已完成"，之后要靠事件/流同步告知上层"可以读了"（见第 5.5/5.6 节）。

---

## 3. 控制面详解

### 3.1 控制面由哪些消息组成

| 消息 | 含义 | 载有大块 KV 吗 |
|---|---|---|
| `lookup(token_hash)` | 引擎问 server："这个前缀有多少 token 命中？" | 否 |
| `store / retrieve(key, block_ids, 方向)` | 让 server 执行一次搬运 | 否（只带"句柄+地址+块号"） |
| KV pool 注册（IPC 句柄交换） | 引擎把 KV 显存"介绍"给 server | 否（句柄 64~几百字节） |
| 完成/失败通知 | 搬运结束或出错 | 否 |

### 3.2 控制面代码在哪（全部在上游，容器里看）

> 上游源码在容器 `/workspace/src/LMCache/lmcache/`。行号引用来自你的测试日志，可逐行核对。

| 环节 | 上游文件 | 你日志里的证据 |
|---|---|---|
| 引擎侧查询客户端 | `lmcache/v1/lookup_client/lmcache_lookup_client.py` | —（测试未覆盖） |
| server 端传输模块（Stored/Retrieved 日志出处） | `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py` | `:834` Stored、`:1100` Retrieved |
| 传输上下文选择 | `lmcache/v1/multiprocess/transfer_context/worker_transfer.py` | `:1069` Creating transfer context (device_type=npu, mode=lmcache_driven) |
| KV 分组（服务端建组） | `lmcache/v1/kv_layer_groups.py` | `:476` KV layer groups 日志 |
| 引擎 KV 格式探测 | `lmcache/v1/gpu_connector/kv_format/detection.py` | `:69` Engine KV Format 日志 |
| 首层张量几何打印 | `lmcache/v1/gpu_connector/utils.py` | `:651` Group 0 first-layer tensor 日志 |
| SGLang 侧控制面入口 | `lmcache/integration/sglang/unified_lmcache_mp_connector.py` | —（下一步 E2E 会走到） |

**结论：控制面的代码你在本仓一行都找不到——因为本仓不实现协议，只实现数据面的"搬运工"。**

---

## 4. 数据面详解（本仓代码，逐步贴）

数据面在测试里被完整走了一遍。顺着调用链，每一步贴关键代码。

### 4.1 第 1 步：引擎 worker 把 KV 池"介绍"给缓存（IPC 句柄导出）

`tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py` 的 `_worker()`，模拟引擎 TP worker 的动作：

```python
# tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py:117-167（节选）
def _worker(device_index: int, conn) -> None:
    torch.npu.set_device(device_index)
    wrappers: list[NpuIPCWrapper] = []
    for layer in range(NL):
        latent = torch.zeros(NB, BS, 1, W_LATENT, device=device)
        rope = torch.zeros(NB, BS, 1, W_ROPE, device=device)
        ...
        # 每层一个聚合 plane 的 wrapper，与 vLLM-Ascend 注册元组时
        # wrap_kv_caches 生成的结构完全一致。
        wrappers.append(NpuIPCWrapper.wrap((latent, rope)))
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for plane in planes:
            plane.mul_(1.0)  # 在 worker 流上执行写入
    # CANN 构建拒绝同进程再次导入，worker 为每次消费各导出一个事件
    producer_a = torch.npu.Event(enable_timing=False, interprocess=True)
    producer_a.record(stream)
    ...
    conn.send({
        "wrappers": encoder.encode(wrappers),       # ← 走控制面的是这段编码
        "producer_a": producer_a.ipc_handle(),      # ← 跨进程同步事件句柄
        ...
    })
```

看懂三件事：

1. **`NpuIPCWrapper.wrap((latent, rope))`**（上游类，本仓通过 `ops.py` 提供背后实现）把一个 MLA 层的两个 plane 打包成一个跨进程可解码的句柄——**传出去的是句柄编码，不是 KV 字节**；
2. **`torch.npu.Event(interprocess=True)` + `ipc_handle()`**：生产者事件句柄，让 server 端能知道"worker 把 KV 写完了没有"——这是数据面的**同步信号**；
3. 这句柄编码之后是塞进 ZMQ 消息的——**控制面载荷里第一次出现"数据面句柄"**，这就是两个面的交汇点。

### 4.2 第 2 步：server 端导入句柄并创建 cache context

父进程（扮演 server）解码后建上下文：

```python
# tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py:378-392（节选）
from lmcache.v1.platform.devices.npu import NpuDeviceSpec

cache_context = NpuDeviceSpec().create_cache_context(
    kv_caches,          # ← 解码出来的 NpuIPCWrapper 列表（每层一个）
    CHUNK,
    layout_hints=LayoutHints(kv_layout="NHD"),
    engine_group_infos=(),
    engine_type=EngineType.VLLM,
)
assert (
    cache_context.get_engine_kv_format(0)
    == lmcache_native.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS
)
```

这一步就是 `lmcache_driven` 模式的灵魂：**server 进程通过句柄直接拿到引擎显存里 KV 的地址**，之后的搬运不再经过引擎进程帮忙（对比 `engine_driven`：搬运动作由引擎 worker 自己做）。

### 4.3 第 3 步：store 发生时，走真实的传输模块

```python
# tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py:434-452（节选）
store_handle, stored = module.store(
    key, 1, block_ids, message["producer_a"]
)
# ↑ 模块提交完成操作后返回；读取主机暂存字节前必须排空传输流
cache_context.stream.synchronize()

stored_obj = storage_manager.objects[CHUNK_KEY]
host = stored_obj.raw_tensor.view(torch.float32)
assert torch.allclose(host.view(NL, NB * BS, HIDDEN), _expected_staging())
```

注意 `module.store` 里的 `producer_a`：**事件句柄沿控制面传到 server，在数据面里兑现为"等 worker 写完再搬"**。这就是你的日志里 `Stored 256 tokens in 0.005 seconds` 那一行背后的动作（`lmcache_driven_transfer.py:834`）。

### 4.4 第 4 步：Python → C++ 边界（本仓 pybind 门面）

上游 `LMCacheDrivenTransferModule` 内部不直接搬数据，它调用 `device_ops.multi_layer_block_kv_transfer`。这个符号由本仓绑定：

```cpp
// csrc/pybind.cpp:219-252（节选）
m.def(
    "multi_layer_block_kv_transfer",
    [](const torch::Tensor &paged_buffer_ptrs_tensor,
       std::vector<int64_t> lmcache_objects_ptrs,
       const torch::Tensor &block_ids, const torch::Device &device,
       const py::object &direction, PageBufferShapeDesc shape_desc,
       int lmcache_chunk_size, const py::object &engine_kv_format,
       int skip_prefix_n_blocks) {
        ...
        // 直接启动路径：先排空 torch_npu 的 ACL 任务队列，再入队
        // kernel + memcpy，期间释放 GIL。
        py::gil_scoped_release release;
        multi_layer_block_kv_transfer(...);
    },
    py::arg("paged_buffer_ptrs_tensor"), py::arg("lmcache_objects_ptrs"),
    py::arg("block_ids"), py::arg("device"), py::arg("direction"),
    py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
    py::arg("engine_kv_format"), py::arg("skip_prefix_n_blocks"));
```

参数签名就是一份"数据面作业单"：

| 参数 | 含义 |
|---|---|
| `paged_buffer_ptrs_tensor` | 引擎侧每层/每 plane KV 的 NPU 地址表 |
| `lmcache_objects_ptrs` | 主机端 L1 对象地址表 |
| `block_ids` | 要搬哪些 block |
| `direction` | `D2H`（store）/ `H2D`（retrieve） |
| `shape_desc` | 布局几何（见 4.5） |
| `engine_kv_format` | 18 种格式之一，决定用哪个 AscendC kernel |

### 4.5 第 5 步：layout 几何怎么进 kernel（plane 描述符）

KV 不止一种排布，kernel 需要知道"每层几个 plane、每个 token 多少字节、block 步长多大"。上游标量描述符不够，本仓在运行时补字段：

```python
# lmcache_ascend/v1/shape_desc.py:141-182（节选）
def make_page_buffer_shape_desc(kv_caches, engine_kv_format, ...):
    desc = original(kv_caches, engine_kv_format, ...)   # 先调上游工厂
    if engine_kv_format in plane_tuple_fmts:             # 格式 16/17
        planes = kv_caches[layer_idx]
        desc.num_planes = len(planes)                    # ← 上游没有
        desc.plane_slot_bytes = tuple(
            int(t.shape[2]) * int(t.shape[3]) * int(t.element_size())
            for t in planes
        )                                                # ← 上游没有
        desc.plane_block_stride_bytes = tuple(
            int(t.stride(0)) * int(t.element_size()) for t in planes
        )                                                # ← 上游没有
    return desc
```

对应 C++ 结构（`csrc/mp_mem_kernels.h:54-83`）就是 `PageBufferShapeDesc` 里 Ascend 专用的 `num_planes / plane_slot_bytes[4] / plane_block_stride_bytes[4]`。**没有这三个字段，AscendC kernel 不知道 MLA 的 latent/rope 两个 plane 各自从哪开始、步长多少——这正是"没插件就跑不了 MLA"的直接原因。**

### 4.6 第 6 步：批量/对象组执行入口

对更复杂的批量场景，走的是带执行计划的入口：

```cpp
// csrc/pybind.cpp:295-316（节选）
m.def(
    "execute_object_group_transfer",
    [](int direction, const torch::Device &device,
       size_t host_buffer_alignment,
       const std::vector<KernelGroupSpec> &kernel_group_specs,
       const std::vector<BatchStep> &batch_steps) {
        return execute_object_group_transfer(...);
    },
    ...);
// 该绑定返回仅表示执行计划已提交，不表示异步操作已完成；
// 调用方需负责 stream 同步以及指针表、block_ids 和 staging 缓冲区的生命周期。
```

`KernelGroupSpec`（静态布局）+ `BatchStep`（staging 拷贝 + launch 参数）把"哪个 kernel、哪段 host staging、哪批 block"打包成执行计划——这也是 Qwen3.5/3.8 混合 Mamba 走 `--separate-object-groups` 时底层用的通路。

### 4.7 兜底：native 事件与完成通知（数据面的同步机制）

异步搬运完成后要通知上层。`csrc/event_recorder.cpp` + `completion_recorder.cpp` 通过 pybind 暴露：

```cpp
// csrc/pybind.cpp:128-147（节选）
m.def("record_event_on_stream", &record_event_on_stream,
      py::arg("cuda_stream_ptr"), py::arg("event_type_name"),
      py::arg("session_id"), py::arg("str_metadata"),
      py::arg("int_metadata"), py::call_guard<py::gil_scoped_release>());
m.def("drain_recorded_events", &drain_recorded_events);
m.def("record_completion_on_stream", &record_completion_on_stream, ...);
m.def("drain_recorded_completions", []() { ... });
```

这是"**kernel 入队 ≠ 完成**"的兑现机制：跨进程的完成通知走控制面，但**记录"哪个流、何时到点"的设备事件走数据面**——两者合起来才能让 server 安全地通知上层"可以读了"。

---

## 5. 你的日志每一行对应到哪一步

| 你的日志 | 属于 | 对应本文 |
|---|---|---|
| `Creating transfer context (device_type=npu, mode=lmcache_driven)` (worker_transfer.py:1069) | 控制面 | 3.2 表 |
| `Engine KV Format: NL_X_NP_X_NB_BS_ONE_HS` (detection.py:69) | 控制面（探测结果）→ 数据面（选 kernel） | 4.2 断言 |
| `Group 0 first-layer tensor: shape=(16,16,1,128) stride=...` (utils.py:651) | 控制面（几何打印）→ 数据面（shape_desc） | 4.5 |
| `KV layer groups: KernelGroupInfo(... hs=144 ...)` (kv_layer_groups.py:476) | 控制面（分组决定） | 3.2 表 |
| `Stored 256 tokens in 0.005 seconds` (lmcache_driven_transfer.py:834) | 控制面发出请求 → **数据面完成搬运** | 2.1 全链路 |
| `Retrieved 256 tokens in 0.004 seconds` (lmcache_driven_transfer.py:1100) | 同上（方向 H2D） | 2.2 |

**判断标准：凡是日志里出现"搬运完成"的速度数字（毫秒级），背后必然已经走完了 4.4→4.6 的本仓原生链。** 如果你哪天看到这个数字变成几十毫秒以上，就要怀疑 native 没绑定、退到了 torch 基线。

---

## 6. 这对 SGLang 适配意味着什么

```text
控制面（上游统一 connector + ZMQ 协议）        → 不用你重写，只需审计
数据面-协议侧（multi_layer_block_kv_transfer 签名）→ 上游定死，本仓已实现
数据面-布局侧（shape_desc / kv_groups / kernel）  → 本仓字段，SGLang 布局不同时在这补
```

SGLang 适配的实质，就是把第 4.1 步"引擎 worker 注册 KV 池"这件事，从 vLLM 的注册形态换成 SGLang 的（K/V pool + RadixAttention + FULL/SWA/MAMBA component）；只要 SGLang 的 KV 布局能被现状的 `EngineKVFormat` + `shape_desc` 表达，数据面一行都不用改——这就是先跑 E2E、看第一个真实报错的原因。