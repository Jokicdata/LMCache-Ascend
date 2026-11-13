# 适配前后完整调用流程对比（掰开揉碎版）

> 本文档回答一个问题：**SGLang + LMCache-Ascend MP mode 这条链路，没有适配时
> 是怎么崩的，适配之后每一次调用到底走了哪些函数、经过了我写的哪些代码。**
>
> 对应补丁：`8bf0444`（xby）/ `093c4ea`（sglang_mp_dev）
> 实测环境：910B4 × TP4 + Qwen2.5-7B-Instruct + lmcache server（chunk 256）

---

## 0. 一图总览：前 vs 后

```text
【适配前】                          【适配后】
─────────────────────────          ─────────────────────────
SGLang scheduler                    SGLang scheduler
  │                                   │
  ▼                                   ▼
import lmcache                      import lmcache
  │                                   │
  ▼                                   ▼
LMCache 平台初始化                   LMCache 平台初始化
  │                                   │
  ▼                                   ▼
import lmcache_ascend               import lmcache_ascend
  │                                   │
  ▼                                   ▼
激活段（只有 vllm 分支）             激活段
                                    │ ├ vllm 分支（原有）
  ⚠ 什么都不为 sglang 做             │ ├ sglang 分支 ★新
  │                                   │ │  └ install_mp_connector_compat()
  │                                   │ │     └ 包装 LMCacheMPConnector.__init__
  ▼                                   │ ▼
LMCRadixCache.__init__              LMCRadixCache.__init__
  │                                   │
  ▼                                   ▼
LMCacheMPConnector(                 LMCacheMPConnector(            ← 现在进入包装器
  k_pool=大张量,                      k_pool=大张量,
  v_pool=大张量)                      v_pool=大张量)
  │                                   │
  ▼                                   ▼
_validate_sglang_kv_pools()         _compat_init ★新（拦截）
  if not k_pool ← 对张量求布尔值      │ _split_stacked_pool(k_pool)
  │                                   │   (28,12203,128,1,128) → 28 个视图
💥 RuntimeError                      │ _split_stacked_pool(v_pool)
   Boolean value of                  │   同上
   Tensor ... ambiguous              ▼
                                    _validate_sglang_kv_pools()
                                    │ if not k_pool ← 现在是列表，合法
                                    ▼
                                    get_event_ipc_backend(npu)
                                    ▼
                                    _wrap_sglang_kv_caches()
                                    │ wrap_one_kv_cache × 56
                                    │   └ NpuIPCWrapper（c_ops）
                                    ▼
                                    register_kv_cache → ZMQ → server
                                    ▼
                                    心跳启动，服务就绪 ✅
```

**全部差异 = 中间插进去的"翻译层"：两张大张量 → 两个逐层视图列表，仅此而已。**

---

## 1. 适配前的调用流程（崩溃版逐帧回放）

### 1.1 启动链（走到崩溃点）

```text
T0  python -m sglang.launch_server ... --enable-lmcache ...
T1  Scheduler.__init__                       （sglang/srt/managers/scheduler.py:559）
T2    └ kv_cache_builder.build_kv_cache()    （mem_cache/kv_cache_builder.py:394）
T3        └ create_tree_cache(tree_context)  （mem_cache/registry.py:234）
T4            └ get_memory().enable_lmcache == True
                └ default_radix_cache_factory(ctx)
                    └ LMCRadixCache(params, model_config, tp_size, rank, tp_group)
                                               （storage/lmcache/lmc_radix_cache.py:125）
```

注意 T4：`enable_lmcache` 分支排在 unified 工厂**之前**，所以只要带了这个开关，
无论什么模型都强制走 `LMCRadixCache`——这就是为什么 Qwen3.8（hybrid）当时
在同一个文件更早的位置崩（k_buffer），而 Qwen2.5-7B（标准模型）能走到更深处。

### 1.2 LMCRadixCache.__init__ 内部（容器里的 SGLang 代码）

```python
# lmc_radix_cache.py:113~127（节选，容器路径 /sgl-workspace/sglang/...）
kvcache = self.token_to_kv_pool_allocator.get_kvcache()
connector_kwargs = dict(
    sgl_config=model_config,
    tp_size=tp_size,
    rank=rank,
    k_pool=getattr(
        kvcache, "k_buffer",
        getattr(self.token_to_kv_pool_allocator._kvcache, "k_buffer"),  # ← 双层 getattr
    ),
    v_pool=...,   # 同构
    tp_group=...,
)
```

两个事实：

1. 标准模型的 `MHATokenToKVPool` **有** `k_buffer` 属性 → 第一层 getattr 成功，
   顺利拿到堆叠大张量（Qwen3.8 的 HybridLinearKVPool 没有它 → 当年崩在这里，
   一步都没走远）。
2. 拿到的是 **一整块** `(28, 12203, 128, 1, 128)` 的 bf16 张量（10.43 GB），
   不是 28 张小张量的列表。

### 1.3 崩溃点

```python
# lmc_radix_cache.py:149
self.lmcache_connector = LMCacheMPConnector(
    page_size=params.page_size, host=..., port=..., **connector_kwargs
)
    │
    ▼
# multi_process_adapter.py:188（LMCache dev，容器路径 /workspace/src/LMCache/...）
def __init__(self, sgl_config, tp_size, rank, page_size, host, port,
             k_pool: list[torch.Tensor], v_pool: list[torch.Tensor], ...):
    device = _validate_sglang_kv_pools(k_pool, v_pool)   # ← 第一行就调校验
        │
        ▼
# multi_process_adapter.py:73
def _validate_sglang_kv_pools(k_pool, v_pool):
    if not k_pool or not v_pool:        # ← 💥 对 12203×128×1×128 个元素求"真假"
        raise ValueError(...)           #     PyTorch：无法回答 → RuntimeError
```

**崩溃的本质**：Python 的 `not x` 会调 `x.__bool__()`。列表的 `__bool__` 是
"是否为空"；多元素张量的 `__bool__` 没有定义（一个张量有 2 亿个值，说"真"
还是"假"？）→ PyTorch 主动抛 `RuntimeError`。**这与 NPU 无关，与显存无关，
纯形状方言问题——CUDA 上同样会挂。**

### 1.4 崩溃之后发生了什么

```text
TP0~3 四个 scheduler 各自抛同样异常 → sigquit → kill_process_tree → 服务退出
lmcache server 侧：什么都还没收到（连接器死在构造第一行，ZMQ 都没建）
```

---

## 2. 适配后的调用流程（完整版，三个阶段）

### 阶段 A：插件激活（import lmcache 的瞬间）

**这一步决定"补丁什么时候装上"——它必须发生在连接器构造之前，否则拦截不到。**

```text
A0  Scheduler 进程启动: import sglang（此时 "sglang" 进入 sys.modules）
A1  Scheduler.__init__ → build_kv_cache（KV 池已建好，k_buffer 已是堆叠张量）
A2  registry 导入 lmc_radix_cache.py → 该文件顶部 import lmcache
A3      └ lmcache/__init__.py 的平台初始化: NpuDeviceOps.ensure_native()
A4          └ import lmcache_ascend.c_ops      ← 插件被上游主动拉起
A5              └ lmcache_ascend/__init__.py 执行到底部激活段（:211）
                    │
A6              _seed_partial_lmcache_device_ops()   （:220，原有，必须最先）
                    │  桥接 device_ops，防止半初始化 ImportError
A7              if _is_vllm_runtime(): ...            （:222，原有，本场景为 False）
A8              if _is_sglang_runtime():              （:225，★新分支）
A9                  │  "sglang" in sys.modules == True（A0 就位了）
A10                 └ try: _patch_sglang_mp_connector_compat()   （:233-234）
A11                     └ except Exception: 只打 warning           （:235）
                        │  ★守卫意义：此刻仍在 import lmcache 中途，任何异常
                        │  都会杀死整个插件 → 退到 torch 基线慢 8 倍
A12             _patch_make_page_buffer_shape_desc()  （:244，原有）
A13             LMCACHE_ASCEND_PATCHED = True          （:246）
```

激活链上的三个新函数（全部在本仓）：

| 函数 | 位置 | 干什么 |
|---|---|---|
| `_is_sglang_runtime()` | [\_\_init\_\_.py:31](file:///d:/proj/LMCache-Ascend/lmcache_ascend/__init__.py#L31) | 两个判据：`"sglang" in sys.modules`（主判据，A0 已保证为真）或 `sys.argv` 含 sglang（兜底 CLI 场景） |
| `_patch_sglang_mp_connector_compat()` | [\_\_init\_\_.py:186](file:///d:/proj/LMCache-Ascend/lmcache_ascend/__init__.py#L186) | 空壳转发（本仓既有模式：`__init__.py` 只管"何时装"，具体逻辑放 integration/）→ 调 `install_mp_connector_compat()` |
| `install_mp_connector_compat()` | [mp_connector_compat.py:78](file:///d:/proj/LMCache-Ascend/lmcache_ascend/integration/sglang/mp_connector_compat.py#L78) | 真正的安装动作，见阶段 B 开头 |

**为什么补丁必然赶在构造之前？** 看 A2：`lmc_radix_cache.py` 的
`import lmcache`（A2）和 `LMCacheMPConnector(...)` 的构造（B0）在同一个函数
`__init__` 的不同行——import 在文件顶部就执行了，构造要等 `__init__` 跑到
第 149 行。A 阶段整个发生在 import 期间 → **必然先于 B**。

### 阶段 B：连接器构造（补丁拦截点，掰开揉碎）

```text
B0  LMCRadixCache.__init__ 第 149 行:
    LMCacheMPConnector(page_size=128, host="tcp://127.0.0.1", port=5556,
                       sgl_config=..., tp_size=4, rank=r,
                       k_pool=<堆叠张量 (28,12203,128,1,128)>,
                       v_pool=<堆叠张量>, tp_group=...)
        │
        ▼  Python 找到的是 _compat_init（A10 换上去的）
B1  _compat_init(self, *args, **kwargs)            （mp_connector_compat.py:102）
        │
        ├─ B1a  "k_pool" in kwargs → True
        │      kwargs["k_pool"] = _split_stacked_pool(k_pool, "k_pool")
        │          │
        │          ├─ isinstance(pool, torch.Tensor) → True
        │          ├─ pool.dim() = 5 ≥ 2 → 不放行，进入拆分
        │          ├─ layers = [pool[i] for i in range(28)]
        │          │    pool[0] → view：形状 (12203,128,1,128)，偏移 0
        │          │    pool[1] → view：同形状，偏移 +1 层的字节数
        │          │    ... 共 28 个，全部指向【同一块 10.43GB 显存】
        │          └─ logger.info("SGLang stacked k_pool buffer shape=(28, 12203, 128, 1, 128)
        │                        dtype=torch.bfloat16 split into 28 per-layer views")
        │                       ↑ 你日志里看到的那行就是这里打的
        │
        ├─ B1b  "v_pool" in kwargs → True，同上拆 v_pool
        │
        └─ B1c  return original_init(self, **kwargs)   ← 原构造，一行不改
                │
                ▼
B2  原始 LMCacheMPConnector.__init__（上游代码，multi_process_adapter.py:174）
        │
        ├─ B2a  _validate_sglang_kv_pools(list, list)
        │        │ if not k_pool → 列表非空 → False → 不抛
        │        │ len(k_pool)==len(v_pool) → 28==28 ✓
        │        │ 逐张量查 device → 全是 npu:N ✓
        │        └ 返回 device = npu:rank
        │
        ├─ B2b  get_event_ipc_backend(npu) + check_event_support
        │        └ 平台注册表分发到 NPU event backend（本仓 c_ops 的 event_recorder）
        │
        ├─ B2c  RequestClientFactory.create("127.0.0.1:5556")   ← ZMQ 建链
        │
        ├─ B2d  get_lmcache_chunk_size(req_client) → 问 server 要 chunk_size
        │        256 % 128(page_size) == 0 ✓ 对齐自检通过
        │
        ├─ B2e  _wrap_sglang_kv_caches(k_pool, v_pool)
        │        │ tensors = [*k_pool, *v_pool]  → 56 个视图
        │        ├─ get_device_spec("npu").is_lmcache_driven_available() ✓
        │        └─ [wrap_one_kv_cache(t) for t in tensors]
        │             └ resolve_kv_wrapper_factory("npu")(t)
        │                  └ NpuIPCWrapper(t)        ← 本仓 NPU IPC，包装的是
        │                                                   共享存储句柄（视图与
        │                                                   原张量同源，语义不变）
        │
        ├─ B2f  req_client.register_kv_cache(
        │          instance_id, 56个wrapper, model_name, tp_size=4,
        │          EngineType.SGLANG,
        │          {"tokens_per_block": 128, "kv_list_layout": "k_v"},
        │          [])   ← ZMQ 发往 server
        │
        └─ B2g  _start_heartbeat()   ← 你日志里的 lmcache-heartbeat 线程
                                   （TP0~3 每个 scheduler 各一条）

B3  server 侧（另一个进程）收到 REGISTER_KV_CACHE:
        ├─ 识别 EngineType.SGLANG + flat-of-2NL 布局（56 = 28K + 28V，中点拆分）
        ├─ 导入 56 个 NPU IPC 句柄
        └─ 每张卡建 cache context
          ↑ 证据：npu-smi 里 lmcache 进程在 NPU 0~3 各持 ~148MB
```

**零拷贝为什么安全（再强调一次）**：`pool[i]` 是视图不是副本。SGLang 运行时
往 KV 池写第 5 层数据 = 写 `pool[5]` 视图 = 写同一块物理显存；server 侧通过
IPC 句柄读的也是同一块 → **永远读到最新值，注册语义与"原生逐层列表"完全等价**。

### 阶段 C：请求运行时（补丁不再参与，全是上游 + 本仓数据面）

补丁只在 B 阶段出场。请求来了之后，链路是纯上游 + 本仓 c_ops：

```text
C1  冷请求（第一次见这个前缀，>256 token）
    │
    ├─ match_prefix → connector.lookup_kv(token_ids, rid)      （同步阻塞）
    │    └ ZMQ LOOKUP → server 查索引 → 命中 0 → 返回 0
    │    └ server 顺手把 L2→L1(DRAM) 预取做好，持读锁
    │
    ├─ 正常 prefill（SGLang 算出全部 KV，写进堆叠张量 = 写 28 个视图）
    │
    └─ 调度收尾 → connector.store_kv(store_md)
         └ ZMQ STORE{block_ids} → server 执行
              multi_layer_block_kv_transfer(...)        ← 上游门面
                └ lmcache_ascend.c_ops（pybind）         ← 本仓接管
                    └ AscendC mp_mem_kernels             ← 本仓 kernel
              从 NPU 显存搬到 L1（DRAM）
              ↑ 日志证据："Stored 256 tokens in 0.00x seconds"（毫秒级=原生路径，
                若退到 torch 基线约慢 8 倍）

C2  热请求（flush 本地 RadixCache 后，同前缀再来）
    │
    ├─ match_prefix → connector.lookup_kv(...)
    │    └ server：命中 N 个 chunk → 返回 host_hit_length
    │
    ├─ scheduler 调度时 → init_load_back → connector.retrieve_kv(load_md)
    │    └ ZMQ RETRIEVE{预分配的 GPU block_ids}
    │         → server 执行 multi_layer_block_kv_transfer（L1 → NPU）
    │           同样走 c_ops → AscendC kernel
    │         → 完成后经 event IPC 通知（torch.npu.Event + ipc_handle）
    │
    └─ prefill 只算未命中的尾部 → TTFT 显著下降
         ↑ 日志证据："Retrieved NNNN tokens in 0.00x seconds"

C3  请求结束 → connector.end_session(rid)   ← 释放残余读锁，防泄漏
```

**两阶段加载**（LOOKUP 与 RETRIEVE 分开）是上游设计：match_prefix 时只查不搬
（快，不阻塞调度决策），真正 dispatch 时才把 KV 灌进已分配的 GPU 槽位。

---

## 3. 函数字典（本仓新增的 4 个函数）

### 3.1 `_is_sglang_runtime()` — [\_\_init\_\_.py:31](file:///d:/proj/LMCache-Ascend/lmcache_ascend/__init__.py#L31)

| 项 | 内容 |
|---|---|
| 谁调用 | 激活段 `:225` |
| 签名 | `() -> bool` |
| 逻辑 | `"sglang" in sys.modules or any("sglang" in a for a in sys.argv)` |
| 为什么可靠 | SGLang scheduler 里 `import lmcache` 发生在 sglang 完整加载之后（阶段 A 的 A0→A2 顺序），检测时 sys.modules 必有 sglang |
| 兄弟函数 | `_is_vllm_runtime()`（:22，完全同构，本仓既有模式） |

### 3.2 `_patch_sglang_mp_connector_compat()` — [\_\_init\_\_.py:186](file:///d:/proj/LMCache-Ascend/lmcache_ascend/__init__.py#L186)

| 项 | 内容 |
|---|---|
| 谁调用 | 激活段 try 块内（:234） |
| 逻辑 | import 具体模块 → 调 `install_mp_connector_compat()` |
| 为什么存在 | 本仓组织约定：`__init__.py` 只做"何时装"的编排，"装什么"独立成文件可单测、可整体删除 |

### 3.3 `install_mp_connector_compat()` — [mp_connector_compat.py:78](file:///d:/proj/LMCache-Ascend/lmcache_ascend/integration/sglang/mp_connector_compat.py#L78)

| 项 | 内容 |
|---|---|
| 谁调用 | 3.2 |
| 六步 | ① `getattr(mpa, "LMCacheMPConnector")`，类不存在 → 静默返回（LMCache 未来删旧类也不炸）② 已装过（查 `_INIT_PATCHED_FLAG`）→ 幂等返回 ③ 存 `original_init` ④ 定义 `_compat_init` ⑤ 打标记 + 改名（traceback 友好）⑥ 类上替换 |
| 替换位置 | **类属性**上（`connector_cls.__init__ = _compat_init`）→ 之后任何代码任何位置构造该类实例都会被拦 |

### 3.4 `_split_stacked_pool()` — [mp_connector_compat.py:52](file:///d:/proj/LMCache-Ascend/lmcache_ascend/integration/sglang/mp_connector_compat.py#L52)

| 项 | 内容 |
|---|---|
| 谁调用 | `_compat_init`（对 k_pool/v_pool 各一次） |
| 守卫 | 不是张量 或 dim<2 → 原样返回（SGLang 修好改回列表时，补丁自动变 no-op） |
| 拆分 | `[pool[i] for i in range(pool.shape[0])]` 沿第 0 维（层维） |
| 副作用 | 一条 info 日志（形状+dtype+层数留痕，换模型时直接读日志验证堆叠顺序） |

### 3.5 `_compat_init`（闭包）— [mp_connector_compat.py:102](file:///d:/proj/LMCache-Ascend/lmcache_ascend/integration/sglang/mp_connector_compat.py#L102)

| 项 | 内容 |
|---|---|
| 拦截范围 | **只碰 `k_pool`/`v_pool` 两个 kwargs**，其余参数原封透传 |
| 为什么拦 kwargs | SGLang 的 `lmc_radix_cache.py` 是唯一构造点，且经 `**connector_kwargs` 展开传入 → 必为关键字参数 |

---

## 4. 前后对照总表

| 环节 | 适配前 | 适配后 | 归属 |
|---|---|---|---|
| 插件激活 | 只认 vllm，sglang 场景零动作 | sglang 分支装连接器补丁（带异常守卫） | ★本仓 `__init__.py` |
| 连接器构造 | 直达 `_validate` → RuntimeError | 先过 `_compat_init` 拆视图列表 | ★本仓 `mp_connector_compat.py` |
| `_validate_sglang_kv_pools` | 对张量求布尔值 💥 | 对列表求布尔值 ✓ | 上游 LMCache（未改一行） |
| 事件后端 | （未到达） | `get_event_ipc_backend(npu)` ✓ | 上游抽象 + 本仓 c_ops |
| KV IPC 包装 | （未到达） | `NpuIPCWrapper` × 56 | 上游门面 + 本仓实现 |
| server 注册 | （未到达） | ZMQ REGISTER → 句柄导入 + 每卡 context | 上游 + 本仓 |
| 请求期 store/retrieve | （未到达） | `multi_layer_block_kv_transfer` → AscendC kernel | 上游门面 + 本仓 c_ops |
| 慢路径风险 | — | 补丁装不上时：warning + 原错误复现（绝不退 torch 基线悄悄慢 8 倍） | ★守卫设计 |

## 5. 日志证据对照（每条日志属于哪个函数）

| 你看到的日志 | 打印它的函数 | 阶段 |
|---|---|---|
| `Seeded lazy lmcache.device_ops proxy` | `_seed_partial_lmcache_device_ops` | A |
| `Installed SGLang MP connector stacked-KV compat patch` | `install_mp_connector_compat` | A |
| `SGLang stacked k_pool buffer shape=(28,...) split into 28 per-layer views` | `_split_stacked_pool` | B |
| `Started PeriodicThread: lmcache-heartbeat` | `_start_heartbeat`（上游构造尾部） | B |
| `Tree cache initialized: impl=LMCRadixCache` | SGLang registry | B 结束 |
| `Stored 256 tokens in 0.00x seconds` | 上游 `lmcache_driven_transfer.py:834`（经 c_ops 执行） | C1 |
| `Retrieved NNNN tokens in 0.00x seconds` | 上游 `lmcache_driven_transfer.py:1100` | C2 |

## 6. 与既有文档的关系

| 文档 | 关系 |
|---|---|
| [stacked_kv_compat_patch_explained.md](stacked_kv_compat_patch_explained.md) | 补丁的"为什么这么设计"（设计视角） |
| 本文 | 补丁的"每一步怎么走"（执行视角，含前/后对比与请求期全链） |
| [control_plane_vs_data_plane.md](control_plane_vs_data_plane.md) | 阶段 C 里控制面/数据面的完整背景 |
| [qwen25_7b_first_light.md](qwen25_7b_first_light.md) | 8.1 部署手册 / 第 11 节问答与 benchmark |
